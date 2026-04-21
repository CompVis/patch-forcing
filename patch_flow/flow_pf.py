import torch
import torch.nn as nn
from torch import Tensor

import einops
from tqdm import tqdm
from jaxtyping import Float
from functools import partial
from typing import Tuple, Optional

from jutils import instantiate_from_config


# ===================================================================================================
# utility functions


def exists(x):
    return x is not None


def pad_v_like_x(v_, x_):
    """
    Reshape or broadcast v_ to match the number of dimensions of x_ by appending singleton dims.
    - x_: (b, c, h, w), v_: (b,) -> (b, 1, 1, 1)
    - x_: (b, c, f, h, w), v_: (b, 1, f) -> (b, 1, f, 1, 1)
    """
    if isinstance(v_, (float, int)):
        return v_
    while v_.ndim < x_.ndim:
        v_ = v_.unsqueeze(-1)
    return v_


def forward_with_cfg(
    x, t, model, cfg_scale=1.0, uc_cond=None, cond_key="y", t_min: float = None, t_max: float = None, **model_kwargs
):
    """Function to include sampling with Classifier-Free Guidance (CFG) and Interval Guidance (IG)"""
    if cfg_scale == 1.0:  # without CFG
        return model(x, t, **model_kwargs)
    else:  # with CFG
        if t_min is not None and t_max is not None:  # with interval guidance
            assert torch.allclose(t, t[0]), "Time t should be the same across the batch for interval guidance"
            assert t_min < t_max, "t_min should be smaller than t_max for interval guidance"
            t_val = t[0].item()
            if not t_min <= t_val <= t_max:  # no cfg outside of the interval
                return model(x, t, **model_kwargs)
        assert cond_key in model_kwargs, f"Condition key '{cond_key}' for CFG not found in model_kwargs"
        assert uc_cond is not None, "Unconditional condition not provided for CFG"
        kwargs = model_kwargs.copy()
        c = kwargs[cond_key]
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        if uc_cond.shape[0] == 1:
            uc_cond = einops.repeat(uc_cond, "1 ... -> bs ...", bs=x.shape[0])
        c_in = torch.cat([uc_cond, c])
        kwargs[cond_key] = c_in
        model_uc, model_c = model(x_in, t_in, **kwargs).chunk(2)
        return model_uc + cfg_scale * (model_c - model_uc)


def compute_xt_patched(
    x1: Tensor,  # (b, c, h, w)  data / target
    t: Tensor,  # (b, n) or (b, gh, gw)   timesteps in [0, 1]
    patch_size: Tuple[int, int],  # (ph, pw)
    x0: Optional[Tensor] = None,  # (b, c, h, w); if None, sampled ~ N(0, I)
):
    assert x1.ndim == 4, f"Expected x1 of shape (b, c, h, w), got {x1.shape}"
    b, c, h, w = x1.shape
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    ph, pw = patch_size
    assert h % ph == 0 and w % pw == 0, f"h,w must be divisible by patch size; got {(h,w)} vs {(ph,pw)}"

    gh, gw = h // ph, w // pw  # grid of patches

    if x0 is None:
        x0 = torch.randn_like(x1)
    assert x0.shape == x1.shape, f"x0 must have shape {x1.shape}, got {x0.shape}"

    # normalize t to (b, gh, gw)
    if t.ndim == 2:
        n = gh * gw
        assert t.shape[1] == n, f"t has {t.shape[1]} tokens but expected {n} (gh*gw)"
        t_grid = t.view(b, gh, gw)
    elif t.ndim == 3:
        assert t.shape[1:] == (gh, gw), f"t must be (b, gh, gw); got {t.shape}"
        t_grid = t
    else:
        raise AssertionError(f"t must be (b, n) or (b, gh, gw); got shape {t.shape}")

    # reshape into (b, c, gh, gw, ph, pw)
    def _patchify(x: Tensor) -> Tensor:
        x = x.view(b, c, gh, ph, gw, pw)  # (b, c, gh, ph, gw, pw)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (b, c, gh, gw, ph, pw)
        return x

    def _unpatchify(xp: Tensor) -> Tensor:
        xp = xp.permute(0, 1, 2, 4, 3, 5).contiguous()  # (b, c, gh, ph, gw, pw)
        return xp.view(b, c, h, w)

    x1_p = _patchify(x1)
    x0_p = _patchify(x0)

    # Broadcast t_grid to patches: (b, 1, gh, gw, 1, 1)
    t_b = t_grid.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

    # Interpolate per patch
    xt_p = t_b * x1_p + (1.0 - t_b) * x0_p

    xt = _unpatchify(xt_p)
    return xt


def pad_v_like_x_patches(
    v: Tensor, x_like: Tensor, patch_size: Tuple[int, int]  # (b, f) or (b, gh, gw)  # (b, c, h, w)  # (ph, pw)
) -> Tensor:
    """
    Broadcast a per-patch tensor v onto an image-like tensor x_like.

    Returns:
        v_img: (b, 1, h, w), where each (ph, pw) patch is filled with the
               corresponding scalar from v.
    """
    assert x_like.ndim == 4, f"x_like should be (b,c,h,w), got {x_like.shape}"
    b, _, h, w = x_like.shape
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    ph, pw = patch_size
    assert h % ph == 0 and w % pw == 0, "h,w must be divisible by patch size"
    gh, gw = h // ph, w // pw

    if v.ndim == 2:
        # (b, f) -> (b, gh, gw)
        f = gh * gw
        assert v.shape[1] == f, f"v has {v.shape[1]} tokens, expected {f}"
        v = v.view(b, gh, gw)
    else:
        assert v.shape == (b, gh, gw), f"v must be (b, gh, gw), got {v.shape}"

    # expand each token into its spatial patch
    v_img = (
        v.unsqueeze(1)  # (b, 1, gh, gw)
        .unsqueeze(-1)
        .unsqueeze(-1)  # (b, 1, gh, gw, 1, 1)
        .expand(b, 1, gh, gw, ph, pw)
    )
    # patch grid back to image
    v_img = einops.rearrange(v_img, "b 1 gh gw ph pw -> b 1 (gh ph) (gw pw)")
    return v_img


# ===================================================================================================
# Patchified Diffusion Forcing


class PatchFlowForcing:
    def __init__(self, timestep_sampler: dict = None, patch_size: int = 2):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        if timestep_sampler is None:
            self.t_sampler = torch.rand
        else:
            self.t_sampler = instantiate_from_config(timestep_sampler)

    """ Training """

    def compute_xt(self, x0: Tensor, x1: Tensor, t: Tensor):
        if x0 is None:
            x0 = torch.randn_like(x1)

        assert x1.shape == x0.shape, f"x0 and x1 must have the same shape, got {x0.shape} vs {x1.shape}"
        assert x1.ndim == 4, f"Expected x1 of shape (b, c, h, w), got {x1.shape}"
        b, c, h, w = x1.shape

        ph, pw = self.patch_size
        assert h % ph == 0 and w % pw == 0, f"(h, w) must be divisible by patch size; got {(h,w)} vs {(ph,pw)}"
        gh, gw = h // ph, w // pw  # grid of patches

        # normalize t to (b, gh, gw)
        if t.ndim == 2:
            n = gh * gw
            assert t.shape[1] == n, f"t has {t.shape[1]} tokens but expected {n} (gh*gw)"
            t_grid = t.view(b, gh, gw)
        elif t.ndim == 3:
            assert t.shape[1:] == (gh, gw), f"t must be (b, gh, gw); got {t.shape}"
            t_grid = t
        else:
            raise AssertionError(f"t must be (b, n) or (b, gh, gw); got shape {t.shape}")

        # reshape into (b, c, gh, gw, ph, pw)
        def _patchify(x: Tensor) -> Tensor:
            x = x.view(b, c, gh, ph, gw, pw)  # (b, c, gh, ph, gw, pw)
            x = x.permute(0, 1, 2, 4, 3, 5)  # (b, c, gh, gw, ph, pw)
            return x

        def _unpatchify(xp: Tensor) -> Tensor:
            xp = xp.permute(0, 1, 2, 4, 3, 5).contiguous()  # (b, c, gh, ph, gw, pw)
            return xp.view(b, c, h, w)

        x1_p = _patchify(x1)
        x0_p = _patchify(x0)

        # Broadcast t_grid to patches: (b, 1, gh, gw, 1, 1)
        t_b = t_grid.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        # Interpolate per patch
        xt_p = t_b * x1_p + (1.0 - t_b) * x0_p

        xt = _unpatchify(xt_p)
        return xt

    def compute_ut(self, x0: Tensor, x1: Tensor, t: Tensor = None):
        return x1 - x0

    def get_interpolants(self, x1: Tensor, x0: Tensor = None, t: Tensor = None):
        b, c, h, w = x1.shape
        if not exists(x0):
            x0 = torch.randn_like(x1)

        ph, pw = self.patch_size
        assert h % ph == 0 and w % pw == 0, f"(h, w) must be divisible by patch size; got {(h,w)} vs {(ph,pw)}"

        f = (h // ph) * (w // pw)  # number of patches
        if not exists(t):
            t = self.t_sampler((b, f), device=x1.device, dtype=x1.dtype)
        assert t.ndim == 2, f"Expected t to have shape (bs, f), got {t.shape}"
        assert t.shape[1] == f, f"Expected t to have {f} timesteps, got {t.shape}"

        xt = self.compute_xt(x0, x1, t)
        ut = self.compute_ut(x0, x1, t)

        return xt, ut, t

    """ Validation and Generation """

    def validation_losses(
        self,
        model: nn.Module,
        x1: Float[Tensor, "bs c h w"],
        x0: Float[Tensor, "bs c h w"] = None,
        num_segments: int = 8,
        **cond_kwargs,
    ):
        """
        SD3 & Meta Movie Gen show that val loss correlates well with human quality. They
        compute the loss in equidistant segments in (0, 1) to reduce variance and average
        them afterwards. Default number of segments: 8 (Esser et al., page 21, ICML 2024).
        """
        assert num_segments > 0, "Number of segments must be greater than 0"

        bs, c, h, w = x1.shape
        ph, pw = self.patch_size
        f = (h // ph) * (w // pw)  # number of patches

        if not exists(x0):
            x0 = torch.randn_like(x1)
        ts = torch.linspace(0, 1, num_segments + 1)[:-1] + 1 / (2 * num_segments)

        losses_per_segment = []
        for t in ts:
            t = torch.ones((bs, f), device=x1.device) * t

            xt, ut, t = self.get_interpolants(x1=x1, x0=x0, t=t)
            vt = model(x=xt, t=t, **cond_kwargs)
            losses_per_segment.append((vt - ut).square().mean())

        losses_per_segment = torch.stack(losses_per_segment)
        return losses_per_segment.mean(), losses_per_segment

    def integrate_conditioning(
        self,
        x: Float[Tensor, "bs c h w"],
        denoise_schedule: Float[Tensor, "t f"],
        x_cond: Float[Tensor, "bs c h w"] = None,
    ):
        first_row = denoise_schedule[0, :]  # (f,)

        # complete denoising, no conditioning
        if torch.all(first_row == 0.0):
            return x

        assert x_cond is not None, "x_cond must be provided to integrate conditioning information"
        assert x_cond.shape == x.shape, f"Expected x_cond to have the same shape as x, got {x_cond.shape} and {x.shape}"

        # mix x and x_cond according to the denoising schedule at t=0
        t_batched = einops.repeat(first_row, "f -> b f", b=x.shape[0])
        xt = self.compute_xt(x0=x, x1=x_cond, t=t_batched)

        return xt

    def generate(
        self,
        model: nn.Module,
        x: Float[Tensor, "bs c h w"],
        x_cond: Float[Tensor, "bs c h w"] = None,  # clean sample for conditioning
        num_steps: int = 50,
        denoise_schedule: Float[Tensor, "t f"] = None,
        return_intermediates: bool = False,
        progress: bool = True,
        allow_negative_dt: bool = False,
        **kwargs,
    ):
        """
        Classic Euler sampling from x0 to x1 in num_steps.

        Args:
            model: nn.Module, the flow model to use for sampling
            x: source minibatch (bs, c, h, w), usually noise
            x_cond: conditioning minibatch (bs, c, h, w), usually clean sample
            num_steps: int, number of steps to take (only if denoise_schedule is None)
            denoise_schedule: shape (num_steps, f), denoise schedule for each step and frame f. If
                None, it creates a full sequence denoise schedule with num_steps
            return_intermediates: bool, if true, return list of intermediate samples
            progress: bool, if true, show tqdm progress bar
            allow_negative_dt: bool, if true, allow negative time steps (e.g. for reverse sampling),
                but otherwise clamp them to 0.0 (e.g. when we use predicted frames as conditioning
                and want to avoid treating them as ground truth)
            kwargs: additional arguments for the network (e.g. conditioning information)
        """
        dev = x.device
        bs, c, h, w = x.shape
        ph, pw = self.patch_size
        f = (h // ph) * (w // pw)  # number of patches

        if denoise_schedule is None:
            denoise_schedule = torch.linspace(0, 1, num_steps + 1)
            denoise_schedule = einops.repeat(denoise_schedule, "t -> t f", f=f)

        assert (
            denoise_schedule.shape[1] == f
        ), f"Expected denoise_schedule to have {f} frames, got {denoise_schedule.shape[1]}"
        denoise_schedule = denoise_schedule.to(dev)

        # integrate conditioning information (e.g. clean frames)
        x = self.integrate_conditioning(x=x, x_cond=x_cond, denoise_schedule=denoise_schedule)

        # include cfg
        sample_fn = partial(forward_with_cfg, model=model)

        xt = x
        intermediates = [xt]
        for t_curr, t_next in tqdm(
            zip(denoise_schedule[:-1], denoise_schedule[1:]), disable=not progress, total=len(denoise_schedule) - 1
        ):
            t = torch.ones((bs, 1), dtype=x.dtype, device=dev) * t_curr
            pred = sample_fn(xt, t, **kwargs)

            dt = t_next - t_curr
            if not allow_negative_dt:
                dt = torch.clamp(dt, min=0.0)
            dt = einops.repeat(dt, "f -> b f", b=bs)

            dt_grid = pad_v_like_x_patches(dt, pred, patch_size=self.patch_size)
            xt = xt + dt_grid * pred

            if return_intermediates:
                intermediates.append(xt)

        if return_intermediates:
            return torch.stack(intermediates, 0)
        return xt
