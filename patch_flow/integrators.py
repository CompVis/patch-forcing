import torch
import einops
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F
from abc import ABC, abstractmethod

from patch_flow.flow_pf import pad_v_like_x_patches


# ===================================================================================================


def forward_with_cfg_and_uncertainty(x, t, model, cfg_scale=1.0, uc_cond=None, cond_key="y", **model_kwargs):
    """Function to include sampling with Classifier-Free Guidance (CFG)"""
    if cfg_scale == 1.0:  # without CFG
        model_output = model(x, t, **model_kwargs, return_uncertainty=True)
        model_vt, model_uq = model_output
        out = {"vt": model_vt, "uq": model_uq, "uq_uc": None, "uq_c": model_uq, "vt_uc": None, "vt_c": model_vt}

    else:  # with CFG
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
        model_output = model(x_in, t_in, **kwargs, return_uncertainty=True)
        model_vt, model_uq = model_output
        model_vt_uc, model_vt_c = model_vt.chunk(2)
        model_uq_uc, model_uq_c = model_uq.chunk(2)
        guided_vt = model_vt_uc + cfg_scale * (model_vt_c - model_vt_uc)
        guided_uq = model_uq_uc + cfg_scale * (model_uq_c - model_uq_uc)

        out = {
            "vt": guided_vt,
            "uq": guided_uq,
            "uq_uc": model_uq_uc,
            "uq_c": model_uq_c,
            "vt_uc": model_vt_uc,
            "vt_c": model_vt_c,
        }

    return out


def forward_with_cfg(x, t, model, cfg_scale=1.0, uc_cond=None, cond_key="y", **model_kwargs):
    """Function to include sampling with Classifier-Free Guidance (CFG)"""
    if cfg_scale == 1.0:  # without CFG
        model_output = model(x, t, **model_kwargs)

    else:  # with CFG
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
        model_output = model_uc + cfg_scale * (model_c - model_uc)

    return model_output


def patch_reduce_pool(x: torch.Tensor, n: int, mode: str = "mean"):
    """Patch reducing with pooling (downsampling)"""
    if mode == "mean":
        return F.avg_pool2d(x, kernel_size=n, stride=n)
    elif mode == "max":
        return F.max_pool2d(x, kernel_size=n, stride=n)
    elif mode == "min":
        return -F.max_pool2d(-x, kernel_size=n, stride=n)
    else:
        raise ValueError("mode must be 'mean', 'max', or 'min'")


def patch_reduce(x: torch.Tensor, n: int, mode: str = "mean"):
    """Patch reduce and upsample to original size"""
    y = patch_reduce_pool(x, n=n, mode=mode)
    y = y.repeat_interleave(n, dim=-1).repeat_interleave(n, dim=-2)
    assert y.shape == x.shape
    return y


# ======================================================================================


class SamplerBase(ABC):
    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __call__(self, model, x, timesteps: list[float], progress: bool = True, **kwargs): ...


# ======================================================================================
# Base samplers


def euler(model, x, timesteps: list[float], progress=True, **kwargs):
    bs, dev = x.shape[0], x.device

    xt = x
    for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), disable=not progress, total=len(timesteps) - 1):
        t = torch.ones((bs,), dtype=x.dtype, device=dev) * t_curr
        pred = model(xt, t, **kwargs)

        dt = t_next - t_curr
        xt = xt + dt * pred

    return xt


class Euler(SamplerBase):
    def __repr__(self):
        return "Euler"

    def __call__(self, model, x, timesteps: list[float], progress=True, **kwargs):
        model_fn = partial(forward_with_cfg, model=model)
        return euler(model_fn, x, timesteps, progress=progress, **kwargs)


# ======================================================================================
# Patch Forcing samplers


class EulerPF(SamplerBase):
    """Default Euler sampler, ignores uncertainty"""

    def __init__(self, patch_size: int = 2):
        self.patch_size = patch_size

    def __repr__(self):
        return "EulerPF"

    def __call__(self, model, x, timesteps: list[float], progress=True, **kwargs):
        dev = x.device
        bs, c, h, w = x.shape
        f = (h // self.patch_size) * (w // self.patch_size)

        # prepare sample function
        sample_fn = partial(forward_with_cfg_and_uncertainty, model=model)

        xt = x
        for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), disable=not progress, total=len(timesteps) - 1):
            t = torch.ones((bs,), dtype=x.dtype, device=dev) * t_curr
            # Here we broadcast to (b, n)
            t = einops.repeat(t, "b -> b f", f=f)
            pred = sample_fn(xt, t, **kwargs)
            pred = pred["vt"]

            dt = t_next - t_curr
            xt = xt + dt * pred

        return xt


# ======================================================================================
# Uncertainty-aware samplers


class DualLoopSampler(SamplerBase):
    def __init__(self, p: float = 0.7, n_inner: int = 4, mode: str = "mean", patch_size: int = 2):
        """
        Args:
            p: percentile for thresholding uncertainty. All patches with uncertainty
                lower than the p-th percentile will be considered certain. So lower
                p -> more restrictive (fewer certain patches), e.g. p=0.8 means 80%
                of patches are considered certain (20% uncertain).
            n_inner: Number of inner steps, per big step.
        """
        self.p = p
        self.n_inner = n_inner  # inner steps for uncertain patches
        self.patch_size = patch_size
        self.mode = mode
        assert 0.0 < p < 1.0, "p must be in (0, 1)"

    def __repr__(self):
        return f"DualLoop-p{self.p*100:.0f}-inner{self.n_inner}"

    def compute_mask(self, uq):
        uq_flat = uq.reshape(uq.shape[0], -1).double()
        thresh = torch.quantile(uq_flat, self.p, dim=-1)  # (bs,)
        thresh_exp = einops.repeat(thresh, "b -> b 1 1 1")
        uq_mask = uq < thresh_exp  # 0: uncertain (inner loop), 1: certain (forward)
        return uq_mask

    def __call__(self, model, x, timesteps: list[float], progress=True, **kwargs):
        dev = x.device
        bs, c, h, w = x.shape
        f = (h // self.patch_size) * (w // self.patch_size)

        # make denoising schedule
        num_steps = len(timesteps) - 1
        denoise_schedule = torch.linspace(0, 1, num_steps + 1)
        denoise_schedule = einops.repeat(denoise_schedule, "t -> t f", f=f).to(dev)
        assert denoise_schedule.shape[1] == f

        # prepare sample function
        sample_fn = partial(forward_with_cfg_and_uncertainty, model=model)

        # sampling loop
        xt = x
        for t_curr, t_next in tqdm(
            zip(denoise_schedule[:-1], denoise_schedule[1:]), total=len(denoise_schedule) - 1, disable=not progress
        ):
            t = einops.repeat(t_curr, "f -> b f", b=bs)

            model_out = sample_fn(xt, t, **kwargs)
            pred = model_out["vt"]

            dt = t_next - t_curr
            dt = torch.clamp(dt, min=0.0)
            dt = einops.repeat(dt, "f -> b f", b=bs)
            dt_grid = pad_v_like_x_patches(dt, pred, patch_size=self.patch_size)

            # x1 prediction from xt (not used during inference)
            # dt_x1 = (1 - t)
            # dt_x1_grid = pad_v_like_x_patches(dt_x1, pred, patch_size=self.patch_size)
            # x1_pred = xt + dt_x1_grid * pred

            # # update xt       # NORMALLY USE THIS
            # xt = xt + dt_grid * pred

            # ============================================= inner loop update xt
            # with mask
            uq = model_out["uq"].exp()
            uq = patch_reduce(uq, n=2, mode=self.mode)
            uq_mask = self.compute_mask(uq)

            dt_inner_grid = dt_grid / self.n_inner

            xt = xt + dt_grid * pred * uq_mask + dt_inner_grid * pred * (~uq_mask)

            t_grid = pad_v_like_x_patches(t, pred, patch_size=self.patch_size)
            t_grid = t_grid + dt_grid * uq_mask + dt_inner_grid * (~uq_mask)

            for _ in range(self.n_inner - 1):
                t_inp = patch_reduce_pool(t_grid, n=2, mode="mean")
                t_inp = einops.rearrange(t_inp, "b 1 h w -> b (h w)")
                model_out_inner = sample_fn(xt, t_inp, **kwargs)
                pred = model_out_inner["vt"]
                xt = xt + dt_inner_grid * pred * (~uq_mask)
                t_grid = t_grid + dt_inner_grid * (~uq_mask)

        return xt


class LookAheadSampler(SamplerBase):
    def __init__(self, p: float = 0.4, mode: str = "mean", patch_size: int = 2, context_t_ratio: int = 1.5):
        """
        Context-guidance on uncertain patches during sampling. For certain patches, use one-step prediction for better context for uncertain patches.
        """
        self.p = p
        self.patch_size = patch_size
        self.mode = mode
        self.context_t_ratio = context_t_ratio
        assert 0.0 < p < 1.0, "p must be in (0, 1)"

    def __repr__(self):
        return f"LookAheadSampler-p{self.p*100:.0f}-context{self.context_t_ratio:.2f}"

    def compute_mask(self, uq):
        uq_flat = uq.reshape(uq.shape[0], -1).double()
        thresh = torch.quantile(uq_flat, self.p, dim=-1)  # (bs,)
        thresh_exp = einops.repeat(thresh, "b -> b 1 1 1")
        uq_mask = uq < thresh_exp  # 0: uncertain (use context), 1: certain (use model)
        return uq_mask

    def __call__(self, model, x, timesteps: list[float], progress=True, **kwargs):
        dev = x.device
        bs, c, h, w = x.shape
        f = (h // self.patch_size) * (w // self.patch_size)

        # make denoising schedule
        num_steps = len(timesteps) - 1
        denoise_schedule = torch.linspace(0, 1, num_steps + 1)
        denoise_schedule = einops.repeat(denoise_schedule, "t -> t f", f=f).to(dev)
        assert denoise_schedule.shape[1] == f

        # prepare sample function
        sample_fn = partial(forward_with_cfg_and_uncertainty, model=model)

        # sampling loop
        xt = x
        for t_curr, t_next in tqdm(
            zip(denoise_schedule[:-1], denoise_schedule[1:]), total=len(denoise_schedule) - 1, disable=not progress
        ):
            t = einops.repeat(t_curr, "f -> b f", b=bs)

            # No CFG for context prediction
            model_out = sample_fn(xt, t, **kwargs)
            pred = model_out["vt"]
            pred_c = model_out["vt_c"]

            dt = t_next - t_curr
            dt = torch.clamp(dt, min=0.0)
            dt = einops.repeat(dt, "f -> b f", b=bs)
            dt_grid = pad_v_like_x_patches(dt, pred, patch_size=self.patch_size)

            # normal step, no context guidance
            if t_curr.mean() <= 0.05:
                xt = xt + dt_grid * pred_c
                continue

            # =============================================
            uq = model_out["uq"].exp()
            uq = patch_reduce(uq, n=2, mode=self.mode)
            low_uq_mask = self.compute_mask(uq)
            high_uq_mask = ~low_uq_mask
            low_uq_pool_mask = patch_reduce_pool(low_uq_mask.float(), n=self.patch_size, mode=self.mode).bool()

            # one step prediction for certain patches
            t_context = t_curr * self.context_t_ratio
            t_context = torch.clamp(t_context, max=1.0)
            dt_context = t_context - t_curr
            dt_context = einops.repeat(dt_context, "f -> b f", b=bs)
            dt_context_grid = pad_v_like_x_patches(dt_context, pred, patch_size=self.patch_size)

            pred_context = pred_c * low_uq_mask
            xt_context = xt + dt_context_grid * pred_context
            t_context = t + dt_context * low_uq_pool_mask.view(bs, -1)

            # context prediction
            pred_context = sample_fn(xt_context, t_context, **kwargs)["vt"]

            # update xt
            xt = xt + dt_grid * pred * low_uq_mask + dt_grid * pred_context * high_uq_mask

        return xt
