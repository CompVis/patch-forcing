import torch
import einops
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
from functools import partial

from jutils import instantiate_from_config


def exists(x):
    return x is not None


def pad_v_like_x(v_, x_):
    """
    Function to reshape the vector by the number of dimensions
    of x. E.g. x (bs, c, h, w), v (bs) -> v (bs, 1, 1, 1).
    """
    if isinstance(v_, float):
        return v_
    return v_.reshape(-1, *([1] * (x_.ndim - 1)))


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


""" Timestep Sampler """


class LogitNormalSampler:
    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        """
        Logit-Normal sampler from the paper 'Scaling Rectified Flow Transformers
        for High-Resolution Image Synthesis' - Esser et al. (ICML 2024)
        """
        self.loc = loc
        self.scale = scale

    def __call__(self, n, device="cpu", dtype=torch.float32):
        return torch.sigmoid(self.loc + self.scale * torch.randn(n)).to(device).to(dtype)


""" Flow Model """


class Flow:
    def __init__(
        self,
        timestep_sampler: dict = None,
    ):
        """
        Flow Matching, Stochastic Interpolants, or Rectified Flow model. :)

        Args:
            sigma_min: a float representing the standard deviation of the
                Gaussian distribution around the mean of the probability
                path N(t * x1 + (1 - t) * x0, sigma), as used in [1].
            timestep_sampler: dict, configuration for the training timestep sampler.

        References:
            [1] Lipman et al. (2023). Flow Matching for Generative Modeling.
            [2] Tong et al. (2023). Improving and generalizing flow-based
                generative models with minibatch optimal transport.
            [3] Ma et al. (2024). SiT: Exploring flow and diffusion-based
                generative models with scalable interpolant transformers.
        """
        if timestep_sampler is not None:
            self.t_sampler = instantiate_from_config(timestep_sampler)
        else:
            self.t_sampler = torch.rand  # default: uniform U(0, 1)

    def generate(
        self,
        model: nn.Module,
        x: Tensor,
        num_steps: int = 50,
        reverse=False,
        return_intermediates=False,
        progress=True,
        **kwargs,
    ):
        """Classic Euler sampling from x0 to x1 in num_steps.

        Args:
            x: source minibatch (bs, *dim)
            num_steps: int, number of steps to take
            reverse: bool, whether to reverse the direction of the flow. If True,
                we map from x1 -> x0, otherwise we map from x0 -> x1.
            return_intermediates: bool, if true, return list of samples
            progress: bool, if true, show tqdm progress bar
            kwargs: additional arguments for the network (e.g. conditioning information).
        """
        bs, dev = x.shape[0], x.device

        # include cfg
        sample_fn = partial(forward_with_cfg, model=model)

        timesteps = torch.linspace(0, 1, num_steps + 1)
        if reverse:
            timesteps = 1 - timesteps

        xt = x
        intermediates = [xt]
        for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), disable=not progress, total=len(timesteps) - 1):
            t = torch.ones((bs,), dtype=x.dtype, device=dev) * t_curr
            pred = sample_fn(xt, t, **kwargs)

            dt = t_next - t_curr
            xt = xt + dt * pred

            if return_intermediates:
                intermediates.append(xt)

        if return_intermediates:
            return torch.stack(intermediates, 0)
        return xt

    """ Training """

    def compute_xt(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Sample from the time-dependent density p_t
            xt ~ N(alpha_t * x1 + sigma_t * x0, sigma_min * I),
        according to Eq. (1) in [3] and for the linear schedule Eq. (14) in [2].

        Args:
            x0 : shape (bs, *dim), represents the source minibatch (noise)
            x1 : shape (bs, *dim), represents the target minibatch (data)
            t  : shape (bs,) represents the time in [0, 1]
        Returns:
            xt : shape (bs, *dim), sampled point along the time-dependent density p_t
        """
        t = pad_v_like_x(t, x0)
        xt = t * x1 + (1 - t) * x0
        return xt

    def compute_ut(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Compute the time-dependent conditional vector field
            ut = alpha_dt_t * x1 + sigma_dt_t * x0,
        see Eq. (7) in [3].

        Args:
            x0 : Tensor, shape (bs, *dim), represents the source minibatch (noise)
            x1 : Tensor, shape (bs, *dim), represents the target minibatch (data)
            t  : FloatTensor, shape (bs,) represents the time in [0, 1]
        Returns:
            ut : conditional vector field
        """
        return x1 - x0

    def get_interpolants(self, x1: Tensor, x0: Tensor = None, t: Tensor = None):
        """
        Args:
            x1: shape (bs, *dim), represents the target minibatch (data)
            x0: shape (bs, *dim), represents the source minibatch. If None,
                we sample x0 from a standard normal distribution.
            t : shape (bs,), represents the time in [0, 1]. If None,
                we sample t using self.t_sampler (default: U(0, 1)).
        Returns:
            xt: shape (bs, *dim), sampled point along the time-dependent density p_t
            ut: shape (bs, *dim), conditional vector field
            t : shape (bs,), represents the time in [0, 1]
        """
        if not exists(x0):
            x0 = torch.randn_like(x1)
        if not exists(t):
            t = self.t_sampler(x1.shape[0], device=x1.device, dtype=x1.dtype)

        xt = self.compute_xt(x0=x0, x1=x1, t=t)
        ut = self.compute_ut(x0=x0, x1=x1, t=t)

        return xt, ut, t

    def training_losses(self, model: nn.Module, x1: Tensor, x0: Tensor = None, **cond_kwargs):
        """
        Args:
            x1: shape (bs, *dim), represents the target minibatch (data)
            x0: shape (bs, *dim), represents the source minibatch, if None
                we sample x0 from a standard normal distribution.
            cond_kwargs: additional arguments for the conditional flow
                network (e.g. conditioning information)
        Returns:
            loss: scalar, the training loss for the flow model
        """
        xt, ut, t = self.get_interpolants(x1=x1, x0=x0)
        vt = model(x=xt, t=t, **cond_kwargs)

        return (vt - ut).square().mean()

    def validation_losses(self, model: nn.Module, x1: Tensor, x0: Tensor = None, num_segments: int = 8, **cond_kwargs):
        """
        SD3 & Meta Movie Gen show that val loss correlates well with human quality. They
        compute the loss in equidistant segments in (0, 1) to reduce variance and average
        them afterwards. Default number of segments: 8 (Esser et al., page 21, ICML 2024).
        """
        assert num_segments > 0, "Number of segments must be greater than 0"

        if not exists(x0):
            x0 = torch.randn_like(x1)
        ts = torch.linspace(0, 1, num_segments + 1)[:-1] + 1 / (2 * num_segments)

        losses_per_segment = []
        for t in ts:
            t = torch.ones(x1.shape[0], device=x1.device) * t
            xt, ut, t = self.get_interpolants(x1=x1, x0=x0, t=t)
            vt = model(x=xt, t=t, **cond_kwargs)
            losses_per_segment.append((vt - ut).square().mean())

        losses_per_segment = torch.stack(losses_per_segment)
        return losses_per_segment.mean(), losses_per_segment
