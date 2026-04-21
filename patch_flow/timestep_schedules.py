import torch
from torch import Tensor
from jaxtyping import Float
from torch.distributions.beta import Beta


class ParallelTimeSampler:
    def __call__(self, shape, device="cpu", dtype=torch.float32):
        bs = shape[0]
        t = torch.rand(bs, device=device, dtype=dtype).view(bs, 1).repeat(1, shape[1])
        return t


class ParallelLogitNormalTimeSampler:
    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        """
        Logit-Normal sampler from the paper 'Scaling Rectified Flow Transformers
        for High-Resolution Image Synthesis' - Esser et al. (ICML 2024)
        """
        self.loc = loc
        self.scale = scale

    def __call__(self, shape, device="cpu", dtype=torch.float32):
        bs = shape[0]
        t = torch.sigmoid(self.loc + self.scale * torch.randn(bs, 1)).to(device).to(dtype)
        t = t.repeat(1, shape[1])
        return t


class SRMSchedule:
    def __init__(self, beta_sharpness: float = 1.0):
        self.beta_sharpness = beta_sharpness
        self.betas: dict[int, Beta] = {}

    def init_betas(self, dim: int) -> None:
        if dim > 1 and dim not in self.betas:
            a = b = (dim - 1 - (dim % 2)) ** 1.05 * self.beta_sharpness
            self.betas[dim] = Beta(a, b)
            half_dim = dim // 2
            self.init_betas(half_dim)
            self.init_betas(dim - half_dim)

    def _get_uniform_l1_conditioned_vector_list(
        self,
        l1_norms: Float[Tensor, "batch"],
        dim: int,
    ) -> list[Float[Tensor, "batch"]]:
        if dim == 1:
            return [l1_norms]

        device = l1_norms.device
        half_cells = dim // 2

        max_first_contribution = l1_norms.clamp(max=half_cells)  # num cells in the first half
        max_second_contribution = l1_norms.clamp(max=dim - half_cells)
        min_first_contribution = (l1_norms - max_second_contribution).clamp_(min=0)

        random_matrix = self.betas[dim].sample((l1_norms.shape[0],)).to(device=device)
        ranges = max_first_contribution - min_first_contribution

        assert ranges.min() >= 0
        first_contribution = min_first_contribution + ranges * random_matrix
        second_contribution = l1_norms - first_contribution

        return self._get_uniform_l1_conditioned_vector_list(
            first_contribution, half_cells
        ) + self._get_uniform_l1_conditioned_vector_list(second_contribution, dim - half_cells)

    def _sample_time_matrix(self, l1_norms: Float[Tensor, "batch"], dim: int) -> Float[Tensor, "batch dim"]:
        vector_list = self._get_uniform_l1_conditioned_vector_list(l1_norms, dim)
        t = torch.stack(vector_list, dim=1)  # [batch_size, dim]
        # shuffle the time matrix (independently for batch elements) to avoid positional biases
        idx = torch.rand_like(t).argsort()
        t = t.gather(1, idx)
        return t

    def get_time_with_mean(self, mean: Float[Tensor, "b"], dim: int) -> Float[Tensor, "b d"]:
        bs = mean.shape[0]
        self.init_betas(dim)
        l1_norms = mean.flatten() * dim
        t = self._sample_time_matrix(l1_norms, dim)
        return t.view(bs, -1)

    def get_time(self, shape, device="cpu", dtype=torch.float32):
        bs, seq_len = shape
        mean = torch.rand((bs,), device=device, dtype=dtype)
        return self.get_time_with_mean(mean, dim=seq_len)

    def __call__(self, shape, device="cpu", dtype=torch.float32):
        return self.get_time(shape, device=device, dtype=dtype)


class GaussianSchedule:
    def __init__(self, std: float = 0.2):
        self.std = std

    def __call__(self, shape, device="cpu", dtype=torch.float32):
        bs, dim = shape
        t_bar = torch.rand(bs, device=device, dtype=dtype)
        t_i = self.get_time_with_mean(t_bar, dim=dim)
        return t_i

    def get_time_with_mean(self, mean, dim: int):
        bs = mean.shape[0]
        std = torch.min(mean, 1 - mean)
        std = torch.min(std / 2, torch.full_like(std, self.std))
        t_i = mean[:, None] + torch.randn(bs, dim, device=mean.device, dtype=mean.dtype) * std[:, None]
        t_i = t_i.clamp(0, 1)
        return t_i


class TruncatedGaussian:
    def __init__(self, std: float = 0.2):
        self.std = std

    def __call__(self, shape, device="cpu", dtype=torch.float32):
        bs, dim = shape
        t_bar = torch.rand(bs, device=device, dtype=dtype)
        t_i = self.get_time_with_mean(t_bar, dim=dim)
        return t_i

    def get_time_with_mean(self, mean, dim: int):
        bs = mean.shape[0]
        std = torch.min(mean / 2, torch.full_like(mean, self.std)) * -1
        t_i = mean[:, None] + torch.randn(bs, dim, device=mean.device, dtype=mean.dtype).abs() * std[:, None]

        # t_i = t_i.clamp(0, 1) <-- Nah we don't do clamping, we reset negative values to uniform samples
        rand = torch.rand_like(t_i)
        t_i = torch.where(t_i < 0, rand * mean[:, None], t_i)
        return t_i


class LogitNormalTruncatedGaussian:
    def __init__(self, std: float = 0.6, loc: float = 0.7, scale: float = 1.0):
        self.std = std
        self.loc = loc
        self.scale = scale

    def get_t_bar(self, bs, device="cpu", dtype=torch.float32):
        return torch.sigmoid(self.loc + self.scale * torch.randn(bs, device=device, dtype=dtype))

    def get_time_with_mean(self, mean, dim: int):
        bs = mean.shape[0]
        std = torch.min(mean / 2, torch.full_like(mean, self.std)) * -1
        t_i = mean[:, None] + torch.randn(bs, dim, device=mean.device, dtype=mean.dtype).abs() * std[:, None]

        # t_i = t_i.clamp(0, 1) <-- Nah we don't do clamping, we reset negative values to uniform samples
        rand = torch.rand_like(t_i)
        t_i = torch.where(t_i < 0, rand * mean[:, None], t_i)
        return t_i

    def __call__(self, shape, device="cpu", dtype=torch.float32):
        bs, dim = shape
        t_bar = self.get_t_bar(bs, device=device, dtype=dtype)
        t_i = self.get_time_with_mean(t_bar, dim=dim)
        return t_i
