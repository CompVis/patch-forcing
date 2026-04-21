import math
import torch
import torch.nn as nn
from jaxtyping import Float
from einops import rearrange, repeat

from jutils.nn.transformer import TimestepEmbedder
from jutils.nn.rope import make_axial_pos_2d, AxialRoPEBase
from jutils.nn.transformer import TransformerLayer, TokenMerge2D, TokenSplitLast2D


def make_axial_pos_2d_with_meta(meta, size, device="cpu", latent_ds_factor=8):
    """
    Args:
        meta: dict with keys 'top', 'left', 'orig_h', 'orig_w'
        size: int, size of the square patch
        device: device to create the tensor on
    """
    top, left = meta["top"], meta["left"]
    orig_h, orig_w = meta["orig_h"], meta["orig_w"]

    # convert to latent space size
    top = math.floor(top / latent_ds_factor)
    left = math.floor(left / latent_ds_factor)
    orig_h = math.floor(orig_h / latent_ds_factor)
    orig_w = math.floor(orig_w / latent_ds_factor)

    pos = make_axial_pos_2d(orig_h, orig_w, device=device, align_corners=False, relative_pos=True)
    pos = rearrange(pos, "(h w) d -> h w d", h=orig_h, w=orig_w)
    pos = pos[top : top + size, left : left + size, :]
    return pos


class AxialRoPETime(AxialRoPEBase):
    """
    Simple 1D RoPE for text/time-like token positions.
    Uses fixed frequencies (non-learnable), matching standard text RoPE behavior.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        learnable_freqs: bool = False,
        relative_canvas: bool = True,  # kept for API compatibility
        in_place: bool = False,
        half_embedding: bool = True,
    ):
        if half_embedding:
            assert dim % 2 == 0, "Half embedding is only supported for even dimensions"
            dim //= 2
        super().__init__(dim, n_heads, in_place=in_place)

        # Best default for text: fixed frequencies, no learned RoPE params.
        min_freq, max_freq = 1 / 10_000, 1.0
        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 2 + 1)[:-1].exp()
        self.freqs = nn.Parameter(
            freqs.view(dim // 2, n_heads).T.contiguous(),
            requires_grad=False,
        )

    def forward(self, pos):
        if pos.shape[-1:] == (1,):
            pos = pos[..., 0]
        return pos[..., None, None] * self.freqs.to(pos.dtype)


class PatchForcingTransformerT2I(nn.Module):
    def __init__(
        self,
        in_dim: int = 4,
        depth: int = 28,
        hidden_dim: int = 1152,
        head_dim: int = 72,
        mapping_dim: int = 384,
        mapping_depth: int = 2,
        patch_size: int = 2,
        txt_in_dim: int = 2048,
        txt_refiner_dim: int = 1536,
        txt_refiner_head_dim: int = 128,
        txt_refiner_depth: int = 2,
        compile: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.depth = depth
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.mapping_dim = mapping_dim
        self.mapping_depth = mapping_depth
        self.patch_size = patch_size

        # timestep embedding
        self.t_embedder = TimestepEmbedder(mapping_dim, mapping_depth, dim_mlp=3 * mapping_dim)

        # model
        self.merge = TokenMerge2D(in_dim, hidden_dim, patch_size)
        self.blocks = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=hidden_dim,
                    d_head=head_dim,
                    d_cond_norm=mapping_dim,
                    d_cross=txt_refiner_dim,  # cross attend to refined txt embs
                    ff_expand=3,
                    rope_cls="jutils.nn.rope.AxialRoPE2D",
                    compile=compile,
                )
                for _ in range(depth)
            ]
        )
        # predict uncertainty per patch, so we have an additional out dim
        self.split = TokenSplitLast2D(hidden_dim, in_dim + 1, patch_size)

        # text embedding refiner
        self.txt_proj = nn.Linear(txt_in_dim, txt_refiner_dim)
        self.txt_refiner = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=txt_refiner_dim,
                    d_head=txt_refiner_head_dim,
                    ff_expand=3,
                    rope_cls="patch_flow.models.pf_transformer_t2i.AxialRoPETime",
                    compile=compile,
                )
                for _ in range(txt_refiner_depth)
            ]
        )

    def forward(
        self,
        x: Float[torch.Tensor, "b c h w"],
        t: Float[torch.Tensor, "b n"],
        txt_emb: Float[torch.Tensor, "b n d"],
        return_uncertainty: bool = False,
        img_meta: dict = None,
    ):
        n_patches = t.shape[1]
        b, c, h, w = x.shape

        # preprocess text with small refiner stack
        txt_emb = self.txt_proj(txt_emb)
        pos_txt = torch.arange(txt_emb.shape[1], device=txt_emb.device)
        pos_txt = repeat(pos_txt, "n -> b n 1", b=txt_emb.shape[0])  # (b, n, 1)
        for block in self.txt_refiner:
            txt_emb = block(txt_emb, pos=pos_txt)

        # timestep conditioning
        t = t[..., None]  # (b, n) -> (b, n, 1)
        t_emb = self.t_embedder(t)  # (b, n, c)

        # positional embeddings
        if img_meta is None:
            pos = make_axial_pos_2d(h, w, device=x.device)
            pos = repeat(pos, "(h w) d -> b h w d", b=b, h=h, w=w)
        else:
            pos = torch.stack([make_axial_pos_2d_with_meta(m, size=h, device=x.device) for m in img_meta], dim=0)

        x = rearrange(x, "b c h w -> b h w c")
        x, pos = self.merge(x, pos)
        nh, nw, _ = x.shape[1:]
        x = rearrange(x, "b h w c -> b (h w) c")
        pos = rearrange(pos, "b h w d -> b (h w) d")
        assert x.shape[1] == pos.shape[1] == n_patches, f"x: {x.shape}, pos: {pos.shape}, t: {t.shape}"

        # model
        for block in self.blocks:
            x = block(x, pos=pos, cond_norm=t_emb, x_cross=txt_emb)
        x = rearrange(x, "b (h w) c -> b h w c", h=nh, w=nw)

        # final layer
        x = self.split(x)

        # switch back to channel first
        x = rearrange(x, "b h w c -> b c h w")

        # split uncertainty head
        logvar_theta = x[:, -1:, :, :]  # (b, 1, h, w)
        x = x[:, :-1, :, :]  # (b, c, h, w)
        if return_uncertainty:
            return x, logvar_theta

        return x
