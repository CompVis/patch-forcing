import torch
import torch.nn as nn

from einops import repeat
from functools import partial

from .dit import DiTBlock, DiT, FinalLayer


COMPILE = True
if torch.cuda.is_available():
    compile_fn = partial(
        torch.compile, fullgraph=True, backend="inductor" if torch.cuda.get_device_capability()[0] >= 7 else "aot_eager"
    )
else:
    compile_fn = lambda f: f


# ===================================================================================================


def pf_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PatchForcingDiTBlock(DiTBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if COMPILE:
            self.forward = compile_fn(self.forward)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(pf_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(pf_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PatchForcingFinalLayer(FinalLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if COMPILE:
            self.forward = compile_fn(self.forward)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = pf_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchForcingDiT(DiT):
    def __init__(
        self,
        *args,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio: float = 4.0,
        predict_uncertainty: bool = True,
        compile: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args, patch_size=patch_size, hidden_size=hidden_size, depth=depth, num_heads=num_heads, **kwargs
        )
        global COMPILE
        COMPILE = compile

        # predict uncertainty per patch (replace dit blocks and last layer)
        self.predict_uncertainty = predict_uncertainty
        if self.predict_uncertainty:
            assert self.learn_sigma is False, "cannot use both learn_sigma and predict_uncertainty!"
            assert self.return_sigma is False, "cannot use both return_sigma and predict_uncertainty!"

            # replace DiT blocks
            self.blocks = nn.ModuleList(
                [PatchForcingDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
            )

            # replace final layer
            self.out_channels = self.out_channels + 1
            self.final_layer = PatchForcingFinalLayer(hidden_size, patch_size, self.out_channels)

            self.initialize_weights()

    def forward(self, x, t, y=None, return_uncertainty: bool = False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N, num_patches) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # patch-level t's
        if self.predict_uncertainty:
            assert x.shape[1] == t.shape[1], f"x: {x.shape}, t: {t.shape}: require patch-level t's!"
            t = t[..., None]  # (N, T) -> (N, T, 1)
            t = self.t_embedder(t)  # (N, 1, T, D)
            t = t.squeeze(1)  # (N, T, D) one embedding per patch
        else:
            t = self.t_embedder(t)  # (N, D)

        cond = t
        if self.y_embedder is not None:
            y = self.y_embedder(y, self.training)  # (N, D)
            if self.predict_uncertainty:
                y = repeat(y, "b c -> b n c", n=x.shape[1])  # (N, D) -> (N, T, D)
            cond = cond + y  # (N, T, D)

        for block in self.blocks:
            x = block(x, cond)  # (N, T, D)
        x = self.final_layer(x, cond)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        # split uncertainty
        if self.predict_uncertainty:
            logvar_theta = x[:, -1:, :, :]  # (b, 1, h, w)
            x = x[:, :-1, :, :]  # (b, c, h, w)
            if return_uncertainty:
                return x, logvar_theta

        if self.learn_sigma and not self.return_sigma:  # LEGACY
            x, _ = x.chunk(2, dim=1)
        return x


# ===================================================================================================


def PF_XL_2(**kwargs):
    return PatchForcingDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def PF_L_2(**kwargs):
    return PatchForcingDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def PF_B_2(**kwargs):
    return PatchForcingDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


PF_models = {
    "PF-XL/2": PF_XL_2,
    "PF-L/2": PF_L_2,
    "PF-B/2": PF_B_2,
}


if __name__ == "__main__":
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    model = PF_models["PF-XL/2"]().to(DEV)
    print(f"{sum([p.numel() for p in model.parameters() if p.requires_grad]):,}")

    inp = dict(
        x=torch.randn((2, 4, 32, 32)).to(DEV),
        t=torch.rand((2,)).to(DEV),
        y=torch.randint(0, 1000, (2,)).to(DEV),
    )
    with torch.no_grad():
        out = model(**inp)
    print(out.shape)
