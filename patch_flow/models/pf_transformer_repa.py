import torch
import torch.nn as nn
from jaxtyping import Float
from functools import partial
from einops import rearrange, repeat

from .pf_transformer import PatchForcingDiT


COMPILE = True
if torch.cuda.is_available():
    compile_fn = partial(
        torch.compile, fullgraph=True, backend="inductor" if torch.cuda.get_device_capability()[0] >= 7 else "aot_eager"
    )
else:
    compile_fn = lambda f: f


def build_mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


# ===================================================================================================


class REPAPatchForcingDiT(PatchForcingDiT):
    def __init__(self, *args, hidden_size=1152, z_dim=768, encoder_depth=8, projector_dim=2048, **kwargs):
        super().__init__(*args, hidden_size=hidden_size, **kwargs)
        self.encoder_depth = encoder_depth
        self.projector = build_mlp(hidden_size, projector_dim, z_dim)
        self.initialize_weights()

        assert self.predict_uncertainty, "REPA PatchForcingDiT requires predict_uncertainty=True"

    def forward(self, x, t, y=None, return_uncertainty: bool = False, return_z=False):
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

        N, T, D = x.shape
        for i, block in enumerate(self.blocks):
            x = block(x, cond)  # (N, T, D)
            if (i + 1) == self.encoder_depth:
                z = self.projector(x.reshape(-1, D)).reshape(N, T, -1)  # (N, T, z_dim)

        x = self.final_layer(x, cond)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        # split uncertainty
        if self.predict_uncertainty:
            logvar_theta = x[:, -1:, :, :]  # (b, 1, h, w)
            x = x[:, :-1, :, :]  # (b, c, h, w)

            if return_uncertainty and return_z:
                return x, logvar_theta, z
            if return_uncertainty:
                return x, logvar_theta

        if self.learn_sigma and not self.return_sigma:  # LEGACY
            x, _ = x.chunk(2, dim=1)
        return x


if __name__ == "__main__":
    pass
