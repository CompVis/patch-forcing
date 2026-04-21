import os
import sys
import torch
import torch.nn as nn
import random
import einops
import argparse
import numpy as np
from PIL import Image
from typing import List
from functools import partial
from omegaconf import OmegaConf

from transformers import AutoTokenizer
from transformers import Qwen3VLForConditionalGeneration

from jutils.nn import FLUX2AutoencoderKL
from jutils import instantiate_from_config

pdir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, pdir)

LATENT_CHANNELS = 32


# ===================================================================================================


class Qwen3VLEmbedder2B(nn.Module):
    def __init__(
        self,
        repo: str = "Qwen/Qwen3-VL-Embedding-2B",
        max_length: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.path = repo
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        full_model = Qwen3VLForConditionalGeneration.from_pretrained(self.path, dtype=dtype)
        text_tower = full_model.model.language_model
        del full_model

        self.model = text_tower
        self.emb_dim = int(self.model.config.hidden_size)
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, txt: List[str]):
        tok_out = self.tokenizer(
            txt, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True
        )
        tok_out = tok_out.to(self.device)
        txt_emb = self.model(**tok_out, output_hidden_states=True)
        if hasattr(txt_emb, "last_hidden_state"):
            return txt_emb.last_hidden_state
        return txt_emb.hidden_states[-1]


def unknowns_to_dict(unknown):
    """Convert a list of 'key=value' strings (dot-notation) into a nested dict."""
    bad = [u for u in unknown if u.startswith("-") or " " in u or u.strip() != u or "=" not in u]
    if bad:
        raise ValueError(f"Invalid override args (expected key=value without spaces): {bad}")
    if not unknown:
        return {}
    # OmegaConf parses values (int, float, bool, lists, null) automatically
    conf = OmegaConf.from_dotlist(unknown)
    return OmegaConf.to_container(conf, resolve=True)


# ===================================================================================================


def main(args, sample_fn_overrides=None):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    assert torch.cuda.is_available(), "CUDA is required to run this script."
    DEV = torch.device("cuda")

    # first stage autoencoder
    vae = FLUX2AutoencoderKL(ckpt_path="checkpoints/flux2_ae.ckpt").to(DEV).eval()
    print(f"{'Autoencoder':<16}: {sum([p.numel() for p in vae.parameters()]):,}")

    # text tower
    text_embedder = Qwen3VLEmbedder2B().to(DEV).eval()
    print(f"{'Text Embedder':<16}: {sum([p.numel() for p in text_embedder.parameters()]):,}")

    # model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["config"]
    state_dict = ckpt["state_dict"]
    model = instantiate_from_config(config).to(DEV)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    print(f"{'Model':<16}: {sum([p.numel() for p in model.parameters()]):,}")

    # sampling function
    timesteps = torch.linspace(0, 1, args.num_steps + 1)
    sample_fn_cfg = OmegaConf.load(args.sample_fn_config)
    if sample_fn_overrides is not None:  # merge with overrides
        sample_fn_cfg = OmegaConf.merge(sample_fn_cfg, sample_fn_overrides)
    sampler = instantiate_from_config(sample_fn_cfg)
    sample_fn = partial(sampler, timesteps=timesteps)
    print("=" * 40)
    print(OmegaConf.to_yaml(sample_fn_cfg))
    print("=" * 40)

    # sampling
    prompt = [args.prompt] * args.num_samples
    latent_shape = (LATENT_CHANNELS, args.resolution // 8, args.resolution // 8)
    noise = torch.randn((args.num_samples, *latent_shape), device=DEV)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        txt_emb = text_embedder(prompt)
        null_txt_emb = text_embedder("")
        samples = sample_fn(
            model=model,
            x=noise,
            txt_emb=txt_emb,
            uc_cond=null_txt_emb,
            progress=True,
            cond_key="txt_emb",
            cfg_scale=args.cfg_scale,
        )
        samples = vae.decode(samples)

    samples = einops.rearrange(samples, "b c h w -> b h w c")
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).cpu().to(torch.uint8).numpy()

    clean_prompt = args.prompt.replace(" ", "-").replace(",", "-")
    save_fn = f"{clean_prompt[:100]}_cfg{args.cfg_scale}_nfe{args.num_steps}_seed{args.seed}"
    for i, img in enumerate(samples):
        Image.fromarray(img).save(f"{save_fn}_{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--sample-fn-config", type=str, default="configs/sampler/euler-pf.yaml")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    known, unknown = parser.parse_known_args()
    unknown = unknowns_to_dict(unknown)
    main(known, unknown)
