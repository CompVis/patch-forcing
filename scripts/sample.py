import os
import sys
import time
import torch
import random
import argparse
import numpy as np
from functools import partial
from omegaconf import OmegaConf
from contextlib import nullcontext
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from jutils import instantiate_from_config

currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


NULL_CLASS = 1000
DATA_SHAPE = (4, 32, 32)  # 256x256 images
CLASS_LABELS = [207, 360, 387, 974, 88, 979, 417, 279]


def unknowns_to_dict(unknown):
    """Convert a list of 'key=value' strings (dot-notation) into a nested dict."""
    bad = [u for u in unknown if u.startswith("-") or " " in u or u.strip() != u or "=" not in u]
    if bad:
        raise ValueError(f"Invalid override args (expected key=value without spaces): {bad}")
    if not unknown:
        return {}
    conf = OmegaConf.from_dotlist(unknown)
    return OmegaConf.to_container(conf, resolve=True)


def main(args, sample_fn_overrides=None):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    assert torch.cuda.is_available(), "CUDA is required to run this script."
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    if args.half_precision:
        inference_context = torch.autocast("cuda")
    else:
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        inference_context = nullcontext()

    timesteps = torch.linspace(0, 1, args.num_sampling_steps + 1, device=device)
    sample_fn_cfg = OmegaConf.load(args.sample_fn_config)
    if sample_fn_overrides is not None:
        sample_fn_cfg = OmegaConf.merge(sample_fn_cfg, sample_fn_overrides)
    sampler = instantiate_from_config(sample_fn_cfg)
    sample_fn = partial(sampler, timesteps=timesteps)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["config"]
    state_dict = ckpt["state_dict"]
    model = instantiate_from_config(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()

    n = len(CLASS_LABELS)
    z = torch.randn(n, *DATA_SHAPE, device=device)
    y = torch.tensor(CLASS_LABELS, device=device)
    y_null = torch.full((n,), NULL_CLASS, device=device)

    model_kwargs = dict(y=y, uc_cond=y_null, cond_key="y", cfg_scale=args.cfg_scale)

    sampler_name = str(sampler)
    cfg_name = str(args.cfg_scale).replace("/", "-")
    save_prefix = f"steps{args.num_sampling_steps}_{sampler_name}_cfg{cfg_name}"
    save_dir = parentdir

    print("=" * 40)
    print(f"{'ckpt':20}: {args.ckpt}")
    print(f"{'output_dir':20}: {save_dir}")
    print(f"{'save_prefix':20}: {save_prefix}")
    print(f"{'class_labels':20}: {CLASS_LABELS}")
    print(f"{'cfg_scale':20}: {args.cfg_scale}")
    print(f"{'num_steps':20}: {args.num_sampling_steps}")
    print(OmegaConf.to_yaml(sample_fn_cfg))
    print("=" * 40)

    start_time = time.time()
    with inference_context:
        samples = sample_fn(
            model=model,
            x=z,
            progress=True,
            **model_kwargs,
        )
        samples = vae.decode(samples / 0.18215).sample
    elapsed = time.time() - start_time

    grid_path = os.path.join(save_dir, f"{save_prefix}_sample.png")
    save_image(samples, grid_path, nrow=4, normalize=True, value_range=(-1, 1))

    print(f"Sampling took {elapsed:.2f} seconds.")
    print(f"Saved grid to {grid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a checkpoint.")
    parser.add_argument("--sample-fn-config", type=str, default="configs/sampler/euler-pf.yaml")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Use TF32 matmuls.")
    parser.add_argument("--half_precision", action="store_true", help="Use this flag to enable bf16.")

    known, unknown = parser.parse_known_args()
    unknown = unknowns_to_dict(unknown)
    main(known, unknown)
