import os
import sys
import math
import torch
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image
from functools import partial
from omegaconf import OmegaConf
from contextlib import nullcontext
from jutils import instantiate_from_config
from diffusers.models import AutoencoderKL

currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import patch_flow.pt_distributed as dist


NUM_CLASSES = 1000
DATA_SHAPE = (4, 32, 32)  # 256x256 images


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}_N{num}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


""" Main """


def main(args, sample_fn_overrides=None):
    """Setup distributed"""
    dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=90))
    GLOBAL_RANK = dist.get_rank()
    LOCAL_RANK = GLOBAL_RANK % torch.cuda.device_count()
    DEV = torch.device(f"cuda:{LOCAL_RANK}")
    WORLD_SIZE = dist.get_world_size()
    is_rank0 = dist.is_primary()
    print(f"[RANK {GLOBAL_RANK} | {WORLD_SIZE}] Initializing on device: {DEV}")

    seed = args.global_seed * dist.get_world_size() + LOCAL_RANK
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    if args.half_precision:
        inference_context = torch.autocast("cuda")
    else:  # DEFAULT from SiT
        torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
        inference_context = nullcontext()
    torch.set_grad_enabled(False)

    dist.print0(f"Global seed set to {seed}")
    dist.print0("=" * 40)
    for k, v in vars(args).items():
        dist.print0(f"{k:20}: {v}")
    dist.print0("=" * 40)

    """ sampling function """
    timesteps = torch.linspace(0, 1, args.num_sampling_steps + 1)
    sample_fn_cfg = OmegaConf.load(args.sample_fn_config)
    if sample_fn_overrides is not None:  # merge with overrides
        sample_fn_cfg = OmegaConf.merge(sample_fn_cfg, sample_fn_overrides)
    sampler = instantiate_from_config(sample_fn_cfg)
    sample_fn = partial(sampler, timesteps=timesteps)
    dist.print0(OmegaConf.to_yaml(sample_fn_cfg))
    dist.print0("=" * 40)

    """ Load model """
    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["config"]
    state_dict = ckpt["state_dict"]
    model = instantiate_from_config(config).to(DEV)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(DEV)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"

    """ Saving folder """
    sample_dir = os.path.join(os.path.dirname(args.ckpt), "samples")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".ckpt", "")
    sample_fn_postfix = f"{sampler}"  # uses __repr__ method of sampler class
    folder_name = (
        f"{ckpt_string_name}-"
        f"cfg-{args.cfg_scale}-"
        f"{args.num_sampling_steps}_seed{args.global_seed}_{sample_fn_postfix}"
    )
    sample_folder_dir = f"{sample_dir}/{folder_name}"
    os.makedirs(sample_folder_dir, exist_ok=True)
    dist.print0(f"Saving samples to {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    dist.print0(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if is_rank0 else pbar
    total = 0

    all_samples = []
    for i in pbar:
        # Sample inputs:
        z = torch.randn(n, *DATA_SHAPE, device=DEV)
        y = torch.randint(0, NUM_CLASSES, (n,), device=DEV)
        y_null = torch.tensor([1000] * n, device=DEV)  # for cfg

        model_kwargs = dict(y=y, uc_cond=y_null, cond_key="y", cfg_scale=args.cfg_scale)

        with inference_context:
            samples = sample_fn(
                model=model,
                x=z,
                progress=False,
                **model_kwargs,
            )
            samples = vae.decode(samples / 0.18215).sample

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        all_samples.append(samples)
        total += global_batch_size
        dist.barrier()

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    all_samples = np.concatenate(all_samples, axis=0)

    # gather all samples over GPUs
    all_samples = torch.tensor(all_samples).to(DEV).contiguous()
    gathered_samples = dist.gather(all_samples)
    gathered_samples = torch.cat(gathered_samples, dim=0).cpu().numpy()

    # build the npz file
    if is_rank0:
        # store the desired number of samples
        npz_path = f"{sample_folder_dir}_N{args.num_fid_samples}.npz"
        arr_0 = gathered_samples[: args.num_fid_samples]
        assert arr_0.shape[0] == args.num_fid_samples, f"Expected {args.num_fid_samples} samples, got {arr_0.shape[0]}"
        np.savez(npz_path, arr_0=arr_0)
        print(f"Saved .npz file to {npz_path} [shape={arr_0.shape}].")

        # store 10k samples
        if args.num_fid_samples > 10000 and gathered_samples.shape[0] > 10000:
            npz_path = f"{sample_folder_dir}_N10000.npz"
            np.savez(npz_path, arr_0=gathered_samples[:10000])
            print(f"Saved .npz file to {npz_path} [shape={gathered_samples[:10000].shape}].")
    dist.barrier()
    dist.destroy_process_group()


""" Parsing utils """


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a checkpoint.")
    parser.add_argument("--sample-fn-config", type=str, default="configs/sampler/euler-pf.yaml")
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Use TF32 matmuls.")
    parser.add_argument("--half_precision", action="store_true", help="Use this flag to enable bf16.")

    # Unknown args will be passed as overrides to the sample function config, e.g. following
    # dot-notation you can pass, e.g. params.p=0.4

    known, unknown = parser.parse_known_args()
    unknown = unknowns_to_dict(unknown)
    main(known, unknown)
