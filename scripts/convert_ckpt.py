"""
Extract model state dict from trainer checkpoint, either "model" or "ema_model",
and store it in a new checkpoint file with corresponding suffix and model config.
"""

import torch
import argparse
from omegaconf import OmegaConf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Path to the checkpoint file")
    parser.add_argument(
        "--prefix", type=str, default="ema_model", help="Prefix for state dict (e.g., 'model' or 'ema_model')"
    )
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    # extract state dict
    clean_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith(args.prefix + "."):
            new_k = k[len(args.prefix) + 1 :]
            clean_state_dict[new_k] = v
    print(f"Extracted {len(clean_state_dict):,} parameters with prefix '{args.prefix}'")

    # extract config
    config = ckpt["hyper_parameters"]["model"]
    print("Extracted model config:")
    print(OmegaConf.to_yaml(config))

    # save model weights and configs
    new_fp = args.ckpt_path.replace(".ckpt", f"_{args.prefix}.ckpt")
    torch.save({"state_dict": clean_state_dict, "config": config}, new_fp)
    print(f"Saved extracted checkpoint to {new_fp}")
