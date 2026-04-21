import wandb
import torch
import einops
import numpy as np
from PIL import Image
from jutils import NullObject
from jutils import ims_to_grid
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger


def log_image(logger, ims, tag, channel_last=True, step=0):
    """
    Args:
        logger: Logger class
        ims: torch.Tensor or np.ndarray of shape (c, h, w) or (h, w, c) in range [0, 255]
        tag: str, key to log the image
        channel_last: bool, whether the channel dimension is last
    """
    assert len(ims.shape) == 3, f"ims shape should be (c, h, w) or (h, w, c), got {ims.shape}"
    if isinstance(ims, torch.Tensor):
        ims = ims.cpu().numpy()

    if not channel_last:
        ims = einops.rearrange(ims, "c h w -> h w c")
    assert ims.shape[-1] in [1, 3], f"ims can have 1 or 3 channels, got {ims.shape[-1]}"

    if isinstance(logger, WandbLogger):
        ims = Image.fromarray(ims)
        ims = wandb.Image(ims)
        logger.experiment.log({tag: ims})

    elif isinstance(logger, (TensorBoardLogger, SummaryWriter)):
        ims = einops.rearrange(ims, "h w c -> c h w")
        if hasattr(logger, "experiment"):
            logger = logger.experiment
        logger.add_image(tag, ims, global_step=step)

    elif isinstance(logger, NullObject):
        pass  # Do nothing if logger is a NullObject

    else:
        raise ValueError(f"Unknown logger type: {type(logger)}")


def log_images(logger, ims, tag, stack="row", split=4, step=0):
    """
    Args:
        logger: Logger class
        ims: torch.Tensor or np.ndarray of shape (b, c, h, w) in range [0, 255]
        tag: str, key to log the images
    """
    assert len(ims.shape) == 4, f"ims shape should be (b, c, h, w), got {ims.shape}"
    assert ims.dtype in [torch.uint8, np.uint8], f"ims dtype should be uint8, got {ims.dtype}"
    ims = ims_to_grid(ims, stack=stack, split=split)
    if isinstance(ims, torch.Tensor):
        ims = ims.cpu().numpy()
    log_image(logger=logger, ims=ims, tag=tag, channel_last=True, step=step)


def log_videos(logger, videos, tag, step=0, fps=4):
    """
    Args:
        logger: Logger class
        videos: torch.Tensor or np.ndarray of shape (b, f, h, w, c) in range [0, 255]
        tag: str, key to log the video
    """
    assert len(videos.shape) == 5, f"videos shape should be (b, f, h, w, c), got {videos.shape}"
    assert videos.dtype in [torch.uint8, np.uint8], f"videos dtype should be uint8, got {videos.dtype}"

    if isinstance(logger, WandbLogger):
        # wandb expects (f c h w) or (b f c h w)
        videos = einops.rearrange(videos, "b f h w c -> b f c h w")
        videos = wandb.Video(videos, fps=fps, format="gif")
        if hasattr(logger, "experiment"):
            logger = logger.experiment
        logger.log({tag: videos})

    elif isinstance(logger, (TensorBoardLogger, SummaryWriter)):
        # convert to numpy and rearrange to (N, T, C, H, W)
        videos = einops.rearrange(videos, "b f h w c -> b f c h w")
        if hasattr(logger, "experiment"):
            logger = logger.experiment
        logger.add_video(tag, videos, global_step=step, fps=fps)

    elif isinstance(logger, NullObject):
        pass  # Do nothing if logger is a NullObject

    else:
        raise ValueError(f"Unknown logger type: {type(logger)}")
