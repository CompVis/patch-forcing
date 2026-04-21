import torch
import torch.nn as nn
from typing import Union
from copy import deepcopy
from omegaconf import DictConfig
from collections import OrderedDict
from lightning import LightningModule
import warnings

from jutils import instantiate_from_config
from jutils import load_partial_from_config
from jutils import exists, freeze, default

from patch_flow.log_utils import log_images
from patch_flow.metrics import ImageMetricTracker
from patch_flow.diagonal_gaussian import DiagonalGaussian
from torchmetrics.aggregation import CatMetric


def un_normalize_ims(ims):
    """Convert from [-1, 1] to [0, 255]"""
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if not param.requires_grad:
            continue
        # unwrap DDP
        if name.startswith("module."):
            name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def instantiate_if_needed(config_or_obj):
    if isinstance(config_or_obj, nn.Module):
        return config_or_obj
    elif isinstance(config_or_obj, dict) or isinstance(config_or_obj, DictConfig):
        return instantiate_from_config(config_or_obj)
    else:
        raise ValueError(f"Expected nn.Module or config dict, got {type(config_or_obj)}")


# ===================================================================================================


class LatentFlowTrainer(LightningModule):
    def __init__(
        self,
        model: Union[dict, DictConfig, nn.Module],
        first_stage: Union[dict, DictConfig, nn.Module],
        flow: Union[dict, DictConfig, object],
        # learning
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        ema_rate: float = 0.9999,
        lr_scheduler_cfg: dict = None,
        # logging
        sample_kwargs: dict = None,
    ):
        super().__init__()

        # flow logic
        self.flow = instantiate_if_needed(flow)

        # unet/transformer model
        self.model = instantiate_if_needed(model)

        # EMA of unet/transformer model
        self.ema_model = None
        self.ema_rate = ema_rate
        if ema_rate > 0:
            if isinstance(model, nn.Module):
                warnings.warn("EMA model with deepcopy, might run into issues with compile.")
                self.ema_model = deepcopy(self.model)
            else:
                self.ema_model = instantiate_if_needed(model)
                self.ema_model.load_state_dict(self.model.state_dict())
            freeze(self.ema_model)
            self.ema_model.eval()
            update_ema(self.ema_model, self.model, decay=0)  # ensure EMA is in sync

        # first stage autoencoder
        self.first_stage = instantiate_if_needed(first_stage)
        self.first_stage.eval().to(self.device)
        freeze(self.first_stage)

        # training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg

        # visualization
        self.sample_kwargs = sample_kwargs or {}
        self.generator = torch.Generator()

        # evaluation
        self.metric_tracker = ImageMetricTracker().to(self.device)

        # SD3 & Meta Movie Gen show that val loss correlates with human quality
        # and compute the loss in equidistant segments in (0, 1) to reduce variance
        self.val_losses = CatMetric().to(self.device)  # sync across GPUs
        self.val_images = None
        self.val_epochs = 0

        self.save_hyperparameters()

        # signal handler for slurm, flag to make sure the signal
        # is not handled at an incorrect state, e.g. during weights update

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr, weight_decay=self.weight_decay
        )
        out = dict(optimizer=opt)
        if exists(self.lr_scheduler_cfg):
            sch = load_partial_from_config(self.lr_scheduler_cfg)
            sch = sch(optimizer=opt)
            out["lr_scheduler"] = sch
        return out

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # first checking for trainer ensures that the module can be also used with accelerate
        if exists(self._trainer) and exists(self.lr_scheduler_cfg):
            self.lr_schedulers().step()
        if exists(self.ema_model):
            update_ema(self.ema_model, self.model, decay=self.ema_rate)

    # ===================================================================================================
    # training logic

    @torch.no_grad()
    def encode(self, x):
        return self.first_stage.encode(x) if exists(self.first_stage) else x

    @torch.no_grad()
    def decode(self, z):
        return self.first_stage.decode(z) if exists(self.first_stage) else z

    def forward(self, batch):
        ims = batch["image"]
        latent = batch.get("latent", None)
        if not exists(latent):
            latent = self.encode(ims)
        label = batch.get("label", None)

        # compute loss
        loss = self.flow.training_losses(model=self.model, x1=latent, y=label)

        return loss

    # ===================================================================================================
    # validation

    def validation_step(self, batch, batch_idx):
        ims = batch["image"]
        label = batch.get("label", None)
        latent = batch.get("latent", None)
        if latent is None:
            latent = self.encode(ims)
        bs = ims.shape[0]

        g = self.generator.manual_seed(batch_idx + self.global_rank * 16102024)
        noise = torch.randn(latent.shape, generator=g, dtype=ims.dtype).to(ims.device)
        sample_model = self.ema_model if exists(self.ema_model) else self.model

        # flow models val loss shows correlation with human quality
        if hasattr(self.flow, "validation_losses"):
            latent = default(latent, self.encode(ims))
            _, val_loss_per_segment = self.flow.validation_losses(model=sample_model, x1=latent, x0=noise, y=label)
            self.val_losses.update(val_loss_per_segment.unsqueeze(0))

        # sample images
        samples = self.flow.generate(model=sample_model, x=noise, y=label, **self.sample_kwargs)
        samples = self.decode(samples)

        # metrics
        self.metric_tracker(ims, samples)

        # save the images for visualization
        if self.val_images is None:
            real_ims = un_normalize_ims(ims)
            fake_ims = un_normalize_ims(samples)
            self.val_images = {
                "real": real_ims[:20],
                "fake": fake_ims[:20],
            }

    def on_validation_epoch_end(self):
        # visualization
        for key, ims in self.val_images.items():
            log_images(self.logger, ims, f"val/{key}/samples", stack="row", split=4, step=self.global_step)

        # reset val images
        self.val_images = None

        # compute metrics
        metrics = self.metric_tracker.aggregate()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, sync_dist=True)
        self.metric_tracker.reset()

        # compute val loss if available (Flow models)
        if len(self.val_losses.value) > 0:
            val_losses = self.val_losses.compute()  # (N batches, segments)
            val_losses = val_losses.mean(0)  # mean per segment
            for i, loss in enumerate(val_losses):
                self.log(f"val/loss_segment_{i}", loss, sync_dist=True)
            self.log("val/loss", val_losses.mean(), sync_dist=True)
            self.val_losses.reset()

        # log some information
        self.val_epochs += 1
        self.print(f"Val epoch {self.val_epochs:,} | Optimizer step {self.global_step:,}")
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.print(metric_str)


# ===================================================================================================


class LatentPatchForcingTrainer(LatentFlowTrainer):
    def __init__(self, *args, uncertainty_weight: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_weight = uncertainty_weight
        assert (
            hasattr(self.model, "predict_uncertainty") and self.model.predict_uncertainty
        ), "Model should be PatchForcingDiT with predict_uncertainty=True."

    def forward(self, batch):
        ims = batch["image"]
        latent = batch.get("latent", None)
        if not exists(latent):
            latent = self.encode(ims)
        label = batch.get("label", None)

        # compute loss
        xt, ut, t = self.flow.get_interpolants(x1=latent)
        vt, logvar_theta = self.model(x=xt, t=t, y=label, return_uncertainty=True)

        # fm loss
        fm_loss = (vt - ut).square().mean()

        # uncertainty loss following SRM
        sigma_theta = torch.exp(0.5 * logvar_theta)
        pred_theta = DiagonalGaussian(mean=vt.detach(), std=sigma_theta)
        sigma_loss = pred_theta.nll(ut).mean()

        loss = fm_loss + self.uncertainty_weight * sigma_loss
        loss_dict = {"flow_loss": fm_loss, "sigma_loss": sigma_loss}

        return loss, loss_dict
