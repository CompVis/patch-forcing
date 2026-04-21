import torch
import einops
import warnings
import numpy as np
import torch.nn as nn
from typing import Union
from copy import deepcopy
from omegaconf import DictConfig
from torchmetrics import CatMetric
from lightning import LightningModule

from jutils import exists, freeze
from jutils import load_partial_from_config
from jutils import tensor2im, text_to_canvas, soft_wrap

from patch_flow.log_utils import log_image
from patch_flow.metrics import Text2ImageMetricTracker
from patch_flow.diagonal_gaussian import DiagonalGaussian
from patch_flow.trainer import update_ema, instantiate_if_needed


class PatchForcingT2ITrainer(LightningModule):
    def __init__(
        self,
        model: Union[dict, DictConfig, nn.Module],
        first_stage: Union[dict, DictConfig, nn.Module],
        flow: Union[dict, DictConfig, object],
        # text conditioning
        text_encoder: Union[dict, DictConfig, nn.Module],
        text_dropout_prob: float = 0.1,
        text_key: str = "txt",
        # learning
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        ema_rate: float = 0.9999,
        lr_scheduler_cfg: dict = None,
        rope_jittering: bool = True,
        uncertainty_weight: float = 0.01,
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

        # text tower
        self.text_key = text_key
        self.text_dropout_prob = text_dropout_prob
        self.text_encoder = instantiate_if_needed(text_encoder)
        self.text_encoder.eval().to(self.device)
        freeze(self.text_encoder)

        # training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.uncertainty_weight = uncertainty_weight
        self.rope_jittering = rope_jittering
        self.sample_kwargs = sample_kwargs or {}
        self.generator = torch.Generator()

        # evaluation
        self.metric_tracker = Text2ImageMetricTracker().eval().to(self.device)

        # SD3 & Meta Movie Gen show that val loss correlates with human quality
        # and compute the loss in equidistant segments in (0, 1) to reduce variance
        self.val_losses = CatMetric().to(self.device)  # sync across GPUs
        self.val_images = None
        self.val_epochs = 0

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

    def encode_text(self, text):
        text = [t.decode() if isinstance(t, bytes) else t for t in text]
        if self.training and self.text_dropout_prob > 0:
            drop_ids = np.random.rand(len(text)) < self.text_dropout_prob
            text = ["" if drop else text for drop, text in zip(drop_ids, text)]
        txt_tokens = self.text_encoder(text)  # no grad
        return txt_tokens

    def forward(self, batch):
        ims = batch["image"]
        latent = batch.get("latent", None)
        if latent is None:
            latent = self.encode(ims)

        # text encoding
        txt = batch[self.text_key]
        txt_emb = self.encode_text(txt)

        # potential rope jit
        kwargs = dict(txt_emb=txt_emb)
        if self.rope_jittering:
            img_meta = batch.get("img_meta", None)
            assert img_meta is not None, "img_meta must be provided in the batch for rope_jittering."
            kwargs["img_meta"] = img_meta

        # compute flow matching loss
        xt, ut, t = self.flow.get_interpolants(x1=latent)
        vt, logvar_theta = self.model(x=xt, t=t, **kwargs, return_uncertainty=True)

        # flow loss
        flow_loss = (vt - ut).square().mean()

        # uncertainty loss following SRM
        sigma_theta = torch.exp(0.5 * logvar_theta)
        pred_theta = DiagonalGaussian(mean=vt.detach(), std=sigma_theta)
        sigma_loss = pred_theta.nll(ut).mean()

        loss = flow_loss + self.uncertainty_weight * sigma_loss
        loss_dict = {"flow_loss": flow_loss, "sigma_loss": sigma_loss}

        return loss, loss_dict

    # ===================================================================================================
    # validation

    def validation_step(self, batch, batch_idx):
        ims = batch["image"]
        latent = batch.get("latent", None)
        if latent is None:
            latent = self.encode(ims)
        bs = ims.shape[0]

        txt = batch[self.text_key]
        txt_emb = self.encode_text(txt)

        g = self.generator.manual_seed(batch_idx + self.global_rank * 16102024)
        noise = torch.randn(latent.shape, generator=g, dtype=ims.dtype).to(ims.device)
        sample_model = self.ema_model if exists(self.ema_model) else self.model

        # flow models val loss shows correlation with human quality
        _, val_loss_per_segment = self.flow.validation_losses(model=sample_model, x1=latent, x0=noise, txt_emb=txt_emb)
        self.val_losses.update(val_loss_per_segment.unsqueeze(0))

        # sample images
        samples = self.flow.generate(model=sample_model, x=noise, txt_emb=txt_emb, **self.sample_kwargs)
        samples = self.decode(samples)

        # metrics
        self.metric_tracker(ims, samples, txt)

        # visualization images
        if self.val_images is None:
            c, h, w = ims.shape[1:]
            out_ims = [ims]
            if exists(self.ema_model):
                non_ema_samples = self.flow.generate(model=self.model, x=noise, txt_emb=txt_emb, **self.sample_kwargs)
                non_ema_samples = self.decode(non_ema_samples)
                out_ims.append(non_ema_samples)
            out_ims.append(samples)

            # CFG images
            cfg_scales = [3, 5, 7]
            uc_txt_emb = self.encode_text([""] * bs)
            for cfg_scale in cfg_scales:
                kwargs = dict(**self.sample_kwargs, cfg_scale=cfg_scale, uc_cond=uc_txt_emb, cond_key="txt_emb")
                samples = self.flow.generate(model=sample_model, x=noise, txt_emb=txt_emb, **kwargs)
                samples = self.decode(samples)
                out_ims.append(samples)

            # out images: [real, non-ema, ema, cfg3, ...] stacked over height
            out_ims = torch.cat(out_ims, dim=2)  # (b, c, n*h, w)
            out_ims = tensor2im(out_ims.float())  # (b, n*h, w, c)

            # generation image with captions
            caption_ims = np.stack(
                [
                    text_to_canvas(
                        soft_wrap(t.decode() if isinstance(t, bytes) else t, 50),
                        h,
                        w,
                        font_size=9.5,
                        background=(255, 255, 255),
                        fontcolor=(0, 0, 0),
                    )
                    for t in txt
                ],
                axis=0,
            )
            out_ims = np.concatenate([out_ims, caption_ims], axis=1)  # (b, n*h+1, w, c)

            # one final image for vis, limit to 20
            out_ims = einops.rearrange(out_ims[:20], "b h w c -> h (b w) c")
            self.val_images = {"gt_non-ema_ema_cfg3-5-7": out_ims}

    def on_validation_epoch_end(self):
        # visualization
        for key, ims in self.val_images.items():
            log_image(self.logger, ims, f"val/{key}", channel_last=True, step=self.global_step)
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
            # for i, loss in enumerate(val_losses):
            #     self.log(f"val/loss_segment_{i}", loss, sync_dist=True)
            self.log("val/loss", val_losses.mean(), sync_dist=True)
            self.val_losses.reset()

        # log some information
        self.val_epochs += 1
        self.print(f"Val epoch {self.val_epochs:,} | Optimizer step {self.global_step:,}")
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.print(metric_str)
