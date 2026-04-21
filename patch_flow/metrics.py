import torch
import torch.nn as nn

from torchmetrics import CatMetric
from torchmetrics import SumMetric  # sum over devices
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore

from jutils.nn import DINOv2, preprocess_for_dinov2
from jutils.nn.metric_kid import kid_features_to_metric


def un_normalize_ims(ims):
    """Convert from [-1, 1] to [0, 255]"""
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims


class ImageMetricTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.total_samples = SumMetric()

        self.fid = FrechetInceptionDistance(
            feature=2048, reset_real_features=True, normalize=False, sync_on_compute=True
        )

    def __call__(self, target, pred):
        """Assumes target and pred in [-1, 1] range"""
        bs = target.shape[0]
        real_ims = un_normalize_ims(target)
        fake_ims = un_normalize_ims(pred)

        self.fid.update(real_ims, real=True)
        self.fid.update(fake_ims, real=False)

        self.total_samples.update(bs)

    def reset(self):
        self.fid.reset()
        self.total_samples.reset()

    def aggregate(self):
        """Compute the final metrics (automatically synced across devices)"""
        n_total_samples = int(self.total_samples.compute())
        return {
            f"fid-{n_total_samples}": self.fid.compute(),
            "n_metric_samples": n_total_samples,
        }


class Text2ImageMetricTracker(nn.Module):
    def __init__(self, kid_subsets: int = 100, kid_subset_size: int = 200):
        super().__init__()
        self.total_samples = SumMetric()
        self.fid = FrechetInceptionDistance(
            feature=2048, reset_real_features=True, normalize=False, sync_on_compute=True
        )

        self.clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

        self.dino = DINOv2(pretrained=True).eval()
        self.dino_features_real = CatMetric()
        self.dino_features_fake = CatMetric()
        self.kid_subsets = kid_subsets
        self.kid_subset_size = kid_subset_size

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, target, pred, txt):
        """Assumes target and pred in [-1, 1] range"""
        bs = target.shape[0]
        real_ims = un_normalize_ims(target)
        fake_ims = un_normalize_ims(pred)

        self.fid.update(real_ims, real=True)
        self.fid.update(fake_ims, real=False)

        txt = [t.decode() if isinstance(t, bytes) else t for t in txt]
        self.clip.update(fake_ims, list(txt))

        real_fts = self.dino(preprocess_for_dinov2(target, safe_mode=False))
        fake_fts = self.dino(preprocess_for_dinov2(pred, safe_mode=False))
        self.dino_features_real.update(real_fts)
        self.dino_features_fake.update(fake_fts)

        self.total_samples.update(bs)

    def reset(self):
        self.fid.reset()
        self.clip.reset()
        self.dino_features_real.reset()
        self.dino_features_fake.reset()
        self.total_samples.reset()

    def aggregate(self):
        """Compute the final metrics (automatically synced across devices)"""
        n_total_samples = int(self.total_samples.compute())

        # compute KDD
        real_fts = self.dino_features_real.compute()
        fake_fts = self.dino_features_fake.compute()
        kdd = kid_features_to_metric(
            real_fts,
            fake_fts,
            kid_subsets=self.kid_subsets,
            kid_subset_size=self.kid_subset_size,
            verbose=False,
        )

        return {
            f"fid-{n_total_samples}": self.fid.compute(),
            f"clip": self.clip.compute(),
            **kdd,
            "n_metric_samples": n_total_samples,
        }
