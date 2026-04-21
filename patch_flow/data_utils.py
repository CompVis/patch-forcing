import torch
import torchvision
import random
from jaxtyping import Float

from jutils import instantiate_from_config


class ResizeCropWithMetaInfo:
    def __init__(self, size: int = 256, antialias: bool = True, img_key: str = "image", meta_key: str = "img_meta"):
        self.size = int(size)
        self.resizer = torchvision.transforms.Resize(size=self.size, antialias=antialias)
        self.img_key = img_key
        self.meta_key = meta_key

    def resize_crop_image(self, img: Float[torch.Tensor, "c h w"]):
        """
        Args:
            img: (c, h, w) torch tensor in [-1, 1]
        """
        assert img.ndim == 3, f"Expected (C,H,W), got {tuple(img.shape)}"

        # resize shorter size to self.size
        img = self.resizer(img)
        _, orig_h, orig_w = img.shape

        # random crop
        top, left = 0, 0
        if orig_h > self.size:
            top = random.randint(0, orig_h - self.size)
        if orig_w > self.size:
            left = random.randint(0, orig_w - self.size)
        img_cropped = img[:, top : top + self.size, left : left + self.size]

        img_meta = dict(orig_h=orig_h, orig_w=orig_w, top=top, left=left)

        return img_cropped, img_meta

    def __call__(self, sample: dict):
        img = sample[self.img_key]
        img_cropped, img_meta = self.resize_crop_image(img)
        sample[self.img_key] = img_cropped
        sample[self.meta_key] = img_meta
        return sample


class CaptionSampler:
    def __init__(self, txt_sampling_cfg: dict, out_txt_key: str = "txt"):
        self.out_txt_key = out_txt_key
        self.text_sampling_cfg = txt_sampling_cfg
        self.total_ratio = sum(self.text_sampling_cfg.values())
        self.txt_keys = list(self.text_sampling_cfg.keys())
        self.txt_probs = [self.text_sampling_cfg[k] / self.total_ratio for k in self.txt_keys]

    def __call__(self, sample: dict):
        txt_key = random.choices(self.txt_keys, weights=self.txt_probs, k=1)[0]
        caption = sample[txt_key]
        if isinstance(caption, bytes):
            caption = caption.decode()
        sample[self.out_txt_key] = caption
        return sample


# ===================================================================================================


class TransformComposer:
    def __init__(self, transforms):
        self.transforms = [instantiate_from_config(t) for t in transforms]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
