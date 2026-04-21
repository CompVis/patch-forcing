import os
import torch
import numpy as np
import torchvision
import webdataset as wds
from collections import deque
from omegaconf import OmegaConf
from omegaconf import ListConfig
from torch.utils.data import DataLoader, Dataset
import lightning as pl
from typing import Dict, Any, Union

from jutils import instantiate_from_config
from jutils import load_partial_from_config


""" WebDataset """


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


def identity(x):
    return x


def safe_rename(sample, renaming):
    """
    Renames keys according to mapping {new_key: old_key}.
    If the old key is missing, warns and continues (skips that key only).
    """
    out = dict(sample)  # copy existing keys
    for new_key, old_key in renaming.items():
        if old_key in sample:
            out[new_key] = sample[old_key]
            if new_key != old_key:
                out.pop(old_key, None)
        else:
            if new_key == "txt":
                if "short" in sample:
                    out[new_key] = sample["short"]
                    continue
                if "caption_internvl3_2b_short" in sample:
                    out[new_key] = sample["caption_internvl3_2b_short"]
                    continue
                wds.warn_and_continue(Exception(f"Could not find alternative keys for missing txt key."))
            wds.warn_and_continue(Exception(f"Missing key '{old_key}' while renaming to '{new_key}'"))
    return out


class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        tar_base,  # can be a list of paths or a single path
        batch_size,
        val_batch_size=None,
        train=None,
        validation=None,
        test=None,
        num_workers=4,
        val_num_workers: int = None,
        multinode=True,
        remove_keys: list = None,  # list of keys to remove from the sample
    ):
        super().__init__()
        if isinstance(tar_base, str):
            self.tar_base = tar_base
        elif isinstance(tar_base, ListConfig) or isinstance(tar_base, list):
            # check which tar_base exists
            for path in tar_base:
                if os.path.exists(path):
                    self.tar_base = path
                    break
            else:
                raise FileNotFoundError("Could not find a valid tarbase.")
        else:
            raise ValueError(f"Invalid tar_base type {type(tar_base)}")
        print(f"[WebDataModuleFromConfig] Setting tar base to {self.tar_base}")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.rm_keys = remove_keys if remove_keys is not None else []

    def make_loader(self, dataset_config, train=True):
        image_transforms = []
        lambda_fn = lambda x: x * 2.0 - 1.0  # normalize to [-1, 1]
        image_transforms.extend([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda_fn)])
        if "image_transforms" in dataset_config:
            image_transforms.extend([instantiate_from_config(tt) for tt in dataset_config.image_transforms])
        image_transforms = torchvision.transforms.Compose(image_transforms)

        if "transforms" in dataset_config:
            transforms_config = OmegaConf.to_container(dataset_config.transforms)
        else:
            transforms_config = dict()

        transform_dict = {
            dkey: (
                load_partial_from_config(transforms_config[dkey]) if transforms_config[dkey] != "identity" else identity
            )
            for dkey in transforms_config
        }
        # this is crucial to set correct image key to get the transofrms applied correctly
        img_key = dataset_config.get("image_key", "image.png")
        transform_dict.update({img_key: image_transforms})

        if "dataset_transforms" in dataset_config:
            dataset_transforms = instantiate_from_config(dataset_config["dataset_transforms"])
        else:
            dataset_transforms = None

        if "postprocess" in dataset_config:
            postprocess = instantiate_from_config(dataset_config["postprocess"])
        else:
            postprocess = None

        shuffle = dataset_config.get("shuffle", 0)
        shardshuffle = shuffle > 0

        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only

        if isinstance(dataset_config.shards, str):
            tars = os.path.join(self.tar_base, dataset_config.shards)
        elif isinstance(dataset_config.shards, list) or isinstance(dataset_config.shards, ListConfig):
            # decompose into lists of shards
            # Turn train-{000000..000002}.tar into ['train-000000.tar', 'train-000001.tar', 'train-000002.tar']
            tars = []
            for shard in dataset_config.shards:
                # Assume that the shard starts from 000000
                if "{" in shard:
                    start, end = shard.split("..")
                    start = start.split("{")[-1]
                    end = end.split("}")[0]
                    start = int(start)
                    end = int(end)
                    tars.extend(
                        [shard.replace(f"{{{start:06d}..{end:06d}}}", f"{i:06d}") for i in range(start, end + 1)]
                    )
                else:
                    tars.append(shard)
            tars = [os.path.join(self.tar_base, t) for t in tars]
            # random shuffle the shards
            if shardshuffle:
                np.random.shuffle(tars)
        else:
            raise ValueError(f"Invalid shards type {type(dataset_config.shards)}")

        dset = (
            wds.WebDataset(tars, nodesplitter=nodesplitter, shardshuffle=shardshuffle, handler=wds.warn_and_continue)
            .repeat()
            .shuffle(shuffle)
        )
        print(f"[WebDataModuleFromConfig] Loading {len(dset.pipeline[0].urls)} shards.")

        dset = (
            dset.decode("rgb", handler=wds.warn_and_continue)
            .map(self.filter_out_keys, handler=wds.warn_and_continue)
            .map_dict(**transform_dict, handler=wds.warn_and_continue)
        )

        # change name of image key to be consistent with other datasets
        renaming = dataset_config.get("rename", None)
        if renaming is not None:
            # dset = dset.rename(**renaming)
            dset = dset.map(lambda sample: safe_rename(sample, renaming), handler=wds.warn_and_continue)

        if dataset_transforms is not None:
            dset = dset.map(dataset_transforms)

        if postprocess is not None:
            dset = dset.map(postprocess)

        bs = self.batch_size if train else self.val_batch_size
        nw = self.num_workers if train else self.val_num_workers
        dset = dset.batched(bs, partial=False, collation_fn=dict_collation_fn)
        loader = wds.WebLoader(dset, batch_size=None, shuffle=False, num_workers=nw, pin_memory=True)

        return loader

    def filter_out_keys(self, sample):
        for key in self.rm_keys:
            sample.pop(key, None)
        return sample

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)


""" Normal Dataset """


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        val_batch_size: int = None,
        train: dict = None,
        validation: dict = None,
        test: dict = None,
        shuffle_validation: bool = False,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train = train
        self.validation = validation
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.shuffle_validation = shuffle_validation

        self.dataset_configs = {}
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def _train_dataloader(self):
        return DataLoader(
            self.datasets["train"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

    def _val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_validation,
        )

    def _test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_validation,
        )

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)


class DummyDataset(Dataset):
    def __init__(self, num_samples=10000000, **kwargs):
        super().__init__()
        self.num_samples = num_samples
        self.keys_shapes = {k: v for k, v in kwargs.items()}

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        return {
            key: (torch.randn(*shape) if len(shape) > 1 else torch.randint(0, 10, (1,)).squeeze())  # e.g. class labels
            for key, shape in self.keys_shapes.items()
        }


class CIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__()
        if transform is None:
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            transform = instantiate_from_config(transform)

        if target_transform is not None:
            target_transform = instantiate_from_config(target_transform)
        self.dataset = torchvision.datasets.CIFAR10(
            root, train=train, transform=transform, target_transform=target_transform, download=download
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return {"image": img, "label": target}


""" Helpers """


class MomentsPreprocessor:
    def __init__(self, moments_key="moments.npy", out_key="latent", scale: float = 0.18215, shift: float = 0.0):
        self.moments_key = moments_key
        self.out_key = out_key
        self.scale = scale
        self.shift = shift

    def __call__(self, sample):
        """
        Helper function for ImageNet first stage sampling using moments.
        https://github.com/joh-schb/jutils/blob/8440e65b6296897ec23f0c1f13199ca0e1be92e9/jutils/nn/kl_autoencoder.py#L45
        """
        moments = torch.tensor(sample[self.moments_key])

        mean, logvar = torch.chunk(moments, 2, dim=0)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)

        latent = mean + std * torch.randn(mean.shape).to(device=moments.device)
        latent = (latent + self.shift) * self.scale
        sample[self.out_key] = latent

        del sample[self.moments_key]

        return sample


def dict_to(d: Dict[str, Union[torch.Tensor, Any]], **to_kwargs) -> Dict[str, Union[torch.Tensor, Any]]:
    return {k: (v.to(**to_kwargs) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


class CUDAPrefetchIterator:
    """Source from diffusion codebase, thanks!"""

    def __init__(
        self,
        iterator,
        device: torch.device,
        prefetch_factor: int = 2,
        enabled: bool = True,
        target_stream: torch.cuda.Stream = None,  # The stream that will use the batch, will be automatically set to torch.cuda.current_stream() in the iterator if not provided
    ):
        self.iterator = iterator
        self.device = device
        self.prefetch_factor = prefetch_factor
        assert self.prefetch_factor > 0, "prefetch_factor must be greater than 0"
        self.enabled = enabled
        self.target_stream = target_stream
        if self.target_stream is not None:
            assert (
                self.target_stream.device == self.device
            ), f"Target stream must be on the same device as the iterator. Got {target_stream.device=} and {device=}"

        self._transfer_stream = torch.cuda.Stream(device)

    def __iter__(self):
        if not self.enabled:
            # Just return synchronously from the iterator
            for batch_cpu in self.iterator:
                yield dict_to(batch_cpu, device=self.device, non_blocking=False)
            return

        batch_buf: deque[tuple[dict, torch.cuda.Event]] = deque()
        target_stream = self.target_stream or torch.cuda.current_stream(self.device)

        def enqueue_batch() -> bool:
            try:
                batch_cpu = next(self.iterator)
            except StopIteration:
                return False

            with torch.cuda.stream(self._transfer_stream):
                batch_gpu = dict_to(batch_cpu, device=self.device, non_blocking=True)
                transfer_event = torch.cuda.Event(blocking=False, enable_timing=False)
                transfer_event.record(self._transfer_stream)

            batch_buf.append((batch_gpu, transfer_event))
            return True

        # Warmup queue
        for _ in range(self.prefetch_factor):
            if not enqueue_batch():
                break
        if not batch_buf:
            return  # Iterator was empty

        # Main loop
        while batch_buf:
            batch_gpu, ready_event = batch_buf.popleft()
            target_stream.wait_event(ready_event)  # Wait for transfer to complete
            enqueue_batch()
            yield batch_gpu
