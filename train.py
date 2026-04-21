import os
import sys
import time
import hydra
import torch
import datetime
from types import MethodType
from functools import partial
from tqdm import tqdm as tqdm_
from lightning import seed_everything
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, DictConfig, ListConfig
from torch.profiler import ProfilerActivity, profile, record_function

from jutils import NullObject
from jutils import instantiate_from_config
from jutils import count_parameters, exists
import patch_flow  # dummy to add omegaconf resolver
from patch_flow.dataloader import CUDAPrefetchIterator

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


# tqdm bar format
BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"
tqdm = partial(tqdm_, bar_format=BAR_FORMAT, dynamic_ncols=True)


# recursive check for `_target_` used with hydra's instantiate (not yet implemented)
def check_for_instantiate_key(cfg_node, path=""):
    if isinstance(cfg_node, dict) or isinstance(cfg_node, DictConfig):
        for k, v in cfg_node.items():
            full_path = f"{path}.{k}" if path else k
            if k == "_target_":
                raise NotImplementedError(
                    f"Unexpected '_target_' key found in config at: '{full_path}'. Hydra instantiate not yet implemented."
                )
            check_for_instantiate_key(v, full_path)
    elif isinstance(cfg_node, (list, ListConfig)):
        for i, item in enumerate(cfg_node):
            check_for_instantiate_key(item, f"{path}[{i}]")


def check_config(cfg):
    if cfg.get("auto_requeue", False):
        raise NotImplementedError("Auto-requeuing not working yet!")
    if exists(cfg.get("resume_checkpoint", None)) and exists(cfg.get("load_weights", None)):
        raise ValueError("Can't resume checkpoint and load weights at the same time.")
    if "experiment" in cfg:
        raise ValueError("Experiment config not merged successfully!")
    if cfg.use_wandb and cfg.use_wandb_offline:
        raise ValueError("Decide either for Online or Offline wandb, not both.")
    check_for_instantiate_key(cfg)

    # check for quick_train missing features
    assert cfg.use_wandb is False, "Wandb is not supported in quick_train.py"
    assert cfg.use_wandb_offline is False, "Wandb is not supported in quick_train.py"
    assert cfg.trainer.params.get("log_grad_norm", False) is False, "Log grad norm is not supported in quick_train.py"
    assert cfg.auto_requeue is False, "Auto-requeue is not supported in quick_train.py"
    assert cfg.deepspeed_stage == 0, "Deepspeed is not supported in quick_train.py"


""" lightning replacement functions """


def log_accelerate(name, value, step=None, writer=None, **kwargs):
    assert exists(writer), "Writer not passed to log function."
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, (float, int)):
        writer.add_scalar(name, value, global_step=step)


def add_global_step_setter(lightning_module):
    """
    Add a global step setter to the lightning module, s.t. we can
    use `self.global_step` within the module hooks.
    """

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, value):
        self._global_step = value

    # apply new property to the instance
    lightning_module.__class__.global_step = global_step


@contextmanager
def temporary_logger(module, logger):
    """create subclass with property override for self.logger"""
    original_class = module.__class__

    def get_logger(self):
        return logger

    TempClass = type(f"Patched{original_class.__name__}", (original_class,), {"logger": property(get_logger)})

    module.__class__ = TempClass
    try:
        yield module
    finally:
        # Restore the original class
        module.__class__ = original_class


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).
    """
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


""" main function """


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Check config"""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    check_config(cfg)

    """ Setup accelerate """
    # translate precision of lightning to accelerate
    lightning_to_accelerate_prec = {
        "16-mixed": "fp16",
        16: "fp16",
        "32-true": "no",
        32: "no",
        "bf16": "bf16",
        "bf16-mixed": "bf16",
    }
    # ddp kwargs
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=cfg.ddp_kwargs.get("find_unused_parameters", False),
        gradient_as_bucket_view=cfg.ddp_kwargs.get("gradient_as_bucket_view", False),
        bucket_cap_mb=cfg.ddp_kwargs.get("bucket_cap_mb", 25),
        broadcast_buffers=cfg.ddp_kwargs.get("broadcast_buffers", True),
    )
    accelerator = Accelerator(
        mixed_precision=lightning_to_accelerate_prec[cfg.train_params.precision],
        gradient_accumulation_steps=cfg.train_params.accumulate_grad_batches,
        kwargs_handlers=[ddp_kwargs],
    )
    seed_everything(2025 + accelerator.process_index)
    is_rank0 = accelerator.is_main_process
    device = accelerator.device

    """ Setup Logging """
    # we store the experiment under: logs/<cfg.name>/<day>/<slurm-id OR timestamp>
    day = datetime.datetime.now().strftime("%Y-%m-%d")
    postfix = str(cfg.slurm_id) if exists(cfg.slurm_id) else datetime.datetime.now().strftime("T%H%M%S")
    exp_name = os.path.join(cfg.name, day, postfix)
    log_dir = os.path.join("logs", exp_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if is_rank0:
        logger = SummaryWriter(log_dir=log_dir)
    else:
        logger = NullObject()

    """ Setup dataloader """
    data = instantiate_from_config(cfg.data)
    if hasattr(data, "prepare_data"):
        data.prepare_data()
    if hasattr(data, "setup"):
        data.setup(None)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    """ Setup module """
    module = instantiate_from_config(cfg.trainer)
    module = module.to(device).train()

    """ Patch lightning logging methods """
    add_global_step_setter(module)

    # printing
    def patched_print(self, *args, **kwargs):
        accelerator.print(*args, **kwargs)

    module.print = MethodType(patched_print, module)

    # logging
    def patched_log(self, name, value, **kwargs):
        log_accelerate(name, value, step=self.global_step, writer=logger, **kwargs)

    module.log = MethodType(patched_log, module)

    """ Setup optimizer """
    out = module.configure_optimizers()
    optimizer = out["optimizer"]
    scheduler = out.get("lr_scheduler", None)

    """ Load from checkpoint """
    resume_step = 0
    if exists(cfg.resume_checkpoint):
        ckpt = torch.load(cfg.resume_checkpoint, map_location=device, weights_only=False)
        resume_step = ckpt["global_step"]
        module.load_state_dict(ckpt["state_dict"], strict=cfg.get("load_strict", True))
        assert len(ckpt["optimizer_states"]) == 1, "Checkpoint should only contain one optimizer state dict."
        optimizer.load_state_dict(ckpt["optimizer_states"][0])
        if exists(scheduler) and len(ckpt["lr_schedulers"]) > 0:
            assert len(ckpt["lr_schedulers"]) == 1, "Checkpoint should only contain one scheduler state dict."
            scheduler.load_state_dict(ckpt["lr_schedulers"][0])
        print(
            f"Rank {accelerator.process_index} ({accelerator.num_processes}): Resumed from checkpoint at step {resume_step}"
        )

    if exists(cfg.load_weights):
        ckpt = torch.load(cfg.load_weights, map_location=device, weights_only=False)
        module.load_state_dict(ckpt["state_dict"], strict=cfg.get("load_strict", True))
        print(f"Rank {accelerator.process_index} ({accelerator.num_processes}): Loaded weights from {cfg.load_weights}")
        if "resume_step" in cfg and cfg.resume_step > 0:
            resume_step = cfg.resume_step
            print(f"Rank {accelerator.process_index} ({accelerator.num_processes}): Set resume step to {resume_step}")

    """ Setup DDP """
    module, optimizer, train_loader, val_loader = accelerator.prepare(module, optimizer, train_loader, val_loader)

    """ Profiling """
    profile_fn = NullObject()
    profile_record_fn = NullObject()
    if cfg.profile:
        profile_fn = partial(
            profile,
            activities=[
                *((ProfilerActivity.CPU,) if cfg.profiling.cpu else ()),
                *((ProfilerActivity.CUDA,) if cfg.profiling.cuda else ()),
            ],
            record_shapes=cfg.profiling.record_shapes,
            profile_memory=cfg.profiling.profile_memory,
            with_flops=cfg.profiling.with_flops,
            with_stack=True,
        )
        profile_record_fn = record_function

    """ print information """
    # log trainer module
    if is_rank0:
        print("-" * 40)
        print(OmegaConf.to_yaml(cfg.trainer))
    bs = cfg.data.params.batch_size
    bs = bs * accelerator.num_processes  # num nodes * num gpus
    bs = bs * cfg.train_params.accumulate_grad_batches  # global batch size
    assert accelerator.num_processes % cfg.num_nodes == 0, "Processes not divisible by nodes."
    # val batch size
    bs_val = cfg.data.params.get("val_batch_size", cfg.data.params.batch_size)
    bs_val = bs_val * accelerator.num_processes
    bs_val = bs_val * cfg.train_params.limit_val_batches
    some_info = {
        "Command": " ".join(["python"] + sys.argv),
        "Name": exp_name,
        "Log dir": log_dir,
        "Trainer Module": cfg.trainer.target,
        "Params": count_parameters(module),
        "Data": cfg.data.get("name", "not set"),
        "Batchsize": cfg.data.params.batch_size,
        "Devices": accelerator.num_processes // cfg.num_nodes,
        "Num nodes": cfg.num_nodes,
        "Gradient accum": cfg.train_params.accumulate_grad_batches,
        "Global batchsize": bs,
        "Val samples": bs_val,
        "LR": cfg.trainer.params.lr,
        "LR scheduler": cfg.lr_scheduler.get("name", "no name") if "lr_scheduler" in cfg else "None",
        "Resume ckpt": cfg.resume_checkpoint,
        "Load weights": cfg.load_weights,
        "Profiling": f"Step {cfg.profiling.warmup}" if cfg.profile else "None",
        "Precision": cfg.train_params.precision,
    }
    if is_rank0:
        OmegaConf.save(cfg, f"{log_dir}/config.yaml")

        # log hyperparameters to tensorboard
        logger.add_text("config", OmegaConf.to_yaml(cfg))
        logger.add_text("summary", OmegaConf.to_yaml(some_info))

        # print and write some info to the config
        with open(f"{log_dir}/config.yaml", "a") as f:
            f.write("\n\n")

            def flush_txt(txt):
                print(f"{txt}")
                f.write(f"# {txt}\n")

            flush_txt("-" * 40)
            for k, v in some_info.items():
                if isinstance(v, float):
                    flush_txt(f"{k:<16}: {v:.5f}")
                elif isinstance(v, int):
                    flush_txt(f"{k:<16}: {v:,}")
                elif isinstance(v, bool):
                    flush_txt(f"{k:<16}: {'True' if v else 'False'}")
                else:
                    flush_txt(f"{k:<16}: {v}")
            flush_txt("-" * 40)

    """ Setup training loop """
    global_step = resume_step
    max_steps = cfg.train_params.get("max_steps", -1)
    use_cuda_prefetch = bool(cfg.get("cuda_prefetch", False)) and device.type == "cuda"
    train_iterable = (
        CUDAPrefetchIterator(
            iterator=iter(train_loader),
            device=device,
            enabled=True,
            prefetch_factor=cfg.get("cuda_prefetch_factor", 2),
        )
        if use_cuda_prefetch
        else train_loader
    )

    # Loop
    for step, batch in enumerate(
        tqdm(train_iterable, desc="Training", miniters=cfg.tqdm_refresh_rate, disable=(not is_rank0))
    ):

        if max_steps > 0 and global_step >= max_steps:
            accelerator.print(f"Finish training after {global_step} steps.")
            accelerator.wait_for_everyone()
            break

        t0 = time.time()
        # ===================== #
        # Training              #
        # ===================== #
        with profile_fn() if cfg.profile and global_step == cfg.profiling.warmup else NullObject() as prof:

            with accelerator.accumulate(module):
                # forward
                with profile_record_fn(f"step_{global_step}/fwd"):
                    with accelerator.autocast():
                        if not use_cuda_prefetch:
                            batch = {
                                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                                for k, v in batch.items()
                            }
                        loss = module.forward(batch)

                        if isinstance(loss, tuple):
                            assert len(loss) == 2, "Loss tuple should be of length 2, shall be (loss, dict)."
                            loss, loss_dict = loss
                        else:
                            loss_dict = {}

                # backward
                with profile_record_fn(f"step_{global_step}/bwd"):
                    accelerator.backward(loss)

                # optimizer step
                with profile_record_fn(f"step_{global_step}/opt"):
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(
                            module.parameters(), max_norm=cfg.train_params.clip_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    if exists(scheduler):
                        scheduler.step()
                    unwrap_model(module).on_train_batch_end(loss, batch, step)  # no sync needed
                    global_step += 1
                    module.global_step = global_step
        step_time = time.time() - t0

        # logging
        if accelerator.sync_gradients and global_step % cfg.train_params.log_every_n_steps == 0:
            logger.add_scalar("train/loss", loss.item(), global_step=global_step)
            for k, v in loss_dict.items():
                logger.add_scalar(f"train/{k}", v.item(), global_step=global_step)
            logger.add_scalar("train/grad_norm", grad_norm.item(), global_step=global_step)
            logger.add_scalar("train/step_time", step_time, global_step=global_step)
            logger.add_scalar("train/it_per_sec", 1.0 / step_time, global_step=global_step)
            logger.add_scalar("train/throughput", bs / step_time, global_step=global_step)
            if exists(scheduler):
                logger.add_scalar("train/lr-AdamW", scheduler.get_last_lr()[0], global_step=global_step)

        if not accelerator.sync_gradients:
            continue

        # ===================== #
        # Profiling             #
        # ===================== #
        if cfg.profile and not isinstance(prof, NullObject):
            accelerator.wait_for_everyone()
            if is_rank0:
                print(f"[Profiling] Enabled after {cfg.profiling.warmup} steps.")
                fn = os.path.join(log_dir, cfg.profiling.filename)
                prof.export_chrome_trace(fn)
                print(f"[Profiling] Exported '{fn}'")
            accelerator.wait_for_everyone()
            break

        # ===================== #
        # Checkpoint            #
        # ===================== #
        if global_step % cfg.checkpoint_params.every_n_train_steps == 0 and global_step > 0:
            accelerator.wait_for_everyone()
            if is_rank0:
                fn = os.path.join(ckpt_dir, f"step{global_step:06d}.ckpt")
                lightning_module = unwrap_model(module)
                lightning_module.eval()
                # align with lightning checkpoints
                checkpoint = {
                    "epoch": 0,
                    "global_step": global_step,
                    "pytorch-lightning_version": "2.5.0.post0",
                    "state_dict": lightning_module.state_dict(),
                    # 'loops': {},                                        # TODO
                    # 'callbacks': {},                                    # TODO
                    "optimizer_states": [optimizer.state_dict()],
                    "lr_schedulers": [scheduler.state_dict()] if exists(scheduler) else [],
                    "hparams_name": "kwargs",
                    "hyper_parameters": OmegaConf.to_object(cfg.trainer.params),
                }
                torch.save(checkpoint, fn)
                print(f"Save checkpoint to {fn}")
                # symlink latest checkpoint
                last_ckpt_symlink = os.path.join(ckpt_dir, "last.ckpt")
                try:
                    if os.path.islink(last_ckpt_symlink) or os.path.exists(last_ckpt_symlink):
                        os.remove(last_ckpt_symlink)
                    relative_ckpt_path = os.path.relpath(fn, start=ckpt_dir)
                    os.symlink(relative_ckpt_path, last_ckpt_symlink)
                except OSError as e:
                    print(f"Failed to update symlink for last.ckpt: {e}")
                lightning_module.train()
            accelerator.wait_for_everyone()

        # ===================== #
        # Validation            #
        # ===================== #
        if global_step % cfg.train_params.val_check_interval == 0 and global_step > 0:

            module.eval()
            n_val_steps = cfg.train_params.limit_val_batches
            sample_module = unwrap_model(module)
            sample_module.global_step = global_step

            for val_step, val_batch in enumerate(
                tqdm(val_loader, desc=f"Validation {global_step}", disable=(not is_rank0), total=n_val_steps)
            ):
                if val_step == n_val_steps:
                    break

                val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                with torch.no_grad(), accelerator.autocast():
                    sample_module.validation_step(val_batch, val_step)

            # gather metrics and log them
            with temporary_logger(sample_module, logger):
                sample_module.on_validation_epoch_end()

            accelerator.wait_for_everyone()
            module.train()

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from einops._torch_specific import allow_ops_in_compiled_graph

    allow_ops_in_compiled_graph()

    try:
        main()
    except KeyboardInterrupt:
        print("[KeyboardInterrupt] Interrupted by user.")
        exit()
