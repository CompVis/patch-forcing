# MIT License Copyright (c) 2022 joh-schb
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


def is_distributed():
    """
    Check if the current process is part of a distributed setup.
    """
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_process_group(*args, **kwargs):
    if not is_distributed():
        return
    dist.init_process_group(*args, **kwargs)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def destroy_process_group():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def cleanup():
    destroy_process_group()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_rank()}")
    return torch.device("cpu")


def is_primary():
    return get_rank() == 0


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


# data loading stuff
def data_sampler(dataset, distributed, shuffle):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    return None


# model wrapping
def prepare_ddp_model(model, device_ids, *args, **kwargs):
    if get_world_size() > 1:
        model = DistributedDataParallel(model, device_ids=device_ids, *args, **kwargs)
    return model


# synchronization functions
def all_reduce(tensor, op="sum"):
    world_size = get_world_size()

    if world_size == 1:
        return tensor

    if op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "avg":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    else:
        raise ValueError(f'"{op}" is an invalid reduce operation!')

    return tensor


def reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = get_world_size()

    if world_size == 1:
        return tensor

    dist.reduce(tensor, dst=0, op=op)

    return tensor


def gather(data, *args, **kwargs):
    world_size = get_world_size()

    if world_size == 1:
        return [data]

    output_list = [torch.zeros_like(data) for _ in range(world_size)]

    if is_primary():
        dist.gather(data, gather_list=output_list, *args, **kwargs)
    else:
        dist.gather(data, *args, **kwargs)

    return output_list


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if is_dist_avail_and_initialized():
        for p in params:
            with torch.no_grad():
                dist.broadcast(p, 0)


def barrier(*args, **kwargs):
    world_size = get_world_size()
    if world_size == 1:
        return
    dist.barrier(*args, **kwargs)


# wrapper with same functionality but better readability as barrier
def wait_for_everyone(*args, **kwargs):
    barrier(*args, **kwargs)


def print_primary(*args, **kwargs):
    if is_primary():
        print(*args, **kwargs)


def print0(*args, **kwargs):
    print_primary(*args, **kwargs)
