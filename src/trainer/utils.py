import torch
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cpu():
    """Get the CPU device."""
    return torch.device('cpu')


def gpu(i=0):
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')


def num_gpus():
    """Get the number of available GPUs."""
    return torch.cuda.device_count()


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]


class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
