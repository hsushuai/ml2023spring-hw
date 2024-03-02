import torch


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

