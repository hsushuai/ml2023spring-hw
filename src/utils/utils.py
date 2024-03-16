import time
import torch
import numpy as np
import random


def same_seeds(seed):
    """Fixed random seed for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Timer:
    """Record multiple running times."""

    def __init__(self, start=True):
        self.times = []
        if start:
            self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def reset(self):
        """Reset the timer."""
        self.times = []
        self.start()

    def stop(self, formatted=False):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        if formatted:
            return seconds_to_hms(self.times[-1])
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
