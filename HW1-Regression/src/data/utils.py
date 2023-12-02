import torch
from torch.utils.data import random_split


def train_valid_split(data_set, valid_ratio, seed):
    r"""Split provided training data into training and validation set

    Returns:
        train_set, valid_set"""
    valid_set_size = int(len(data_set) * valid_ratio)
    train_set_size = len(data_set) - valid_set_size
    train_indices, valid_indices = random_split(data_set,
                                                [train_set_size, valid_set_size],
                                                torch.Generator().manual_seed(seed))
    train_set = data_set[train_indices.indices]
    valid_set = data_set[valid_indices.indices]

    return train_set, valid_set


def select_feat(data_set):
    """Select useful features to perform regression.

    Returns:
        data_set: Data after selecting useful features."""
    cor = data_set.corr()["tested_positive"]
    best_cor = cor[cor > 0.5]
    data_set = data_set[best_cor.index]
    return data_set.values
