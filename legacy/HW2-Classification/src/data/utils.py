import torch
import os
import zipfile
from tqdm import tqdm


def unzip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)


def shift(X, n):
    """Shift tensor up or down.

    Args:
        X (torch.Tensor): A 2D tensor.
        n (int): Shift distance. n > 0 shift up, n < 0 shift down.
    """
    if n > 0:  # shift up
        top = X[n:]
        bottom = X[-1].repeat(n, 1)
    elif n < 0:  # shift down
        bottom = X[:n]
        top = X[0].repeat(-n, 1)
    else:
        return X
    return torch.concat((top, bottom), dim=0)


def concat_feat(feat, concat_n):
    r"""Concatenate past and future features k frames.

    Args:
        feat (torch.Tensor): The Tensor with shape of (T, 39).
        concat_n (int): Totals frame after concatenation (concat_n = 2k + 1).
    """
    assert concat_n % 2 == 1  # n must be odd
    if concat_n < 2:
        return feat
    seq_len, feat_dim = feat.size(0), feat.size(1)
    feat = feat.repeat(1, concat_n)
    feat = feat.view(seq_len, concat_n, feat_dim).permute(
        1, 0, 2)  # (concat_n, seq_len, feat_dim)
    mid = concat_n // 2
    for r_idx in range(1, mid + 1):
        feat[mid + r_idx, :] = shift(feat[mid + r_idx], r_idx)
        feat[mid - r_idx, :] = shift(feat[mid - r_idx], -r_idx)
    return feat.permute(1, 0, 2).view(seq_len, concat_n * feat_dim)


def preprocess_data(data_dir, output_dir, concat_nframes, train, valid_ratio):
    r"""Load and preprocess train and valid data or test data.

    Args:
        data_dir (str): Path to `ml2023spring-hw2.zip` directory.
        output_dir (str): Path to unzip the `ml2023spring-hw2.zip`.
        concat_nframes (int): Totals frame after concat past and future features.
        train (bool): Preprocess the training or test dataset.
        valid_ratio (float): The ratio in the train data set used for validation.

    Returns:
        (train_X, train_y, valid_X, valid_y) or test_X
    """
    if not os.path.isdir(os.path.join(output_dir, "libriphone")):
        unzip_file(os.path.join(data_dir, "ml2023spring-hw2.zip"), output_dir)
    data_dir = os.path.join(output_dir, "libriphone")
    # load label dict
    label_dict = {}
    if train:
        for line in open(os.path.join(data_dir, "train_labels.txt")).readlines():
            line = line.strip("\n").split(" ")
            label_dict[line[0]] = [int(p) for p in line[1:]]
        # load features id
        with open(os.path.join(data_dir, "train_split.txt")) as f:
            feat_ids = f.readlines()
    else:
        with open(os.path.join(data_dir, "test_split.txt")) as f:
            feat_ids = f.readlines()
    # load features and label
    feat_ids = [i.strip("\n") for i in feat_ids]
    max_len = 3000000
    # original feature dimensions is 39
    X = torch.empty(max_len, 39 * concat_nframes)
    if train:
        y = torch.empty(max_len, dtype=torch.long)
    feat_dir = os.path.join(data_dir, "feat", "train") if train else os.path.join(
        data_dir, "feat", "test")
    idx = 0
    print("")
    for feat_id in tqdm(feat_ids, desc="Load features"):
        feat = torch.load(os.path.join(feat_dir, f"{feat_id}.pt"))
        feat = concat_feat(feat, concat_nframes)
        if idx + feat.shape[0] > max_len:
            raise ValueError(
                f"The number of features exceeds the maximum allowed number of {max_len}.")
        X[idx: idx + feat.shape[0], :] = feat
        if train:
            label = torch.LongTensor(label_dict[feat_id])
            y[idx: idx + feat.shape[0]] = label
        idx += feat.shape[0]
    if train:  # split train and valid dataset
        train_len = int(idx * (1 - valid_ratio))
        print(f"\nTraining dataset size: ({train_len}, {X.shape[1]})")
        print(f"Validation dataset size: ({idx - train_len}, {X.shape[1]})")
        return X[:train_len], y[:train_len], X[train_len:idx], y[train_len:idx]
    else:
        print(f"\nTest dataset size: ({idx}, {X.shape[1]})")
        return X[:idx]
