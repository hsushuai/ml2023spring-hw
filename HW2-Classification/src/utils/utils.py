import numpy as np
import torch
import random
import os
from tqdm import tqdm
import gdown


def same_seeds(seed):
    """Fixes random number generator seed for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        feat (torch.Tensor): Tensor of features.
        concat_n (int): Totals frame after concatenation (concat_n = 2k + 1).
    """
    assert concat_n % 2 == 1  # n must be odd
    if concat_n < 2:
        return feat
    seq_len, feat_dim = feat.size(0), feat.size(1)
    feat = feat.repeat(1, concat_n)
    feat = feat.view(seq_len, concat_n, feat_dim).permute(1, 0, 2)  # (concat_n, seq_len, feat_dim)
    mid = concat_n // 2
    for r_idx in range(1, mid + 1):
        feat[mid + r_idx, :] = shift(feat[mid + r_idx], r_idx)
        feat[mid - r_idx, :] = shift(feat[mid - r_idx], -r_idx)
    return feat.permute(1, 0, 2).view(seq_len, concat_n * feat_dim)


def preprocess_data(data_dir, concat_nframes, train, valid_ratio=0.2):
    r"""Load and preprocess train and valid data or test data

    Args:
        data_dir (str): Path to `libriphone` folder
        concat_nframes (int): Totals frame after concat past and future features
        train (bool): Whether to preprocess train dataset
        valid_ratio (float): The ratio in the train data set used for validation

    Returns:
        (train_X, train_y, valid_X, valid_y) or test_X
    """
    if not os.path.exists(data_dir):
        if not os.path.exists(os.path.dirname(data_dir)):
            os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        output = os.path.join(os.path.dirname(data_dir), "ml2023spring-hw2.zip")
        gdown.extractall(gdown.download(id="1qzCRnywKh30mTbWUEjXuNT2isOCAPdO1", output=output))
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
    feat_ids = [i.strip('\n') for i in feat_ids]
    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)  # original feature dimensions is 39
    if train:
        y = torch.empty(max_len, dtype=torch.long)
    feat_dir = os.path.join(data_dir, "feat", "train") if train else os.path.join(data_dir, "feat", "test")
    idx = 0
    for i, feat_id in tqdm(enumerate(feat_ids), total=len(feat_ids), desc=f'Loading data'):
        feat = torch.load(os.path.join(feat_dir, f"{feat_id}.pt"))
        feat = concat_feat(feat, concat_nframes)
        if idx + feat.shape[0] > max_len:
            raise ValueError(f"The number of features exceeds the maximum allowed number of {max_len}.")
        X[idx: idx + feat.shape[0], :] = feat
        if train:
            label = torch.LongTensor(label_dict[feat_id])
            y[idx: idx + feat.shape[0]] = label
        idx += feat.shape[0]

    if train:  # split train and valid dataset
        train_len = int(idx * (1 - valid_ratio))
        print(f"\n[INFO] - # training dataset size: ({train_len}, {X.shape[1]})")
        print(f"[INFO] - # validation dataset size: ({idx - train_len}, {X.shape[1]})\n")
        return X[:train_len], y[:train_len], X[train_len:idx], y[train_len:idx]
    else:
        print(f"\n[INFO] - # testing dataset size: ({idx}, {X.shape[1]})\n")
        return X[:idx]


def predict_and_save(test_loader, model, filename):
    """Predict the test dataset and save the results to csv file"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    preds = []
    for x in tqdm(test_loader, desc="Testing     "):
        x = x.to(device)
        with torch.no_grad():
            outputs = model(x)
        _, pred = torch.max(outputs, dim=1)
        preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()

    with open(filename, 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(preds):
            f.write(f"{i},{y}\n")


if __name__ == "__main__":
    train_X, train_y, valid_X, valid_y = preprocess_data("../../data/libriphone", 3, True)
