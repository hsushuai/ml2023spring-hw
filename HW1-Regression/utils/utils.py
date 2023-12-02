import torch
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    """Split provided training data into training and validation set"""
    valid_set_size = int(len(data_set) * valid_ratio)
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set,
                                        [train_set_size, valid_set_size],
                                        torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    """Predict on test set"""
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


def feature_select(train_data, valid_data, test_data, select_all=True):
    """Select useful features to perform regression"""
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # TODO: Select suitable features columns
        feat_idx = list(range(35, raw_x_train.shape[1]))
    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:], y_train, y_valid
