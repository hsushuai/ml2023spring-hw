import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from ..base import DataModule
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Libriphone(DataModule):
    concat_nframes: int  # n must be odd.

    def __init__(self, batch_size, valid_ratio, concat_nframes, root):
        super().__init__(batch_size, root)
        self.save_hyperparameters()

        label_dict = {}
        logger.info(f"Loading dataset from '{self.root}'.")
        try:
            for line in open(os.path.join(self.root, "train_labels.txt")):
                line = line.strip("\n").split(" ")
                label_dict[line[0]] = [int(p) for p in line[1:]]
            with open(os.path.join(self.root, "train_split.txt")) as f:
                feat_ids = f.readlines()
            feat_ids = [i.strip("\n") for i in feat_ids]
            self.X, self.y = torch.Tensor([]), torch.LongTensor([])
            for feat_id in tqdm(feat_ids):
                feat = torch.load(os.path.join(self.root, "feat", "train", f"{feat_id}.pt"))
                feat = self._preprocess(feat)
                self.X = torch.cat((self.X, feat), dim=0)
                label = torch.LongTensor(label_dict[feat_id])
                self.y = torch.cat((self.y, label), dim=0)
            self.X = self.X.view(-1, self.concat_nframes, 39)  # feature dim is 39
            logger.info(f"Dataset loaded successfully. Data size: {self.X.shape}.")
        except Exception as e:
            logger.critical(e)
            raise e
        self.num_valid = int(self.X.shape[0] * valid_ratio)
        self.num_train = self.X.shape[0] - self.num_valid

    def _preprocess(self, feat):
        """Concatenate past and future features frames."""
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

        assert self.concat_nframes % 2 == 1, "n must be odd."
        if self.concat_nframes < 2:
            return feat
        seq_len, feat_dim = feat.size(0), feat.size(1)
        feat = feat.repeat(1, self.concat_nframes)
        feat = feat.view(seq_len, self.concat_nframes, feat_dim).permute(1, 0, 2)  # (concat_n, seq_len, feat_dim)
        mid = self.concat_nframes // 2
        for r_idx in range(1, mid + 1):
            feat[mid + r_idx, :] = shift(feat[mid + r_idx], r_idx)
            feat[mid - r_idx, :] = shift(feat[mid - r_idx], -r_idx)
        return feat.permute(1, 0, 2).view(seq_len, self.concat_nframes * feat_dim)

    def test_dataloader(self, num_workers=0, pin_memory=False):
        logger.info(f"Loading dataset from '{self.root}'.")
        try:
            with open(os.path.join(self.root, "test_split.txt")) as f:
                feat_ids = f.readlines()
            feat_ids = [i.strip("\n") for i in feat_ids]
            X = torch.Tensor([])
            for feat_id in tqdm(feat_ids):
                feat = torch.load(os.path.join(self.root, "feat", "test", f"{feat_id}.pt"))
                feat = self._preprocess(feat)
                X = torch.cat((X, feat), dim=0)
            X = X.view(-1, self.concat_nframes, 39)  # feature dim is 39
            logger.info(f"Dataset loaded successfully. Data size: {X.shape}.")
        except Exception as e:
            logger.critical(e)
            raise e
        dataset = TensorDataset(X)
        return DataLoader(dataset, self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)