import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from ..base import DataModule
import logging

logger = logging.getLogger(__name__)


class COVID19(DataModule):
    """The Covid-19 dataset of hw1."""
    def __init__(self, batch_size, valid_ratio, root):
        super().__init__(root, batch_size)
        self.save_hyperparameters()
        filepath = os.path.join(self.root, "covid_train.csv")
        logger.info(f"Loading dataset from '{filepath}'.")
        try:
            data = pd.read_csv(filepath)
        except Exception as e:
            logger.critical(e)
            raise e
        logger.info(f"Dataset loaded successfully. Total samples: {len(data)}, data size: {data.shape}.")
        data = torch.Tensor(self._preprocess(data))
        self.X, self.y = data[:, :-1], data[:, -1]
        self.num_valid = int(len(data) * valid_ratio)
        self.num_train = len(data) - self.num_valid
        self.shape = data.shape

    def __sizeof__(self):
        return self.shape

    def _preprocess(self, data, shuffle=True):
        """Select useful features to perform regression."""
        logger.info("Preprocessing data: selecting features.")
        cor = data.corr()["tested_positive"]
        best_cor = cor[cor > 0.5]
        data = data[best_cor.index]
        # shuffle
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
        logger.info(f"Data preprocessing completed. Data size: {data.shape}.")
        return data.values

    def test_dataloader(self, num_workers=0, pin_memory=False):
        filepath = os.path.join(self.root, "covid_test.csv")
        logger.info(f"Loading dataset from '{filepath}'.")
        try:
            data = pd.read_csv(filepath)
        except Exception as e:
            logger.critical(e)
            raise e
        logger.info(f"Dataset loaded successfully. Total samples: {len(data)}, data size: {data.shape}.")
        data = torch.Tensor(self._preprocess(data, shuffle=False))
        dataset = TensorDataset(data)
        return DataLoader(dataset, self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)