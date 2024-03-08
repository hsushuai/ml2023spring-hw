import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..base import Module, Classifier
from tqdm import tqdm
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PhonemeClassifier(Classifier):
    concat_nframes: int
    weight_decay: float

    def __init__(self, hidden_size, num_layers, concat_nframes, lr, dropout, weight_decay):
        super().__init__(lr)
        self.save_hyperparameters()
        self.lstm = nn.LSTM(39, hidden_size, num_layers, dropout=dropout, bidirectional=True,
                            batch_first=True)  # feature_dim = 39
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.LazyLinear(41)  # 41 classes.
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, X):
        X = X.view(-1, self.concat_nframes, 39)  # feature_dim = 39
        out, (h_n, c_n) = self.lstm(X)
        return self.fc(torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1))

    def predict(self, test_loader):
        logger.info("Predicting on the test set.")
        self.eval()
        preds = []
        with torch.no_grad():
            for X in test_loader:
                X = X[0].cuda()
                logits = self(X)
                _, pred = torch.max(logits, 1)
                preds.extend(pred.cpu().numpy().tolist())
        return preds
