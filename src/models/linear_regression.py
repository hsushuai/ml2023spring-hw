import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..base import Module, Classifier
from tqdm import tqdm
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LinearRegression(Module):
    weight_decay: float

    def __init__(self, hidden_size, num_layers, lr, weight_decay):
        super().__init__(lr)
        self.save_hyperparameters()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.LazyLinear(hidden_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.LazyLinear(1))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X).squeeze(1)  # (B, 1) -> (B)

    def predict(self, test_loader):
        logger.info("Predicting on the test set.")
        self.eval()
        preds = []
        with torch.no_grad():
            for X in tqdm(test_loader):
                X = X[0].cuda()
                y_hat = self(X)
                preds.extend(y_hat.cpu().numpy())
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)