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

    def __init__(self, hidden_size, num_layers, batch_size, concat_nframes, lr, dropout, weight_decay):
        super().__init__(lr)
        self.save_hyperparameters()
        self.lstm = nn.LSTM(39, hidden_size, num_layers, dropout=dropout, bidirectional=True, batch_first=True)  # feature_dim = 39
        # self.fc = nn.LazyLinear(41)  # 41 classes.
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.LazyLinear(41)
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, X):
        # output, _ = self.lstm(X)
        # return self.fc(output[:, -1, :])  # (batch_size, 41)
        out, (h_n, c_n) = self.lstm(X)
        return self.fc(torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1))

    def step(self, batch):
        X, y = batch
        logits = self(X)
        return self.loss(logits, y), self.accuracy(logits, y)

    def predict(self, test_loader):
        logger.info("Predicting on the test set.")
        self.eval()
        preds = []
        with torch.no_grad():
            for X in tqdm(test_loader):
                X = X[0].cuda()
                logits = self(X)
                _, pred = torch.max(logits, 1)
                preds.extend(pred.cpu().numpy().tolist())
        return preds


class FoodClassifier(Classifier):
    def __init__(self, lr):
        super().__init__(lr)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
