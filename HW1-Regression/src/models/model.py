import torch
import torch.nn as nn


class COVIDPerceiver(nn.Module):
    def __init__(self, hidden_size, hidden_layers):
        super(COVIDPerceiver, self).__init__()
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.LazyLinear(hidden_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.LazyLinear(1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B, 1) -> (B)
