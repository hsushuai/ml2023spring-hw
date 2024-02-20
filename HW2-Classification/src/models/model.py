from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class RNNPhonemeClassifier(nn.Module):
    def __init__(self, lstm_num_layers, lstm_hidden_size, mlp_hidden_layers, mlp_hidden_size,  dropout):
        super().__init__()
        self.rnn = nn.LSTM(39, lstm_hidden_size, lstm_num_layers, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.mlp = nn.Sequential(
            *[BasicBlock(mlp_hidden_size, dropout)
              for _ in range(mlp_hidden_layers)],
            nn.LazyLinear(41)
        )

    def forward(self, x):
        # x.shape: (batch_size, seq_len, RNN_input_size)
        x, _ = self.rnn(x)  # => (batch_size, seq_len, RNN_hidden_size)
        x = x[:, -1]  # => (batch_size, RNN_hidden_size)
        x = self.mlp(x)  # => (batch_size, labels)
        return x
