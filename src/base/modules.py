import inspect
from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes."""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


class Module(nn.Module, HyperParameters):
    """The base class of models."""
    train_loss: float
    valid_loss: float

    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.metrics = ["loss", float("inf")]

    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is not defined"
        return self.net(X)

    def step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1]), None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss(self, input, target, averaged=True):
        return F.mse_loss(input, target, reduction="mean" if averaged else "none")


class Classifier(Module):
    """The base class of classification models."""
    def __init__(self, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.metrics = ["acc", 0]

    def accuracy(self, logits, labels, averaged=True):
        """Compute the number of correct predictions."""
        compare = (logits.argmax(dim=-1) == labels).float()
        return compare.mean() if averaged else compare

    def step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1]), self.accuracy(self(*batch[:-1]), batch[-1])

    def loss(self, logits, labels, averaged=True):
        logits = logits.reshape((-1, logits.shape[-1]))
        labels = labels.reshape((-1,))
        return F.cross_entropy(logits, labels, reduction="mean" if averaged else "none")

    def layer_summary(self, X_shape):
        """Print each layer's output shape."""
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


class DataModule(HyperParameters):
    """The base class of data."""
    num_train: int
    num_valid: int
    X: torch.Tensor
    y: torch.Tensor
    batch_size: int
    root: str

    def __init__(self, root, batch_size):
        self.save_hyperparameters()

    def get_dataloader(self, train, num_workers, pin_memory):
        idx = slice(0, self.num_train) if train else slice(self.num_train, self.num_train + self.num_valid)
        return self.get_tensorloader((self.X, self.y), train, num_workers, pin_memory, idx)

    def train_dataloader(self, num_workers=0, pin_memory=False):
        return self.get_dataloader(True, num_workers, pin_memory)

    def valid_dataloader(self, num_workers=0, pin_memory=False):
        return self.get_dataloader(False, num_workers, pin_memory)

    def test_dataloader(self, num_workers=0, pin_memory=False):
        raise NotImplementedError

    def get_tensorloader(self, tensors, train, num_workers, pin_memory, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, self.batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin_memory)
