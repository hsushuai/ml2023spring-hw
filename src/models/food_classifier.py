import torch.optim
from torch import nn
from ..base import Classifier
import logging
from torchvision.models import alexnet, resnet18, resnet50, resnet101, squeezenet1_0, densenet121

logger = logging.getLogger(__name__)


def get_net(net: str):
    if net == "alexnet":
        return alexnet(num_classes=11)
    elif net == "resnet18":
        return resnet18(num_classes=11)
    elif net == "resnet50":
        return resnet50(num_classes=11)
    elif net == "resnet101":
        return resnet101(num_classes=11)
    elif net == "squeezenet1_0":
        return squeezenet1_0(num_classes=11)
    elif net == "densenet121":
        return densenet121(num_classes=11)


class FoodClassifier(Classifier):
    weight_dacey: float

    def __init__(self, lr, weight_dacey, net: str):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = get_net(net)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_dacey)

    def predict(self, test_loader, test_tfm_loader):
        logger.info("Predicting on the test set.")
        self.eval()
        preds = []
        with torch.no_grad():
            for X, X_tfm in zip(test_loader, test_tfm_loader):
                X, X_tfm = X[0].cuda(), X_tfm[0].cuda()
                logits = 0.8 * self(X) + 0.2 * self(X_tfm)
                _, pred = torch.max(logits, 1)
                preds.extend(pred.cpu().numpy().tolist())
        return preds
