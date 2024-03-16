import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
from ..base import DataModule
from PIL import Image

logger = logging.getLogger(__name__)


class Food11Dataset(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.transform = transform
        self.files = [os.path.join(root, x) for x in os.listdir(root) if x.endswith(".jpg")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        fp = self.files[item]
        img = self.transform(Image.open(fp))
        try:
            label = int(fp.split("/")[-1].split("_")[0])
        except:
            label = -1
        return img, label


class Food11(DataModule):
    def __init__(self, root, batch_size):
        super().__init__(root, batch_size)
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(244, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        logger.info(f"Loading dataset from {root}")

        self.train_ds = Food11Dataset(os.path.join(self.root, "train"), self.transform_train)
        self.valid_ds = Food11Dataset(os.path.join(self.root, "valid"), self.transform_test)

        self.num_train = len(self.train_ds)
        self.num_valid = len(self.valid_ds)

        logger.info(f"Dataset loading completed. "
                    f"Number of training samples: {self.num_train}, number of validation samples: {self.num_valid}.")

    def train_dataloader(self, num_workers=4, pin_memory=True):
        return DataLoader(self.train_ds, self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    def valid_dataloader(self, num_workers=4, pin_memory=True):
        return DataLoader(self.valid_ds, self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    def test_dataloader(self, num_workers=4, pin_memory=False):
        logger.info(f"Loading dataset from {self.root}")
        test_ds = Food11Dataset(os.path.join(self.root, "test"), self.transform_test)
        test_ds_tfm = Food11Dataset(os.path.join(self.root, "test"), self.transform_train)
        logger.info(f"Dataset loading completed. Number of test samples: {len(test_ds)}.")
        return DataLoader(test_ds, self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory), \
            DataLoader(test_ds_tfm, self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
