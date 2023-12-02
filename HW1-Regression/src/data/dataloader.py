import os
import pandas as pd
from .dataset import COVID19Dataset
from .utils import select_feat, train_valid_split
from torch.utils.data import DataLoader


def create_dataloader(data_dir, batch_size, valid_ratio, seed):
    r"""Creat training, validation and test data loader.

    Returns:
        train_loader, valid_loader, test_loader"""

    # train_data size: 3009 x 89 (35 states + 18 features x 3 days)
    # test_data size: 997 x 88 (without last day's positive rate)
    train_data = pd.read_csv(os.path.join(data_dir, "covid_train.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "covid_test.csv"))

    train_data, test_data = select_feat(train_data), select_feat(test_data)
    train_data, valid_data = train_valid_split(train_data, valid_ratio, seed)

    # Print out the data size
    print(f"""train_data size: {train_data.shape}
valid_data size: {valid_data.shape}
test_data size: {test_data.shape}""")

    x_train, x_valid, x_test, y_train, y_valid = train_data[:, :-1], valid_data[:, :-1], test_data, train_data[:, -1], valid_data[:, -1]

    train_dataset = COVID19Dataset(x_train, y_train)
    valid_dataset = COVID19Dataset(x_valid, y_valid)
    test_dataset = COVID19Dataset(x_test)

    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, valid_loader, test_loader
