import gc
from torch.utils.data import DataLoader
from .dataset import LibriDataset
from .utils import preprocess_data


def create_dataloader(data_dir, output_dir, batch_size, train, concat_nframes, valid_ratio):
    r"""Create training and validation dataloader or testing dataloader

    Args:
        data_dir (str): `libriphone` folder path.
        output_dir (str): Path to unzip the `ml2023spring-hw2.zip`.
        batch_size (int): The number of samples included in each batch.
        train (bool): Training and validation dataloader or test dataloader.
        concat_nframes (int): Totals frame after concat past and future features, n must be odd (total 2k + 1 = n frames).
        valid_ratio (float): Ratio of validation dataset in training data.
    Returns:
        (train_loader, valid_loader) or test_loader.
    """
    if train:
        # preprocess data
        train_X, train_y, valid_X, valid_y = preprocess_data(data_dir, output_dir, concat_nframes, train, valid_ratio)

        # get dataset
        train_set = LibriDataset(train_X, train_y)
        valid_set = LibriDataset(valid_X, valid_y)

        # remove raw feature to save memory
        del train_X, train_y, valid_X, valid_y
        gc.collect()

        # get dataloader
        train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, valid_loader
    else:
        test_X = preprocess_data(data_dir, output_dir, concat_nframes, train, valid_ratio)
        test_set = LibriDataset(test_X)
        test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return test_loader
