import random

import torch
from ..base import DataModule
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import json
import os
from torch.nn.utils.rnn import pad_sequence
import logging

logger = logging.getLogger(__name__)


class Voxceleb2Dataset(Dataset):
    def __init__(self, root, segment_len=128):
        self.root = root
        self.segment_len = segment_len

        mapping = json.load(open(os.path.join(root, "mapping.json")))
        self.speaker2id = mapping["speaker2id"]

        metadata = json.load(open(os.path.join(root, "metadata.json")))["speakers"]
        self.num_speakers = len(metadata)
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_path, speaker = self.data[idx]
        mel = torch.load(os.path.join(self.root, feature_path))

        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = mel[start : start + self.segment_len]
        return torch.FloatTensor(mel), torch.LongTensor([speaker])


class Voxceleb2TestSet(Dataset):
    def __init__(self, root):
        metadata = json.load(open(os.path.join(root, "testdata.json")))
        self.root = root
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_path = self.data[idx]["feature_path"]
        return feature_path, torch.load(os.path.join(self.root, feature_path))


class Voxceleb2(DataModule):
    segment_len: int
    train_indices: None
    valid_indices: None

    def __init__(self, root, batch_size, segment_len, valid_ratio):
        super().__init__(root, batch_size)
        self.save_hyperparameters()
        self.dataset = Voxceleb2Dataset(root, segment_len)
        logger.info(f"Loaded dataset with size of {len(self.dataset)}.")
        self.num_valid = int(valid_ratio * len(self.dataset))
        self.num_train = len(self.dataset) - self.num_valid
        self.train_valid_split()

    def train_valid_split(self):
        train_set, valid_set = random_split(
            self.dataset, [self.num_train, self.num_valid]
        )
        self.train_indices, self.valid_indices = train_set.indices, valid_set.indices

    def train_dataloader(self, num_workers=0, pin_memory=False):
        if self.train_indices is None:
            self.train_valid_split()
        return DataLoader(
            Subset(self.dataset, self.train_indices),
            self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_train_batch,
        )

    def valid_dataloader(self, num_workers=0, pin_memory=False):
        if self.valid_indices is None:
            self.train_valid_split()
        return DataLoader(
            Subset(self.dataset, self.valid_indices),
            self.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_train_batch,
        )

    def test_dataloader(self, num_workers=0, pin_memory=False):
        test_ds = Voxceleb2TestSet(self.root)
        logger.info(f"Loaded dataset with size of {len(test_ds)}.")
        return DataLoader(
            test_ds,
            1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_test_batch,
        )


def collate_train_batch(batch):
    # Process features within a batch.
    """Collate a batch of data."""
    mels, speaker = zip(*batch)
    # Because we train the model batch by batch,
    # we need to pad the features in the same batch to make their lengths the same.
    # pad log 10^(-20) which is very small value.
    # Besides, Conformer model requires both inputs and lengths.
    input_lengths = torch.LongTensor([len(mel) for mel in mels])
    mels = pad_sequence(mels, batch_first=True, padding_value=-20)
    # mel: (batch size, length, 40)
    return mels, torch.FloatTensor(speaker).long(), input_lengths


def collate_test_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)
    input_lengths = torch.LongTensor([len(mel) for mel in mels])
    return feat_paths, torch.stack(mels), input_lengths
