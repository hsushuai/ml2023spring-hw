import torch
from torch.utils.tensorboard import SummaryWriter
from .utils import gpu, num_gpus as device_count
import os
from ..base import Module, DataModule, HyperParameters
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class Trainer(HyperParameters):
    """The base class for training models with data."""
    model: Module
    max_epochs: int
    early_stopping: int
    grad_clip_threshold: float
    output_dir: str

    def __init__(self, max_epochs, output_dir, early_stopping, num_gpus=1, grad_clip_threshold=0):
        self.save_hyperparameters()
        self.gpus = [gpu(i) for i in range(min(num_gpus, device_count()))]
        self.writer = SummaryWriter(log_dir=output_dir)

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.valid_dataloader = data.valid_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_valid_batches = len(self.valid_dataloader) if self.valid_dataloader is not None else 0

    def prepare_model(self, model):
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model

    def fit(self, model, data: DataModule):
        start_time = time.time()
        logger.info("Training the model.")
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.gloabel_step = 0
        self.stale = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            if self.early_stopping < self.stale:  # early stopping
                training_time = (time.time() - start_time) / 60
                logger.info(f"Training halted as the model did not exhibit improvement. Best validation {self.model.metrics[0]}: {self.model.metrics[1]:.3f}, Total time: {training_time:.2f} min.")
                return
        training_time = (time.time() - start_time) / 60
        logger.info(f"All epochs training complete. Best validation {self.model.metrics[0]}: {self.model.metrics[1]:.3f}, Total time: {training_time:.2f} min.")

    def fit_epoch(self):
        loss_list, acc_list = [], []
        # training
        self.model.train()
        for idx, batch in enumerate(tqdm(self.train_dataloader)):
            # step
            loss, acc = self.model.step(self.prepare_batch(batch))
            self.optim.zero_grad()
            loss.backward()
            if self.grad_clip_threshold > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_threshold)
            self.optim.step()

            # add scalar
            self.writer.add_scalar("Loss/train", loss.item(), self.epoch * self.num_train_batches + idx + 1)
            if acc is not None:
                self.writer.add_scalar("Accuracy/train", acc, self.epoch * self.num_train_batches + idx + 1)
            loss_list.append(loss.item())
            acc_list.append(acc)

        # logging
        avg_acc = sum(acc_list) / len(acc_list) if acc_list[0] is not None else None
        avg_loss = sum(loss_list) / len(loss_list)
        if acc_list[0] is None:
            msg = f"Epoch {self.epoch + 1}/{self.max_epochs} - Train loss: {avg_loss:.3f}"
        else:
            msg = f"Epoch {self.epoch + 1}/{self.max_epochs} - Train loss: {avg_loss:.3f}, acc: {avg_acc:.3f}"
        logger.info(msg)

        # validation
        self.model.eval()
        loss_list, acc_list = [], []
        for idx, batch in enumerate(tqdm(self.valid_dataloader)):
            with torch.no_grad():
                loss, acc = self.model.step(self.prepare_batch(batch))

            # add scalar
            self.writer.add_scalar("Loss/valid", loss.item(), self.epoch * self.num_valid_batches + idx + 1)
            if acc is not None:
                self.writer.add_scalar("Accuracy/valid", acc, self.epoch * self.num_valid_batches + idx + 1)
            loss_list.append(loss.item())
            acc_list.append(acc)

        # logging
        avg_acc = sum(acc_list) / len(acc_list) if acc_list[0] is not None else None
        avg_loss = sum(loss_list) / len(loss_list)
        if acc_list[0] is None:
            msg = f"Epoch {self.epoch + 1}/{self.max_epochs} - Valid loss: {avg_loss:.3f}"
        else:
            msg = f"Epoch {self.epoch + 1}/{self.max_epochs} - Valid loss: {avg_loss:.3f}, acc: {avg_acc:.3f}"
        logger.info(msg)

        # save best model
        if self.model.metrics[0] == "loss" and self.model.metrics[1] > avg_loss:
            fp = os.path.join(self.output_dir, "model.pt")
            torch.save(self.model, fp)
            self.model.metrics[1] = avg_loss
            logger.info(f"Save the best model in '{fp}'.")
            self.stale = 0
        elif self.model.metrics[0] == "acc" and self.model.metrics[1] < avg_acc:
            fp = os.path.join(self.output_dir, "model.pt")
            torch.save(self.model, fp)
            self.model.metrics[1] = avg_acc
            logger.info(f"Save the best model in '{fp}'.")
            self.stale = 0
        else:
            self.stale += 1

    def prepare_batch(self, batch):
        if self.gpus:
            batch = [_.to(self.gpus[0]) for _ in batch]
        return batch
