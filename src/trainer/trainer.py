import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from .utils import gpu, num_gpus as device_count
import os
from ..base import Module, DataModule, HyperParameters
import logging
from torch.utils.data import DataLoader
from ..utils import Metrics, Timer, seconds_to_hms

logger = logging.getLogger(__name__)


class Trainer(HyperParameters):
    """The base class for training models with data."""
    model: Module
    max_epochs: int
    early_stopping: int
    grad_clip_threshold: float
    output_dir: str
    optim:  Optimizer
    epoch: int
    stale: int
    best_metric: float
    train_dataloader: DataLoader
    valid_dataloader: DataLoader
    num_train_batches: int
    num_valid_batches: int

    def __init__(self, max_epochs, output_dir, early_stopping, num_gpus=1, grad_clip_threshold=0):
        self.save_hyperparameters()
        self.gpus = [gpu(i) for i in range(min(num_gpus, device_count()))]
        self.writer = SummaryWriter(log_dir=output_dir)

    def prepare_data(self, data: DataModule):
        self.train_dataloader = data.train_dataloader()
        self.valid_dataloader = data.valid_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_valid_batches = len(self.valid_dataloader) if self.valid_dataloader is not None else 0

    def prepare_model(self, model):
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model

    def fit(self, model: Module, data: DataModule):
        timer = Timer()
        logger.info("Training the model.")
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.stale = 0
        self.best_metric = float("inf") if model.metrics.downward else 0.0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            # early stopping
            if self.early_stopping <= self.stale:
                logger.info(f"Training halted as the model did not exhibit improvement. Best validation {self.model.metrics.main_key}: {self.best_metric:.4f}, Total time: {timer.stop() / 60:.2f} min.")
                return
        logger.info(f"All epochs training complete. Best validation {self.model.metrics.main_key}: {self.best_metric:.4f}, Total time: {timer.stop() / 60:.2f} min.")

    def fit_epoch(self):
        # training
        timer = Timer()
        self.model.train()
        self.model.metrics.reset()
        for idx, batch in enumerate(self.train_dataloader):
            # step
            loss = self.model.step(self.prepare_batch(batch))
            self.optim.zero_grad()
            loss.backward()
            if self.grad_clip_threshold > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_threshold)
            self.optim.step()
        self.log_metrics(self.model.metrics, timer.stop(), True)
        # validation
        self.model.eval()
        self.model.metrics.reset()
        timer.reset()
        for idx, batch in enumerate(self.valid_dataloader):
            with torch.no_grad():
                loss = self.model.step(self.prepare_batch(batch))
        self.log_metrics(self.model.metrics, timer.stop(), False)
        self.compare_and_save(self.model.metrics)

    def log_metrics(self, metrics: Metrics, time_cost: float, train: bool):
        msg = f"Epoch {self.epoch + 1}/{self.max_epochs}, "
        num_steps = self.num_train_batches if train else self.num_valid_batches
        split_type = "Train" if train else "Valid"
        msg += f"step={num_steps:<{len(str(self.num_train_batches))}} | {split_type} "
        for i, kv in enumerate(metrics):
            msg += f"{kv[0]}: {kv[1] / len(metrics):.4f}" if i == 0 else f", {kv[0]}: {kv[1] / len(metrics):.4f}"
            self.writer.add_scalars(kv[0], {split_type: kv[1] / len(metrics)}, self.epoch + 1)
        msg += f" | {seconds_to_hms(time_cost)}, {num_steps / time_cost:.1f} steps/sec"
        logger.info(msg)

    def compare_and_save(self, metrics: Metrics):
        if (metrics.downward and metrics.main / len(metrics) < self.best_metric) or \
                (not metrics.downward and metrics.main / len(metrics) > self.best_metric):
            torch.save(self.model, os.path.join(self.output_dir, "model.pt"))
            logger.info(f"Saved current best model.")
            self.best_metric = metrics.main / len(metrics)
            self.stale = 0
        else:
            self.stale += 1

    def prepare_batch(self, batch):
        if self.gpus:
            batch = [_.to(self.gpus[0]) for _ in batch]
        return batch
