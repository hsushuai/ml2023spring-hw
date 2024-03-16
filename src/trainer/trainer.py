import os
import logging
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import torch
from ..base import Module, DataModule, HyperParameters
from ..utils import Metrics, Timer, seconds_to_hms

logger = logging.getLogger(__name__)


class Trainer(HyperParameters):
    """The Trainer class."""

    model: Module
    train_dataloader: DataLoader
    valid_dataloader: DataLoader
    optimizer: Optimizer
    writer: SummaryWriter
    best_state_dict: dict
    best_metric: float
    output_dir: str

    max_epochs: Optional[int] = (None,)
    max_steps: Optional[int] = (None,)
    validation_after_n_steps: Optional[int] = (None,)
    scheduler: Optional[LRScheduler] = (None,)
    schedule_by_epoch: Optional[bool] = (True,)
    early_stopping: Optional[int] = (None,)
    gradient_clip_val: Optional[float] = (None,)
    save_best_freq: int = 1

    def __init__(
        self,
        output_dir: str,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        validation_after_n_steps: Optional[int] = None,
        scheduler: Optional[LRScheduler] = None,
        schedule_by_epoch: bool = True,
        early_stopping: Optional[int] = None,
        gradient_clip_val: Optional[float] = None,
        save_best_freq: int = 1,
    ):
        """Initialize the awesome trainer.

        Args:
            output_dir (str):
                The output directory where the model and logs will be saved.
            max_epochs (int, optional):
                The maximum number of epochs to train the model.
            max_steps (int, optional):
                The maximum number of steps to train the model.
            validation_after_n_steps (int, optional):
                Perform validation after every n steps.
            scheduler (LRScheduler, optional):
                The learning rate scheduler to use during training.
            schedule_by_epoch (bool, optional):
                Whether to schedule the learning rate by epoch or by step.
            early_stopping (int, optional):
                The number of epochs/steps to wait before early stopping if no improvement is observed.
            gradient_clip_val (float, optional):
                The value to clip the gradients to prevent exploding gradients.
            save_best_freq (int, optional):
                The frequency (in epochs/steps) at which to save the best model checkpoint.
        """
        self.save_hyperparameters()
        self.stale = 0
        self.timer = Timer(start=False)
        self.update_state = (
            False  # Flag the best model state dictionary updated but not saved
        )

    def fit(self, model: Module, data: DataModule):
        """Train the given model on the provided data.

        Args:
           model (Module): The model to be trained.
           data (DataModule): The data module containing the training and validation data loaders.

        Raises:
           ValueError: If neither max_epochs nor max_steps is provided.
           ValueError: If both max_epochs and max_steps are provided (only one should be specified).
        """
        if self.max_epochs is None and self.max_steps is None:
            raise ValueError("Either num_epochs or num_steps must be provided.")

        if self.max_epochs is not None and self.max_steps is not None:
            raise ValueError("Only one of num_epochs or num_steps should be provided.")

        self.prepare_training(model, data)

        logger.info("Start training...")
        self.timer.start()

        # training by epochs
        if self.max_epochs is not None:
            self.fit_by_epoch()

        # training by steps
        if self.max_steps is not None:
            self.fit_by_step()

        # save best model
        if self.update_state:
            self.save_checkpoint()

    def prepare_training(self, model, data):
        """Prepare the trainer for training by setting up the data loaders, model, optimizer, and writer.
        Args:
            model (Module): The model to be trained.
            data (DataModule): The data module containing the training and validation data loaders.
        """
        self.train_dataloader = data.train_dataloader()
        self.valid_dataloader = data.valid_dataloader()
        self.model = model.cuda()
        self.optimizer = model.configure_optimizers()
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.best_metric = float("inf") if model.metrics.downward else 0.0

    def prepare_batch(self, batch):
        """Move the batch data to GPU.

        Args:
            batch: The batch of data to be moved to GPU.

        Returns:
            A list of tensors on the GPU.
        """
        batch = [_.cuda() for _ in batch]
        return batch

    def fit_by_epoch(self):
        """Train the model by epochs."""
        self.cur_epoch = 0
        for self.cur_epoch in range(self.max_epochs):
            # prepare training
            timer = Timer()
            self.model.train()
            self.model.metrics.reset()

            # fit one epoch
            self.fit_one_epoch()

            # lr schedule
            if self.scheduler is not None and self.schedule_by_epoch:
                self.scheduler.step()

            # log training info
            self.log_metrics(self.model.metrics, timer.stop(), True)

            # prepare validation
            self.model.metrics.reset()
            timer.reset()

            # validation
            self.valid()

            # log validation info
            self.log_metrics(self.model.metrics, timer.stop(), False)

            # save current best model dict
            self.keep_best()

            # early stopping
            if self.early_stopping is not None and self.early_stopping <= self.stale:
                logger.info(
                    f"Training halted as the model did not exhibit improvement. "
                    f"Best validation {self.model.metrics.main_key}: {self.best_metric:.4f}, "
                    f"Total time: {self.timer.stop() / 60:.2f} min."
                )
                return
        logger.info(
            f"All epochs training complete. "
            f"Best validation {self.model.metrics.main_key}: {self.best_metric:.4f}, "
            f"Total time: {self.timer.stop() / 60:.2f} min."
        )

    def fit_by_step(self):
        """Train the model by steps."""
        train_iterator = iter(self.train_dataloader)
        self.cur_step = 0
        timer = Timer()
        for self.cur_step in range(self.max_steps):
            # iterate a batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)

            # prepare training
            self.model.train()

            # fit one step
            self.fit_one_step(batch)

            # lr schedule
            if self.scheduler is not None and not self.schedule_by_epoch:
                self.scheduler.step()

            if (
                self.validation_after_n_steps is not None
                and (self.cur_step + 1) % self.validation_after_n_steps == 0
            ):
                # log training info
                self.log_metrics(self.model.metrics, timer.stop(), True)

                # prepare validation
                self.model.metrics.reset()
                timer.reset()

                # validation
                self.valid()

                # log validation info
                self.log_metrics(self.model.metrics, timer.stop(), False)

                # save current best model dict
                self.keep_best()

                # rest
                timer.reset()
                self.model.metrics.reset()

                # early stopping
                if (
                    self.early_stopping is not None
                    and self.early_stopping <= self.stale
                ):
                    logger.info(
                        f"Training halted as the model did not exhibit improvement. "
                        f"Best validation {self.model.metrics.main_key}: {self.best_metric:.4f}, "
                        f"Total time: {self.timer.stop() / 60:.2f} min."
                    )
                    return
        logger.info(
            f"All steps training complete. "
            f"Best validation {self.model.metrics.main_key}: {self.best_metric:.4f}, "
            f"Total time: {self.timer.stop() / 60:.2f} min."
        )

    def fit_one_epoch(self):
        for batch in self.train_dataloader:
            self.fit_one_step(batch)

    def fit_one_step(self, batch):
        loss = self.model.step(self.prepare_batch(batch))
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_val
            )
        self.optimizer.step()

    def valid(self):
        self.model.eval()
        for batch in self.valid_dataloader:
            with torch.no_grad():
                self.model.step(self.prepare_batch(batch))

    def keep_best(self):
        if self.model.metrics.compare(self.best_metric):
            self.best_state_dict = self.model.state_dict()
            self.best_metric = self.model.metrics.main_value
            self.stale = 0
            self.update_state = True
        else:
            self.stale += 1

        cur_idx = self.cur_epoch if self.max_epochs is not None else self.cur_step
        if (cur_idx + 1) % self.save_best_freq == 0 and self.update_state:
            self.save_checkpoint()

    def save_checkpoint(self):
        self.update_state = False
        torch.save(self.best_state_dict, os.path.join(self.output_dir, "model.ckpt"))
        logger.info(
            f"Saved current best model "
            f"with validation {self.model.metrics.main_key} of {self.best_metric:.4f}."
        )

    def log_metrics(self, metrics: Metrics, cost_time, train: bool):
        if self.max_epochs is not None:
            cur_idx = self.cur_epoch + 1
            max_idx = self.max_epochs
            iter_typ = "Epoch"
        else:
            cur_idx = self.cur_step + 1
            max_idx = self.max_steps
            iter_typ = "Step"
        msg = f"{iter_typ} {cur_idx}/{max_idx}"
        train_or_valid = "train" if train else "valid"
        msg += f" | {train_or_valid}"
        for i, (k, v) in enumerate(metrics):
            msg += f" {k} {v:.4f}" if i == 0 else f", {k} {v:.4f}"
            self.writer.add_scalars(k, {train_or_valid: v}, cur_idx)
        msg += f" | {seconds_to_hms(cost_time)}, {metrics.num_samples / cost_time:.1f} steps/sec"
        logger.info(msg)
