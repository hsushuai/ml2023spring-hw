from torch import nn, optim
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import socket
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, valid_loader, training_config, concat_nframes, device):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.concat_nframes = concat_nframes
        self.device = device

        # Get the parameters from the configuration file
        self.batch_size = training_config["batch_size"]
        self.num_epochs = training_config["num_epochs"]
        self.learning_rate = training_config["learning_rate"]
        self.optimizer = training_config["optimizer"]
        self.weight_decay = training_config["weight_decay"]
        self.output_dir = training_config["output_dir"]
        self.criterion = training_config["criterion"]
        self.early_stop = training_config["early_stop"]

        # Define the optimizer
        if self.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(
            ), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(
            ), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        elif self.optimizer == "AdamW":
            self.optimizer = optim.AdamW(self.model.parameters(
            ), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Define the criterion
        if self.criterion == "mse":
            self.criterion = nn.MSELoss()
        elif self.criterion == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()

        # Define the path to save models
        model_dir = os.path.join(self.output_dir, "models")
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        self.model_filename = os.path.join(model_dir, "model.ckpt")

        # Define the SummaryWriter of tensorboard
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = os.path.join(self.output_dir, "runs",
                               current_time + "_" + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self):
        best_acc, early_stop_count = 0.0, 0
        for epoch in range(self.num_epochs):
            train_acc, train_loss = 0.0, 0.0
            self.model.train()
            print("")
            for features, labels in tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{self.num_epochs}"):
                features, labels = features.to(self.device), labels.to(self.device)
                features = features.view(-1, self.concat_nframes, 39).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, pred = torch.max(outputs, 1)
                train_acc += (pred.detach() == labels.detach()).sum().item()
                train_loss += loss.item()
            valid_loss, valid_acc = self.valid(epoch)

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}"
                  f"\nTrain Acc: {train_acc / len(self.train_loader.dataset):3.5f} Loss: {train_loss / len(self.train_loader):3.5f}"
                  f"\nValid Acc: {valid_acc / len(self.valid_loader.dataset):3.5f} loss: {valid_loss / len(self.valid_loader):3.5f}")
            self.writer.add_scalar(
                "Train/Loss", train_loss / len(self.train_loader), epoch)
            self.writer.add_scalar(
                "Train/Acc", train_acc / len(self.train_loader.dataset), epoch)
            self.writer.add_scalar(
                "Valid/Loss", valid_loss / len(self.valid_loader), epoch)
            self.writer.add_scalar(
                "Valid/Acc", valid_acc / len(self.valid_loader.dataset), epoch)

            if valid_acc > best_acc:
                early_stop_count = 0
                best_acc = valid_acc
                torch.save(self.model.state_dict(), self.model_filename)
                print(
                    f"\nSaved model with acc {best_acc/len(self.valid_loader.dataset):.5f}")
            else:
                early_stop_count += 1
            if early_stop_count >= self.early_stop:
                print("\nModel is not improving, so we halt the training session.")
                break
        print(
            f"\nThe best validation accuracy is {best_acc/len(self.valid_loader.dataset)}")

    def valid(self, epoch):
        valid_acc, valid_loss = 0.0, 0.0
        self.model.eval()
        for features, labels in tqdm(self.valid_loader, desc=f"Valid Epoch {epoch+1}/{self.num_epochs}"):
            features, labels = features.to(self.device), labels.to(self.device)
            features = features.view(-1, self.concat_nframes,
                                     39).to(self.device)
            with torch.no_grad():
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
            _, pred = torch.max(outputs, 1)
            valid_acc += (pred.cpu() == labels.cpu()).sum().item()
            valid_loss += loss.item()
        return valid_loss, valid_acc
