import math
import torch
import torch.optim as optim
import os
import torch.nn as nn
import socket
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, train_loader, valid_loader, training_config, device):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

        # Get the parameters from the configuration file
        self.batch_size = training_config["batch_size"]
        self.epochs = training_config["epochs"]
        self.learning_rate = training_config["learning_rate"]
        self.optimizer = training_config["optimizer"]
        self.weight_decay = training_config["weight_decay"]
        self.output_dir = training_config["output_dir"]
        self.criterion = training_config["criterion"]
        self.early_stop = training_config["early_stop"]

        # Define the optimizer
        if self.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)

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
        log_dir = os.path.join(self.output_dir, "runs", current_time + "_" + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self):
        best_loss, early_stop_count = math.inf, 0
        for epoch in range(self.epochs):
            # Train
            self.model.train()
            total_loss = 0
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach().item()
            avg_train_loss = total_loss / len(self.train_loader)

            # Validate
            self.model.eval()
            total_loss = 0
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                with torch.no_grad():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                total_loss += loss.detach().item()
            avg_valid_loss = total_loss / len(self.valid_loader)

            # Record and print related information
            print(f"\nEpoch {epoch+1}/{self.epochs}\nTrain Loss: {avg_train_loss:.4f}\nValid Loss: {avg_valid_loss:.4f}")
            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            self.writer.add_scalar("Loss/valid", avg_valid_loss, epoch)

            # Whether to early stop
            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                early_stop_count = 0
                torch.save(self.model.state_dict(), self.model_filename)
                print(f"\nSaving model with loss {avg_valid_loss:.4f}...")
            else:
                early_stop_count += 1

            if early_stop_count >= self.early_stop:
                print("\nModel is not improving, so we halt the training session.")
                break
        print(f"\nThe Best validation loss is {best_loss}")

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.valid_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy}")
