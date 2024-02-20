import os
import numpy as np
import torch
import random
import yaml


def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file {file_path}: {e}")
            return None


def same_seeds(seed):
    """Fixes random number generator seed for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_and_save(test_loader, model, output_dir, concat_nframes, device):
    """Predict the test dataset and save the results as a csv file to the output directory."""
    print("\nPredicting on the test dataset...")
    model.eval()
    preds = []
    for x in test_loader:
        x = x.to(device)
        x = x.view(-1, concat_nframes, 39).to(device)
        with torch.no_grad():
            outputs = model(x)
        _, pred = torch.max(outputs, dim=1)
        preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    with open(os.path.join(output_dir, "pred.csv"), "w") as f:
        f.write("Id,Class\n")
        for i, y in enumerate(preds):
            f.write(f"{i},{y}\n")
