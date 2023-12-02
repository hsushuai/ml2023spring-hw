import warnings
import torch
from src.models import COVIDPerceiver
from src.training import Trainer
from src.data import create_dataloader
from src.utils import same_seed, predict, save_pred, load_yaml_file
import argparse
import os


if __name__ == "__main__":
    config = load_yaml_file("./configs/config.yaml")
    parser = argparse.ArgumentParser(description="HW1 Regression Training")
    parser.add_argument("--epochs", "-ep", type=int, default=config["training"]["epochs"], help="Number of epochs for training")
    parser.add_argument("--batch_size", "-bs", type=int, default=config["training"]["batch_size"], help="Batch size for training")
    parser.add_argument("--learning_rate", "-lr", type=float, default=config["training"]["learning_rate"], help="Learning rate for training")
    parser.add_argument("--data_dir", "-data", type=str, default=config["data"]["data_dir"], help="Directory containing data")
    parser.add_argument("--output_dir", "-output", type=str, default=config["training"]["output_dir"], help="Directory for storing output")
    args = parser.parse_args()
    # Update config with command-line values
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.learning_rate
    config["data"]["data_dir"] = args.data_dir
    config["training"]["output_dir"] = args.output_dir

    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.lazy")

    model_filename = os.path.join(config["training"]["output_dir"], "models", "model.ckpt")
    pred_filename = os.path.join(config["training"]["output_dir"], "pred.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    same_seed(config["data"]["seed"])
    model = COVIDPerceiver(config["model"]["hidden_size"], config["model"]["hidden_layers"]).to(device)
    train_loader, valid_loader, test_loader = create_dataloader(config["data"]["data_dir"], config["training"]["batch_size"], config["data"]["valid_ratio"], config["data"]["seed"])
    trainer = Trainer(model, train_loader, valid_loader, config["training"], device)
    trainer.train()

    model = COVIDPerceiver(config["model"]["hidden_size"], config["model"]["hidden_layers"]).to(device)
    model.load_state_dict(torch.load(model_filename))
    preds = predict(test_loader, model, device)
    save_pred(preds, pred_filename)
    print(f"\n🎉 ALL DONE!\n")

# Open terminal and run `tensorboard --logdir=output/runs` to visualize training process
