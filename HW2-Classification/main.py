import os
import warnings
from src.data import create_dataloader
from src.models import RNNPhonemeClassifier
from src.utils import predict_and_save, load_yaml_file, same_seeds
from src.training import Trainer
import torch
import argparse
import gc


def main():
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="torch.nn.modules.lazy")

    config = load_yaml_file("configs/config.yaml")
    parser = argparse.ArgumentParser(description="HW2-Classification: Phoneme Recognition")
    parser.add_argument("--num_epochs", "-ep", type=int, default=config["training"]["num_epochs"],
                        help="Number of epochs for training")
    parser.add_argument("--batch_size", "-bs", type=int, default=config["training"]["batch_size"],
                        help="Batch size for training")
    parser.add_argument("--learning_rate", "-lr", type=float, default=config["training"]["learning_rate"],
                        help="Learning rate for training")
    parser.add_argument("--data_dir", "-data", type=str, default=config["data"]["data_dir"],
                        help="Directory containing data")
    parser.add_argument("--output_dir", "-output", type=str, default=config["training"]["output_dir"],
                        help="Directory for storing output")
    parser.add_argument("--concat_nframes", type=int, default=config["data"]["concat_nframes"],
                        help="Totals frame after concat past and future features, n must be odd (total 2k + 1 = n frames).")
    args = parser.parse_args()
    # Update config with command-line values
    config["training"]["epochs"] = args.num_epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.learning_rate
    config["data"]["data_dir"] = args.data_dir
    config["training"]["output_dir"] = args.output_dir
    config["data"]["concat_nframes"] = args.concat_nframes

    device = "cuda" if torch.cuda.is_available() else "cpu"

    same_seeds(config["data"]["seed"])
    train_loader, valid_loader = create_dataloader(config["data"]["data_dir"], config["training"]["output_dir"],
                                                   config["training"]["batch_size"], True,
                                                   config["data"]["concat_nframes"], config["data"]["valid_ratio"])

    model = RNNPhonemeClassifier(config["model"]["lstm_num_layers"], config["model"]["lstm_hidden_size"],
                                 config["model"]["mlp_hidden_layers"], config["model"]["mlp_hidden_size"], config["training"]["dropout"]).to(device)
    trainer = Trainer(model, train_loader, valid_loader,
                      config["training"], config["data"]["concat_nframes"], device)
    del train_loader, valid_loader
    gc.collect()
    trainer.train()

    model_filename = os.path.join(
        config["training"]["output_dir"], "models", "model.ckpt")
    model = RNNPhonemeClassifier(config["model"]["lstm_num_layers"], config["model"]["lstm_hidden_size"],
                                 config["model"]["mlp_hidden_layers"], config["model"]["mlp_hidden_size"], config["training"]["dropout"]).to(device)
    model.load_state_dict(torch.load(model_filename))
    test_loader = create_dataloader(config["data"]["data_dir"], config["training"]["output_dir"],
                                    config["training"]["batch_size"], False,
                                    config["data"]["concat_nframes"], config["data"]["valid_ratio"])
    predict_and_save(test_loader, model,
                     config["training"]["output_dir"], config["data"]["concat_nframes"], device)

    print(f"\n🎉 ALL DONE!\n")


if __name__ == "__main__":
    main()
