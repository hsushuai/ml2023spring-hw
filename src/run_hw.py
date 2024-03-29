import os.path
import time

from src.trainer import Trainer
from src.models import (
    LinearRegression,
    PhonemeClassifier,
    FoodClassifier,
    SpeakerClassifier,
)
from src.utils import same_seeds
from src.utils.scheduler import get_cosine_schedule_with_warmup
from src.data import COVID19, Libriphone, Food11, Voxceleb2
import torch
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def hw1(configs):
    same_seeds(configs["seed"])
    data = COVID19(
        configs["batch_size"], configs["valid_ratio"], root=configs["data_dir"]
    )

    model = LinearRegression(
        configs["hidden_size"],
        configs["num_layers"],
        configs["learning_rate"],
        configs["weight_decay"],
    )
    logger.info(model)

    trainer = Trainer(
        configs["output_dir"],
        max_epochs=configs["max_epochs"],
        early_stopping=configs["early_stopping"],
    )
    trainer.fit(model, data)

    test_loader = data.test_dataloader()
    fp = os.path.join(configs["output_dir"], "model.ckpt")
    model.load_state_dict(torch.load(fp))
    logger.info(f"Loaded the pre-trained model from '{fp}'.")

    preds = model.predict(test_loader)
    fp = os.path.join(configs["output_dir"], "submission.csv")
    pd.DataFrame({"tested_positive": preds}).to_csv(fp, index_label="Id")
    logger.info(f"Saved the inference results in {fp}.")


def hw2(configs):
    same_seeds(configs["seed"])
    data = Libriphone(
        configs["batch_size"],
        configs["valid_ratio"],
        configs["concat_nframes"],
        configs["data_dir"],
    )

    model = PhonemeClassifier(
        configs["hidden_size"],
        configs["num_layers"],
        configs["concat_nframes"],
        configs["learning_rate"],
        configs["dropout"],
        configs["weight_decay"],
    )
    logger.info(model)

    trainer = Trainer(
        configs["output_dir"],
        max_epochs=configs["max_epochs"],
        early_stopping=configs["early_stopping"],
    )
    trainer.fit(model, data)

    test_loader = data.test_dataloader()
    fp = os.path.join(configs["output_dir"], "model.ckpt")
    model.load_state_dict(torch.load(fp))
    logger.info(f"Loaded the pre-trained model from '{fp}'.")

    preds = model.predict(test_loader)
    fp = os.path.join(configs["output_dir"], "submission.csv")
    pd.DataFrame({"Class": preds}).to_csv(fp, index_label="Id")
    logger.info(f"Predict completed, saved the results in {fp}.")


def hw3(configs):
    same_seeds(configs["seed"])
    data = Food11(configs["data_dir"], configs["batch_size"])

    model = FoodClassifier(
        configs["learning_rate"], configs["weight_decay"], configs["model"]
    )
    logger.info(model)

    trainer = Trainer(
        configs["output_dir"],
        max_epochs=configs["max_epochs"],
        early_stopping=configs["early_stopping"],
    )
    trainer.fit(model, data)

    test_loader, test_tfm_loader = data.test_dataloader()
    fp = os.path.join(configs["output_dir"], "model.ckpt")
    model.load_state_dict(torch.load(fp))
    logger.info(f"Loaded the pre-trained model from '{fp}'.")

    preds = model.predict(test_loader, test_tfm_loader)
    fp = os.path.join(configs["output_dir"], "submission.csv")
    pd.DataFrame({"Category": preds}).to_csv(fp, index_label="Id")
    logger.info(f"Predict completed, saved the results in {fp}.")


def hw4(configs):
    same_seeds(configs["seed"])
    data = Voxceleb2(
        configs["data_dir"],
        configs["batch_size"],
        configs["segment_len"],
        configs["valid_ratio"],
    )

    logger.info("Building neural network model.")
    model = SpeakerClassifier(
        configs["d_model"],
        configs["learning_rate"],
        configs["dropout"],
        configs["weight_decay"],
    )
    logger.info(model)

    scheduler = get_cosine_schedule_with_warmup(
        model.configure_optimizers(), configs["warmup_steps"], configs["max_steps"]
    )

    trainer = Trainer(
        configs["output_dir"],
        max_steps=configs["max_steps"],
        validation_after_n_steps=configs["validation_after_n_steps"],
        save_best_freq=configs["save_best_freq"],
        scheduler=scheduler,
    )
    trainer.fit(model, data)

    test_loader = data.test_dataloader()
    fp = os.path.join(configs["output_dir"], "model.ckpt")
    model.load_state_dict(torch.load(fp))
    logger.info(f"Loaded the pre-trained model from '{fp}'.")

    results = model.inference(test_loader, configs["data_dir"])
    results = np.array(results)
    fp = os.path.join(configs["output_dir"], "submission.csv")
    pd.DataFrame({"Id": results[:, 0], "Category": results[:, 1]}).to_csv(
        fp, index=False
    )
    logger.info(f"Inference completed, saved the results in {fp}.")


def hw5(configs):
    print("Running hw5")
    print(configs)


def hw6(configs):
    print("Running hw6")
    print(configs)


def hw7(configs):
    print("Running hw7")
    print(configs)


def hw8(configs):
    print("Function hw8")


def hw9(configs):
    print("Function hw9")


def hw10(configs):
    print("Function hw10")


def hw11(configs):
    print("Function hw11")


def hw12(configs):
    print("Function hw12")


def hw13(configs):
    print("Function hw13")


def hw14(configs):
    print("Function hw14")


def hw15(configs):
    print("Function hw15")


hw_funcs = {
    "hw1": hw1,
    "hw2": hw2,
    "hw3": hw3,
    "hw4": hw4,
    "hw5": hw5,
    "hw6": hw6,
    "hw7": hw7,
    "hw8": hw8,
    "hw9": hw9,
    "hw10": hw10,
    "hw11": hw11,
    "hw12": hw12,
    "hw13": hw13,
    "hw14": hw14,
    "hw15": hw15,
}


def run_hw(hw, configs: dict):
    if hw in hw_funcs:
        try:
            start_time = time.time()
            logger.info(f"Start Running the {hw} process, with configs:\n{configs}")
            hw_funcs[hw](configs)
            running_time = (time.time() - start_time) / 60
            logger.info(
                f"The running process completed. Total time: {running_time:.2f} min."
            )
        except Exception as e:
            logger.critical(e)
            raise e
