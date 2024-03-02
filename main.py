import torch
import yaml
import argparse
import os
import socket
import time
import logging
from src.run_hw import run_hw
import warnings

assert torch.cuda.is_available(), "No GPUs available."


def flatten_dict(d, keep_parent_key=False, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, keep_parent_key, new_key, sep=sep).items())
        else:
            if keep_parent_key:
                items.append((new_key, v))
            else:
                items.append((k, v))
    return dict(items)


def load_configs(filepath, flatten=True):
    """Load configs from yaml file."""
    with open(filepath, "r") as f:
        configs = yaml.safe_load(f)
    if not flatten:
        return configs
    return flatten_dict(configs)


# ====================
#    Argument Parse
# ====================
parser = argparse.ArgumentParser(description="Run homework assignments.")
parser.add_argument("hw", choices=[f"hw{i}" for i in range(1, 16)],
                    help="Specify the homework assignment. Choose from 'hw1' to 'hw15'.")
args, remaining_args = parser.parse_known_args()


# ====================
#     Load Configs
# ====================
configs = load_configs(os.path.join("configs", f"{args.hw}_config.yaml"))


# ===================
#    Add Argument
# ===================
parser = argparse.ArgumentParser(description="Set the configuration parameters.")
for key, value in configs.items():
    parser.add_argument(f"--{key}", type=type(value), default=value)


# ===================
#   Update Configs
# ===================
configs = vars(parser.parse_args(remaining_args))
configs["output_dir"] = os.path.join(configs["output_dir"], f"{args.hw}-{socket.gethostname().replace('-', '_')}-{time.strftime('%Y%m%d%H%M%S')}")
os.makedirs(configs["output_dir"], exist_ok=True)

# ===================
#       Logging
# ===================
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.lazy")
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = logging.FileHandler(os.path.join(configs["output_dir"], "running_log.txt"))
console_handler = logging.StreamHandler()
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


# ===================
#         RUN
# ===================
run_hw(args.hw, configs)
