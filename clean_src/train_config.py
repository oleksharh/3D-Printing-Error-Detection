import os
from datetime import datetime
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torchvision import transforms
from pathlib import Path

DATE = datetime.now().strftime("%d%m%Y")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    BASE_DIR.parent / "stages"
).resolve()  # This is the dataset with telemetry/additonal metadata can be found

#######################
### TRAINING CONFIG ###
#######################
INITIAL_LR = 0.003

BATCH_SIZE = 192
MAX_EPOCHS = 50

NUM_NODES = 1
NUM_GPUS = 1
ACCELERATOR = "gpu"
#######################


def get_dataset_config(dataset_switch):
    if dataset_switch == 0:
        # INITIAL LAYER DATASET (STAGE 1)
        return {
            "name": "initial_layer_dataset",
            "csv_path": os.path.join(
                DATA_DIR,
                "stage1/initial_layer_dataset.csv",
            ),
            "mean": [0.1523144692182541, 0.15893325209617615, 0.08561990410089493],
            "std": [0.10251416265964508, 0.10434339195489883, 0.07971586287021637],
        }

    elif dataset_switch == 1:
        # FULL DATASET (STAGE 2)
        return {
            "name": "full_dataset",
            "csv_path": os.path.join(
                DATA_DIR,
                "stage2/full_dataset.csv",
            ),
            "mean": [0.2916452884674072, 0.2713455855846405, 0.13948898017406464],
            "std": [0.12008921056985855, 0.11583106219768524, 0.10315410792827606],
        }

    elif dataset_switch == 2:
        # BALANCED DATASET (STAGE 3)
        return {
            "name": "balanced_dataset",
            "csv_path": os.path.join(
                DATA_DIR,
                "stage3/balanced_dataset.csv",
            ),
            "mean": [0.2925814, 0.2713622, 0.14409496],
            "std": [0.0680447, 0.06964592, 0.0779964],
        }

    elif dataset_switch == 3:
        # Refactoring dataset/ Helper
        return {
            "name": "refactor",
            "csv_path": os.path.join(
                DATA_DIR,
                "refactor.csv",
            ),
            "mean": [0.5, 0.5, 0.5],
            "std": [0.05, 0.05, 0.05],
        }


def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.Generator().manual_seed(seed) # this prevents data leakage possibly # TODO




# sample to the original smoll dataset
# preprocess = transforms.Compose(
#     [
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.2915257, 0.27048784, 0.14393276],
#             [0.2915257, 0.27048784, 0.14393276],
#         ),
#     ],
# )
