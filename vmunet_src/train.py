from vmunet_src.model.vmamba import VSSMEncoderOnly

print(VSSMEncoderOnly)

import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from vmunet_src.data.data_module import ParametersDataModule
from vmunet_src.model.vmamba_classifier import VMambaMultiHeadClassifier
import torch


torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())


INITIAL_LR = 0.0003

BATCH_SIZE = 32
MAX_EPOCHS = 100

NUM_NODES = 1
NUM_GPUS = 1
ACCELERATOR = "gpu"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR.parent / "stages").resolve()
print(f"DATA_DIR: {DATA_DIR}")

dataset = {
    "name": "initial_layer_dataset",
    "csv_path": os.path.join(
        DATA_DIR,
        "stage1/initial_layer_dataset.csv",
    ),
    "mean": [4444.555555, 0.5, 0.5],
    "std": [0.05, 0.05, 0.05],
}

from pytorch_lightning import seed_everything


def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.Generator().manual_seed(seed)


# parser = argparse.ArgumentParser()

if __name__ == "__main__":
    set_seed(1234)

    data = ParametersDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,  # images themselves
        csv_file=dataset["csv_path"],  # print telemetry and image labels
        dataset_name=dataset["name"],
        mean=dataset["mean"],
        std=dataset["std"],
        workers=4,
    )

    model = VMambaMultiHeadClassifier(
        encoder=VSSMEncoderOnly(
            d_model=96,
            d_state=16,
            d_conv=3,
            expand=2,
        ),
        num_classes=3,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, data)
