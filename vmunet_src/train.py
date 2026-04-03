from vmunet_src.model.vmamba import VSSMEncoderOnly

print(VSSMEncoderOnly)

import os
import argparse
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
MAX_EPOCHS = 50

NUM_NODES = 1
NUM_GPUS = 1
ACCELERATOR = "gpu"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    BASE_DIR.parent / "stages"
).resolve()
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

dataset = {
            "name": "full_dataset",
            "csv_path": os.path.join(
                DATA_DIR,
                "stage2/full_dataset.csv",
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

    torch.Generator().manual_seed(seed) # this prevents data leakage possibly # TODO


import numpy as np

# parser = argparse.ArgumentParser()

if __name__ == "__main__":
    set_seed(1234)

    data = ParametersDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR, # images themselves
        csv_file=dataset["csv_path"], # print telemetry and image labels
        dataset_name=dataset["name"],
        mean=dataset["mean"],
        std=dataset["std"],
        workers=6,
    )

    model = VMambaMultiHeadClassifier(
        encoder=VSSMEncoderOnly(
            d_model=96,
            d_state=16,
            d_conv=3,
            expand=2,
        ),
        num_classes=3,
        # checkpoint_path="/home/alex/FYP/lightning_logs/version_55/checkpoints/epoch=19-step=26480.ckpt",
        # checkpoint_path="/home/alex/FYP/lightning_logs/version_56/checkpoints/epoch=1-step=41494.ckpt",
        checkpoint_path="/home/alex/FYP/lightning_logs/version_59/checkpoints/epoch=1-step=41494.ckpt"
    )


    profiler = pl.profilers.AdvancedProfiler(dirpath="./profiler_logs")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        enable_progress_bar=True,      # ensures terminal progress bar
        log_every_n_steps=10,          # prints logs every N batches
        # check_val_every_n_epoch=1,     # run validation every epoch
        profiler=profiler,             # enable profiler
    )

    trainer.fit(model, data)

# initially lr encoder 1e-4 and heads 1e-3 then added cosine annealing and weights to losses and reduced lr to 1e-5 and 1e-4 now, increase lr to 3e-5 



# python -m vmunet_src.train