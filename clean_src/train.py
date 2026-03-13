import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier
import torch

torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

from train_config import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s", "--seed", default=1234, type=int, help="Set seed for training"
)
parser.add_argument(
    "-e",
    "--epochs",
    default=MAX_EPOCHS,
    type=int,
    help="Number of epochs to train the model for",
)

parser.add_argument(
    "-lr",
    "--learning_rate",
    default=INITIAL_LR,
    type=float,
    help="Learning rate for the optimizer",
)

parser.add_argument(
    "-ds",
    "--dataset_switch",
    default=0,
    type=int,
    help="Switch to select which dataset to train on."
    " 0: initial_layer_dataset, "
    " 1: full_dataset, "
    " 2: balanced_dataset, "
    " 3: refactor dataset",
)

parser.add_argument(
    "-w",
    "--workers",
    default=8,
    type=int,
    help="Number of workers for data loading",
)

args = parser.parse_args()
seed = args.seed
set_seed(seed)


stage = args.dataset_switch
dataset_cfg = get_dataset_config(stage)

tb_logger = pl_loggers.TensorBoardLogger(
    save_dir="logs/",
    name=f"{DATE}-{stage}-{seed}",
)

log_path = tb_logger.log_dir

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=os.path.join(log_path, "checkpoints/"),
    filename=f"MHResAttNet-{dataset_cfg['name']}-"
    + "{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
    save_top_k=3,
    mode="min",
)


if __name__ == "__main__":
    model = ParametersClassifier(
        num_classes=3,
        lr=args.learning_rate,
        gpus=NUM_GPUS,
        transfer=False,
    )

    # model = ParametersClassifier.load_from_checkpoint(
    #     "C:/FYP/checkpoints/stage1/MHResAttNet-initial_layer_dataset-11032026-epoch=39-val_loss=0.62-val_acc=0.95.ckpt"
    # )

    data = ParametersDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR, # images themselves
        csv_file=dataset_cfg["csv_path"], # print telemetry and image labels
        dataset_name=dataset_cfg["name"],
        per_img_normalisation=True,
        mean=dataset_cfg["mean"],
        std=dataset_cfg["std"],
        workers=args.workers,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=args.epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )

    trainer.fit(model, data)
