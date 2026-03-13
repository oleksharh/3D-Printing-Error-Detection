import os
import argparse
import pytorch_lightning as pl
from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier
from train_config import *



if __name__ == "__main__":  

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--seed", default=1234, type=int, help="Set seed for training"
    )

    args = parser.parse_args()
    seed = args.seed

    set_seed(seed)

    model = ParametersClassifier.load_from_checkpoint(
        checkpoint_path="C:\\FYP\\checkpoints\\stage1\\MHResAttNet-initial_layer_dataset-11032026-epoch=39-val_loss=0.62-val_acc=0.95.ckpt",
        num_classes=3,
        lr=INITIAL_LR,
        gpus=1,
        transfer=False,
    )
    model.eval()

    dataset_cfg = get_dataset_config(0)

    data = ParametersDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        csv_file=dataset_cfg["csv_path"],
        dataset_name=dataset_cfg["name"],
        per_img_normalisation=True,
        mean=dataset_cfg["mean"],
        std=dataset_cfg["std"],
        transform=False,
    )
    data.setup('test')

    trainer = pl.Trainer(
        num_nodes=1,
        precision="16-mixed",
    )

    trainer.test(model, datamodule=data)