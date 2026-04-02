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

    set_seed(1234)

    model = ParametersClassifier.load_from_checkpoint(
        # checkpoint_path="C:\\FYP\\logs\\30032026-2-1234\\version_0\\checkpoints\\MHResAttNet-balanced_dataset-epoch=17-val_loss=1.20-val_acc=0.88.ckpt",
        # checkpoint_path="C:/FYP/logs/16032026-1-1234/version_2/checkpoints/MHResAttNet-full_dataset-epoch=07-val_loss=1.88-val_acc=0.82.ckpt",
        checkpoint_path="C:/FYP/logs/30032026-2-1234/version_0/checkpoints/MHResAttNet-balanced_dataset-epoch=17-val_loss=1.20-val_acc=0.88.ckpt",
        # checkpoint_path="C:/FYP/logs/01042026-4-3482/version_6/checkpoints/MHResAttNet-initial_layer_dataset_reduced-epoch=09-val_loss=1.93-val_acc=0.81.ckpt",
        num_classes=3,
        lr=INITIAL_LR,
        gpus=1,
        transfer=False,
        per_img_normalisation=True
    )
    model.eval() # uses mean and std from training, not that it matters for testing but just to be safe

    dataset_cfg = get_dataset_config(1)
    
    data = ParametersDataModule(
        batch_size=192,
        data_dir=DATA_DIR,
        csv_file=dataset_cfg["csv_path"],
        dataset_name=dataset_cfg["name"],
        per_img_normalisation=True,
        mean=dataset_cfg["mean"],
        std=dataset_cfg["std"],
    )

    trainer = pl.Trainer(
        num_nodes=1,
        precision="16-mixed",
    )

    trainer.test(model, datamodule=data)