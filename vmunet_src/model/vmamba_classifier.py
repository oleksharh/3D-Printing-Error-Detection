import torch
from torch import nn
import pytorch_lightning as pl
from vmunet_src.model.vmamba import VSSM


class VMambaMultiHeadClassifier(pl.LightningModule):
    def __init__(self, encoder, num_classes=3, lr=0.003, per_img_normalisation=True):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.lr = lr
        self.per_img_normalisation = per_img_normalisation

        hidden_dim = encoder.num_features  # embed_dim from VSSM
        self.head_flow = nn.Linear(hidden_dim, num_classes)
        self.head_speed = nn.Linear(hidden_dim, num_classes)
        self.head_zoff = nn.Linear(hidden_dim, num_classes)
        self.head_temp = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is [B,C,H,W]
        x = self.encoder(x)  # [B, H, W, embed_dim]
        x = x.mean(dim=(1, 2))

        return {
            "flow": self.head_flow(x),
            "speed": self.head_speed(x),
            "zoff": self.head_zoff(x),
            "temp": self.head_temp(x),
        }

    def _shared_step(self, batch):
        images, labels = batch  # labels: [B, 4]
        outputs = self.forward(images)

        loss_fn = nn.CrossEntropyLoss()

        l_flow = loss_fn(outputs["flow"], labels[:, 0])
        l_speed = loss_fn(outputs["speed"], labels[:, 1])
        l_zoff = loss_fn(outputs["zoff"], labels[:, 2])
        l_temp = loss_fn(outputs["temp"], labels[:, 3])

        total_loss = l_flow + l_speed + l_zoff + l_temp

        acc_flow = (
            (outputs["flow"].argmax(1) == labels[:, 0]).float().mean()
        )  # batch accuracy as [False, True, False, ...] -> float 1.0 or 0.0 -> mean across batch
        acc_speed = (outputs["speed"].argmax(1) == labels[:, 1]).float().mean()
        acc_zoff = (outputs["zoff"].argmax(1) == labels[:, 2]).float().mean()
        acc_temp = (outputs["temp"].argmax(1) == labels[:, 3]).float().mean()

        return {
            "loss": total_loss,
            "acc": (acc_flow + acc_speed + acc_zoff + acc_temp) / 4,
            "metrics": {
                "flow_acc": acc_flow,
                "speed_acc": acc_speed,
                "zoff_acc": acc_zoff,
                "temp_acc": acc_temp,
            },
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": 5e-4},  # backbone
                {
                    "params": list(self.head_flow.parameters())
                    + list(self.head_speed.parameters())
                    + list(self.head_zoff.parameters())
                    + list(self.head_temp.parameters()),
                    "lr": 3e-3,
                },  # heads
            ]
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        step_out = self._shared_step(batch)

        self.log("train_loss", step_out["loss"], on_step=True, prog_bar=True)
        self.log("train_acc", step_out["acc"], on_step=True, prog_bar=True)

        self.log(
            "train_flow_acc",
            step_out["metrics"]["flow_acc"],
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_speed_acc",
            step_out["metrics"]["speed_acc"],
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_zoff_acc",
            step_out["metrics"]["zoff_acc"],
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_temp_acc",
            step_out["metrics"]["temp_acc"],
            on_epoch=True,
            prog_bar=False,
        )

        return step_out["loss"]

    def validation_step(self, batch, batch_idx):
        step_out = self._shared_step(batch)

        self.log("val_loss", step_out["loss"], on_step=True, prog_bar=True)
        self.log("val_acc", step_out["acc"], on_step=True, prog_bar=True)

        self.log(
            "val_flow_acc",
            step_out["metrics"]["flow_acc"],
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val_speed_acc",
            step_out["metrics"]["speed_acc"],
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val_zoff_acc",
            step_out["metrics"]["zoff_acc"],
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val_temp_acc",
            step_out["metrics"]["temp_acc"],
            on_epoch=True,
            prog_bar=False,
        )

        return step_out["loss"]

    def on_test_epoch_end(self):
        results = {}
        for key in ["flow", "speed", "zoff", "temp"]:
            head_preds = torch.cat(
                [out["preds"][key] for out in self.test_step_outputs], dim=0
            )
            results[f"preds_{key}"] = head_preds

            head_targets = torch.cat(
                [out["targets"][key] for out in self.test_step_outputs], dim=0
            )
            results[f"targets_{key}"] = head_targets

        torch.save(results, "test/full_results.pt")
        self.test_step_outputs.clear()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        imgs, labels = batch
        if self.per_img_normalisation:
            imgs = imgs.to(self.device)
            mean = torch.mean(imgs, dim=[2, 3], keepdim=True)
            std = torch.std(imgs, dim=[2, 3], keepdim=True)
            imgs = (imgs - mean) / (std + 1e-8)
        labels = labels.to(self.device)
        return imgs, labels
