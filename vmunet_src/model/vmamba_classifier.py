import torch
from torch import nn
import pytorch_lightning as pl
from vmunet_src.model.vmamba import VSSM


class VMambaMultiHeadClassifier(pl.LightningModule):
    def __init__(self, encoder, num_classes=3, lr=0.003, per_img_normalisation=True, checkpoint_path=None):
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

        self.log_vars = nn.Parameter(torch.zeros(4))

        # due to overfitting at epoch 4, changed back to get the same results of epoch 3
        self.log_vars.requires_grad = False

        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
            self.load_from()

        

    def forward(self, x):
        # x: [B,H,W,C]
        x = self.encoder(x)
        print(x.shape)
        x = x.mean(dim=(1, 2)) # global avg pool

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
##############################################################################
        weighted_losses = []
        raw_losses = [l_flow, l_speed, l_zoff, l_temp]
        for i, loss in enumerate(raw_losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_losses.append(precision * loss + self.log_vars[i])

        total_loss = sum(weighted_losses)

        for i, name in enumerate(['flow', 'speed', 'zoff', 'temp']):
            self.log(f"weight_{name}", torch.exp(-self.log_vars[i]))
            self.log(f"raw_loss_{name}", raw_losses[i])
##############################################################################
        acc_flow = (
            (outputs["flow"].argmax(1) == labels[:, 0]).float().mean()
        )  # batch accuracy as [False, True, False, ...] -> float 1.0 or 0.0 -> mean across batch
        acc_speed = (outputs["speed"].argmax(1) == labels[:, 1]).float().mean()
        acc_zoff = (outputs["zoff"].argmax(1) == labels[:, 2]).float().mean()
        acc_temp = (outputs["temp"].argmax(1) == labels[:, 3]).float().mean()

        return {
            "loss": total_loss,
            "acc": (acc_flow + acc_speed + acc_zoff + acc_temp) / 4,
            "individual_losses": {
                "flow_loss": l_flow,
                "speed_loss": l_speed,
                "zoff_loss": l_zoff,
                "temp_loss": l_temp,
            },
            "metrics": {
                "flow_acc": acc_flow,
                "speed_acc": acc_speed,
                "zoff_acc": acc_zoff,
                "temp_acc": acc_temp,
            },
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.encoder.parameters(), 'lr': 3e-5},
            {'params': list(self.head_flow.parameters()) + 
                       list(self.head_speed.parameters()) + 
                       list(self.head_zoff.parameters()) + 
                       list(self.head_temp.parameters()), 'lr': 1e-4},
            {'params': [self.log_vars], 'lr': 1e-3}
        ], weight_decay=0.01)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min',
        #     factor=0.5,
        #     patience=5, 
        #     min_lr=1e-6,
        #     verbose=True
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss_epoch",
        #         "interval": "epoch",
        #         "frequency": 1,
        #     },
        # }

        # HELPED AT EPOCH 3 BUt not epoch 4
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 
        #     T_max=self.trainer.estimated_stepping_batches,
        #     eta_min=1e-6
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch",
        #     },
        # }

        return optimizer


    def training_step(self, batch, batch_idx):
        step_out = self._shared_step(batch)

        opt = self.optimizers()

        self.log("lr_encoder", opt.param_groups[0]['lr'], on_step=True, prog_bar=True)
        self.log("lr_heads", opt.param_groups[1]['lr'], on_step=True, prog_bar=True)


        self.log("train_loss", step_out["loss"], on_step=True, prog_bar=True)
        self.log("train_acc", step_out["acc"], on_step=True, prog_bar=True)

        self.log_dict({
            f"train_{k}": v for k, v in step_out["individual_losses"].items()
        }, on_epoch=True, prog_bar=False)

        self.log("train_flow_acc", step_out["metrics"]["flow_acc"], on_epoch=True, prog_bar=False)
        self.log("train_speed_acc", step_out["metrics"]["speed_acc"], on_epoch=True, prog_bar=False)
        self.log("train_zoff_acc", step_out["metrics"]["zoff_acc"], on_epoch=True, prog_bar=False)
        self.log("train_temp_acc", step_out["metrics"]["temp_acc"], on_epoch=True, prog_bar=False)

        return step_out["loss"]

    def validation_step(self, batch, batch_idx):
        step_out = self._shared_step(batch)

        self.log("val_loss", step_out["loss"], on_step=True, prog_bar=True)
        self.log("val_acc", step_out["acc"], on_step=True, prog_bar=True)

        self.log_dict({
            f"val_{k}": v for k, v in step_out["individual_losses"].items()
        }, on_epoch=True)
        

        self.log("val_flow_acc", step_out["metrics"]["flow_acc"], on_epoch=True, prog_bar=False)
        self.log("val_speed_acc", step_out["metrics"]["speed_acc"], on_epoch=True, prog_bar=False)
        self.log("val_zoff_acc", step_out["metrics"]["zoff_acc"], on_epoch=True, prog_bar=False)
        self.log("val_temp_acc", step_out["metrics"]["temp_acc"], on_epoch=True, prog_bar=False)

        return step_out["loss"]

    def on_test_epoch_end(self):
        # preds will be a list of dicts from each test_step
        results = {}
        for key in ["flow", "speed", "zoff", "temp"]:
            # Extract all predictions for this specific head across all batches
            head_preds = torch.cat(
                [out["preds"][key] for out in self.test_step_outputs], dim=0
            )
            results[f"preds_{key}"] = head_preds

            head_targets = torch.cat(
                [out["targets"][key] for out in self.test_step_outputs], dim=0
            )
            results[f"targets_{key}"] = head_targets

        # Save the whole dictionary of tensors
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

    def load_from(self):
        print("Loading checkpoint from:", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path)
        print("Checkpoint keys:", checkpoint.keys())

        self.load_state_dict(checkpoint["state_dict"])
        print("Model loaded successfully from checkpoint.")
        
        
        
