import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy

from eeg_online.utils.lr_scheduler import linear_warmup_cosine_decay


class ClassificationModule(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 optimizer: str = "adam",
                 scheduler: bool = False,
                 max_epochs: int = 1000,
                 warmup_epochs: int = 20,
                 **kwargs):
        super(ClassificationModule, self).__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, x: torch.tensor):
        return self.model(x)

    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                         betas=betas,
                                         weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError
        if self.hparams.scheduler:
            scheduler = LambdaLR(optimizer,
                                 linear_warmup_cosine_decay(self.hparams.warmup_epochs,
                                                            self.hparams.max_epochs))
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="val")
        return {"val_loss": loss, "val_acc": acc}

    def shared_step(self, batch, batch_idx, mode: str = "train"):
        x, y = batch
        y_hat = torch.sigmoid(self.forward(x))

        loss = F.binary_cross_entropy(y_hat.mean(dim=-1), y.float())

        acc = accuracy(y_hat.mean(dim=-1), y, task="binary")

        # b x n_windows
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return torch.sigmoid(self.forward(x))
