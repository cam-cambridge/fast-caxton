import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
import torchvision
import torchvision.models as models
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from src.model.augmentations import (
    TestAugmentation,
    SimpleTestAugmentation,
    TransferAugmentation,
    TrainAugmentation
)

class RegNetRegression(pl.LightningModule):
    def __init__(
        self,
        max_epochs,
        lr=1e-3,
        gpus=1,
        test_overwrite_filename=False,
        mad_scale_factor=5.0,
        test='simple',
        train_file= None
        ):
    
        super().__init__()
        self.lr = lr
        self.__dict__.update(locals())

        print(torchvision.__version__)

        self.model = models.regnet_x_400mf(pretrained=True)
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 1),nn.ReLU())
     
        self.save_hyperparameters()

        self.name = "RegNetRegression"
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.sync_dist = True if self.gpus > 1 else False
        self.test_overwrite_filename = test_overwrite_filename
        self.mad_scale_factor = mad_scale_factor
        self.test_aug = TestAugmentation()
        self.train_aug= TrainAugmentation()
        self.simple_test_aug= SimpleTestAugmentation()
        self.transfer_aug= TransferAugmentation()
        self.test_mode= test
        self.test_name= None

        self.train_file = train_file
    def set_test_name(self,name):
        self.test_name=name
    
    def forward(self, X):
        X = self.model(X)
        return X

    def test_inference(self, x):

        b, n, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        batch_preds = self.forward(x)
        batch_preds = batch_preds.reshape(b, n, -1)
        batch_preds = batch_preds.squeeze(dim=2)
        meaned_lst = [pred.mean() for pred in batch_preds]
        std_lst= [pred.std() for pred in batch_preds]
        y_pred = torch.tensor(meaned_lst).unsqueeze(1).cuda()
        std_lst = torch.tensor(std_lst).unsqueeze(1).cuda()
        return y_pred, std_lst

    def reject_outliers(self, data):
        median = torch.median(data, dim=0, keepdim=True)[0]
        d = torch.abs(data - median)
        mdev = torch.median(d, dim=0, keepdim=True)[0]
        s = d / mdev if mdev else 0.0
        out = data[s < self.mad_scale_factor]
        return out

    def mae(self, y_true, predictions):
        y_true, predictions = np.array(y_true), np.array(predictions)
        return np.mean(np.abs(y_true - predictions))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        # scheduler = CosineAnnealingLR(
        #     optimizer, T_max=self.max_epochs, eta_min=0, last_epoch=-1
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, train_batch, batch_idx):

        x, y, _ = train_batch

        if self.transfer==True:
            x=self.transfer_aug(x)
            y_pred = self.forward(x)
            loss = F.mse_loss(y, y_pred)
        elif self.transfer=="il":
            x=self.transfer_aug(x)
            y_pred = self.forward(x)
            loss = self.il_loss(y, y_pred)
        else:
            x= self.train_aug(x)
            y_pred = self.forward(x)
            loss = F.mse_loss(y, y_pred)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            reduce_fx="mean",
            batch_size=128
        )

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            reduce_fx="mean",
            batch_size=128
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, _ = val_batch

        x= self.train_aug(x)
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            reduce_fx="mean",
            sync_dist=True,
            batch_size=128
        )

        self.log(
            "val_loss_truespace",
            torch.mean(torch.abs(torch.exp(y) - torch.exp(y_pred)))*100,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            reduce_fx="mean",
            sync_dist=True,
            batch_size=128
        )

        return loss
        
    def test_step(self, test_batch, batch_idx):

        x, y, _ = test_batch

        if self.test_mode=='inference':
            x= self.test_aug(x)
            y_pred,std = self.test_inference(x)
        else:
            x= self.simple_test_aug(x)
            y_pred = self.forward(x)
        
        loss = F.mse_loss(y_pred, y)

        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            reduce_fx="mean",
        )

        self.log(
            "test_loss_truespace",
            torch.mean(torch.abs(torch.exp(y) - torch.exp(y_pred)))*100,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            reduce_fx="mean",
            sync_dist=True
        )
        
        return {"loss": loss, "y_pred": y_pred, "y_true": y, "std":std}