from statistics import variance
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
import torchvision
import torchvision.models as models
import torchmetrics
from model.augmentations import (
    TestAugmentation,
    SimpleTestAugmentation,
    TransferAugmentation,
    TrainAugmentation
)
from dataset.data_module import RegressionDataModule

from torchvision import transforms as T
import numpy as np
import pandas as pd
import wandb
import csv

import copy

from matplotlib import pyplot as plt
from PIL import Image

class RegNetClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        max_epochs,
        lr=1e-3,
        transfer=False,
        trainable_layers=1,
        gpus=1,
        test_overwrite_filename=False,
    ):
        super().__init__()
        self.lr = lr
        self.__dict__.update(locals())

        print(torchvision.__version__)

        self.model = models.regnet_y_200mf(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        if transfer:
            for child in list(self.model.children())[:-trainable_layers]:
                for param in child.parameters():
                    param.requires_grad = False
        self.save_hyperparameters()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.name = "RegNetClassifier"
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.sync_dist = True if self.gpus > 1 else False
        self.test_overwrite_filename = test_overwrite_filename

        # self.preprocess = Preprocess()
        # self.train_aug = TrainAugmentation()
        # self.val_aug = ValAugmentation()
        # self.test_aug = TestAugmentation()

    def forward(self, X):
        X = self.model(X)
        return X

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.lr, momentum=0.9)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=0, last_epoch=-1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.squeeze()
        x = self.train_aug(x)
        y_pred = self.forward(x)
        _, preds = torch.max(y_pred, 1)
        loss = F.cross_entropy(y_pred, y)
        self.train_acc(preds, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )
        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.squeeze()
        y_pred = self.forward(x)
        _, preds = torch.max(y_pred, 1)
        loss = F.cross_entropy(y_pred, y)
        self.val_acc(preds, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        print(x.size())
        x = self.test_aug(x)
        print(x.size())
        y = y.squeeze()
        y_pred = self.forward(x)
        _, preds = torch.max(y_pred, 1)
        loss = F.cross_entropy(y_pred, y)
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )
        self.test_acc(preds, y)
        self.log(
            "test_acc",
            self.test_acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )
        return {"loss": loss, "preds": preds, "targets": y}


class RegNetRegression(pl.LightningModule):
    def __init__(
        self,
        max_epochs,
        lr=1e-3,
        transfer=False,
        gpus=1,
        test_overwrite_filename=False,
        mad_scale_factor=5.0,
        scheme=None,
        test='simple',
        train_file= None
        ):
    
        super().__init__()
        self.lr = lr
        self.__dict__.update(locals())

        print(torchvision.__version__)

        self.model = models.regnet_x_400mf(pretrained=True)
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,1),
            )

        # self.model.fc = nn.Sequential(
        #     nn.Linear(num_ftrs, 1024),
        #     nn.Linear(1024, 512),
        #     nn.Linear(512, 256),
        #     nn.Linear(256, 128),
        #     nn.Linear(128,1),
        #     )


        # self.model.fc = nn.Sequential(
        #     nn.Linear(num_ftrs, num_ftrs),
        #     nn.Linear(num_ftrs, 1)
        #     )

        if self.transfer==True:

            if scheme=='full':
                for child in list(self.model.children()):
                    for param in child.parameters():
                        param.requires_grad = True

            else:
                for child in list(self.model.children()):
                    for param in child.parameters():
                        param.requires_grad = False

                for child in list(self.model.children())[2:]:
                    for param in child.parameters():
                        param.requires_grad = True

                proxy= len(list(list(list(self.model.children())[1].children())[3].children()))
                prop= int(scheme*proxy)
                
                for child in list(list(list(self.model.children())[1].children())[3].children())[proxy-prop:-1]:
                    for param in child.parameters():
                        param.requires_grad = True
        
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

    def weighted_loss(self, y_hat, y):
        '''
        Boundary weighted squared error
        '''
        se= (y_hat-y)**2
        weights= abs(y)*.3 + 1
        # weights = torch.where(abs(y_hat-y)>0.2, abs(y_hat-y), abs(y_hat-y)/2)
        loss= torch.mean(se*weights)
        return loss

    def il_loss(self, y, y_hat):
        '''
        Boundary weighted squared error
        '''

        num_bins= len(self.normalised_weights)
        bin_edges= torch.linspace(0,252,num_bins+1).cuda()
        # print(bin_edges)
        bin_indices= torch.bucketize(torch.exp(y)*100,bin_edges)-1
        # print(torch.exp(y)*100,bin_indices)

        se= (y_hat-y)**2


        weighted_loss= torch.mean(self.normalised_weights[bin_indices]*se).cuda()

        return weighted_loss

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

    def create_img(self, x):
        '''
        Reverse normalisation for image visualisation
        '''
        
        mean = torch.tensor([0.16963361, 0.1761256, 0.16955556])
        std = torch.tensor([0.27373913, 0.27497604, 0.27459216])

        unnormalize = T.Compose([
            T.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
            T.ToPILImage()
        ])
        x = unnormalize(x.squeeze())
        return x

    def adversarial_expansion(self, x, y, rate=0, n=5):
        '''
        adversarial sample augmentation
            '''
        
        aug_x= self.test_aug(x)
        _,adversarial_rating = self.test_inference(aug_x)
        expansion_indices= torch.argsort(adversarial_rating,axis=0)[-int(adversarial_rating.shape[0]*rate)-5:-5]
        expansion= torch.index_select(x, dim=0, index=expansion_indices.squeeze())
        expansion_labels= torch.index_select(y, dim=0, index=expansion_indices.squeeze())

        # combine old batch with re-augmented samples
        x=torch.vstack([x,expansion.repeat(n,1,1,1)])
        y=torch.vstack([y,expansion_labels.repeat(n,1)])

        # for i,img in enumerate(expansion):
        #     img= self.create_img(img)
        #     img.save(f"/home/cam/Documents/Git/locaxton/src/model/training_samples/{i}.jpeg")

        return x,y

    def training_step(self, train_batch, batch_idx):
        x, y, _ = train_batch

        if self.transfer==True:
            # x,y = self.adversarial_expansion(x,y,0)
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
    
    def on_train_start(self):
        continuous_labels=[]

        continuous_labels= torch.tensor(pd.read_csv(self.train_file)["flow_rate"]).cuda()
        num_bins=10
        bin_counts= torch.histc(continuous_labels, bins=num_bins, min=0.0, max=250.0)
        valid_bins= bin_counts>=100
        
        print("bin counts ", bin_counts.tolist())
        bin_weights= torch.zeros_like(bin_counts).float()
        
        bin_weights[valid_bins]+= 1.0 / (bin_counts[valid_bins])
        self.normalised_weights= bin_weights/bin_weights.sum()+1
        
        print("weights ", [round(value,2) for value in self.normalised_weights.tolist()])

    def test_epoch_end(self, outputs):
        y_true = np.array([])
        y_pred = np.array([])
        std = np.array([])

        def to_np(x):
            try:
                return x.detach().cpu().numpy()
            except: 
                return np.asarray(x)
            

        for results_dict in outputs:
            y_true = np.append(y_true, to_np(results_dict["y_true"]))
            y_pred = np.append(y_pred, to_np(results_dict["y_pred"]))            
            std = np.append(std, to_np(results_dict["std"]))

        print(y_true.shape, y_pred.shape, std.shape)

        data = {
            "y_true": y_true,
            "y_pred": y_pred,
            "std": std,
        }
        df = pd.DataFrame(data)
        df.to_csv(self.test_name, index=False)
        
    def hat_vs_true(self, y_hat, y_true, version):
        plt.ioff()
        fig, ax = plt.subplots(1, 1, dpi=500, figsize=[6,6])
        ax.scatter(y_true, y_hat, alpha=0.1, color="greenyellow", marker=",", s=0.005)
        ax.plot(list(range(-3,3)), list(range(-3,3)), color="black", linewidth=0.5, linestyle='dashed')
        ax.set_ylim(-1,1)
        ax.set_xlim(-1,1)
        fig.tight_layout()
        plt.savefig(f"/home/cam/Documents/Git/locaxton/src/model/in_version_results/{version}.jpg")

class RegNetRegressionUncertainty(pl.LightningModule):
    def __init__(
        self,
        max_epochs,
        lr=1e-3,
        transfer=False,
        trainable_layers=1,
        gpus=1,
        test_overwrite_filename=False,
        epsilon=0.01,
    ):
        super().__init__()
        self.lr = lr
        self.__dict__.update(locals())

        print(torchvision.__version__)

        self.model = models.regnet_y_400mf(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

        if transfer:
            for child in list(self.model.children())[:-trainable_layers]:
                for param in child.parameters():
                    param.requires_grad = False
        self.save_hyperparameters()

        self.name = "RegNetRegressionUncertaintyAdv"
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.sync_dist = True if self.gpus > 1 else False
        self.test_overwrite_filename = test_overwrite_filename

        # self.preprocess = Preprocess()
        # self.train_aug = TrainAugmentation()
        # self.val_aug = ValAugmentation()

        self.epsilon = epsilon

    def forward(self, X):
        X = self.model(X)
        mean, variance = torch.split(X, 1, dim=1)
        variance = F.softplus(variance) + 1e-6  # positive constraint
        return mean, variance

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=0, last_epoch=-1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    # fast graident sign method adversarial attack
    def fgsm(self, x, data_grad):
        sign_data_grad = data_grad.sign()
        x_prime = x + self.epsilon * sign_data_grad
        return x_prime

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = self.train_aug(x)
        x.requires_grad = True
        mean, variance = self.forward(x)
        loss_for_attack = F.gaussian_nll_loss(mean, y, variance)
        grad = torch.autograd.grad(loss_for_attack, x, retain_graph=False)[0]
        x, y = x.detach(), y.detach()
        mean, variance = mean.detach(), variance.detach()
        loss_for_attack.detach_()
        x_prime = self.fgsm(x, grad)
        mean_prime, variance_prime = self.forward(x_prime)
        loss = F.gaussian_nll_loss(mean, y, variance)
        loss_prime = F.gaussian_nll_loss(mean_prime, y, variance_prime)
        combined_loss = loss + loss_prime
        self.log(
            "train_loss",
            combined_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )
        return combined_loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        mean, variance = self.forward(x)
        loss = F.gaussian_nll_loss(mean, y, variance)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        mean, variance = self.forward(x)
        loss = F.gaussian_nll_loss(mean, y, variance)
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )
        return {"loss": loss, "mean": mean, "target": y, "variance": variance}
