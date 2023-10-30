import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from PIL import ImageFile
from dataset.dataset import ClassificationDataset, RegressionDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        data_dir,
        val_dir=None,
        train_csv=None,
        val_csv=None,
        test_csv=None,
        dataset_name=None,
        image_dim=(224, 224),
        precropped=True,
        raw=False,
        test=False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.val_dir = val_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        self.size = (3, 224, 224)
        self.num_classes = 3
        self.image_dim = image_dim

        self.raw = raw
        self.precropped = precropped
        self.test= test
    def setup(self, stage=None, test_all=False):
        if stage == "fit" or stage is None:
            # Assign train/val datasets for use in dataloaders
            self.train_dataset = ClassificationDataset(
                csv_file=self.train_csv,
                root_dir=self.data_dir,
                val_dir= self.val_dir,
                image_dim=self.image_dim,
            )
            self.val_dataset = ClassificationDataset(
                csv_file=self.val_csv,
                root_dir=self.data_dir,
                val_dir= self.val_dir,
                image_dim=self.image_dim,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = ClassificationDataset(
                csv_file=self.test_csv,
                root_dir=self.data_dir,
                val_dir= self.val_dir,
                image_dim=self.image_dim,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=16,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
        )

class RegressionDataModule(ClassificationDataModule):
    def __init__(
        self,
        batch_size,
        data_dir,
        val_dir=None,
        train_csv=None,
        val_csv=None,
        test_csv=None,
        dataset_name=None,
        image_dim=(350, 350),
        precropped=True,
        raw=False,
        test=False
    ):
        super().__init__(
            batch_size,
            data_dir,
            val_dir,
            train_csv,
            val_csv,
            test_csv,
            dataset_name,
            image_dim,
            precropped=precropped,
            raw=raw,
            test=test
        )

    def setup(self, stage=None, test_all=False):
        if stage == "fit" or stage is None:
            # Assign train/val datasets for use in dataloaders
            self.train_dataset = RegressionDataset(
                csv_file=self.train_csv,
                root_dir=self.data_dir,
                image_dim=self.image_dim,
                precropped=self.precropped,
                raw=self.raw,
            )
            self.val_dataset = RegressionDataset(
                csv_file=self.val_csv,
                root_dir=self.val_dir,
                image_dim=self.image_dim,
                precropped=self.precropped,
                raw=self.raw,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = RegressionDataset(
                csv_file=self.test_csv,
                root_dir=self.data_dir,
                image_dim=self.image_dim,
                precropped=self.precropped,
                raw=self.raw,
                test=self.test
            )
