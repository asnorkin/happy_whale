from argparse import ArgumentParser

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

from pipeline.dataset import HappyDataset


class HappyLightningDataModule(pl.LightningDataModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.pre_transforms = []
        self.augmentations = [
            A.HorizontalFlip(),
        ]
        self.post_transforms = [
            A.Resize(self.hparams.input_size, self.hparams.input_size, always_apply=True),
            A.Normalize(always_apply=True),
            ToTensorV2(always_apply=True),
        ]

        self.train_dataset = None
        self.train_sampler = None

        self.val_dataset = None
        self.val_sampler = None

    def setup(self, stage=None):
        items = HappyDataset.load_items(
            images_dir=self.hparams.images_dir,
            labels_csv=self.hparams.labels_csv,
            debug=self.hparams.debug)

        train_items = [item for item in items if item["fold"] == self.hparams.fold]
        val_items = [item for item in items if item["fold"] == self.hparams.fold]

        # Train dataset
        train_transform = A.Compose(self.pre_transforms + self.augmentations + self.post_transforms)
        self.train_dataset = HappyDataset(train_items, train_transform)

        # Val dataset
        val_transform = A.Compose(self.pre_transforms + self.post_transforms)
        self.val_dataset = HappyDataset(val_items, val_transform)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Paths
        parser.add_argument("--images_dir", type=str, default="../data/train_images")
        parser.add_argument("--labels_csv", type=str, default="../data/train.csv")

        # General
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--input_size", type=int, default=256)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--fold", type=int, default=0)

        return parser

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, self.train_sampler, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, self.val_sampler)

    def _dataloader(self, dataset, batch_sampler=None, shuffle=False):
        params = {
            "pin_memory": True,
            "num_workers": self.hparams.num_workers,
        }

        if batch_sampler is not None:
            params["batch_sampler"] = batch_sampler
        else:
            params["batch_size"] = self.hparams.batch_size
            params["shuffle"] = shuffle
            params["drop_last"] = False

        return DataLoader(dataset, **params)
