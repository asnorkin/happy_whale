from argparse import ArgumentParser

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

from camera_viewpoint.augmentations import CameraHorizontalFlip
from camera_viewpoint.dataset import CameraDataset


class CameraLightningDataModule(pl.LightningDataModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.pre_transforms = []
        self.augmentations = [
            CameraHorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, border_mode=0, value=0, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=20, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ToGray(p=0.05),
            ], p=0.75),
        ]
        self.post_transforms = [
            A.Resize(height=self.hparams.input_height, width=self.hparams.input_width, always_apply=True),
            A.Normalize(always_apply=True),
            ToTensorV2(always_apply=True),
        ]

        self.train_dataset = None
        self.train_sampler = None

        self.val_dataset = None
        self.val_sampler = None

    def setup(self, stage=None):
        items = CameraDataset.load_items(
            images_dir=self.hparams.images_dir,
            labels_csv=self.hparams.labels_csv,
            debug=self.hparams.debug)

        train_items = [item for item in items if item["fold"] != self.hparams.fold]
        val_items = [item for item in items if item["fold"] == self.hparams.fold]

        # Train dataset
        train_transform = A.Compose(self.pre_transforms + self.augmentations + self.post_transforms)
        self.train_dataset = CameraDataset(train_items, train_transform)

        # Val dataset
        val_transform = A.Compose(self.pre_transforms + self.post_transforms)
        self.val_dataset = CameraDataset(val_items, val_transform)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Paths
        parser.add_argument("--images_dir", type=str, default="../data/train_images")
        parser.add_argument("--labels_csv", type=str, default="../data/camera_viewpoint_labels.csv")

        # General
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--input_height", type=int, default=224)
        parser.add_argument("--input_width", type=int, default=224)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--fold", type=int, default=0)

        return parser

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, self.train_sampler, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, self.val_sampler)

    def _dataloader(self, dataset, batch_sampler=None, shuffle=False, batch_size=None):
        if batch_size is None:
            batch_size = self.hparams.batch_size

        params = {
            "pin_memory": True,
            "persistent_workers": True,
            "num_workers": self.hparams.num_workers,
        }

        if batch_sampler is not None:
            params["batch_sampler"] = batch_sampler
        else:
            params["batch_size"] = batch_size
            params["shuffle"] = shuffle
            params["drop_last"] = False

        return DataLoader(dataset, **params)
