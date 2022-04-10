from argparse import ArgumentParser
from collections import Counter

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

from pipeline.augmentations import CameraHorizontalFlip, StackImages
from pipeline.dataset import BalancedHappyDataset, HappyDataset


class HappyLightningDataModule(pl.LightningDataModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.pre_transforms = []
        self.augmentations = [
            # A.HorizontalFlip(p=0.5),
            CameraHorizontalFlip(p=0.5),
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, border_mode=0, value=0, p=0.5),
                A.Perspective(p=0.1),
            ], p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=20, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ToGray(p=0.05),
                A.ChannelShuffle(p=0.05),
            ], p=0.75),
            A.OneOf([
                A.MultiplicativeNoise(p=0.5, elementwise=True, per_channel=True),
                A.GaussianBlur(p=0.5),
                A.ImageCompression(p=0.5, quality_lower=50),
            ], p=0.2),
        ]

        self.post_transforms = [
            A.Resize(height=self.hparams.input_height, width=self.hparams.input_width, always_apply=True),
            A.Normalize(always_apply=True),
            ToTensorV2(always_apply=True),
        ]

        if self.hparams.all_images:
            self.post_transforms.append(StackImages(always_apply=True))

        self.train_dataset = None
        self.train_sampler = None

        self.val_dataset = None
        self.val_sampler = None

        self.test_dataset = None

    def setup(self, stage=None):
        # DataModule already configured
        if stage == "test":
            return

        items = HappyDataset.load_items(
            images_dir=self.hparams.images_dir,
            labels_csv=self.hparams.labels_csv,
            debug=self.hparams.debug)

        train_items = [item for item in items if item["fold"] != self.hparams.fold]
        val_items = [item for item in items if item["fold"] == self.hparams.fold]

        train_counts = Counter([item["individual_id"] for item in train_items])
        train_items = [item for item in train_items if train_counts[item["individual_id"]] >= self.hparams.min_count]
        for i, item in enumerate(val_items):
            if train_counts[item["individual_id"]] < self.hparams.min_count:
                val_items[i]["new"] = 1

        additional_targets = dict()
        if self.hparams.all_images:
            additional_targets["image_fish"] = "image"
            # additional_targets["image_fin"] = "image"

        # Train dataset
        train_transform = A.Compose(self.pre_transforms + self.augmentations + self.post_transforms, additional_targets=additional_targets)
        self.train_dataset = BalancedHappyDataset(
            train_items,
            train_transform,
            load_all_images=self.hparams.all_images,
            load_random_image=self.hparams.random_image,
            max_count=self.hparams.max_count,
        )

        # Val dataset
        val_transform = A.Compose(self.pre_transforms + self.post_transforms, additional_targets=additional_targets)
        self.val_dataset = HappyDataset(val_items, val_transform, load_all_images=self.hparams.all_images, load_random_image=False)

        # Test dataset
        test_transform = A.Compose(self.pre_transforms + self.post_transforms, additional_targets={"image_fish": "image"})
        self.test_dataset = HappyDataset(items, test_transform, load_all_images=True, load_random_image=False)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Paths
        parser.add_argument("--images_dir", type=str, default="../data/train_images")
        parser.add_argument("--labels_csv", type=str, default="../data/train.csv")

        # General
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--input_height", type=int, default=384)
        parser.add_argument("--input_width", type=int, default=768)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--fold", type=int, default=0)
        parser.add_argument("--min_count", type=int, default=2)
        parser.add_argument("--max_count", type=int, default=5)

        return parser

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, self.train_sampler, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, self.val_sampler)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset, batch_sampler=None, shuffle=False, batch_size=None, drop_last=False):
        if batch_size is None:
            batch_size = self.hparams.batch_size

        params = {
            "pin_memory": True,
            "persistent_workers": self.hparams.num_workers > 0,
            "num_workers": self.hparams.num_workers,
        }

        if batch_sampler is not None:
            params["batch_sampler"] = batch_sampler
        else:
            params["batch_size"] = batch_size
            params["shuffle"] = shuffle
            params["drop_last"] = drop_last

        return DataLoader(dataset, **params)
