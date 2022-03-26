from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from fuse.dataset import FuseDataset


class FuseLightningDataModule(pl.LightningDataModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        items = FuseDataset.load_items(
            embeddings_file=self.hparams.embeddings_file,
            image_ids_file=self.hparams.image_ids_file,
            labels_csv=self.hparams.labels_csv,
            debug=self.hparams.debug,
            fold=self.hparams.fold,
        )

        train_items = [item for item in items if item["fold"] != self.hparams.fold]
        val_items = [item for item in items if item["fold"] == self.hparams.fold]

        self.train_dataset = FuseDataset(train_items)
        self.val_dataset = FuseDataset(val_items)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Paths
        parser.add_argument("--embeddings_file", type=str, default="../data/tf_effb7_ns_m0.4_yolov5_9c_v3_augv3_2ldp0.4_512x512_4g20x3b60e0.001lr_embeddings.pt")
        parser.add_argument("--image_ids_file", type=str, default="../data/image_ids.npy")
        parser.add_argument("--labels_csv", type=str, default="../data/train.csv")

        # General
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--fold", type=int, default=0)

        return parser

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

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
