from argparse import ArgumentParser
from math import ceil

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report
from torch import distributed as dist
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from camera_viewpoint.model import CameraModel


class CameraLightningModule(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.viewpoint_criterion = nn.CrossEntropyLoss()
        self.klass_criterion = nn.BCEWithLogitsLoss()
        self.specie_criterion = nn.CrossEntropyLoss()

        self.model = CameraModel(
            model_name=self.hparams.model_name,
            num_classes=self.hparams.num_classes,
            num_species=self.hparams.num_species,
            viewpoint_hidden=self.hparams.viewpoint_hidden,
            specie_hidden=self.hparams.specie_hidden,
            dropout=self.hparams.dropout,
        )

    def forward(self, images):
        return self.model.forward(images)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        scheduler = self._configure_scheduler(optimizer)
        return [optimizer], [scheduler]

    def _configure_scheduler(self, optimizer):
        dataset_size = len(self.trainer.datamodule.train_dataset)
        step_period = self.hparams.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size
        steps_per_epoch = ceil(dataset_size / step_period)
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches < steps_per_epoch:
            steps_per_epoch = self.trainer.limit_train_batches

        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                pct_start=self.hparams.lr_pct_start,
                steps_per_epoch=steps_per_epoch,
                epochs=self.trainer.max_epochs,
            ),
            "interval": "step",
        }

        return scheduler

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model
        parser.add_argument("--model_name", type=str, default="tf_efficientnet_b0_ns")
        parser.add_argument("--num_classes", type=int, default=3)
        parser.add_argument("--viewpoint_hidden", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--num_species", type=int, default=26)
        parser.add_argument("--specie_hidden", type=int, default=64)
        parser.add_argument("--viewpoint_alpha", type=float, default=10.0)

        # Learning rate
        parser.add_argument("--num_epochs", type=int, default=20)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--lr_pct_start", type=float, default=0.1)

        # Monitor
        parser.add_argument("--monitor", type=str, default="vf1")
        parser.add_argument("--monitor_mode", type=str, default="max")

        # Work dir
        parser.add_argument("--work_dir", type=str, default="camera_viewpoint")

        return parser

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="val")

    def validation_epoch_end(self, outputs):
        stage = "val"

        def _gather(key, outs=outputs):
            node_values = torch.concat([torch.as_tensor(output[key]).to(self.device) for output in outs])
            if self.trainer.world_size == 1:
                return node_values

            all_values = [torch.zeros_like(node_values) for _ in range(self.trainer.world_size)]
            dist.barrier()
            dist.all_gather(all_values, node_values)
            all_values = torch.cat(all_values)
            return all_values

        viewpoint_probabilities = _gather("viewpoint_probabilities").cpu().numpy()
        viewpoint_labels = _gather("viewpoint_labels").cpu().numpy()
        klass_probabilities = _gather("klass_probabilities").cpu().numpy()
        klass_labels = _gather("klass_labels").cpu().numpy()
        specie_probabilities = _gather("specie_probabilities").cpu().numpy()
        specie_labels = _gather("specie_labels").cpu().numpy()

        # Viewpoint metrics
        viewpoint_predictions = np.argmax(viewpoint_probabilities, axis=1)
        viewpoint_metrics = classification_report(
            viewpoint_labels, viewpoint_predictions, output_dict=True, zero_division=0)

        # Klass metrics
        klass_threshold = 0.5
        klass_predictions = (klass_probabilities > klass_threshold).astype(int)
        klass_metrics = classification_report(klass_labels, klass_predictions, output_dict=True, zero_division=0)

        # Specie metrics
        specie_predictions = np.argmax(specie_probabilities, axis=1)
        specie_metrics = classification_report(specie_labels, specie_predictions, output_dict=True, zero_division=0)

        # Progress Bar
        self.log(f"vf1", viewpoint_metrics["macro avg"]["f1-score"], prog_bar=True, logger=False)
        self.log(f"kf1", klass_metrics["macro avg"]["f1-score"], prog_bar=True, logger=False)
        self.log(f"sf1", specie_metrics["macro avg"]["f1-score"], prog_bar=True, logger=False)

        # Logger
        self.log(f"metrics/{stage}_viewpoint_f1", viewpoint_metrics["macro avg"]["f1-score"], prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_klass_f1", klass_metrics["macro avg"]["f1-score"], prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_specie_f1", specie_metrics["macro avg"]["f1-score"], prog_bar=False, logger=True)

    def _step(self, batch, _batch_idx, stage):
        viewpoint_logits, klass_logits, specie_logits = self.forward(batch["image"])

        # Losses
        viewpoint_loss = self.viewpoint_criterion(viewpoint_logits, batch["viewpoint_label"])
        klass_loss = self.klass_criterion(klass_logits, batch["klass_label"].float())
        specie_loss = self.specie_criterion(specie_logits, batch["specie_label"])
        losses = {
            "total": 10 * viewpoint_loss + klass_loss + specie_loss,
            "arcface": viewpoint_loss,
            "klass": klass_loss,
            "specie": specie_loss
        }

        metrics = dict()
        self._log(losses, metrics, stage=stage)

        if stage == "train":
            return losses["total"]

        return {
            "viewpoint_probabilities": viewpoint_logits.detach().softmax(dim=1).cpu(),
            "klass_probabilities": klass_logits.detach().sigmoid().cpu(),
            "specie_probabilities": specie_logits.detach().softmax(dim=1).cpu(),
            "viewpoint_labels": batch["viewpoint_label"].detach().cpu(),
            "klass_labels": batch["klass_label"].detach().cpu(),
            "specie_labels": batch["specie_label"].detach().cpu(),
        }

    def _log(self, losses, metrics, stage):
        # Progress bar
        progress_bar = dict()
        if stage != "train":
            progress_bar[f"{stage}_loss"] = losses["total"]
        self.log_dict(progress_bar, prog_bar=True, logger=False)

        # Logs
        logs = dict()
        for lname, lval in losses.items():
            logs[f"losses/{stage}_{lname}"] = lval
        self.log_dict(logs, prog_bar=False, logger=True)
