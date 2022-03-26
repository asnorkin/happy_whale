from argparse import ArgumentParser
from math import ceil

import pytorch_lightning as pl
import torch
from torch import nn
from torch import distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from fuse.model import FuseModel
from pipeline.module import find_best_map


class FuseLightningModule(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.arcface_criterion = nn.CrossEntropyLoss()

        self.model = FuseModel(
            num_classes=self.hparams.num_classes,
            emb_size=self.hparams.emb_size,
            hidden_input=self.hparams.hidden_input,
            output_size=self.hparams.output_size,
            s=self.hparams.s,
            m=self.hparams.m,
            easy_margin=self.hparams.easy_margin,
            ls_eps=self.hparams.ls_eps,
        )

        # Placeholders
        self.train_outputs = None

    def forward(self, fin_emb, fish_emb, labels):
        return self.model.forward(fin_emb, fish_emb, labels)

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
        parser.add_argument("--emb_size", type=int, default=512)
        parser.add_argument("--hidden_input", type=int, default=512)
        parser.add_argument("--output_size", type=int, default=512)
        parser.add_argument("--num_classes", type=int, default=15587)

        # Loss
        parser.add_argument("--s", type=float, default=30.0)
        parser.add_argument("--m", type=float, default=0.4)
        parser.add_argument("--easy_margin", type=int, default=0)
        parser.add_argument("--ls_eps", type=float, default=0.0)

        # Learning rate
        parser.add_argument("--num_epochs", type=int, default=20)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--lr_pct_start", type=float, default=0.1)

        # Monitor
        parser.add_argument("--monitor", type=str, default="map@5")
        parser.add_argument("--monitor_mode", type=str, default="max")

        # Work dir
        parser.add_argument("--work_dir", type=str, default="fuse")

        return parser

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="val")

    def on_train_epoch_start(self):
        self.train_outputs = []

    def validation_epoch_end(self, outputs):
        stage = "val"

        if not self.train_outputs:
            self.print("self.train_outputs is empty!! Skip validation metrics.")
            return

        train_embeddings = self._gather("embeddings", outs=self.train_outputs).cpu().float()
        train_labels = self._gather("individual_labels", outs=self.train_outputs).cpu().numpy()

        embeddings = self._gather("embeddings", outs=outputs).cpu().float()
        individual_labels = self._gather("individual_labels", outs=outputs).cpu().numpy()
        new = self._gather("new", outs=outputs).cpu().numpy()

        best_thresh, best_map5, best_map1 = find_best_map(
            embeddings, train_embeddings, individual_labels, train_labels, new)

        # Progress Bar
        self.log(f"map@1", best_map1, prog_bar=True, logger=False)
        self.log(f"map@5", best_map5, prog_bar=True, logger=False)

        # Logger
        self.log(f"metrics/{stage}_map@1", best_map1, prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_map@5", best_map5, prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_new_threshold", best_thresh, prog_bar=False, logger=True)

    def _step(self, batch, _batch_idx, stage):
        arcface_logits, embeddings = self.forward(
            batch["embedding_fin"], batch["embedding_fish"], batch["individual_label"])

        # Losses
        arcface_loss = self.arcface_criterion(arcface_logits, batch["individual_label"])

        losses = {
            "total": arcface_loss,
            "arcface": arcface_loss,
        }

        metrics = dict()
        self._log(losses, metrics, stage=stage)

        if stage == "train":
            self.train_outputs.append({
                "embeddings": embeddings.detach().cpu(),
                "individual_labels": batch["individual_label"].detach().cpu(),
            })
            return losses["total"]

        return {
            "embeddings": embeddings.detach().cpu(),
            "individual_labels": batch["individual_label"].detach().cpu(),
            "new": batch["new"].detach().cpu(),
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

    def _gather(self, key, outs):
        node_values = torch.concat([torch.as_tensor(output[key]).to(self.device) for output in outs])
        if self.trainer.world_size == 1:
            return node_values

        all_values = [torch.zeros_like(node_values) for _ in range(self.trainer.world_size)]
        dist.barrier()
        dist.all_gather(all_values, node_values)
        all_values = torch.cat(all_values)
        return all_values
