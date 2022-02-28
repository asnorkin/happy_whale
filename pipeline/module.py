from argparse import ArgumentParser
from math import ceil

import numpy as np
import pytorch_lightning as pl
import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from pipeline.metric import map_per_set
from pipeline.model import HappyModel


class HappyLightningModule(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = nn.CrossEntropyLoss()
        self.model = HappyModel(
            model_name=self.hparams.model_name,
            num_classes=self.hparams.num_classes,
            embedding_size=self.hparams.embedding_size,
            s=self.hparams.s,
            m=self.hparams.m,
            easy_margin=self.hparams.easy_margin,
            ls_eps=self.hparams.ls_eps)

        # Placeholders
        self.train_outputs = None

    def forward(self, images, labels):
        return self.model.forward(images, labels)

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
        parser.add_argument("--embedding_size", type=int, default=512)
        parser.add_argument("--num_classes", type=int, default=15587)

        # Loss
        parser.add_argument("--s", type=float, default=30.0)
        parser.add_argument("--m", type=float, default=0.5)
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
        parser.add_argument("--work_dir", type=str, default="pipeline")

        return parser

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="val")

    def on_train_epoch_start(self):
        self.train_outputs = []

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

        if not self.train_outputs:
            self.print("self.train_outputs is empty!! Skip validation metrics.")
            return

        train_embeddings = _gather("embeddings", outs=self.train_outputs).cpu()
        train_labels = _gather("labels", outs=self.train_outputs).cpu().numpy()

        embeddings = _gather("embeddings").cpu()
        labels = _gather("labels").cpu().numpy()
        new = _gather("new").cpu().numpy()

        k = 5
        new_threshold = 0.5
        _distance = 1 - torch.mm(F.normalize(embeddings), F.normalize(train_embeddings).T)
        topk_distances, topk_predictions = torch.topk(_distance, k=10 * k, largest=False, dim=1)
        topk_distances, topk_predictions = topk_distances.numpy(), topk_predictions.numpy()

        predictions = []
        for i, (preds, dists) in enumerate(zip(topk_predictions, topk_distances)):
            pred_ids = train_labels[preds].tolist()

            # Remove duplicates
            seen = set()
            _pred_ids, _dists = [], []
            for pid, di in zip(pred_ids, dists):
                if pid in seen:
                    continue
                seen.add(pid)
                _pred_ids.append(pid)
                _dists.append(di)
            pred_ids, dists = _pred_ids[:k], _dists[:k]

            # Add new_individual
            if dists[-1] > new_threshold:
                new_i = min(i for i, d in enumerate(dists) if d > new_threshold)
                new_pred = labels[i] if new[i] else -1
                pred_ids = pred_ids[:new_i] + [new_pred] + pred_ids[new_i:-1]

            predictions.append(pred_ids)

        map1 = map_per_set(labels, predictions, topk=1)
        map5 = map_per_set(labels, predictions, topk=5)

        self.log(f"map@1", map1, prog_bar=True, logger=False)
        self.log(f"map@5", map5, prog_bar=True, logger=False)
        self.log(f"metrics/{stage}_map@1", map1, prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_map@5", map5, prog_bar=False, logger=True)

    def _step(self, batch, _batch_idx, stage):
        logits, pooled_features = self.forward(batch["image"], batch["label"])

        losses = {"total": self.criterion(logits, batch["label"])}
        metrics = dict()
        self._log(losses, metrics, stage=stage)

        if stage == "train":
            self.train_outputs.append({
                "embeddings": pooled_features.detach().cpu(),
                "labels": batch["label"].detach().cpu(),
            })
            return losses["total"]

        return {
            "embeddings": pooled_features.detach().cpu(),
            "labels": batch["label"].detach().cpu(),
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
        logs[f"losses/{stage}_total"] = losses["total"]
        self.log_dict(logs, prog_bar=False, logger=True)
