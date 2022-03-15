from argparse import ArgumentParser
from math import ceil

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report
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

        self.arcface_criterion = nn.CrossEntropyLoss()
        self.klass_criterion = nn.BCEWithLogitsLoss()
        self.specie_criterion = nn.CrossEntropyLoss()

        num_classes = self.hparams.num_classes
        if self.hparams.flip_id:
            num_classes *= 2
        self.model = HappyModel(
            model_name=self.hparams.model_name,
            num_classes=num_classes,
            num_species=self.hparams.num_species,
            specie_hidden=self.hparams.specie_hidden,
            embedding_size=self.hparams.embedding_size,
            dropout=self.hparams.dropout,
            s=self.hparams.s,
            m=self.hparams.m,
            easy_margin=self.hparams.easy_margin,
            ls_eps=self.hparams.ls_eps,
            in_chans=9 if self.hparams.all_images else 3,
        )

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
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--num_classes", type=int, default=15587)
        parser.add_argument("--num_species", type=int, default=26)
        parser.add_argument("--specie_hidden", type=int, default=128)
        parser.add_argument("--flip_id", type=int, default=0)
        parser.add_argument("--all_images", type=int, default=1)

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

        train_embeddings = _gather("embeddings", outs=self.train_outputs).cpu().float()
        train_labels = _gather("individual_labels", outs=self.train_outputs).cpu().numpy()

        embeddings = _gather("embeddings").cpu().float()
        klass_probabilities = _gather("klass_probabilities").cpu().numpy()
        klass_labels = _gather("klass_labels").cpu().numpy()
        specie_probabilities = _gather("specie_probabilities").cpu().numpy()
        specie_labels = _gather("specie_labels").cpu().numpy()
        individual_labels = _gather("individual_labels").cpu().numpy()
        new = _gather("new").cpu().numpy()

        _distance = 1 - torch.mm(F.normalize(embeddings), F.normalize(train_embeddings).T)
        topk_distances, topk_predictions = torch.topk(_distance, k=100, largest=False, dim=1)
        topk_distances, topk_predictions = topk_distances.numpy(), topk_predictions.numpy()

        k = 5
        best_thresh, best_map5, best_map1 = 0.2, 0, 0
        for new_thresh in np.arange(0.2, 1.01, 0.05):
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
                if dists[-1] > new_thresh:
                    new_i = min(i for i, d in enumerate(dists) if d > new_thresh)
                    new_pred = individual_labels[i] if new[i] else -1
                    pred_ids = pred_ids[:new_i] + [new_pred] + pred_ids[new_i:-1]

                predictions.append(pred_ids)

            map1 = map_per_set(individual_labels, predictions, topk=1)
            map5 = map_per_set(individual_labels, predictions, topk=5)
            if (map5, map1) > (best_map5, best_map1):
                best_thresh, best_map5, best_map1 = new_thresh, map5, map1

        # Klass metrics
        klass_threshold = 0.5
        klass_predictions = (klass_probabilities > klass_threshold).astype(int)
        klass_metrics = classification_report(klass_labels, klass_predictions, output_dict=True, zero_division=0)

        # Specie metrics
        specie_predictions = np.argmax(specie_probabilities, axis=1)
        specie_metrics = classification_report(specie_labels, specie_predictions, output_dict=True, zero_division=0)

        # Progress Bar
        self.log(f"map@1", best_map1, prog_bar=True, logger=False)
        self.log(f"map@5", best_map5, prog_bar=True, logger=False)
        self.log(f"kf1", klass_metrics["macro avg"]["f1-score"], prog_bar=True, logger=False)
        self.log(f"sf1", specie_metrics["macro avg"]["f1-score"], prog_bar=True, logger=False)

        # Logger
        self.log(f"metrics/{stage}_map@1", best_map1, prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_map@5", best_map5, prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_new_threshold", best_thresh, prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_klass_f1", klass_metrics["macro avg"]["f1-score"], prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_specie_f1", specie_metrics["macro avg"]["f1-score"], prog_bar=False, logger=True)

    def _step(self, batch, _batch_idx, stage):
        arcface_logits, klass_logits, specie_logits, embeddings = self.forward(batch["image"], batch["individual_label"])

        # Losses
        arcface_loss = self.arcface_criterion(arcface_logits, batch["individual_label"])
        klass_loss = self.klass_criterion(klass_logits, batch["klass_label"].float())
        specie_loss = self.specie_criterion(specie_logits, batch["specie_label"])
        losses = {
            "total": arcface_loss + klass_loss + specie_loss,
            "arcface": arcface_loss,
            "klass": klass_loss,
            "specie": specie_loss
        }

        metrics = dict()
        self._log(losses, metrics, stage=stage)

        if stage == "train":
            self.train_outputs.append({
                "embeddings": embeddings.detach().cpu(),
                "individual_labels": batch["individual_label"].detach().cpu(),
                "klass_labels": batch["klass_label"].detach().cpu(),
            })
            return losses["total"]

        return {
            "embeddings": embeddings.detach().cpu(),
            "klass_probabilities": klass_logits.detach().sigmoid().cpu(),
            "specie_probabilities": specie_logits.detach().softmax(dim=1).cpu(),
            "individual_labels": batch["individual_label"].detach().cpu(),
            "klass_labels": batch["klass_label"].detach().cpu(),
            "specie_labels": batch["specie_label"].detach().cpu(),
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
