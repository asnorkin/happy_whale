import os.path as osp
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

from pipeline.loss import FocalLoss
from pipeline.metric import map_per_set
from pipeline.model import HappyModel


def find_best_map(
        val_embeddings,
        train_embeddings,
        val_labels,
        train_labels,
        new,
        t_min=0.2,
        t_max=0.66,
        t_step=0.05
):
    K = 5

    _distance = 1 - torch.mm(F.normalize(val_embeddings), F.normalize(train_embeddings).T)
    topk_distances, topk_predictions = torch.topk(_distance, k=min(100, *_distance.shape), largest=False, dim=1)
    topk_distances, topk_predictions = topk_distances.numpy(), topk_predictions.numpy()

    best_thresh, best_map5, best_map1 = t_min, 0, 0
    for new_thresh in np.arange(t_min, t_max, t_step):
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
            pred_ids, dists = _pred_ids[:K], _dists[:K]

            # Add new_individual
            if dists[-1] > new_thresh:
                new_i = min(i for i, d in enumerate(dists) if d > new_thresh)
                new_pred = val_labels[i] if new[i] else -1
                pred_ids = pred_ids[:new_i] + [new_pred] + pred_ids[new_i:-1]

            predictions.append(pred_ids)

        map1 = map_per_set(val_labels, predictions, topk=1)
        map5 = map_per_set(val_labels, predictions, topk=K)
        if (map5, map1) > (best_map5, best_map1):
            best_thresh, best_map5, best_map1 = new_thresh, map5, map1

    return best_thresh, best_map5, best_map1


class HappyLightningModule(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.arcface_criterion = nn.CrossEntropyLoss()
        self.klass_criterion = nn.BCEWithLogitsLoss()
        self.specie_criterion = nn.CrossEntropyLoss()
        self.viewpoint_criterion = nn.CrossEntropyLoss(label_smoothing=self.hparams.viewpoint_smoothing)
        self.crop_criterion = nn.BCEWithLogitsLoss()
        self.crop_weight = 1.0 - int(self.hparams.all_images)  # Do not use crop loss for --all_images=1 case

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
            all_images=self.hparams.all_images,
        )

        # Placeholders
        self.train_outputs = None
        self.best_model_score = None
        self.best_model_thresh = None

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
        parser.add_argument("--all_images", type=int, default=0)
        parser.add_argument("--random_image", type=int, default=0)

        # Loss
        parser.add_argument("--s", type=float, default=30.0)
        parser.add_argument("--m", type=float, default=0.5)
        parser.add_argument("--easy_margin", type=int, default=0)
        parser.add_argument("--ls_eps", type=float, default=0.0)
        parser.add_argument("--focal_gamma", type=float, default=2.0)
        parser.add_argument("--viewpoint_smoothing", type=float, default=0.05)

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

    def test_step(self, batch, batch_idx):
        klass_probs_fin, specie_probs_fin, *_, embeddings_fin = self.model.predict(batch["image"])
        klass_probs_fish, specie_probs_fish, *_, embeddings_fish = self.model.predict(batch["image_fish"])

        def _hflip(tensor):
            return torch.flip(tensor, dims=(3,))

        klass_probs_fin_hflip, specie_probs_fin_hflip, *_, embeddings_fin_hflip = self.model.predict(_hflip(batch["image"]))
        klass_probs_fish_hflip, specie_probs_fish_hflip, *_, embeddings_fish_hflip = self.model.predict(_hflip(batch["image_fish"]))

        klass_probs = torch.stack((klass_probs_fin, klass_probs_fish, klass_probs_fin_hflip, klass_probs_fish_hflip), dim=1)
        specie_probs = torch.stack((specie_probs_fin, specie_probs_fish, specie_probs_fin_hflip, specie_probs_fish_hflip), dim=1)
        embeddings = torch.stack((embeddings_fin, embeddings_fish, embeddings_fin_hflip, embeddings_fish_hflip), dim=1)

        return {
            "klass_probs": klass_probs,
            "specie_probs": specie_probs,
            "embeddings": embeddings,
            "individual_labels": batch["individual_label"],
            "fold": batch["fold"],
            "new": batch["new"],
        }

    def test_epoch_end(self, outputs):
        stage = "test"

        klass_probs = self._gather("klass_probs", outs=outputs).cpu().float()
        specie_probs = self._gather("specie_probs", outs=outputs).cpu().float()
        embeddings = self._gather("embeddings", outs=outputs).cpu().float()
        individual_labels = self._gather("individual_labels", outs=outputs).cpu().numpy()
        new = self._gather("new", outs=outputs).cpu().numpy()
        fold = self._gather("fold", outs=outputs).cpu().numpy()

        torch.save(embeddings, osp.join(self.hparams.checkpoints_dir, "embeddings.pt"))
        torch.save(klass_probs, osp.join(self.hparams.checkpoints_dir, "klass_probs.pt"))
        torch.save(specie_probs, osp.join(self.hparams.checkpoints_dir, "specie_probs.pt"))

        train_indices = np.where(fold != self.hparams.fold)[0]
        val_indices = np.where(fold == self.hparams.fold)[0]

        embeddings = F.normalize(embeddings, dim=2).mean(dim=1)

        best_thresh, best_map5, best_map1 = find_best_map(
            embeddings[train_indices],
            embeddings[val_indices],
            individual_labels[train_indices],
            individual_labels[val_indices],
            new
        )

        self.best_model_score = best_map5
        self.best_model_thresh = best_thresh

        self.log(f"metrics/{stage}_map@1", best_map1, prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_map@5", best_map5, prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_new_threshold", best_thresh, prog_bar=False, logger=True)

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
        klass_probabilities = self._gather("klass_probabilities", outs=outputs).cpu().numpy()
        klass_labels = self._gather("klass_labels", outs=outputs).cpu().numpy()
        specie_probabilities = self._gather("specie_probabilities", outs=outputs).cpu().numpy()
        specie_labels = self._gather("specie_labels", outs=outputs).cpu().numpy()
        crop_probabilities = self._gather("crop_probabilities", outs=outputs).cpu().numpy()
        crop_labels = self._gather("crop_labels", outs=outputs).cpu().numpy()
        viewpoint_probabilities = self._gather("viewpoint_probabilities", outs=outputs).cpu().numpy()
        viewpoint_labels = self._gather("viewpoint_labels", outs=outputs).cpu().numpy()
        individual_labels = self._gather("individual_labels", outs=outputs).cpu().numpy()
        new = self._gather("new", outs=outputs).cpu().numpy()

        best_thresh, best_map5, best_map1 = find_best_map(
            embeddings, train_embeddings, individual_labels, train_labels, new)

        # Klass metrics
        klass_threshold = 0.5
        klass_predictions = (klass_probabilities > klass_threshold).astype(int)
        klass_metrics = classification_report(klass_labels, klass_predictions, output_dict=True, zero_division=0)

        # Specie metrics
        specie_predictions = np.argmax(specie_probabilities, axis=1)
        specie_metrics = classification_report(specie_labels, specie_predictions, output_dict=True, zero_division=0)

        # Crop metrics
        crop_threshold = 0.5
        crop_predictions = (crop_probabilities > crop_threshold).astype(int)
        crop_metrics = classification_report(crop_labels, crop_predictions, output_dict=True, zero_division=0)

        # Viewpoint metrics
        viewpoint_predictions = np.argmax(viewpoint_probabilities, axis=1)
        viewpoint_metrics = classification_report(viewpoint_labels, viewpoint_predictions, output_dict=True, zero_division=0)

        # Progress Bar
        self.log(f"map@1", best_map1, prog_bar=True, logger=False)
        self.log(f"map@5", best_map5, prog_bar=True, logger=False)
        self.log(f"kf1", klass_metrics["macro avg"]["f1-score"], prog_bar=True, logger=False)
        self.log(f"sf1", specie_metrics["macro avg"]["f1-score"], prog_bar=True, logger=False)
        self.log(f"cf1", crop_metrics["macro avg"]["f1-score"], prog_bar=True, logger=False)
        self.log(f"vf1", viewpoint_metrics["macro avg"]["f1-score"], prog_bar=True, logger=False)

        # Logger
        self.log(f"metrics/{stage}_map@1", best_map1, prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_map@5", best_map5, prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_new_threshold", best_thresh, prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_klass_f1", klass_metrics["macro avg"]["f1-score"], prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_specie_f1", specie_metrics["macro avg"]["f1-score"], prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_crop_f1", crop_metrics["macro avg"]["f1-score"], prog_bar=False, logger=True)
        self.log(f"metrics/{stage}_viewpoint_f1", viewpoint_metrics["macro avg"]["f1-score"], prog_bar=False, logger=True)

    def _step(self, batch, _batch_idx, stage):
        arcface_logits, klass_logits, specie_logits, crop_logits, viewpoint_logits, embeddings = \
            self.forward(batch["image"], batch["individual_label"])

        # Losses
        arcface_loss = self.arcface_criterion(arcface_logits, batch["individual_label"])
        klass_loss = self.klass_criterion(klass_logits, batch["klass_label"].float())
        specie_loss = self.specie_criterion(specie_logits, batch["specie_label"])
        crop_loss = self.crop_criterion(crop_logits, batch["crop_label"].float())
        viewpoint_loss = self.viewpoint_criterion(viewpoint_logits, batch["viewpoint_label"])

        losses = {
            "total": arcface_loss + klass_loss + specie_loss + self.crop_weight * crop_loss + viewpoint_loss,
            "arcface": arcface_loss,
            "klass": klass_loss,
            "specie": specie_loss,
            "crop": crop_loss,
            "viewpoint": viewpoint_loss,
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
            "crop_probabilities": crop_logits.detach().sigmoid().cpu(),
            "viewpoint_probabilities": viewpoint_logits.detach().softmax(dim=1).cpu(),
            "individual_labels": batch["individual_label"].detach().cpu(),
            "klass_labels": batch["klass_label"].detach().cpu(),
            "specie_labels": batch["specie_label"].detach().cpu(),
            "crop_labels": batch["crop_label"].detach().cpu(),
            "viewpoint_labels": batch["viewpoint_label"].detach().cpu(),
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
