import timm
import torch
from torch import nn

from pipeline.arcface import ArcMarginProduct, GeM


class HappyModel(nn.Module):
    def __init__(self, model_name, num_classes, num_species, embedding_size=512, dropout=0.2, pretrained=True,
                 s=30.0, m=0.5, easy_margin=False, ls_eps=0.0, **timm_kwargs):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, **timm_kwargs)
        if hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        else:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Dropout(dropout),
        )
        self.fc = ArcMarginProduct(embedding_size, num_classes, s=s, m=m, easy_margin=easy_margin, ls_eps=ls_eps)
        self.klass_head = nn.Sequential(
            nn.Linear(in_features, 1),
        )
        self.specie_head = nn.Sequential(
            nn.Linear(in_features, num_species)
        )

    def forward(self, images, labels):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)

        # ArcFace
        embeddings = self.embedding(pooled_features)
        arcface_logits = self.fc(embeddings, labels)

        # Klass
        klass_logits = self.klass_head(pooled_features)[:, 0]  # (B, 1) -> (B,)

        # Specie
        specie_logits = self.specie_head(pooled_features)

        return arcface_logits, klass_logits, specie_logits, embeddings

    def predict(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)

        # Embeddings
        embeddings = self.embedding(pooled_features)

        # Klass
        klass_logits = self.klass_head(pooled_features)[:, 0]  # (B, 1) -> (B,)
        klass_probabilities = klass_logits.sigmoid()

        # Specie
        specie_logits = self.specie_head(pooled_features)
        specie_probabilities = specie_logits.softmax(dim=1)

        return klass_probabilities, specie_probabilities, embeddings

    @classmethod
    def from_checkpoint(cls, checkpoint_file):
        ckpt = torch.load(checkpoint_file, map_location="cpu")
        state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
        hparams = ckpt["hyper_parameters"]

        model = cls(
            model_name=hparams["model_name"],
            num_classes=hparams["num_classes"],
            num_species=hparams["num_species"],
            embedding_size=hparams["embedding_size"],
            dropout=hparams["dropout"],
            s=hparams["s"],
            m=hparams["m"],
            easy_margin=hparams["easy_margin"],
            ls_eps=hparams["ls_eps"],
        )
        model.load_state_dict(state_dict)

        return model
