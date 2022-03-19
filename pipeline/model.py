import timm
import torch
from timm.models.layers.conv2d_same import Conv2dSame
from torch import nn

from pipeline.arcface import ArcMarginProduct, GeM


class HappyModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        num_species,
        embedding_size=512,
        specie_hidden=128,
        dropout=0.2,
        pretrained=True,
        s=30.0,
        m=0.5,
        easy_margin=False,
        ls_eps=0.0,
        all_images=False,
        **timm_kwargs
    ):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained, **timm_kwargs)
        for head_name in ["fc", "head", "classifier"]:
            if hasattr(self.model, head_name):
                in_features = getattr(self.model, head_name).in_features
                setattr(self.model, head_name, nn.Identity())
                break
        else:
            raise ValueError(f"Model has no expected head")

        if all_images:
            n_images = 2
            conv_stem_grouped = Conv2dSame(
                in_channels=self.model.conv_stem.in_channels * n_images,
                out_channels=self.model.conv_stem.out_channels,
                kernel_size=self.model.conv_stem.kernel_size,
                stride=self.model.conv_stem.stride,
                padding=self.model.conv_stem.padding,
                dilation=self.model.conv_stem.dilation,
                groups=n_images,
                bias=(self.model.conv_stem.bias is not None),
            )
            conv_stem_grouped.weight = self.model.conv_stem.weight
            conv_stem_grouped.bias = self.model.conv_stem.bias
            self.model.conv_stem = conv_stem_grouped

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
            nn.Linear(in_features, specie_hidden),
            nn.BatchNorm1d(specie_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(specie_hidden, num_species),
        )
        self.crop_head = nn.Sequential(
            nn.Linear(in_features, 1),
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

        # Crop
        crop_logits = self.crop_head(pooled_features)[:, 0] # (B, 1) -> (B,)

        return arcface_logits, klass_logits, specie_logits, crop_logits, embeddings

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

        # Crop
        crop_logits = self.crop_head(pooled_features)[:, 0]  # (B, 1) -> (B,)
        crop_probabilities = crop_logits.sigmoid()

        return klass_probabilities, specie_probabilities, crop_probabilities, embeddings

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
            specie_hidden=hparams["specie_hidden"],
            dropout=hparams["dropout"],
            s=hparams["s"],
            m=hparams["m"],
            easy_margin=hparams["easy_margin"],
            ls_eps=hparams["ls_eps"],
        )
        model.load_state_dict(state_dict)

        return model
