import timm
import torch
from torch import nn

from pipeline.arcface import ArcMarginProduct, GeM


class HappyModel(nn.Module):
    def __init__(self, model_name, num_classes, embedding_size=512, pretrained=True,
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
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = ArcMarginProduct(embedding_size, num_classes, s=s, m=m, easy_margin=easy_margin, ls_eps=ls_eps)

    def forward(self, images, labels):
        embeddings = self.embed(images)
        output = self.fc(embeddings, labels)
        return output, embeddings

    def embed(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embeddings = self.embedding(pooled_features)
        return embeddings

    @classmethod
    def from_checkpoint(cls, checkpoint_file):
        ckpt = torch.load(checkpoint_file, map_location="cpu")
        state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
        hparams = ckpt["hyper_parameters"]

        model = cls(
            model_name=hparams["model_name"],
            num_classes=hparams["num_classes"],
            embedding_size=hparams["embedding_size"],
            s=hparams["s"],
            m=hparams["m"],
            easy_margin=hparams["easy_margin"],
            ls_eps=hparams["ls_eps"],
        )
        model.load_state_dict(state_dict)

        return model
