import timm
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
        self.fc = ArcMarginProduct(in_features, num_classes, s=s, m=m, easy_margin=easy_margin, ls_eps=ls_eps)

    def forward(self, images, labels):
        embeddings = self.embed(images)
        output = self.fc(embeddings, labels)
        return output, embeddings

    def embed(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embeddings = self.embedding(pooled_features)
        return embeddings
