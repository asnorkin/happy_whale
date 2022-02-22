import timm
from torch import nn

from pipeline.arcface import ArcMarginProduct, GeM


class HappyModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True, s=30.0, m=0.5,
                 easy_margin=False, ls_eps=0.0, **timm_kwargs):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, **timm_kwargs)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.fc = ArcMarginProduct(in_features, num_classes, s=s, m=m, easy_margin=easy_margin, ls_eps=ls_eps)

    def forward(self, images, labels):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.fc(pooled_features, labels)
        return output, pooled_features
