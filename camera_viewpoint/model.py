import timm
import torch
from torch import nn


class CameraModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        num_species,
        viewpoint_hidden=64,
        specie_hidden=64,
        dropout=0.5,
        pretrained=True,
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

        self.viewpoint_head = nn.Sequential(
            nn.Linear(in_features, viewpoint_hidden),
            nn.BatchNorm1d(viewpoint_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(viewpoint_hidden, num_classes),
        )
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

    def forward(self, images):
        x = self.model(images)
        viewpoint_logits = self.viewpoint_head(x)
        klass_logits = self.klass_head(x)[:, 0]  # (B, 1) -> (B,)
        specie_logits = self.specie_head(x)
        return viewpoint_logits, klass_logits, specie_logits

    def predict(self, images):
        viewpoint_logits, klass_logits, specie_logits = self.forward(images)
        viewpoint_probabilities = viewpoint_logits.softmax(dim=1)
        klass_probabilities = klass_logits.sigmoid()
        specie_probabilities = specie_logits.softmax(dim=1)

        return viewpoint_probabilities, klass_probabilities, specie_probabilities

    @classmethod
    def from_checkpoint(cls, checkpoint_file):
        ckpt = torch.load(checkpoint_file, map_location="cpu")
        state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
        hparams = ckpt["hyper_parameters"]

        model = cls(
            model_name=hparams["model_name"],
            num_classes=hparams["num_classes"],
            num_species=hparams["num_species"],
            viewpoint_hidden=hparams["viewpoint_hidden"],
            specie_hidden=hparams["specie_hidden"],
            dropout=hparams["dropout"],
        )
        model.load_state_dict(state_dict)

        return model
