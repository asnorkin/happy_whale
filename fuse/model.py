import torch
from torch import nn

from pipeline.arcface import ArcMarginProduct


class FuseModel(nn.Module):
    def __init__(
        self,
        num_classes,
        emb_size=512,
        hidden_input=512,
        output_size=512,
        s=30.0,
        m=0.5,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super().__init__()

        self.fin_input = nn.Sequential(
            nn.Linear(emb_size, hidden_input),
        )
        self.fish_input = nn.Sequential(
            nn.Linear(emb_size, hidden_input),
        )

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_input, output_size),
        )

        self.fc = ArcMarginProduct(output_size, num_classes, s=s, m=m, easy_margin=easy_margin, ls_eps=ls_eps)

    def forward(self, fin_emb, fish_emb, labels):
        fin_emb = self.fin_input(fin_emb)
        fish_emb = self.fish_input(fish_emb)
        emb = self.head(torch.cat((fin_emb, fish_emb), dim=-1))

        arcface_logits = self.fc(emb, labels)

        return arcface_logits, emb
