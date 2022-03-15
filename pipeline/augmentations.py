import random

import albumentations as A
import torch


class HorizontalFlip(A.HorizontalFlip):
    def __init__(self, num_classes, flip_id=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    def __call__(self, *args, force_apply=False, **kwargs):
        if (random.random() < self.p) or self.always_apply or force_apply:
            kwargs["image"] = self.apply(kwargs["image"])
            kwargs["individual_label"] += self.num_classes

        return kwargs


class StackImages(A.BasicTransform):
    def __init__(self, always_apply=True, *args, **kwargs):
        super().__init__(always_apply=always_apply, *args, **kwargs)

    def __call__(self, *args, force_apply=False, **kwargs):
        if (random.random() < self.p) or self.always_apply or force_apply:
            kwargs["image"] = torch.cat((kwargs["image"], kwargs["image_fish"], kwargs["image_fin"]), dim=0)

        return kwargs
