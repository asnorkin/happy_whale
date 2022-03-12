import random

import albumentations as A


class CameraHorizontalFlip(A.HorizontalFlip):
    FLIP = {
        0: 1,
        1: 0,
        2: 2,
        "port": "starboard",
        "starboard": "port",
        "unclear": "unclear",
    }

    def __call__(self, *args, force_apply=False, **kwargs):
        if (random.random() < self.p) or self.always_apply or force_apply:
            kwargs["image"] = self.apply(kwargs["image"])
            kwargs["viewpoint_label"] = self.FLIP[kwargs["viewpoint_label"]]
            if "viewpoint" in kwargs:
                kwargs["viewpoint"] = self.FLIP[kwargs["viewpoint"]]

        return kwargs
