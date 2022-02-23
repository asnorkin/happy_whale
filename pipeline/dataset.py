import os
import os.path as osp
from copy import deepcopy

import cv2
import pandas as pd
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class HappyDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sample = deepcopy(self.items[index])
        sample["image"] = self.load_image(sample["image_file"])

        if self.transform:
            sample = self.transform(**sample)

        return sample

    @classmethod
    def create(cls, images_dir, labels_csv=None, debug=False):
        items = cls.load_items(images_dir, labels_csv, debug)
        return cls(items)

    @classmethod
    def load_items(cls, images_dir, labels_csv=None, debug=False):
        if labels_csv is not None:
            labels_df = pd.read_csv(labels_csv)

        else:
            # Only images case
            # Create dummy labels dataframe
            labels_df = pd.DataFrame([{
                "image": image_file,
                "label": -1,
                "specie": -1,
                "class": -1,
                "individual_id": -1,
                "fold": -1
            } for image_file in os.listdir(images_dir)])

        items, not_found = [], 0
        for i, row in enumerate(tqdm(labels_df.itertuples(), desc="Loading items", unit="item", total=len(labels_df))):
            image_file = osp.join(images_dir, row.image)
            if not osp.exists(image_file):
                not_found += 1
                continue

            if debug and len(items) >= 1000:
                break

            item = {
                "image_file": image_file,
                "label": row.label,
                "specie": row.species,
                "class": row.klass,
                "individual_id": row.individual_id,
                "fold": row.fold,
            }
            items.append(item)

        if not_found > 0:
            print(f"Not found: {not_found}")

        return items

    @classmethod
    def load_image(cls, image_file: str, fmt: str = "rgb", image_size=None):
        if fmt == "gray":
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        elif fmt == "rgb":
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image format: {fmt}. Supported are: gray, rgb")

        if image_size is not None:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            image = cv2.resize(image, image_size)

        return image


if __name__ == "__main__":
    items = HappyDataset.load_items("../data/train_images", "../data/train.csv", debug=True)
    dataset = HappyDataset(items)
    print(1)
