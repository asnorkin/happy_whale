import os
import os.path as osp

import cv2
import pandas as pd
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class HappyDataset(Dataset):
    def __init__(self, items, transform=None, load_all_fields=False):
        self.items = items
        self.transform = transform
        self.load_all_fields = load_all_fields

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        sample = {
            "image": self.load_image(item["image_file"]),
            "klass_label": item["klass_label"],
            "species_label": item["species_label"],
            "individual_label": item["individual_label"],
            "new": item["new"],
        }

        if self.load_all_fields:
            for key in item.keys():
                if key not in sample:
                    sample[key] = item[key]

        if self.transform:
            sample = self.transform(**sample)

        return sample

    @classmethod
    def create(cls, images_dir, labels_csv=None, transform=None, load_all_fields=False, debug=False):
        items = cls.load_items(images_dir, labels_csv, debug)
        return cls(items, transform=transform, load_all_fields=load_all_fields)

    @classmethod
    def load_items(cls, images_dir, labels_csv=None, debug=False):
        if labels_csv is not None:
            labels_df = pd.read_csv(labels_csv)

        else:
            # Only images case
            # Create dummy labels dataframe
            labels_df = pd.DataFrame([{
                "image": image_file,
                "klass": -1,
                "species": -1,
                "individual_id": -1,
                "klass_label": -1,
                "species_label": -1,
                "individual_label": -1,
                "fold": -1,
                "new": -1,
            } for image_file in os.listdir(images_dir)])

        items, not_found = [], 0
        for i, row in enumerate(tqdm(labels_df.itertuples(), desc="Loading items", unit="item", total=len(labels_df))):
            image_file = osp.join(images_dir, row.image)
            if not osp.exists(image_file):
                not_found += 1
                continue

            if debug and len(items) >= 100:
                break

            item = {
                "image_file": image_file,
                "klass": row.klass,
                "species": row.species,
                "individual_id": row.individual_id,
                "klass_label": row.klass_label,
                "species_label": row.species_label,
                "individual_label": row.individual_label,
                "fold": row.fold,
                "new": row.new,
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
