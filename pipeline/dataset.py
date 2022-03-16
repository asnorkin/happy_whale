import os
import os.path as osp

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.dataset import ImageItemsDataset


class HappyDataset(ImageItemsDataset):
    def __init__(self, *args, load_all_images=False, load_random_image=False, p_fin=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_all_images = load_all_images
        self.load_random_image = load_random_image
        self.p_fin = p_fin if load_random_image else 1.0

    def get_image_file(self, item):
        if item["image_file_fin"] and np.random.random() < self.p_fin:
            image_file = item["image_file_fin"]
        elif item["image_file_fish"]:
            image_file = item["image_file_fish"]
        else:
            image_file = item["image_file"]

        return image_file

    def __getitem__(self, index):
        item = self.items[index]
        if self.load_all_images:
            image_fish = self.load_image(item["image_file_fish"] if item["image_file_fish"] else item["image_file"])
            image_fin = self.load_image(item["image_file_fin"]) if item["image_file_fin"] else image_fish

            item = self.items[index]
            sample = {
                "image": image_fin,  # For albumentations compatibility
                "image_fish": image_fish,
                "image_fin": image_fin,
            }

        else:
            return super().__getitem__(index)

        for key in ["klass_label", "specie_label", "individual_label", "new"]:
            sample[key] = item[key]

        if self.load_all_fields:
            for key in item.keys():
                if key not in sample:
                    sample[key] = item[key]

        if self.transform:
            sample = self.transform(**sample)

        return sample

    @classmethod
    def load_items(cls, images_dir, labels_csv=None, debug=False):
        if labels_csv is not None:
            labels_df = pd.read_csv(labels_csv)

        else:
            # Only images case
            # Create dummy labels dataframe
            image_files = []
            for image_file in os.listdir(images_dir):
                name, ext = osp.splitext(image_file)
                if not name.endswith(("_fish", "_fin")):
                    image_files.append(image_file)

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
            } for image_file in image_files])

        items, not_found = [], 0
        for i, row in enumerate(tqdm(labels_df.itertuples(), desc="Loading items", unit="item", total=len(labels_df))):
            image_file = osp.join(images_dir, row.image)
            if not osp.exists(image_file):
                not_found += 1
                continue

            if debug and len(items) >= 100:
                break

            name, ext = osp.splitext(row.image)
            image_file_fish = osp.join(images_dir, name + "_fish" + ext)
            image_file_fin = osp.join(images_dir, name + "_fin" + ext)

            item = {
                "image_file": image_file,
                "image_file_fish": image_file_fish if osp.exists(image_file_fish) else "",
                "image_file_fin": image_file_fin if osp.exists(image_file_fin) else "",
                "klass": row.klass,
                "species": row.species,
                "individual_id": row.individual_id,
                "klass_label": row.klass_label,
                "specie_label": row.species_label,
                "individual_label": row.individual_label,
                "fold": row.fold,
                "new": row.new,
            }
            items.append(item)

        if not_found > 0:
            print(f"Not found: {not_found}")

        return items


if __name__ == "__main__":
    items = HappyDataset.load_items("../data/train_images", "../data/train.csv", debug=True)
    dataset = HappyDataset(items)
    sample = dataset[13]
    print(1)
