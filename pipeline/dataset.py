import os
import os.path as osp

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.dataset import ImageItemsDataset


class HappyDataset(ImageItemsDataset):
    def __init__(self, *args, load_all_images=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_all_images = load_all_images

    def __getitem__(self, index):
        if not self.load_all_images:
            return super().__getitem__(index)

        item = self.items[index]
        images_dir, fname = osp.split(item["image_file"])
        name, ext = osp.splitext(fname)

        image_full = self.load_image(item["image_file"])

        fish_file = osp.join(images_dir, name + "_fish" + ext)
        image_fish = np.zeros_like(image_full) if not osp.exists(fish_file) else self.load_image(fish_file)

        fin_file = osp.join(images_dir, name + "_fin" + ext)
        image_fin = np.zeros_like(image_full) if not osp.exists(fin_file) else self.load_image(fin_file)

        item = self.items[index]
        sample = {
            "image": image_full,
            "image_fish": image_fish,
            "image_fin": image_fin,
            "klass_label": item["klass_label"],
            "specie_label": item["specie_label"],
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
