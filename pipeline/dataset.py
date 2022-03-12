import os
import os.path as osp

import pandas as pd
from tqdm import tqdm

from utils.dataset import ImageItemsDataset


class HappyDataset(ImageItemsDataset):
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
    print(1)
