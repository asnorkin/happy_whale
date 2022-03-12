import os.path as osp

import pandas as pd
from tqdm import tqdm

from utils.dataset import ImageItemsDataset


class CameraDataset(ImageItemsDataset):
    def __getitem__(self, index):
        item = self.items[index]
        sample = {
            "image": self.load_image(item["image_file"]),
            "klass_label": item["klass_label"],
            "specie_label": item["specie_label"],
            "viewpoint_label": item["viewpoint_label"],
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
        if labels_csv is None:
            raise ValueError("labels_csv should not be None")

        labels_df = pd.read_csv(labels_csv)
        labels_df = labels_df[~labels_df.viewpoint.isna()]
        labels_df.viewpoint_label = labels_df.viewpoint_label.astype(int)

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
                "viewpoint": row.viewpoint,
                "klass": row.klass,
                "species": row.species,
                "viewpoint_label": row.viewpoint_label,
                "klass_label": row.klass_label,
                "specie_label": row.species_label,
                "fold": row.fold,
            }
            items.append(item)

        if not_found > 0:
            print(f"Not found: {not_found}")

        return items


if __name__ == "__main__":
    dataset = CameraDataset.create("../data/train_images", "../data/camera_viewpoint_labels.csv", debug=True)
