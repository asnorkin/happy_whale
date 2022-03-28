import os
import os.path as osp

import pandas as pd
from tqdm import tqdm

from utils.dataset import ImageItemsDataset


class CameraDataset(ImageItemsDataset):
    def __getitem__(self, index):
        item = self.items[index]
        sample = {
            "image": self.load_image(self.get_image_file(item)),
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

    def get_image_file(self, item):
        if item["image_file_fish"]:
            return item["image_file_fish"]
        elif item["image_file_fin"]:
            return item["image_file_fin"]
        return item["image_file"]

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
                "viewpoint": -1,
                "klass": -1,
                "species": -1,
                "viewpoint_label": -1,
                "klass_label": -1,
                "species_label": -1,
                "fold": -1,
            } for image_file in image_files])

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

            name, ext = osp.splitext(row.image)
            image_file_fish = osp.join(images_dir, name + "_fish" + ext)
            image_file_fin = osp.join(images_dir, name + "_fin" + ext)

            item = {
                "image_file": image_file,
                "image_file_fish": image_file_fish if osp.exists(image_file_fish) else "",
                "image_file_fin": image_file_fin if osp.exists(image_file_fin) else "",
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
