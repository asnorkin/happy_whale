import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class FuseDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        sample_keys = [
            "klass_label",
            "specie_label",
            "individual_label",
            "new",
            "fold",
            "embedding_fin",
            "embedding_fish",
            "has_fin"
        ]
        sample = {key: item[key] for key in sample_keys}
        return sample

    @classmethod
    def load_items(cls, labels_csv, image_ids_file, embeddings_file, fold=None, debug=False):
        embeddings = torch.load(embeddings_file)
        image_ids = np.load(image_ids_file)
        labels_df = pd.read_csv(labels_csv)

        image2ind = {image_id: i for i, image_id in enumerate(image_ids)}
        if fold is not None:
            embeddings = embeddings[fold]

        items = []
        for i, row in enumerate(tqdm(labels_df.itertuples(), desc="Loading items", unit="item", total=len(labels_df))):
            if debug and len(items) >= 100:
                break

            embedding_fin = embeddings[image2ind[row.image], 0]
            embedding_fish = embeddings[image2ind[row.image], 1]
            has_fin = True
            if torch.isclose(embedding_fin, embedding_fish).all():
                embedding_fin = torch.zeros_like(embedding_fin)
                has_fin = False

            item = {
                "embedding_fin": embedding_fin,
                "embedding_fish": embedding_fish,
                "has_fin": has_fin,
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

        return items


if __name__ == "__main__":
    labels_csv = "../data/train.csv"
    image_ids_file = "../data/image_ids.npy"
    embeddings_file = "../data/tf_effb7_ns_m0.4_yolov5_9c_v3_augv3_2ldp0.4_512x512_4g20x3b60e0.001lr_embeddings.pt"
    items = FuseDataset.load_items(labels_csv, image_ids_file, embeddings_file, fold=0, debug=True)
