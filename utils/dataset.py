import cv2
from torch.utils.data.dataset import Dataset


class ImageItemsDataset(Dataset):
    def __init__(self, items, transform=None, load_all_fields=False):
        self.items = items
        self.transform = transform
        self.load_all_fields = load_all_fields

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        sample = {
            "image": self.load_image(self.get_image_file(item)),
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

    def get_image_file(self, item):
        return item["image_file"]

    @classmethod
    def create(cls, images_dir, labels_csv=None, transform=None, load_all_fields=False, debug=False):
        items = cls.load_items(images_dir, labels_csv, debug)
        return cls(items, transform=transform, load_all_fields=load_all_fields)

    @classmethod
    def load_items(cls, images_dir, labels_csv=None, debug=False):
        raise NotImplementedError("Method load_items should be implemented in ancestors")

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
