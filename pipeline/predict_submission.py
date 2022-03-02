from argparse import ArgumentParser

import torch

from pipeline.model import HappyModel


CONFIG = {
    "model_name": "tf_efficientnet_b0_ns",
    "num_classes": 15587,
    "embedding_size": 512,
}


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--submission_csv", type=str, default="../data/sample_submission.csv")
    parser.add_argument("--train_images_dir", type=str, default="../data/train_images")
    parser.add_argument("--test_images_dir", type=str, default="../data/test_images")
    parser.add_argument("--checkpoint", type=str, default="../")

    args = parser.parse_args()
    return args


def load_model(args):
    ckpt = torch.load(args.checkpoint)
    print(1)


def main(args):
    model = load_model(args)


if __name__ == "__main__":
    main(parse_args())
