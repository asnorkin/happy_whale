import pytorch_lightning as pl

from pipeline.datamodule import HappyLightningDataModule
from pipeline.module import HappyLightningModule
from utils.fs_utils import create_if_not_exist
from utils.pl_utils import (
    checkpoint_callback,
    config_args,
    lr_monitor_callback,
    tensorboard_logger
)

import cv2
cv2.setNumThreads(1)


def train(args):
    pl.seed_everything(args.seed)

    datamodule = HappyLightningDataModule(**vars(args))
    module = HappyLightningModule(**vars(args))

    logger = tensorboard_logger(args)

    callbacks = []
    ckpt_callback = checkpoint_callback(args)
    callbacks.append(ckpt_callback)
    callbacks.append(lr_monitor_callback())

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)
    trainer.fit(module, datamodule=datamodule)

    print(f"Best model score: {ckpt_callback.best_model_score:.3f}")
    print(f"Best model path: {ckpt_callback.best_model_path}")


def main(args):
    create_if_not_exist(args.checkpoints_dir)
    create_if_not_exist(args.log_dir)
    train(args)


if __name__ == '__main__':
    main(config_args(HappyLightningModule, HappyLightningDataModule))
