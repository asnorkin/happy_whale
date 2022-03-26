import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from fuse.datamodule import FuseLightningDataModule
from fuse.module import FuseLightningModule
from utils.fs_utils import create_if_not_exist
from utils.pl_utils import (
    checkpoint_callback,
    config_args,
    lr_monitor_callback,
    wandb_logger
)

import cv2
cv2.setNumThreads(1)


def train(args):
    pl.seed_everything(args.seed)

    datamodule = FuseLightningDataModule(**vars(args))
    module = FuseLightningModule(**vars(args))

    callbacks = []
    ckpt_callback = checkpoint_callback(args)
    callbacks.append(ckpt_callback)
    callbacks.append(lr_monitor_callback())

    logger = wandb_logger(args)
    plugins = DDPPlugin(find_unused_parameters=False)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, plugins=plugins)
    trainer.fit(module, datamodule=datamodule)

    # module = HappyLightningModule.load_from_checkpoint(ckpt_callback.best_model_path)
    # trainer.test(module, dataloaders=datamodule.test_dataloader())

    # print(f"Best model thresh: {module.best_model_thresh:.3f}")
    print(f"Best model score: {ckpt_callback.best_model_score:.3f}")
    print(f"Best model path: {ckpt_callback.best_model_path}")


def main(args):
    create_if_not_exist(args.checkpoints_dir)
    create_if_not_exist(args.log_dir)
    train(args)


if __name__ == '__main__':
    main(config_args(FuseLightningModule, FuseLightningDataModule))
