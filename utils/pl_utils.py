import yaml
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class ModelConfig:
    def __init__(self, config_file: str):
        self.params = yaml.safe_load(open(config_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def tensorboard_logger(args) -> TensorBoardLogger:
    return TensorBoardLogger(save_dir=args.log_dir, name=args.experiment, default_hp_metric=False)


def wandb_logger(args) -> WandbLogger:
    return WandbLogger(name=args.experiment, project=args.project)


def checkpoint_callback(args):
    return ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        save_top_k=1,
        save_last=True,
        monitor=args.monitor,
        mode=args.monitor_mode,
    )


def lr_monitor_callback() -> LearningRateMonitor:
    return LearningRateMonitor(log_momentum=False)


def parse_args(parser):
    args = parser.parse_args()

    args.checkpoints_dir = f"{args.work_dir}/checkpoints/{args.experiment}"
    args.log_dir = f"{args.work_dir}/logs"

    if args.num_epochs is not None:
        args.max_epochs = args.num_epochs

    if args.seed is not None:
        args.benchmark = False
        args.deterministic = True

    return args


def add_program_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument("--project", type=str, default="happy_whale")
    parser.add_argument("--experiment", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    return parser


def config_args(module_class, datamodule_class=None):
    parser = ArgumentParser()

    parser = add_program_specific_args(parser)
    if datamodule_class is not None:
        parser = datamodule_class.add_data_specific_args(parser)
    parser = module_class.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    return parse_args(parser)
