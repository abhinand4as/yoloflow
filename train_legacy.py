"""
Legacy training script - wrapper around original train.py
This script maintains backward compatibility with the original CLI interface
while using the new modular YoloFlow package internally.
"""

import argparse
import sys
from pathlib import Path

from yoloflow.core.config import (
    TrainingConfig,
    ModelConfig,
    DatasetConfig,
    OptimizerConfig,
    AugmentationConfig,
    TrainingHyperparameters,
    ExperimentConfig,
)
from yoloflow.training.orchestrator import TrainingOrchestrator
from yoloflow.utils.logging import setup_logging


def parse_legacy_args():
    """
    Parse legacy command-line arguments.
    This maintains compatibility with the original train.py interface.
    """
    parser = argparse.ArgumentParser(
        description="Train YOLO model (legacy compatibility wrapper)"
    )

    # Original arguments from train.py
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="yolo26n.pt")
    parser.add_argument("--data", type=str, default="yolo_cac_dataset_quadrants/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--cache", type=str, default="False")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="yolo26_cac_quadrants")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--val", type=bool, default=True)
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--plots", type=bool, default=True)
    parser.add_argument("--comet-api-key", type=str, default=None)
    parser.add_argument("--comet-project", type=str, default=None)
    parser.add_argument("--comet-workspace", type=str, default=None)
    parser.add_argument("--comet-experiment-name", type=str, default=None)
    parser.add_argument("--comet-experiment-tag", type=str, default=None)

    return parser.parse_args()


def main():
    """Main entry point for legacy training."""
    args = parse_legacy_args()

    # Setup logging
    setup_logging(level="INFO")

    # Load config from file if provided
    if args.config:
        config = TrainingConfig.from_yaml(Path(args.config))
    else:
        # Create config from command-line arguments
        config = TrainingConfig()

    # Map legacy args to new config structure
    config.model.name = args.model
    config.model.pretrained = args.pretrained
    config.model.dropout = args.dropout

    config.dataset.path = args.data
    config.dataset.imgsz = args.imgsz
    config.dataset.cache = args.cache
    config.dataset.fraction = args.fraction

    config.hyperparameters.epochs = args.epochs
    config.hyperparameters.batch = args.batch
    config.hyperparameters.device = args.device
    config.hyperparameters.workers = args.workers
    config.hyperparameters.patience = args.patience
    config.hyperparameters.save_period = args.save_period
    config.hyperparameters.close_mosaic = args.close_mosaic
    config.hyperparameters.amp = args.amp
    config.hyperparameters.seed = args.seed

    config.optimizer.optimizer = args.optimizer
    config.optimizer.lr0 = args.lr0
    config.optimizer.lrf = args.lrf
    config.optimizer.momentum = args.momentum
    config.optimizer.weight_decay = args.weight_decay

    config.augmentation.hsv_h = args.hsv_h
    config.augmentation.hsv_s = args.hsv_s
    config.augmentation.hsv_v = args.hsv_v
    config.augmentation.degrees = args.degrees
    config.augmentation.translate = args.translate
    config.augmentation.scale = args.scale
    config.augmentation.shear = args.shear
    config.augmentation.perspective = args.perspective
    config.augmentation.flipud = args.flipud
    config.augmentation.fliplr = args.fliplr
    config.augmentation.mosaic = args.mosaic
    config.augmentation.mixup = args.mixup
    config.augmentation.copy_paste = args.copy_paste

    config.experiment.project = args.project
    config.experiment.name = args.name
    config.experiment.exist_ok = args.exist_ok
    config.experiment.resume = args.resume
    config.experiment.verbose = args.verbose
    config.experiment.val = args.val
    config.experiment.save = args.save
    config.experiment.plots = args.plots
    config.experiment.comet_api_key = args.comet_api_key
    config.experiment.comet_project = args.comet_project
    config.experiment.comet_workspace = args.comet_workspace
    config.experiment.comet_experiment_name = args.comet_experiment_name or args.name
    config.experiment.comet_experiment_tag = args.comet_experiment_tag

    # Run training using new orchestrator
    orchestrator = TrainingOrchestrator(config)
    results = orchestrator.run()

    return results


if __name__ == "__main__":
    main()
