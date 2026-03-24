"""
Configuration management for YoloFlow training pipeline.
Supports YAML configs and command-line arguments.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "yolo26n.pt"
    pretrained: bool = True
    dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    path: str = "dataset.yaml"
    imgsz: int = 640
    cache: Literal["True", "False", "disk", "ram"] = "False"
    fraction: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerConfig:
    """Optimizer and learning rate configuration."""

    optimizer: Literal["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"] = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""

    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingHyperparameters:
    """Training hyperparameters."""

    epochs: int = 100
    batch: int = 16
    workers: int = 4
    device: str = "0"
    patience: int = 100
    save_period: int = -1
    close_mosaic: int = 10
    amp: bool = True
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""

    project: str = "runs/train"
    name: str = "yolo_experiment"
    exist_ok: bool = False
    resume: Optional[str] = None
    verbose: bool = True
    val: bool = True
    save: bool = True
    plots: bool = True

    # Comet ML specific
    comet_api_key: Optional[str] = None
    comet_project: Optional[str] = None
    comet_workspace: Optional[str] = None
    comet_experiment_name: Optional[str] = None
    comet_experiment_tag: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    Combines all sub-configurations for a training run.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    hyperparameters: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            dataset=DatasetConfig(**config_dict.get("dataset", {})),
            optimizer=OptimizerConfig(**config_dict.get("optimizer", {})),
            augmentation=AugmentationConfig(**config_dict.get("augmentation", {})),
            hyperparameters=TrainingHyperparameters(**config_dict.get("hyperparameters", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "augmentation": self.augmentation.to_dict(),
            "hyperparameters": self.hyperparameters.to_dict(),
            "experiment": self.experiment.to_dict(),
        }

    def to_yaml(self, output_path: Path) -> None:
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def merge_with_args(self, args: Dict[str, Any]) -> None:
        """Merge command-line arguments with config (args take precedence)."""
        for key, value in args.items():
            if value is None:
                continue

            # Navigate nested config structure
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.dataset, key):
                setattr(self.dataset, key, value)
            elif hasattr(self.optimizer, key):
                setattr(self.optimizer, key, value)
            elif hasattr(self.augmentation, key):
                setattr(self.augmentation, key, value)
            elif hasattr(self.hyperparameters, key):
                setattr(self.hyperparameters, key, value)
            elif hasattr(self.experiment, key):
                setattr(self.experiment, key, value)

    def get_train_args(self) -> Dict[str, Any]:
        """
        Get training arguments formatted for YOLO model.train() method.
        """
        train_args = {
            # Dataset
            "data": self.dataset.path,
            "imgsz": self.dataset.imgsz,
            "cache": self.dataset.cache,
            "fraction": self.dataset.fraction,

            # Training
            "epochs": self.hyperparameters.epochs,
            "batch": self.hyperparameters.batch,
            "device": self.hyperparameters.device,
            "workers": self.hyperparameters.workers,
            "patience": self.hyperparameters.patience,
            "save_period": self.hyperparameters.save_period,
            "close_mosaic": self.hyperparameters.close_mosaic,
            "amp": self.hyperparameters.amp,
            "seed": self.hyperparameters.seed,

            # Optimizer
            "optimizer": self.optimizer.optimizer,
            "lr0": self.optimizer.lr0,
            "lrf": self.optimizer.lrf,
            "momentum": self.optimizer.momentum,
            "weight_decay": self.optimizer.weight_decay,

            # Augmentation
            "hsv_h": self.augmentation.hsv_h,
            "hsv_s": self.augmentation.hsv_s,
            "hsv_v": self.augmentation.hsv_v,
            "degrees": self.augmentation.degrees,
            "translate": self.augmentation.translate,
            "scale": self.augmentation.scale,
            "shear": self.augmentation.shear,
            "perspective": self.augmentation.perspective,
            "flipud": self.augmentation.flipud,
            "fliplr": self.augmentation.fliplr,
            "mosaic": self.augmentation.mosaic,
            "mixup": self.augmentation.mixup,
            "copy_paste": self.augmentation.copy_paste,

            # Model
            "dropout": self.model.dropout,
            "pretrained": self.model.pretrained,

            # Experiment
            "project": self.experiment.project,
            "name": self.experiment.name,
            "exist_ok": self.experiment.exist_ok,
            "verbose": self.experiment.verbose,
            "val": self.experiment.val,
            "save": self.experiment.save,
            "plots": self.experiment.plots,
        }

        # Add resume if specified
        if self.experiment.resume:
            train_args["resume"] = self.experiment.resume

        return train_args
