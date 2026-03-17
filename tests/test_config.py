"""
Tests for configuration management.
"""

import tempfile
from pathlib import Path
import pytest

from yoloflow.core.config import (
    TrainingConfig,
    ModelConfig,
    DatasetConfig,
    OptimizerConfig,
    AugmentationConfig,
    TrainingHyperparameters,
    ExperimentConfig,
)


def test_default_config_creation():
    """Test creating config with defaults."""
    config = TrainingConfig()

    assert config.model.name == "yolo26n.pt"
    assert config.dataset.imgsz == 640
    assert config.hyperparameters.epochs == 100
    assert config.optimizer.lr0 == 0.01


def test_config_to_dict():
    """Test converting config to dictionary."""
    config = TrainingConfig()
    config_dict = config.to_dict()

    assert "model" in config_dict
    assert "dataset" in config_dict
    assert "optimizer" in config_dict
    assert config_dict["model"]["name"] == "yolo26n.pt"


def test_config_from_dict():
    """Test creating config from dictionary."""
    config_dict = {
        "model": {"name": "yolo26s.pt", "pretrained": False},
        "dataset": {"path": "custom_dataset.yaml", "imgsz": 1024},
        "hyperparameters": {"epochs": 200, "batch": 32},
    }

    config = TrainingConfig.from_dict(config_dict)

    assert config.model.name == "yolo26s.pt"
    assert config.model.pretrained is False
    assert config.dataset.imgsz == 1024
    assert config.hyperparameters.epochs == 200


def test_config_yaml_roundtrip():
    """Test saving and loading config from YAML."""
    config = TrainingConfig()
    config.model.name = "test_model.pt"
    config.hyperparameters.epochs = 50

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "config.yaml"

        # Save to YAML
        config.to_yaml(yaml_path)
        assert yaml_path.exists()

        # Load from YAML
        loaded_config = TrainingConfig.from_yaml(yaml_path)

        assert loaded_config.model.name == "test_model.pt"
        assert loaded_config.hyperparameters.epochs == 50


def test_get_train_args():
    """Test getting training arguments for YOLO."""
    config = TrainingConfig()
    train_args = config.get_train_args()

    assert "data" in train_args
    assert "epochs" in train_args
    assert "batch" in train_args
    assert train_args["epochs"] == 100
    assert train_args["batch"] == 16


def test_merge_with_args():
    """Test merging command-line arguments with config."""
    config = TrainingConfig()
    args = {
        "epochs": 200,
        "batch": 32,
        "lr0": 0.001,
        "name": "custom_experiment",
    }

    config.merge_with_args(args)

    assert config.hyperparameters.epochs == 200
    assert config.hyperparameters.batch == 32
    assert config.optimizer.lr0 == 0.001
    assert config.experiment.name == "custom_experiment"


def test_model_config():
    """Test ModelConfig."""
    model_config = ModelConfig(name="yolo26l.pt", pretrained=False, dropout=0.1)

    assert model_config.name == "yolo26l.pt"
    assert model_config.pretrained is False
    assert model_config.dropout == 0.1


def test_dataset_config():
    """Test DatasetConfig."""
    dataset_config = DatasetConfig(
        path="test_dataset.yaml",
        imgsz=1024,
        cache="ram",
        fraction=0.5,
    )

    assert dataset_config.path == "test_dataset.yaml"
    assert dataset_config.imgsz == 1024
    assert dataset_config.cache == "ram"
    assert dataset_config.fraction == 0.5


def test_optimizer_config():
    """Test OptimizerConfig."""
    optimizer_config = OptimizerConfig(
        optimizer="Adam",
        lr0=0.001,
        lrf=0.1,
        momentum=0.9,
        weight_decay=0.001,
    )

    assert optimizer_config.optimizer == "Adam"
    assert optimizer_config.lr0 == 0.001
    assert optimizer_config.lrf == 0.1
