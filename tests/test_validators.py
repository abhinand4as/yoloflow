"""
Tests for validation utilities.
"""

import pytest

from yoloflow.core.config import TrainingConfig
from yoloflow.utils.validators import validate_config, validate_device


def test_validate_valid_config():
    """Test validation of a valid configuration."""
    config = TrainingConfig()
    is_valid, errors = validate_config(config)

    assert is_valid
    assert len(errors) == 0


def test_validate_invalid_epochs():
    """Test validation with invalid epochs."""
    config = TrainingConfig()
    config.hyperparameters.epochs = -1

    is_valid, errors = validate_config(config)

    assert not is_valid
    assert any("Epochs must be positive" in error for error in errors)


def test_validate_invalid_batch():
    """Test validation with invalid batch size."""
    config = TrainingConfig()
    config.hyperparameters.batch = 0

    is_valid, errors = validate_config(config)

    assert not is_valid
    assert any("Batch size must be positive" in error for error in errors)


def test_validate_invalid_lr():
    """Test validation with invalid learning rate."""
    config = TrainingConfig()
    config.optimizer.lr0 = -0.01

    is_valid, errors = validate_config(config)

    assert not is_valid
    assert any("learning rate must be positive" in error for error in errors)


def test_validate_invalid_fraction():
    """Test validation with invalid dataset fraction."""
    config = TrainingConfig()
    config.dataset.fraction = 1.5

    is_valid, errors = validate_config(config)

    assert not is_valid
    assert any("fraction must be between 0 and 1" in error for error in errors)


def test_validate_device_cpu():
    """Test device validation with CPU."""
    assert validate_device("cpu")
    assert validate_device("CPU")


def test_validate_device_gpu():
    """Test device validation with GPU."""
    assert validate_device("0")
    assert validate_device("1")
    assert validate_device("0,1,2,3")


def test_validate_device_invalid():
    """Test device validation with invalid input."""
    assert not validate_device("invalid")
    assert not validate_device("-1")
