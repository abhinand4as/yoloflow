"""
Validation utilities for configurations and paths.
"""

from pathlib import Path
from typing import List, Tuple

from yoloflow.core.config import TrainingConfig
from yoloflow.utils.logging import get_logger

logger = get_logger(__name__)


def validate_config(config: TrainingConfig) -> Tuple[bool, List[str]]:
    """
    Validate training configuration.

    Args:
        config: Training configuration to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Validate model
    if not config.model.name:
        errors.append("Model name is required")

    # Validate dataset
    if not config.dataset.path:
        errors.append("Dataset path is required")

    if config.dataset.imgsz <= 0:
        errors.append("Image size must be positive")

    if not 0 <= config.dataset.fraction <= 1:
        errors.append("Dataset fraction must be between 0 and 1")

    # Validate training hyperparameters
    if config.hyperparameters.epochs <= 0:
        errors.append("Epochs must be positive")

    if config.hyperparameters.batch <= 0:
        errors.append("Batch size must be positive")

    if config.hyperparameters.workers < 0:
        errors.append("Workers must be non-negative")

    # Validate optimizer
    if config.optimizer.lr0 <= 0:
        errors.append("Initial learning rate must be positive")

    if not 0 <= config.optimizer.lrf <= 1:
        errors.append("Final learning rate factor must be between 0 and 1")

    if not 0 <= config.optimizer.momentum <= 1:
        errors.append("Momentum must be between 0 and 1")

    if config.optimizer.weight_decay < 0:
        errors.append("Weight decay must be non-negative")

    # Validate augmentation
    if not 0 <= config.augmentation.fliplr <= 1:
        errors.append("Flip left-right probability must be between 0 and 1")

    if not 0 <= config.augmentation.flipud <= 1:
        errors.append("Flip up-down probability must be between 0 and 1")

    if not 0 <= config.augmentation.mosaic <= 1:
        errors.append("Mosaic probability must be between 0 and 1")

    # Validate experiment
    if not config.experiment.name:
        errors.append("Experiment name is required")

    is_valid = len(errors) == 0

    if not is_valid:
        for error in errors:
            logger.error(f"Validation error: {error}")

    return is_valid, errors


def validate_paths(config: TrainingConfig, check_existence: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate file paths in configuration.

    Args:
        config: Training configuration
        check_existence: If True, check if paths actually exist

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if check_existence:
        # Check model path
        model_path = Path(config.model.name)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path

        if not model_path.exists() and not config.model.name.endswith('.pt'):
            errors.append(f"Model file not found: {model_path}")

        # Check dataset path
        data_path = Path(config.dataset.path)
        if not data_path.is_absolute():
            data_path = Path.cwd() / data_path

        if not data_path.exists() and not data_path.suffix in ['.yaml', '.yml']:
            errors.append(f"Dataset file not found: {data_path}")

        # Check resume checkpoint if specified
        if config.experiment.resume:
            resume_path = Path(config.experiment.resume)
            if not resume_path.exists():
                errors.append(f"Resume checkpoint not found: {resume_path}")

    is_valid = len(errors) == 0

    if not is_valid:
        for error in errors:
            logger.error(f"Path validation error: {error}")

    return is_valid, errors


def validate_device(device: str) -> bool:
    """
    Validate device specification.

    Args:
        device: Device string (e.g., '0', 'cpu', '0,1,2,3')

    Returns:
        True if valid
    """
    if device.lower() == 'cpu':
        return True

    # Check if it's a valid GPU specification
    try:
        gpu_ids = [int(x.strip()) for x in device.split(',')]
        return all(x >= 0 for x in gpu_ids)
    except ValueError:
        return False
