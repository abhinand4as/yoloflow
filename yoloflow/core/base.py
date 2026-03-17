"""
Base classes for training infrastructure.
Enables future extensibility for cloud training, distributed training, etc.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, Dict
from yoloflow.core.config import TrainingConfig


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    Allows easy extension for different training backends (local, Azure, GCP, etc.)
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration object
        """
        self.config = config
        self.model = None
        self.results = None

    @abstractmethod
    def setup(self) -> None:
        """Setup training environment (load model, validate paths, etc.)"""
        pass

    @abstractmethod
    def train(self) -> Any:
        """Execute training process."""
        pass

    @abstractmethod
    def validate_environment(self) -> bool:
        """Validate that the training environment is properly configured."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources after training."""
        pass

    def get_model_path(self) -> Path:
        """Resolve model path."""
        model_path = Path(self.config.model.name)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        return model_path

    def get_data_path(self) -> Path:
        """Resolve dataset path."""
        data_path = Path(self.config.dataset.path)
        if not data_path.is_absolute():
            data_path = Path.cwd() / data_path
        return data_path


class BaseExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking.
    Allows support for multiple tracking backends (Comet ML, MLflow, Weights & Biases, etc.)
    """

    def __init__(self, config: TrainingConfig):
        """Initialize experiment tracker with configuration."""
        self.config = config
        self.experiment = None

    @abstractmethod
    def setup(self) -> Optional[Any]:
        """Setup and start experiment tracking."""
        pass

    @abstractmethod
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics during training."""
        pass

    @abstractmethod
    def log_model(self, model_path: Path, model_name: str) -> None:
        """Log trained model artifact."""
        pass

    @abstractmethod
    def end(self) -> None:
        """End experiment tracking."""
        pass

    @abstractmethod
    def get_experiment_url(self) -> Optional[str]:
        """Get URL to view experiment."""
        pass
