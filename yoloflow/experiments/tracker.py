"""
Unified experiment tracker interface.
Provides a simple API that works with multiple tracking backends.
"""

from pathlib import Path
from typing import Optional, Any, Dict, Literal

from yoloflow.core.config import TrainingConfig
from yoloflow.experiments.comet_tracker import CometMLTracker


class ExperimentTracker:
    """
    Unified experiment tracking interface.
    Automatically selects appropriate tracker based on configuration.
    """

    SUPPORTED_BACKENDS = {
        "comet": CometMLTracker,
        # Future: Add more backends
        # "mlflow": MLflowTracker,
        # "wandb": WandbTracker,
        # "tensorboard": TensorboardTracker,
    }

    def __init__(
        self,
        config: TrainingConfig,
        backend: Literal["comet", "mlflow", "wandb", "tensorboard"] = "comet",
    ):
        """
        Initialize experiment tracker.

        Args:
            config: Training configuration
            backend: Tracking backend to use
        """
        self.config = config
        self.backend = backend

        # Initialize the appropriate tracker
        tracker_class = self.SUPPORTED_BACKENDS.get(backend)
        if not tracker_class:
            raise ValueError(f"Unsupported backend: {backend}. Supported: {list(self.SUPPORTED_BACKENDS.keys())}")

        self.tracker = tracker_class(config)
        self.experiment = None

    def setup(self) -> Optional[Any]:
        """Setup and start experiment tracking."""
        return self.tracker.setup()

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.tracker.log_parameters(params)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics during training."""
        self.tracker.log_metrics(metrics, step)

    def log_model(self, model_path: Path, model_name: str) -> None:
        """Log trained model artifact."""
        self.tracker.log_model(model_path, model_name)

    def end(self) -> None:
        """End experiment tracking."""
        self.tracker.end()

    def get_experiment_url(self) -> Optional[str]:
        """Get URL to view experiment."""
        return self.tracker.get_experiment_url()

    @property
    def is_active(self) -> bool:
        """Check if tracker is active."""
        return self.tracker.experiment is not None
