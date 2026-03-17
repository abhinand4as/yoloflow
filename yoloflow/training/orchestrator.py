"""
Training orchestrator - coordinates training, experiment tracking, and callbacks.
Central point for managing the entire training workflow.
"""

from pathlib import Path
from typing import Optional, Any, Type

from yoloflow.core.base import BaseTrainer, BaseExperimentTracker
from yoloflow.core.config import TrainingConfig
from yoloflow.training.local import LocalTrainer
from yoloflow.experiments.comet_tracker import CometMLTracker
from yoloflow.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates the complete training workflow.
    Coordinates trainer, experiment tracking, and callbacks.
    """

    def __init__(
        self,
        config: TrainingConfig,
        trainer_class: Optional[Type[BaseTrainer]] = None,
        tracker_class: Optional[Type[BaseExperimentTracker]] = None,
    ):
        """
        Initialize training orchestrator.

        Args:
            config: Training configuration
            trainer_class: Trainer implementation class (defaults to LocalTrainer)
            tracker_class: Experiment tracker class (defaults to CometMLTracker)
        """
        self.config = config

        # Default to local trainer if not specified
        self.trainer_class = trainer_class or LocalTrainer
        self.tracker_class = tracker_class or CometMLTracker

        # Initialize components
        self.trainer: Optional[BaseTrainer] = None
        self.tracker: Optional[BaseExperimentTracker] = None
        self.results: Optional[Any] = None

    def setup(self) -> None:
        """Setup all training components."""
        logger.info("Initializing training orchestrator...")

        # Initialize trainer
        self.trainer = self.trainer_class(self.config)
        self.trainer.setup()

        # Initialize experiment tracker
        self.tracker = self.tracker_class(self.config)
        experiment = self.tracker.setup()

        if experiment:
            logger.info(f"Experiment tracking enabled")
            if hasattr(self.tracker, 'get_experiment_url'):
                url = self.tracker.get_experiment_url()
                if url:
                    logger.info(f"  Experiment URL: {url}")

            # Log hyperparameters
            self._log_hyperparameters()
        else:
            logger.info("Experiment tracking disabled")

    def _log_hyperparameters(self) -> None:
        """Log all hyperparameters to experiment tracker."""
        if not self.tracker or not self.tracker.experiment:
            return

        hyper_params = {
            # Model
            "model": self.config.model.name,
            "pretrained": self.config.model.pretrained,
            "dropout": self.config.model.dropout,

            # Training
            "epochs": self.config.hyperparameters.epochs,
            "batch_size": self.config.hyperparameters.batch,
            "imgsz": self.config.dataset.imgsz,
            "device": self.config.hyperparameters.device,
            "workers": self.config.hyperparameters.workers,
            "patience": self.config.hyperparameters.patience,
            "close_mosaic": self.config.hyperparameters.close_mosaic,
            "amp": self.config.hyperparameters.amp,
            "fraction": self.config.dataset.fraction,
            "seed": self.config.hyperparameters.seed,

            # Optimizer
            "optimizer": self.config.optimizer.optimizer,
            "learning_rate": self.config.optimizer.lr0,
            "final_lr": self.config.optimizer.lr0 * self.config.optimizer.lrf,
            "momentum": self.config.optimizer.momentum,
            "weight_decay": self.config.optimizer.weight_decay,

            # Augmentation
            "hsv_h": self.config.augmentation.hsv_h,
            "hsv_s": self.config.augmentation.hsv_s,
            "hsv_v": self.config.augmentation.hsv_v,
            "degrees": self.config.augmentation.degrees,
            "translate": self.config.augmentation.translate,
            "scale": self.config.augmentation.scale,
            "shear": self.config.augmentation.shear,
            "perspective": self.config.augmentation.perspective,
            "flipud": self.config.augmentation.flipud,
            "fliplr": self.config.augmentation.fliplr,
            "mosaic": self.config.augmentation.mosaic,
            "mixup": self.config.augmentation.mixup,
            "copy_paste": self.config.augmentation.copy_paste,
        }

        self.tracker.log_parameters(hyper_params)
        logger.info("Hyperparameters logged to experiment tracker")

    def train(self) -> Any:
        """
        Execute the complete training workflow.

        Returns:
            Training results
        """
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        try:
            # Execute training
            self.results = self.trainer.train()

            # Log final results
            self._log_final_results()

            return self.results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        finally:
            # Cleanup
            self.cleanup()

    def _log_final_results(self) -> None:
        """Log final training results to experiment tracker."""
        if not self.tracker or not self.tracker.experiment:
            return

        try:
            # Log model artifact
            if hasattr(self.trainer, 'get_best_model_path'):
                best_model_path = self.trainer.get_best_model_path()
                if best_model_path and best_model_path.exists():
                    model_name = f"{self.config.experiment.name}_best"
                    self.tracker.log_model(best_model_path, model_name)
                    logger.info("Model logged to experiment tracker")

            # Log final metrics if available
            if self.results and hasattr(self.results, 'results_dict'):
                self.tracker.log_metrics(self.results.results_dict)

        except Exception as e:
            logger.error(f"Error logging final results: {e}")

    def cleanup(self) -> None:
        """Cleanup all resources."""
        if self.trainer:
            self.trainer.cleanup()

        if self.tracker:
            self.tracker.end()

    def run(self) -> Any:
        """
        Complete training workflow: setup -> train -> cleanup.
        Convenience method for simple training runs.

        Returns:
            Training results
        """
        self.setup()
        return self.train()

    @classmethod
    def from_config_file(cls, config_path: Path, **kwargs) -> "TrainingOrchestrator":
        """
        Create orchestrator from YAML config file.

        Args:
            config_path: Path to config YAML file
            **kwargs: Additional trainer/tracker class overrides

        Returns:
            Initialized TrainingOrchestrator
        """
        config = TrainingConfig.from_yaml(config_path)
        return cls(config, **kwargs)
