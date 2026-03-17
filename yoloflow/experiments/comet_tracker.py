"""
Comet ML experiment tracking implementation.
"""

import os
from pathlib import Path
from typing import Optional, Any, Dict

from comet_ml import start, ExperimentConfig
from comet_ml.integration.pytorch import log_model

from yoloflow.core.base import BaseExperimentTracker
from yoloflow.core.config import TrainingConfig
from yoloflow.utils.logging import get_logger

logger = get_logger(__name__)


class CometMLTracker(BaseExperimentTracker):
    """
    Comet ML experiment tracking implementation.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize Comet ML tracker."""
        super().__init__(config)

    def setup(self) -> Optional[Any]:
        """
        Setup and start Comet ML experiment.

        Returns:
            Comet experiment object or None if setup fails
        """
        # Check if API key is available
        api_key = self.config.experiment.comet_api_key or os.environ.get("COMET_API_KEY")
        if not api_key:
            logger.warning("COMET_API_KEY not set. Comet ML tracking will be disabled.")
            logger.info("Set it via --comet-api-key argument or COMET_API_KEY environment variable.")
            return None

        # Validate required fields
        if not self.config.experiment.comet_project:
            logger.warning("Comet project name not specified. Tracking disabled.")
            return None

        try:
            # Start Comet ML experiment
            self.experiment = start(
                api_key=api_key,
                project_name=self.config.experiment.comet_project,
                workspace=self.config.experiment.comet_workspace,
                experiment_config=ExperimentConfig(
                    name=self.config.experiment.comet_experiment_name or self.config.experiment.name,
                    tags=[self.config.experiment.comet_experiment_tag] if self.config.experiment.comet_experiment_tag else [],
                    parse_args=False
                ),
            )
            logger.info("Comet ML experiment started successfully!")
            return self.experiment

        except Exception as e:
            logger.error(f"Failed to start Comet ML experiment: {e}")
            return None

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters to Comet ML.

        Args:
            params: Dictionary of parameters to log
        """
        if not self.experiment:
            return

        try:
            self.experiment.log_parameters(params)
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to Comet ML.

        Args:
            metrics: Dictionary of metrics to log
            step: Training step/epoch (optional)
        """
        if not self.experiment:
            return

        try:
            self.experiment.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def log_model(self, model_path: Path, model_name: str) -> None:
        """
        Log trained model to Comet ML.

        Args:
            model_path: Path to model file
            model_name: Name for the model artifact
        """
        if not self.experiment:
            return

        try:
            logger.info(f"Logging model to Comet ML: {model_path}")
            log_model(self.experiment, model=str(model_path), model_name=model_name)
            logger.info("Model logged successfully")
        except Exception as e:
            logger.error(f"Error logging model: {e}")

    def end(self) -> None:
        """End Comet ML experiment."""
        if not self.experiment:
            return

        try:
            self.experiment.end()
            logger.info(f"Comet ML experiment ended. View at: {self.get_experiment_url()}")
        except Exception as e:
            logger.error(f"Error ending experiment: {e}")

    def get_experiment_url(self) -> Optional[str]:
        """
        Get URL to view experiment in Comet ML.

        Returns:
            Experiment URL or None
        """
        if not self.experiment:
            return None

        try:
            return self.experiment.url
        except:
            return None
