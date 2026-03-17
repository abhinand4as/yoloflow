"""
Local training implementation for YOLO models.
Runs training on local machine (GPU or CPU).
"""

from pathlib import Path
from typing import Optional, Any
from ultralytics import YOLO

from yoloflow.core.base import BaseTrainer
from yoloflow.core.config import TrainingConfig
from yoloflow.utils.logging import get_logger

logger = get_logger(__name__)


class LocalTrainer(BaseTrainer):
    """
    Local YOLO trainer implementation.
    Trains models on local hardware (GPU/CPU).
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize local trainer.

        Args:
            config: Training configuration
        """
        super().__init__(config)
        self.model_path = None
        self.data_path = None

    def validate_environment(self) -> bool:
        """
        Validate that the local environment is ready for training.

        Returns:
            bool: True if environment is valid
        """
        try:
            # Check model path
            self.model_path = self.get_model_path()
            if not self.model_path.exists():
                logger.warning(f"Model file not found at {self.model_path}")
                logger.info(f"Will attempt to download {self.config.model.name} from Ultralytics hub")

            # Check dataset path
            self.data_path = self.get_data_path()
            if not self.data_path.exists():
                logger.warning(f"Dataset YAML not found at {self.data_path}")
                logger.info(f"Will attempt to download {self.config.dataset.path} from Ultralytics hub")
                self.data_path = self.config.dataset.path  # Use as-is for Ultralytics to handle

            return True

        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            return False

    def setup(self) -> None:
        """Setup local training environment."""
        logger.info("=" * 80)
        logger.info("Local YOLO Training Setup")
        logger.info("=" * 80)

        # Validate environment
        if not self.validate_environment():
            raise RuntimeError("Environment validation failed")

        # Load model
        logger.info(f"Loading model: {self.config.model.name}")
        self.model = YOLO(str(self.model_path))

        logger.info(f"Model: {self.config.model.name}")
        logger.info(f"Dataset: {self.data_path}")
        logger.info(f"Epochs: {self.config.hyperparameters.epochs}")
        logger.info(f"Batch Size: {self.config.hyperparameters.batch}")
        logger.info(f"Image Size: {self.config.dataset.imgsz}")
        logger.info(f"Device: {self.config.hyperparameters.device}")
        logger.info(f"Optimizer: {self.config.optimizer.optimizer}")
        logger.info(f"Learning Rate: {self.config.optimizer.lr0} -> {self.config.optimizer.lr0 * self.config.optimizer.lrf}")
        logger.info("=" * 80)

    def train(self) -> Any:
        """
        Execute local training.

        Returns:
            Training results object from Ultralytics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call setup() first.")

        # Get training arguments
        train_args = self.config.get_train_args()

        # Start training
        save_path = Path(self.config.experiment.project) / self.config.experiment.name
        logger.info(f"Starting training...")
        logger.info(f"Results will be saved to: {save_path}")
        logger.info("=" * 80)

        self.results = self.model.train(**train_args)

        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info(f"Results saved to: {save_path}")
        logger.info("=" * 80)

        return self.results

    def cleanup(self) -> None:
        """Cleanup local training resources."""
        # For local training, typically no cleanup needed
        # But this can be extended for temporary file cleanup, etc.
        logger.info("Local training cleanup complete")

    def get_best_model_path(self) -> Optional[Path]:
        """
        Get path to the best trained model.

        Returns:
            Path to best.pt file if it exists
        """
        save_path = Path(self.config.experiment.project) / self.config.experiment.name
        best_model = save_path / "weights" / "best.pt"

        if best_model.exists():
            return best_model
        return None

    def get_last_model_path(self) -> Optional[Path]:
        """
        Get path to the last checkpoint.

        Returns:
            Path to last.pt file if it exists
        """
        save_path = Path(self.config.experiment.project) / self.config.experiment.name
        last_model = save_path / "weights" / "last.pt"

        if last_model.exists():
            return last_model
        return None
