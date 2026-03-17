"""
Example: Local training with YoloFlow
Shows how to train a model on local hardware using the Python API.
"""

from pathlib import Path
from yoloflow import TrainingConfig, TrainingOrchestrator
from yoloflow.utils.logging import setup_logging


def main():
    # Setup logging
    setup_logging(level="INFO")

    # Load configuration from YAML
    config_path = Path("configs/example_local_training.yaml")
    config = TrainingConfig.from_yaml(config_path)

    # Optional: Override configuration programmatically
    config.hyperparameters.epochs = 50  # Train for fewer epochs
    config.hyperparameters.batch = 8    # Smaller batch size

    # Create orchestrator and run training
    orchestrator = TrainingOrchestrator(config)
    results = orchestrator.run()

    print(f"\nTraining completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
