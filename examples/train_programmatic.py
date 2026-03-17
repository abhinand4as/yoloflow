"""
Example: Programmatic configuration
Shows how to create configuration entirely in Python without YAML files.
"""

from yoloflow.core.config import (
    TrainingConfig,
    ModelConfig,
    DatasetConfig,
    OptimizerConfig,
    AugmentationConfig,
    TrainingHyperparameters,
    ExperimentConfig,
)
from yoloflow.training.orchestrator import TrainingOrchestrator
from yoloflow.utils.logging import setup_logging


def main():
    # Setup logging
    setup_logging(level="INFO")

    # Create configuration programmatically
    config = TrainingConfig(
        model=ModelConfig(
            name="yolo26n.pt",
            pretrained=True,
            dropout=0.0,
        ),
        dataset=DatasetConfig(
            path="yolo_cac_dataset_quadrants/dataset.yaml",
            imgsz=640,
            cache="False",
            fraction=1.0,
        ),
        hyperparameters=TrainingHyperparameters(
            epochs=100,
            batch=16,
            workers=4,
            device="0",
            patience=100,
            amp=True,
            seed=42,
        ),
        optimizer=OptimizerConfig(
            optimizer="auto",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
        ),
        augmentation=AugmentationConfig(
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
        ),
        experiment=ExperimentConfig(
            project="runs/train",
            name="programmatic_experiment",
            exist_ok=True,
            # Comet ML configuration
            comet_project="yolo_experiments",
            comet_experiment_name="test_run",
            comet_experiment_tag="local",
        ),
    )

    # Optional: Save configuration for future use
    config.to_yaml("configs/saved_config.yaml")
    print("Configuration saved to: configs/saved_config.yaml")

    # Create orchestrator and run training
    orchestrator = TrainingOrchestrator(config)
    results = orchestrator.run()

    print(f"\nTraining completed!")


if __name__ == "__main__":
    main()
