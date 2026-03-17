# YoloFlow Quick Start Guide

## Installation

### From Source (Development)

```bash
# Clone the repository
cd /home/abhinand/myws/ai/yolo_ws/yoloflow

# Install in development mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check version
yoloflow --version

# Get help
yoloflow --help
```

## Usage Modes

YoloFlow supports three ways to run training:

### 1. CLI with YAML Config (Recommended)

```bash
# Generate a config template
yoloflow config template my_config.yaml

# Edit my_config.yaml with your settings

# Validate configuration
yoloflow config validate my_config.yaml

# Start training
yoloflow train --config my_config.yaml
```

### 2. Python API

```python
from yoloflow import TrainingConfig, TrainingOrchestrator

# Load config
config = TrainingConfig.from_yaml("my_config.yaml")

# Run training
orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()
```

### 3. Legacy CLI (Backward Compatible)

```bash
# Use original train.py interface
python train_legacy.py \
    --model yolo26n.pt \
    --data dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0 \
    --comet-project my_project
```

## Example Workflows

### Basic Local Training

1. **Prepare your dataset** in YOLO format
2. **Create config file**:

```yaml
# config.yaml
model:
  name: yolo26n.pt
  pretrained: true

dataset:
  path: path/to/dataset.yaml
  imgsz: 640

hyperparameters:
  epochs: 100
  batch: 16
  device: "0"

experiment:
  project: runs/train
  name: my_first_training
```

3. **Start training**:

```bash
yoloflow train --config config.yaml
```

### Training with Comet ML Tracking

1. **Set Comet API key**:

```bash
export COMET_API_KEY=your_api_key_here
```

2. **Update config**:

```yaml
experiment:
  project: runs/train
  name: comet_tracked_training
  comet_project: my-yolo-project
  comet_workspace: my-workspace
  comet_experiment_name: experiment_001
  comet_experiment_tag: baseline
```

3. **Run training** (same command as before)

### Programmatic Training

```python
from yoloflow.core.config import (
    TrainingConfig,
    ModelConfig,
    DatasetConfig,
    TrainingHyperparameters,
    ExperimentConfig,
)
from yoloflow.training.orchestrator import TrainingOrchestrator

# Create config
config = TrainingConfig(
    model=ModelConfig(name="yolo26n.pt"),
    dataset=DatasetConfig(
        path="dataset.yaml",
        imgsz=640,
    ),
    hyperparameters=TrainingHyperparameters(
        epochs=100,
        batch=16,
        device="0",
    ),
    experiment=ExperimentConfig(
        project="runs/train",
        name="programmatic_training",
    ),
)

# Train
orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()

print(f"Training complete! Best model saved.")
```

## Configuration Examples

### Minimal Config

```yaml
model:
  name: yolo26n.pt

dataset:
  path: dataset.yaml

hyperparameters:
  epochs: 100
  batch: 16
  device: "0"

experiment:
  name: my_experiment
```

### Full Config with All Options

See [configs/example_local_training.yaml](../configs/example_local_training.yaml)

## Common Tasks

### Resume Training

```yaml
experiment:
  resume: runs/train/my_experiment/weights/last.pt
```

Or via CLI:

```bash
yoloflow train --config config.yaml
# (Make sure resume is set in the config)
```

### Multi-GPU Training

```yaml
hyperparameters:
  device: "0,1,2,3"  # Use 4 GPUs
  batch: 64          # Increase batch size
```

### CPU Training

```yaml
hyperparameters:
  device: "cpu"
  batch: 8  # Use smaller batch size
```

### Custom Augmentation

```yaml
augmentation:
  fliplr: 0.5      # 50% horizontal flip
  mosaic: 1.0      # Always use mosaic
  mixup: 0.1       # 10% mixup
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
```

## Troubleshooting

### Model Not Found

```
Warning: Model file not found at yolo26n.pt
Will attempt to download from Ultralytics hub
```

**Solution**: YoloFlow will automatically download the model. Ensure you have internet connection.

### Dataset Not Found

```
Warning: Dataset YAML not found at dataset.yaml
```

**Solution**: Check the path in your config file. Use absolute paths or paths relative to where you run the command.

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or image size:

```yaml
hyperparameters:
  batch: 8  # Reduce from 16

dataset:
  imgsz: 512  # Reduce from 640
```

### Comet ML Not Working

```
Warning: COMET_API_KEY not set. Comet ML tracking will be disabled.
```

**Solution**: Set your API key:

```bash
export COMET_API_KEY=your_key_here
```

Or in config:

```yaml
experiment:
  comet_api_key: your_key_here
```

## Next Steps

- Read [Architecture Documentation](ARCHITECTURE.md) to understand the design
- Check [examples/](../examples/) for more use cases
- Explore [configs/](../configs/) for configuration templates
- Run tests: `pytest tests/`

## Getting Help

- Check the [README.md](../README.md) for general information
- Look at example scripts in [examples/](../examples/)
- Review the [Architecture](ARCHITECTURE.md) documentation
- Check configuration templates in [configs/](../configs/)
