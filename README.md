# YoloFlow

Production-ready YOLO training pipeline for custom datasets with support for local and cloud training.

## Features

- **Modular Architecture**: Clean separation of concerns with extensible components
- **Configuration Management**: YAML-based configuration with command-line overrides
- **Experiment Tracking**: Built-in support for Comet ML (extensible to MLflow, Weights & Biases, etc.)
- **Local Training**: Full support for GPU/CPU training on local machines
- **Cloud-Ready**: Architecture designed for future cloud platform integration (Azure ML, GCP Vertex AI, AWS SageMaker)
- **CLI Interface**: User-friendly command-line tools for training and configuration management
- **Type-Safe**: Full type hints for better IDE support and code quality

## Installation

### Basic Installation

```bash
pip install yoloflow
```

### Development Installation

```bash
git clone https://github.com/yourusername/yoloflow.git
cd yoloflow
pip install -e ".[dev]"
```

### With Cloud Support (Future)

```bash
pip install "yoloflow[cloud]"
```

## Quick Start

### 1. Generate a Configuration Template

```bash
yoloflow config template my_config.yaml
```

### 2. Edit Configuration

Edit `my_config.yaml` to match your dataset and training requirements:

```yaml
model:
  name: yolo26n.pt
  pretrained: true
  dropout: 0.0

dataset:
  path: path/to/your/dataset.yaml
  imgsz: 640
  cache: "False"
  fraction: 1.0

hyperparameters:
  epochs: 100
  batch: 16
  device: "0"
  workers: 4

optimizer:
  optimizer: auto
  lr0: 0.01
  lrf: 0.01

experiment:
  project: runs/train
  name: my_experiment
  comet_api_key: your_comet_api_key
  comet_project: your_project_name
```

### 3. Validate Configuration

```bash
yoloflow config validate my_config.yaml
```

### 4. Start Training

```bash
yoloflow train --config my_config.yaml
```

## Usage Examples

### Using Python API

```python
from yoloflow import TrainingConfig, TrainingOrchestrator

# Load configuration
config = TrainingConfig.from_yaml("my_config.yaml")

# Create and run orchestrator
orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()
```

### Custom Training Backend

```python
from yoloflow import TrainingConfig
from yoloflow.training.orchestrator import TrainingOrchestrator
from yoloflow.training.local import LocalTrainer

config = TrainingConfig.from_yaml("my_config.yaml")

# Use specific trainer
orchestrator = TrainingOrchestrator(
    config=config,
    trainer_class=LocalTrainer,
)
results = orchestrator.run()
```

### Programmatic Configuration

```python
from yoloflow.core.config import (
    TrainingConfig,
    ModelConfig,
    DatasetConfig,
    OptimizerConfig,
    TrainingHyperparameters,
    ExperimentConfig,
)

config = TrainingConfig(
    model=ModelConfig(name="yolo26n.pt", pretrained=True),
    dataset=DatasetConfig(path="dataset.yaml", imgsz=640),
    hyperparameters=TrainingHyperparameters(epochs=100, batch=16),
    optimizer=OptimizerConfig(optimizer="auto", lr0=0.01),
    experiment=ExperimentConfig(
        project="runs/train",
        name="my_experiment",
    ),
)

from yoloflow.training.orchestrator import TrainingOrchestrator
orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()
```

## Project Structure

```
yoloflow/
├── yoloflow/
│   ├── __init__.py
│   ├── core/                    # Core configuration and base classes
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration dataclasses
│   │   └── base.py              # Abstract base classes
│   ├── training/                # Training implementations
│   │   ├── __init__.py
│   │   ├── local.py             # Local training
│   │   ├── cloud.py             # Cloud training (future)
│   │   └── orchestrator.py      # Training orchestration
│   ├── experiments/             # Experiment tracking
│   │   ├── __init__.py
│   │   ├── tracker.py           # Unified tracker interface
│   │   └── comet_tracker.py     # Comet ML implementation
│   ├── deployment/              # Model deployment (future)
│   │   └── __init__.py
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── logging.py           # Logging setup
│   │   └── validators.py        # Configuration validators
│   └── cli/                     # Command-line interface
│       ├── __init__.py
│       └── main.py              # CLI entry point
├── tests/                       # Unit tests
├── examples/                    # Example scripts and configs
├── docs/                        # Documentation
├── configs/                     # Example configurations
├── pyproject.toml              # Package configuration
├── setup.py                    # Setup script
└── README.md                   # This file
```

## Architecture

YoloFlow is designed with modularity and extensibility in mind:

### Core Components

1. **Configuration System** ([yoloflow/core/config.py](yoloflow/core/config.py))
   - Type-safe configuration dataclasses
   - YAML serialization/deserialization
   - Command-line argument merging

2. **Base Classes** ([yoloflow/core/base.py](yoloflow/core/base.py))
   - `BaseTrainer`: Abstract trainer interface
   - `BaseExperimentTracker`: Abstract tracker interface
   - Easy to extend for new implementations

3. **Training Orchestrator** ([yoloflow/training/orchestrator.py](yoloflow/training/orchestrator.py))
   - Coordinates training workflow
   - Manages experiment tracking
   - Handles cleanup and error recovery

### Extensibility

#### Adding Cloud Training Support

```python
from yoloflow.core.base import BaseTrainer

class AzureMLTrainer(BaseTrainer):
    def setup(self):
        # Setup Azure ML environment
        pass

    def train(self):
        # Submit training job to Azure ML
        pass

    def validate_environment(self):
        # Validate Azure credentials
        pass

    def cleanup(self):
        # Cleanup Azure resources
        pass
```

#### Adding New Experiment Tracker

```python
from yoloflow.core.base import BaseExperimentTracker

class MLflowTracker(BaseExperimentTracker):
    def setup(self):
        # Initialize MLflow
        pass

    def log_parameters(self, params):
        # Log to MLflow
        pass

    # Implement other abstract methods
```

## Future Roadmap

- [ ] Cloud training support
  - [ ] Azure ML
  - [ ] GCP Vertex AI
  - [ ] AWS SageMaker
- [ ] Additional experiment trackers
  - [ ] MLflow
  - [ ] Weights & Biases
  - [ ] TensorBoard
- [ ] Deployment tools
  - [ ] REST API server
  - [ ] ONNX export
  - [ ] TensorRT optimization
  - [ ] Docker containerization
- [ ] Advanced features
  - [ ] Distributed training
  - [ ] Hyperparameter tuning
  - [ ] AutoML capabilities
  - [ ] Model versioning

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Citation

If you use YoloFlow in your research, please cite:

```bibtex
@software{yoloflow,
  title = {YoloFlow: Production-ready YOLO Training Pipeline},
  author = {Your Team},
  year = {2024},
  url = {https://github.com/yourusername/yoloflow}
}
```
