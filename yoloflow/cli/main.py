"""
Main CLI entry point for YoloFlow.
Provides command-line interface for training and managing experiments.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from yoloflow import __version__
from yoloflow.core.config import TrainingConfig
from yoloflow.training.orchestrator import TrainingOrchestrator
from yoloflow.utils.logging import setup_logging, get_logger
from yoloflow.utils.validators import validate_config

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="YoloFlow - Production-ready YOLO training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"YoloFlow {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a YOLO model")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    train_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration files")
    config_subparsers = config_parser.add_subparsers(dest="config_action")

    # Config validate
    validate_parser = config_subparsers.add_parser("validate", help="Validate a config file")
    validate_parser.add_argument(
        "config_file",
        type=str,
        help="Path to config file to validate",
    )

    # Config template
    template_parser = config_subparsers.add_parser("template", help="Generate a config template")
    template_parser.add_argument(
        "output_file",
        type=str,
        help="Output path for template config",
    )

    return parser


def handle_train(args) -> int:
    """
    Handle train command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    try:
        # Load configuration
        logger.info(f"Loading configuration from: {config_path}")
        config = TrainingConfig.from_yaml(config_path)

        # Validate configuration
        is_valid, errors = validate_config(config)
        if not is_valid:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return 1

        # Create and run orchestrator
        orchestrator = TrainingOrchestrator(config)
        orchestrator.run()

        logger.info("Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


def handle_config_validate(args) -> int:
    """
    Handle config validate command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    config_path = Path(args.config_file)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    try:
        config = TrainingConfig.from_yaml(config_path)
        is_valid, errors = validate_config(config)

        if is_valid:
            logger.info(f"Configuration is valid: {config_path}")
            return 0
        else:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return 1

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1


def handle_config_template(args) -> int:
    """
    Handle config template command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    output_path = Path(args.output_file)

    try:
        # Create default configuration
        config = TrainingConfig()
        config.to_yaml(output_path)

        logger.info(f"Configuration template created: {output_path}")
        return 0

    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        return 1


def main(argv: Optional[list] = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging
    log_level = getattr(args, 'log_level', 'INFO')
    setup_logging(level=log_level)

    # Handle commands
    if args.command == "train":
        return handle_train(args)
    elif args.command == "config":
        if args.config_action == "validate":
            return handle_config_validate(args)
        elif args.config_action == "template":
            return handle_config_template(args)
        else:
            parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
