"""
Cloud training implementations.
Future support for Azure ML, GCP Vertex AI, AWS SageMaker, etc.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Optional, Any

from yoloflow.core.base import BaseTrainer
from yoloflow.core.config import TrainingConfig
from yoloflow.utils.logging import get_logger

logger = get_logger(__name__)


class CloudTrainer(BaseTrainer):
    """
    Base class for cloud training implementations.
    Extend this for specific cloud providers.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.job_id = None

    @abstractmethod
    def submit_job(self) -> str:
        """
        Submit training job to cloud platform.

        Returns:
            Job ID
        """
        pass

    @abstractmethod
    def monitor_job(self, job_id: str) -> dict:
        """
        Monitor training job status.

        Args:
            job_id: Job identifier

        Returns:
            Job status dictionary
        """
        pass

    @abstractmethod
    def download_results(self, job_id: str, output_dir: Path) -> None:
        """
        Download training results from cloud.

        Args:
            job_id: Job identifier
            output_dir: Local directory to save results
        """
        pass


# Future implementations:
#
# class AzureMLTrainer(CloudTrainer):
#     """Azure ML training implementation."""
#     pass
#
# class GCPVertexTrainer(CloudTrainer):
#     """GCP Vertex AI training implementation."""
#     pass
#
# class AWSSageMakerTrainer(CloudTrainer):
#     """AWS SageMaker training implementation."""
#     pass
