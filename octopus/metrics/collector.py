"""
PipelineMetricsCollector - Main class for collecting and managing pipeline metrics.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .enums import PipelineStatus, MetricType
from .metric_entry import MetricEntry
from .epoch_summary import EpochSummary
from .pipeline_run import PipelineRun


class PipelineMetricsCollector:
    def __init__(
        self,
        pipeline_name: str,
        pipeline_type: str,
        run_id: Optional[str] = None,
        output_dir: str = "mlops_data/metrics",
        config: Optional[Dict[str, Any]] = None,
        auto_save: bool = True
    ):
        """
        Initialize the metrics collector.

        Args:
            pipeline_name: Name of the pipeline (e.g., "MNIST_Training", "CNN14_AudioTagging")
            pipeline_type: Type of pipeline (e.g., "supervised_classification", "gan_training")
            run_id: Unique run identifier (auto-generated if not provided)
            output_dir: Directory to save metrics
            config: Pipeline configuration dictionary
            auto_save: Automatically save metrics after each epoch
        """
        self.pipeline_name = pipeline_name
        self.pipeline_type = pipeline_type
        self.run_id = run_id or self._generate_run_id()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        # Initialize pipeline run with sanitized config
        self.pipeline_run = PipelineRun(
            run_id=self.run_id,
            pipeline_name=pipeline_name,
            pipeline_type=pipeline_type,
            status=PipelineStatus.PENDING.value,
            start_time=datetime.now().isoformat(),
            config=self._sanitize_config(config or {}),
            environment=self._collect_environment_info()
        )

        # Tracking variables
        self.current_epoch = None
        self.current_epoch_data = None
        self.metrics_buffer: List[MetricEntry] = []
        self.epoch_start_time = None

    @staticmethod
    def _generate_run_id() -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    @staticmethod
    def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize config to make it JSON serializable.
        Converts non-serializable objects (like torch.device) to strings.
        """
        import torch
        sanitized = {}
        for key, value in config.items():
            # Convert torch.device to string
            if isinstance(value, torch.device):
                sanitized[key] = str(value)
            # Convert other torch types
            elif hasattr(value, '__module__') and 'torch' in value.__module__:
                sanitized[key] = str(value)
            # Handle nested dicts
            elif isinstance(value, dict):
                sanitized[key] = PipelineMetricsCollector._sanitize_config(value)
            # Handle lists
            elif isinstance(value, list):
                sanitized[key] = [
                    str(item) if (hasattr(item, '__module__') and 'torch' in getattr(item, '__module__', ''))
                    else item
                    for item in value
                ]
            # Keep primitive types as-is
            else:
                try:
                    json.dumps(value)  # Test if serializable
                    sanitized[key] = value
                except (TypeError, ValueError):
                    sanitized[key] = str(value)  # Convert to string if not serializable
        return sanitized

    @staticmethod
    def _collect_environment_info() -> Dict[str, Any]:
        """Collect environment information."""
        import sys
        import platform

        env_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

        try:
            import torch
            env_info["pytorch_version"] = torch.__version__
            env_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env_info["cuda_version"] = torch.version.cuda
                env_info["cudnn_version"] = torch.backends.cudnn.version()
                env_info["gpu_count"] = torch.cuda.device_count()
                env_info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        except ImportError:
            pass

        return env_info

    def start_pipeline(self, additional_config: Optional[Dict[str, Any]] = None):
        """
        Mark pipeline as started.

        Args:
            additional_config: Additional configuration to merge with existing config
        """
        self.pipeline_run.status = PipelineStatus.RUNNING.value
        self.pipeline_run.start_time = datetime.now().isoformat()

        if additional_config:
            sanitized_additional = self._sanitize_config(additional_config)
            self.pipeline_run.config.update(sanitized_additional)

        print(f"[PipelineMetrics] Pipeline '{self.pipeline_name}' started (Run ID: {self.run_id})")

    def start_epoch(self, epoch: int):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.current_epoch_data = EpochSummary(
            epoch=epoch,
            start_time=datetime.now().isoformat()
        )
        print(f"[PipelineMetrics] Epoch {epoch} started")

    def end_epoch(self, epoch: int, epoch_metrics: Optional[Dict[str, Any]] = None):
        
        if self.current_epoch_data is None:
            print(f"[PipelineMetrics] Warning: end_epoch called but epoch {epoch} was not started")
            return

        duration = time.time() - self.epoch_start_time
        self.current_epoch_data.end_time = datetime.now().isoformat()
        self.current_epoch_data.duration_seconds = duration

        if epoch_metrics:
            self.current_epoch_data.metrics.update(epoch_metrics)

        # Update best metrics
        self._update_best_metrics(epoch_metrics or {})

        # Add to epoch summaries
        self.pipeline_run.epoch_summaries.append(self.current_epoch_data.to_dict())

        print(f"[PipelineMetrics] Epoch {epoch} completed in {duration:.2f}s")

        # Auto-save if enabled
        if self.auto_save:
            self.save_to_json()

        # Reset epoch tracking
        self.current_epoch_data = None
        self.epoch_start_time = None

    def log_metric(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metric_type: MetricType = MetricType.SCALAR,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a single metric.

        Args:
            name: Metric name (e.g., "train_loss", "val_accuracy")
            value: Metric value
            step: Training step/iteration
            epoch: Epoch number (uses current_epoch if not provided)
            metric_type: Type of metric
            metadata: Additional metadata
        """
        entry = MetricEntry(
            name=name,
            value=value,
            step=step,
            epoch=epoch or self.current_epoch,
            metric_type=metric_type.value,
            metadata=metadata or {}
        )

        self.metrics_buffer.append(entry)
        self.pipeline_run.all_metrics.append(entry.to_dict())

        # Also add to current epoch data if available
        if self.current_epoch_data is not None:
            self.current_epoch_data.metrics[name] = value

    def log_phase_metrics(self, phase: str, metrics: Dict[str, Any]):
        """
        Log metrics for a specific phase (train/val/test).

        Args:
            phase: Phase name ("train", "val", "test")
            metrics: Dictionary of metrics for this phase
        """
        if self.current_epoch_data is not None:
            self.current_epoch_data.phase_metrics[phase] = metrics

        # Log individual metrics with phase prefix
        for key, value in metrics.items():
            metric_name = f"{phase}_{key}" if not key.startswith(phase) else key
            self.log_metric(metric_name, value)

    def log_batch_metrics(self, batch_idx: int, metrics: Dict[str, Any]):
        """
        Log metrics for a single batch.

        Args:
            batch_idx: Batch index
            metrics: Dictionary of batch metrics
        """
        for key, value in metrics.items():
            self.log_metric(
                name=key,
                value=value,
                step=batch_idx,
                metadata={"batch_idx": batch_idx}
            )

    def _update_best_metrics(self, epoch_metrics: Dict[str, Any]):
        """
        Update best metrics tracking.

        Args:
            epoch_metrics: Metrics from the current epoch
        """
        for key, value in epoch_metrics.items():
            if not isinstance(value, (int, float)):
                continue

            # Determine if higher is better (accuracy) or lower is better (loss)
            is_better = False
            if key not in self.pipeline_run.best_metrics:
                is_better = True
            elif "loss" in key.lower() or "error" in key.lower():
                # Lower is better
                is_better = value < self.pipeline_run.best_metrics[key]["value"]
            else:
                # Higher is better (accuracy, precision, etc.)
                is_better = value > self.pipeline_run.best_metrics[key]["value"]

            if is_better:
                self.pipeline_run.best_metrics[key] = {
                    "value": value,
                    "epoch": self.current_epoch,
                    "timestamp": datetime.now().isoformat()
                }

    def add_artifact(self, artifact_name: str, artifact_path: str):
        """
        Add an artifact (model file, checkpoint, etc.).

        Args:
            artifact_name: Name of the artifact (e.g., "model_path", "onnx_path")
            artifact_path: Path to the artifact
        """
        self.pipeline_run.artifacts[artifact_name] = artifact_path
        print(f"[PipelineMetrics] Artifact added: {artifact_name} -> {artifact_path}")

    def end_pipeline(
        self,
        status: Union[str, PipelineStatus] = PipelineStatus.COMPLETED,
        final_metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None
    ):
        """
        Mark pipeline as completed/failed.

        Args:
            status: Final status of the pipeline
            final_metrics: Final metrics summary
            error_message: Error message if failed
            error_traceback: Error traceback if failed
        """
        if isinstance(status, PipelineStatus):
            status = status.value

        self.pipeline_run.status = status
        self.pipeline_run.end_time = datetime.now().isoformat()

        # Calculate duration
        start = datetime.fromisoformat(self.pipeline_run.start_time)
        end = datetime.fromisoformat(self.pipeline_run.end_time)
        self.pipeline_run.duration_seconds = (end - start).total_seconds()

        if final_metrics:
            self.pipeline_run.final_metrics.update(final_metrics)

        if error_message:
            self.pipeline_run.error_message = error_message
            self.pipeline_run.error_traceback = error_traceback

        print(f"[PipelineMetrics] Pipeline '{self.pipeline_name}' {status}")
        print(f"[PipelineMetrics] Duration: {self.pipeline_run.duration_seconds:.2f}s")

        # Final save
        self.save_to_json()

    def save_to_json(self, filename: Optional[str] = None):
        """
        Save metrics to JSON file.

        Args:
            filename: Custom filename (auto-generated if not provided)
        """
        if filename is None:
            filename = f"{self.pipeline_name}_{self.run_id}.json"

        filepath = self.output_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(self.pipeline_run.to_dict(), f, indent=2)
            print(f"[PipelineMetrics] Metrics saved to: {filepath}")
        except Exception as e:
            print(f"[PipelineMetrics] Error saving metrics: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline run.

        Returns:
            Dictionary containing summary information
        """
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "pipeline_type": self.pipeline_type,
            "status": self.pipeline_run.status,
            "duration_seconds": self.pipeline_run.duration_seconds,
            "total_epochs": len(self.pipeline_run.epoch_summaries),
            "best_metrics": self.pipeline_run.best_metrics,
            "final_metrics": self.pipeline_run.final_metrics,
            "artifacts": self.pipeline_run.artifacts
        }

    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get history of a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of metric entries for the specified metric
        """
        return [
            m for m in self.pipeline_run.all_metrics
            if m["name"] == metric_name
        ]

    def get_epoch_summary(self, epoch: int) -> Optional[Dict[str, Any]]:
        """
        Get summary for a specific epoch.

        Args:
            epoch: Epoch number

        Returns:
            Epoch summary dictionary or None if not found
        """
        for summary in self.pipeline_run.epoch_summaries:
            if summary["epoch"] == epoch:
                return summary
        return None


def create_collector(
    pipeline_name: str,
    pipeline_type: str,
    config: Optional[Dict[str, Any]] = None,
    output_dir: str = "mlops_data/metrics"
) -> PipelineMetricsCollector:
    """
    Factory function to create a metrics collector.

    Args:
        pipeline_name: Name of the pipeline
        pipeline_type: Type of pipeline
        config: Configuration dictionary
        output_dir: Output directory for metrics

    Returns:
        Initialized PipelineMetricsCollector instance

    Example:
        ```python
        collector = create_collector("MNIST_Training", "supervised_classification")
        ```
    """
    return PipelineMetricsCollector(
        pipeline_name=pipeline_name,
        pipeline_type=pipeline_type,
        config=config,
        output_dir=output_dir
    )
