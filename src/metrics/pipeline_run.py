"""
PipelineRun data class for storing complete pipeline execution records.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class PipelineRun:
    """Complete pipeline execution record."""
    run_id: str
    pipeline_name: str
    pipeline_type: str  # supervised_classification, gan_training, object_detection, etc.
    status: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Environment info
    environment: Dict[str, Any] = field(default_factory=dict)

    # Metrics
    all_metrics: List[Dict[str, Any]] = field(default_factory=list)
    epoch_summaries: List[Dict[str, Any]] = field(default_factory=list)
    final_metrics: Dict[str, Any] = field(default_factory=dict)

    # Best metrics tracking
    best_metrics: Dict[str, Any] = field(default_factory=dict)

    # Artifacts
    artifacts: Dict[str, str] = field(default_factory=dict)  # model_path, onnx_path, etc.

    # Error tracking
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)
