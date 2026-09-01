"""
Enumerations for pipeline metrics tracking.
"""

from enum import Enum


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricType(Enum):
    """Type of metric being tracked."""
    SCALAR = "scalar"  # Single numeric value (loss, accuracy, etc.)
    HISTOGRAM = "histogram"  # Distribution of values
    IMAGE = "image"  # Image data
    TEXT = "text"  # Text data
    CUSTOM = "custom"  # Custom data type
