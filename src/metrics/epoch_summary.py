"""
EpochSummary data class for tracking epoch-level metrics.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class EpochSummary:
    epoch: int
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    phase_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # train/val/test

    def to_dict(self):
        return asdict(self)
