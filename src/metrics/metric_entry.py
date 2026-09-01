"""
MetricEntry data class for storing individual metric measurements.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional
from .enums import MetricType


@dataclass
class MetricEntry:
    name: str
    value: Any
    step: Optional[int] = None
    epoch: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metric_type: str = MetricType.SCALAR.value
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
