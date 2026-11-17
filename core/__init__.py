"""
Core modeling and algorithmic library for Physics-Informed Dual-Stream
SCADA Prognostics in Wind Turbines.
"""

from .ontology import (
    FEATURE_NAMES,
    CORE_8_INDICES,
    AERO_STREAM_INDICES,
    THERM_STREAM_INDICES,
    SCALING_CONSTANTS,
    PHYSICAL_LIMITS,
    raw_to_normalized_tensor,
    apply_core8_zeropad
)

from .model import StreamEnc, DualStreamV5
from .physics import DiscreteEulerLoss
from .mmd import MultiKernelMMDLoss
from .scoring import ChannelMADScorer, RollingMinGate
from .metrics import compute_care_metrics, calculate_prognostic_lead_time

__all__ = [
    "FEATURE_NAMES",
    "CORE_8_INDICES",
    "AERO_STREAM_INDICES",
    "THERM_STREAM_INDICES",
    "SCALING_CONSTANTS",
    "PHYSICAL_LIMITS",
    "raw_to_normalized_tensor",
    "apply_core8_zeropad",
    "StreamEnc",
    "DualStreamV5",
    "DiscreteEulerLoss",
    "MultiKernelMMDLoss",
    "ChannelMADScorer",
    "RollingMinGate",
    "compute_care_metrics",
    "calculate_prognostic_lead_time",
]
