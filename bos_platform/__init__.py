"""
In-tree BOS platform stubs for BMAC integration tests and toy demos.

When BOS_PLATFORM_PATH points at a compatible external checkout, classes are
loaded from that tree instead (see bos_platform.loader).
"""
import sys

from .loader import BosPlatformBundle, apply_bundle_to_module, load_bos_platform

_bundle = load_bos_platform()
apply_bundle_to_module(_bundle, sys.modules[__name__])

__all__ = [
    "SignalAPI",
    "ControlAPI",
    "Kalman",
    "TemporalWorkflow",
    "OPA",
    "DigitalTwin",
    "BosPlatformBundle",
    "load_bos_platform",
    "__bos_platform_source__",
]

# ControlAPI stays in-tree (not part of external bundle contract yet)
from .signal_control import ControlAPI  # noqa: E402

__all__.append("ControlAPI")
