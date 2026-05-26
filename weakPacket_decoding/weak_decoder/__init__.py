"""弱包 LoRa 检测与同步估计原型。"""

from .chirp import build_upchirp, dechirp_fft
from .initial_state import (
    InitialStateEstimate,
    InitialStateSearchConfig,
    InitialStateSeed,
    estimate_initial_state,
)
from .frame_locator import (
    FrameLocation,
    FrameLocatorConfig,
    SymbolPeak,
    locate_frame_from_event,
    sync_word_to_symbols,
)
from .preamble_detector import (
    DetectionEvent,
    PreambleDetectorConfig,
    WindowPeak,
    detect_preamble_runs,
    scan_preamble_windows,
)

__all__ = [
    "DetectionEvent",
    "InitialStateEstimate",
    "InitialStateSearchConfig",
    "InitialStateSeed",
    "FrameLocation",
    "FrameLocatorConfig",
    "PreambleDetectorConfig",
    "SymbolPeak",
    "WindowPeak",
    "build_upchirp",
    "dechirp_fft",
    "detect_preamble_runs",
    "estimate_initial_state",
    "locate_frame_from_event",
    "scan_preamble_windows",
    "sync_word_to_symbols",
]
