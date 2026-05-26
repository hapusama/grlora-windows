"""弱包 LoRa 检测与帧同步原型。"""

from .chirp import build_upchirp, dechirp_fft, signed_fft_bin
from .frame_locator import (
    FrameLocation,
    FrameLocatorConfig,
    SymbolPeak,
    locate_frame_from_event,
    sync_word_to_symbols,
)
from .grlora_frame_sync import (
    FrameSyncPeak,
    GrloraFrameSyncResult,
    build_grlora_corrected_preamble_chirps,
    run_grlora_frame_sync_validation,
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
    "FrameLocation",
    "FrameLocatorConfig",
    "FrameSyncPeak",
    "GrloraFrameSyncResult",
    "PreambleDetectorConfig",
    "SymbolPeak",
    "WindowPeak",
    "build_upchirp",
    "build_grlora_corrected_preamble_chirps",
    "dechirp_fft",
    "detect_preamble_runs",
    "locate_frame_from_event",
    "run_grlora_frame_sync_validation",
    "scan_preamble_windows",
    "signed_fft_bin",
    "sync_word_to_symbols",
]
