"""仿照 gr-lora_sdr 的前导码粗同步验证逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .chirp import build_upchirp, positive_mod, signed_fft_bin
from .frame_locator import FrameLocation, sync_word_to_symbols
from .preamble_detector import PreambleDetectorConfig, circular_bin_distance


@dataclass(frozen=True)
class FrameSyncPeak:
    """同步后某个符号的 dechirp+FFT 主峰观测。"""

    stage: str
    symbol_index: int
    start_sample: int
    peak_bin: int
    signed_peak_bin: int
    expected_signed_bin: int | None
    distance_to_expected: int | None
    peak_power: float
    second_power: float
    confidence_db: float
    peak_share: float


@dataclass(frozen=True)
class GrloraFrameSyncResult:
    """gr-lora_sdr 风格粗同步后的帧边界与验证结果。"""

    event_index: int
    preamble_ref_bin: int
    preamble_ref_signed_bin: int
    coarse_offset_chips: int
    coarse_offset_samples: int
    synced_preamble_start_sample: int
    synced_sfd_start_sample: int
    synced_payload_start_sample: int
    preamble_peak_mean_signed_bin: float
    preamble_peak_max_abs_signed_bin: int
    preamble_bin0_count: int
    preamble_peak_count: int
    sync1_peak_signed_bin: int
    sync2_peak_signed_bin: int
    sync1_expected_signed_bin: int
    sync2_expected_signed_bin: int
    sync1_distance: int
    sync2_distance: int
    sfd1_peak_signed_bin: int
    sfd2_peak_signed_bin: int
    sfd_mean_signed_bin: float
    cfo_int_est: int
    valid: bool
    peaks: tuple[FrameSyncPeak, ...]


def _measure_peak(
    samples: np.ndarray,
    start_sample: int,
    reference: np.ndarray,
    stage: str,
    symbol_index: int,
    expected_signed_bin: int | None,
) -> FrameSyncPeak | None:
    n = int(reference.size)
    start = int(start_sample)
    stop = start + n
    if start < 0 or stop > samples.size:
        return None

    segment = np.asarray(samples[start:stop], dtype=np.complex64)
    spectrum = np.fft.fft(segment * reference)
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power, dtype=np.float64))
    if total_power <= 0.0:
        return None

    peak_bin = int(np.argmax(power))
    peak_power = float(power[peak_bin])
    second_power = float(np.partition(power, -2)[-2]) if power.size > 1 else 0.0
    signed_peak = signed_fft_bin(peak_bin, n)
    if expected_signed_bin is None:
        distance = None
    else:
        expected_bin = positive_mod(int(expected_signed_bin), n)
        distance = circular_bin_distance(peak_bin, expected_bin, n)

    return FrameSyncPeak(
        stage=str(stage),
        symbol_index=int(symbol_index),
        start_sample=start,
        peak_bin=peak_bin,
        signed_peak_bin=int(signed_peak),
        expected_signed_bin=expected_signed_bin,
        distance_to_expected=distance,
        peak_power=peak_power,
        second_power=second_power,
        confidence_db=float(10.0 * math.log10((peak_power + 1e-30) / (second_power + 1e-30))),
        peak_share=float(peak_power / total_power),
    )


def cfo_int_from_down_val(signed_down_val: int) -> int:
    """复刻 gr-lora_sdr 对 SFD downchirp 解调值除以 2 得到整数 CFO 的逻辑。"""

    return int(math.floor(float(signed_down_val) / 2.0))


def run_grlora_frame_sync_validation(
    samples: np.ndarray,
    frame_location: FrameLocation,
    detector_config: PreambleDetectorConfig,
    preamble_len: float,
    sync_word: int,
    bin0_tol: int = 0,
) -> GrloraFrameSyncResult:
    """把帧定界结果按 gr-lora_sdr 粗同步方式挪窗，并验证前导码是否落到 bin0。"""

    detector_config.validate()
    preamble_symbols = int(round(float(preamble_len)))
    if preamble_symbols <= 0:
        raise ValueError("preamble_len must be positive.")

    chirp_samples = detector_config.chirp_samples
    os_factor = detector_config.os_factor
    ref_signed = signed_fft_bin(frame_location.preamble_ref_bin, chirp_samples)

    # gr-lora_sdr 的 k_hat 粗同步会把 upchirp 主峰对应的时间偏移挪回 bin0。
    coarse_offset_chips = -int(ref_signed)
    coarse_offset_samples = int(coarse_offset_chips * os_factor)
    synced_preamble_start = int(frame_location.preamble_start_sample + coarse_offset_samples)
    synced_sfd_start = int(synced_preamble_start + (preamble_symbols + 2) * chirp_samples)
    synced_payload_start = int(round(synced_sfd_start + 2.25 * chirp_samples))

    upchirp = build_upchirp(
        detector_config.sf,
        symbol_id=0,
        os_factor=detector_config.os_factor,
    )
    down_ref = np.conjugate(upchirp).astype(np.complex64)
    up_ref = upchirp.astype(np.complex64)
    sync1_expected, sync2_expected = sync_word_to_symbols(sync_word)

    peaks: list[FrameSyncPeak] = []
    for symbol_index in range(preamble_symbols):
        peak = _measure_peak(
            samples,
            synced_preamble_start + symbol_index * chirp_samples,
            down_ref,
            "preamble",
            symbol_index,
            0,
        )
        if peak is not None:
            peaks.append(peak)

    sync1_peak = _measure_peak(
        samples,
        synced_preamble_start + preamble_symbols * chirp_samples,
        down_ref,
        "sync",
        preamble_symbols,
        int(sync1_expected),
    )
    sync2_peak = _measure_peak(
        samples,
        synced_preamble_start + (preamble_symbols + 1) * chirp_samples,
        down_ref,
        "sync",
        preamble_symbols + 1,
        int(sync2_expected),
    )
    sfd1_peak = _measure_peak(
        samples,
        synced_sfd_start,
        up_ref,
        "sfd",
        preamble_symbols + 2,
        None,
    )
    sfd2_peak = _measure_peak(
        samples,
        synced_sfd_start + chirp_samples,
        up_ref,
        "sfd",
        preamble_symbols + 3,
        None,
    )

    required = [sync1_peak, sync2_peak, sfd1_peak, sfd2_peak]
    if any(item is None for item in required):
        raise ValueError(f"event {frame_location.event_index} cannot be validated after frame sync.")
    peaks.extend(item for item in required if item is not None)

    preamble_peaks = [item for item in peaks if item.stage == "preamble"]
    if not preamble_peaks:
        raise ValueError(f"event {frame_location.event_index} has no valid synced preamble peak.")

    preamble_signed = [item.signed_peak_bin for item in preamble_peaks]
    bin0_count = sum(abs(item) <= int(bin0_tol) for item in preamble_signed)
    sfd_signed = [sfd1_peak.signed_peak_bin, sfd2_peak.signed_peak_bin]  # type: ignore[union-attr]
    down_val = int(round(float(np.mean(sfd_signed, dtype=np.float64))))
    sync1_distance = int(sync1_peak.distance_to_expected)  # type: ignore[union-attr]
    sync2_distance = int(sync2_peak.distance_to_expected)  # type: ignore[union-attr]
    valid = (
        bin0_count == len(preamble_peaks)
        and sync1_distance == 0
        and sync2_distance == 0
    )

    return GrloraFrameSyncResult(
        event_index=int(frame_location.event_index),
        preamble_ref_bin=int(frame_location.preamble_ref_bin),
        preamble_ref_signed_bin=int(ref_signed),
        coarse_offset_chips=int(coarse_offset_chips),
        coarse_offset_samples=int(coarse_offset_samples),
        synced_preamble_start_sample=synced_preamble_start,
        synced_sfd_start_sample=synced_sfd_start,
        synced_payload_start_sample=synced_payload_start,
        preamble_peak_mean_signed_bin=float(np.mean(preamble_signed, dtype=np.float64)),
        preamble_peak_max_abs_signed_bin=int(max(abs(item) for item in preamble_signed)),
        preamble_bin0_count=int(bin0_count),
        preamble_peak_count=len(preamble_peaks),
        sync1_peak_signed_bin=int(sync1_peak.signed_peak_bin),  # type: ignore[union-attr]
        sync2_peak_signed_bin=int(sync2_peak.signed_peak_bin),  # type: ignore[union-attr]
        sync1_expected_signed_bin=int(sync1_expected),
        sync2_expected_signed_bin=int(sync2_expected),
        sync1_distance=sync1_distance,
        sync2_distance=sync2_distance,
        sfd1_peak_signed_bin=int(sfd1_peak.signed_peak_bin),  # type: ignore[union-attr]
        sfd2_peak_signed_bin=int(sfd2_peak.signed_peak_bin),  # type: ignore[union-attr]
        sfd_mean_signed_bin=float(np.mean(sfd_signed, dtype=np.float64)),
        cfo_int_est=cfo_int_from_down_val(down_val),
        valid=bool(valid),
        peaks=tuple(peaks),
    )
