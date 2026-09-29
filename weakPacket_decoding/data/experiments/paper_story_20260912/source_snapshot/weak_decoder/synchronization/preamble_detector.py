"""弱包前导码滑窗检测原型。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from ..chirp import build_upchirp


PreambleDetectionMode = Literal["noncoherent", "chip_phase", "virtual_simo"]


@dataclass(frozen=True)
class PreambleDetectorConfig:
    """滑窗前导码检测所需的最小参数集合。"""

    sf: int
    bw: float
    samp_rate: float
    win_chirps: int
    hop_samples: int | None
    min_periodic_peaks: int
    bin_tol: int

    @property
    def n_bins(self) -> int:
        return 1 << int(self.sf)

    @property
    def os_factor(self) -> int:
        ratio = float(self.samp_rate) / float(self.bw)
        rounded = int(round(ratio))
        if rounded <= 0 or not math.isclose(ratio, rounded, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"samp_rate / bw must be an integer oversampling factor, got {ratio:.9g}"
            )
        return rounded

    @property
    def chirp_samples(self) -> int:
        return int(self.n_bins * self.os_factor)

    @property
    def resolved_hop_samples(self) -> int:
        return int(self.hop_samples or self.chirp_samples)

    @property
    def window_samples(self) -> int:
        return int(self.win_chirps * self.chirp_samples)

    def validate(self) -> None:
        if int(self.sf) <= 0:
            raise ValueError("sf must be positive.")
        if float(self.bw) <= 0.0:
            raise ValueError("bw must be positive.")
        if float(self.samp_rate) <= 0.0:
            raise ValueError("samp_rate must be positive.")
        if int(self.win_chirps) <= 0:
            raise ValueError("win_chirps must be positive.")
        if self.resolved_hop_samples <= 0:
            raise ValueError("hop_samples must be positive.")
        if int(self.min_periodic_peaks) <= 1:
            raise ValueError("min_periodic_peaks should be at least 2.")
        if int(self.bin_tol) < 0:
            raise ValueError("bin_tol must be non-negative.")
        _ = self.os_factor


@dataclass(frozen=True)
class WindowPeak:
    """一个滑动窗口内多 chirp 能量累加后的 peak 观测。"""

    window_index: int
    start_sample: int
    end_sample: int
    peak_bin: int
    peak_power: float
    second_power: float
    total_power: float
    confidence_db: float
    peak_share: float
    valid: bool

    @property
    def peak_lora_bin(self) -> float:
        return float(self.peak_bin)


@dataclass(frozen=True)
class DetectionEvent:
    """连续窗口 peak bin 稳定后形成的粗前导码检测事件。"""

    event_index: int
    start_sample: int
    end_sample: int
    first_window_index: int
    last_window_index: int
    window_count: int
    reference_bin: int
    bin_min: int
    bin_max: int
    mean_peak_power: float
    mean_confidence_db: float
    max_peak_share: float


def circular_bin_distance(lhs: int, rhs: int, n_bins: int) -> int:
    """计算循环 FFT bin 距离，避免 0 和末尾 bin 被误认为相距很远。"""

    diff = abs(int(lhs) - int(rhs)) % int(n_bins)
    return int(min(diff, int(n_bins) - diff))


def _build_detection_downchirp(config: PreambleDetectorConfig) -> np.ndarray:
    # 检测阶段保留过采样 chirp，不先降到 BW 采样率。
    upchirp = build_upchirp(config.sf, symbol_id=0, os_factor=config.os_factor)
    return np.conjugate(upchirp).astype(np.complex64)


def scan_preamble_windows(
    samples: np.ndarray,
    config: PreambleDetectorConfig,
    sample_limit: int | None = None,
    max_windows: int | None = None,
) -> list[WindowPeak]:
    """在接收序列上滑窗扫描，输出每个窗口的累加 FFT peak。"""

    config.validate()
    limit = samples.size if sample_limit is None else min(samples.size, int(sample_limit))
    if limit < config.window_samples:
        return []

    downchirp = _build_detection_downchirp(config)
    hop = config.resolved_hop_samples
    chirp_samples = config.chirp_samples
    window_samples = config.window_samples
    win_chirps = int(config.win_chirps)

    peaks: list[WindowPeak] = []
    max_start = limit - window_samples
    for window_index, start in enumerate(range(0, max_start + 1, hop)):
        if max_windows is not None and window_index >= int(max_windows):
            break

        window = np.asarray(samples[start : start + window_samples], dtype=np.complex64)
        chirps = window.reshape(win_chirps, chirp_samples)
        dechirped = chirps * downchirp[np.newaxis, :]
        spectrum = np.fft.fft(dechirped, axis=1)
        energy = np.sum(np.abs(spectrum) ** 2, axis=0, dtype=np.float64)

        if energy.size == 0:
            continue
        total_power = float(np.sum(energy, dtype=np.float64))
        if total_power <= 0.0:
            # 全零窗口没有可比较的 peak；这里不引入能量阈值，只排除数学上没有峰值的窗口。
            peak_bin = -1
            peak_power = 0.0
            second_power = 0.0
            confidence_db = float("nan")
            peak_share = float("nan")
            valid = False
        else:
            peak_bin = int(np.argmax(energy))
            peak_power = float(energy[peak_bin])
            if energy.size > 1:
                second_power = float(np.partition(energy, -2)[-2])
            else:
                second_power = 0.0
            confidence_db = 10.0 * math.log10((peak_power + 1e-30) / (second_power + 1e-30))
            peak_share = peak_power / total_power
            valid = True
        peaks.append(
            WindowPeak(
                window_index=window_index,
                start_sample=int(start),
                end_sample=int(start + window_samples),
                peak_bin=peak_bin,
                peak_power=peak_power,
                second_power=second_power,
                total_power=total_power,
                confidence_db=float(confidence_db),
                peak_share=float(peak_share),
                valid=valid,
            )
        )

    return peaks


def virtual_simo_preamble_spectrum(
    dechirped_chirps: np.ndarray,
    os_factor: int,
) -> np.ndarray:
    """Coherently combine temporal preamble snapshots and OSR branches.

    Repeated LoRa preamble upchirps become one continuous tone after
    dechirping.  Splitting the full-rate observation into ``R`` polyphase
    branches gives a virtual-SIMO array.  For long-FFT candidate ``k``, branch
    ``q`` has the known steering phase ``exp(-j*2*pi*k*q/(M*N*R))``.  Applying
    this steering vector and summing the branch DFTs reconstructs the full-rate
    coherent matched filter while making explicit which SIMO property is used.

    The branches are correlated samples of one RF chain, so this function does
    not assume ``R`` independent antennas or claim an ``R``-fold diversity gain.
    """

    values = np.asarray(dechirped_chirps, dtype=np.complex64)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("dechirped_chirps must be a non-empty 2-D array")
    os_value = int(os_factor)
    if os_value <= 0 or int(values.shape[1]) % os_value:
        raise ValueError("chirp length must be divisible by os_factor")

    flattened = values.reshape(-1)
    total_samples = int(flattened.size)
    branch_samples = total_samples // os_value
    candidate_bins = np.arange(total_samples, dtype=np.int64)
    branch_bins = np.mod(candidate_bins, branch_samples)
    combined = np.zeros(total_samples, dtype=np.complex128)
    for phase in range(os_value):
        branch_spectrum = np.fft.fft(flattened[phase::os_value])
        steering = np.exp(
            -2j
            * np.pi
            * candidate_bins.astype(np.float64)
            * float(phase)
            / float(total_samples)
        )
        combined += branch_spectrum[branch_bins] * steering
    return combined.astype(np.complex64)


def scan_virtual_simo_preamble_windows(
    samples: np.ndarray,
    config: PreambleDetectorConfig,
    sample_limit: int | None = None,
    max_windows: int | None = None,
) -> list[WindowPeak]:
    """Scan repeated LoRa preambles with coherent virtual-SIMO integration."""

    config.validate()
    limit = samples.size if sample_limit is None else min(samples.size, int(sample_limit))
    if limit < config.window_samples:
        return []

    downchirp = _build_detection_downchirp(config)
    chirp_samples = int(config.chirp_samples)
    win_chirps = int(config.win_chirps)
    window_samples = int(config.window_samples)
    peaks: list[WindowPeak] = []
    max_start = int(limit - window_samples)
    for window_index, start in enumerate(
        range(0, max_start + 1, int(config.resolved_hop_samples))
    ):
        if max_windows is not None and window_index >= int(max_windows):
            break
        window = np.asarray(
            samples[start : start + window_samples], dtype=np.complex64
        )
        chirps = window.reshape(win_chirps, chirp_samples)
        spectrum = virtual_simo_preamble_spectrum(
            chirps * downchirp[np.newaxis, :],
            os_factor=int(config.os_factor),
        )
        energy = np.abs(spectrum).astype(np.float64) ** 2
        total_power = float(np.sum(energy, dtype=np.float64))
        if total_power <= 0.0:
            peak_bin = -1
            peak_power = 0.0
            second_power = 0.0
            confidence_db = float("nan")
            peak_share = float("nan")
            valid = False
        else:
            peak_bin = int(np.argmax(energy))
            peak_power = float(energy[peak_bin])
            second_power = (
                float(np.partition(energy, -2)[-2]) if energy.size > 1 else 0.0
            )
            confidence_db = float(
                10.0
                * math.log10((peak_power + 1e-30) / (second_power + 1e-30))
            )
            peak_share = float(peak_power / total_power)
            valid = True
        peaks.append(
            WindowPeak(
                window_index=int(window_index),
                start_sample=int(start),
                end_sample=int(start + window_samples),
                peak_bin=int(peak_bin),
                peak_power=float(peak_power),
                second_power=float(second_power),
                total_power=float(total_power),
                confidence_db=float(confidence_db),
                peak_share=float(peak_share),
                valid=bool(valid),
            )
        )
    return peaks


def scan_chip_phase_coherent_preamble_windows(
    samples: np.ndarray,
    config: PreambleDetectorConfig,
    sample_limit: int | None = None,
    max_windows: int | None = None,
    sample_phase: int | None = None,
) -> list[WindowPeak]:
    """Coherently integrate repeated preamble chirps from one ADC phase."""

    config.validate()
    limit = samples.size if sample_limit is None else min(samples.size, int(sample_limit))
    if limit < config.window_samples:
        return []
    phase = (
        int(config.os_factor // 2) if sample_phase is None else int(sample_phase)
    )
    if phase < 0 or phase >= int(config.os_factor):
        raise ValueError("sample_phase is outside the OSR polyphase range")

    downchirp = _build_detection_downchirp(config)
    chirp_samples = int(config.chirp_samples)
    win_chirps = int(config.win_chirps)
    window_samples = int(config.window_samples)
    peaks: list[WindowPeak] = []
    for window_index, start in enumerate(
        range(
            0,
            int(limit - window_samples) + 1,
            int(config.resolved_hop_samples),
        )
    ):
        if max_windows is not None and window_index >= int(max_windows):
            break
        window = np.asarray(
            samples[start : start + window_samples], dtype=np.complex64
        ).reshape(win_chirps, chirp_samples)
        dechirped = window * downchirp[np.newaxis, :]
        spectrum = np.fft.fft(dechirped[:, phase:: int(config.os_factor)].reshape(-1))
        energy = np.abs(spectrum).astype(np.float64) ** 2
        total_power = float(np.sum(energy, dtype=np.float64))
        if total_power <= 0.0:
            peak_bin = -1
            peak_power = 0.0
            second_power = 0.0
            confidence_db = float("nan")
            peak_share = float("nan")
            valid = False
        else:
            peak_bin = int(np.argmax(energy))
            peak_power = float(energy[peak_bin])
            second_power = (
                float(np.partition(energy, -2)[-2]) if energy.size > 1 else 0.0
            )
            confidence_db = float(
                10.0
                * math.log10((peak_power + 1e-30) / (second_power + 1e-30))
            )
            peak_share = float(peak_power / total_power)
            valid = True
        peaks.append(
            WindowPeak(
                window_index=int(window_index),
                start_sample=int(start),
                end_sample=int(start + window_samples),
                peak_bin=int(peak_bin),
                peak_power=float(peak_power),
                second_power=float(second_power),
                total_power=float(total_power),
                confidence_db=float(confidence_db),
                peak_share=float(peak_share),
                valid=bool(valid),
            )
        )
    return peaks


def find_periodic_peak_runs(
    windows: Iterable[WindowPeak],
    config: PreambleDetectorConfig,
    mode: PreambleDetectionMode = "noncoherent",
) -> list[DetectionEvent]:
    """按文档里的周期性 peak-bin 规则，把稳定窗口串成检测事件。"""

    config.validate()
    window_list = list(windows)
    events: list[DetectionEvent] = []
    if not window_list:
        return events

    if str(mode) not in {"noncoherent", "chip_phase", "virtual_simo"}:
        raise ValueError(f"unknown preamble detection mode: {mode}")
    coherent = str(mode) in {"chip_phase", "virtual_simo"}
    if str(mode) == "virtual_simo":
        n_fft_bins = config.window_samples
    elif str(mode) == "chip_phase":
        n_fft_bins = int(config.win_chirps * config.n_bins)
    else:
        n_fft_bins = config.chirp_samples
    bin_tolerance = int(config.bin_tol) * (int(config.win_chirps) if coherent else 1)
    run: list[WindowPeak] = []
    reference_bin: int | None = None

    def close_run() -> None:
        nonlocal run, reference_bin
        if len(run) >= int(config.min_periodic_peaks) and reference_bin is not None:
            event_index = len(events)
            bins = [item.peak_bin for item in run]
            peak_powers = [item.peak_power for item in run]
            confidence = [item.confidence_db for item in run]
            peak_shares = [item.peak_share for item in run]
            events.append(
                DetectionEvent(
                    event_index=event_index,
                    start_sample=int(run[0].start_sample),
                    end_sample=int(run[-1].end_sample),
                    first_window_index=int(run[0].window_index),
                    last_window_index=int(run[-1].window_index),
                    window_count=len(run),
                    reference_bin=int(reference_bin),
                    bin_min=int(min(bins)),
                    bin_max=int(max(bins)),
                    mean_peak_power=float(np.mean(peak_powers, dtype=np.float64)),
                    mean_confidence_db=float(np.mean(confidence, dtype=np.float64)),
                    max_peak_share=float(np.nanmax(peak_shares)),
                )
            )
        run = []
        reference_bin = None

    for window in window_list:
        if not window.valid:
            close_run()
            continue
        if not run:
            run = [window]
            reference_bin = int(window.peak_bin)
            continue
        assert reference_bin is not None
        if circular_bin_distance(window.peak_bin, reference_bin, n_fft_bins) <= bin_tolerance:
            run.append(window)
        else:
            close_run()
            run = [window]
            reference_bin = int(window.peak_bin)

    close_run()
    return events


def detect_preamble_runs(
    samples: np.ndarray,
    config: PreambleDetectorConfig,
    sample_limit: int | None = None,
    max_windows: int | None = None,
    mode: PreambleDetectionMode = "noncoherent",
) -> tuple[list[WindowPeak], list[DetectionEvent]]:
    """执行完整的滑窗扫描和周期性 peak 判定。"""

    if str(mode) == "noncoherent":
        windows = scan_preamble_windows(
            samples,
            config,
            sample_limit=sample_limit,
            max_windows=max_windows,
        )
    elif str(mode) == "chip_phase":
        windows = scan_chip_phase_coherent_preamble_windows(
            samples,
            config,
            sample_limit=sample_limit,
            max_windows=max_windows,
        )
    elif str(mode) == "virtual_simo":
        windows = scan_virtual_simo_preamble_windows(
            samples,
            config,
            sample_limit=sample_limit,
            max_windows=max_windows,
        )
    else:
        raise ValueError(f"unknown preamble detection mode: {mode}")
    events = find_periodic_peak_runs(windows, config, mode=mode)
    return windows, events


def load_complex64_file(path: Path) -> np.memmap:
    """读取 GNU Radio complex64 IQ 文件。"""

    size_bytes = path.stat().st_size
    if size_bytes % np.dtype(np.complex64).itemsize != 0:
        raise ValueError(f"{path} is not a raw complex64 file.")
    return np.memmap(path, dtype=np.complex64, mode="r")
