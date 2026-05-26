"""前导码相干叠加初始状态估计。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .chirp import build_upchirp
from .preamble_detector import PreambleDetectorConfig


@dataclass(frozen=True)
class InitialStateSeed:
    """检测阶段输出的粗前导码事件。"""

    event_index: int
    start_sample: int
    end_sample: int
    reference_bin: int
    window_count: int


@dataclass(frozen=True)
class InitialStateSearchConfig:
    """相干估计搜索参数。"""

    estimate_chirps: int
    preamble_len: float
    tau_min: float | None = None
    tau_max: float | None = None
    tau_step: float = 4.0
    beta_min: float = -64.0
    beta_max: float = 64.0
    beta_step: float = 1.0
    fine_tau_radius: float = 4.0
    fine_tau_step: float = 1.0
    fine_beta_radius: float = 2.0
    fine_beta_step: float = 0.25
    zeta_span: float = 0.0
    zeta_step: float = 1e-6
    frequency_chunk: int = 128

    def validate(self) -> None:
        if self.estimate_chirps <= 0:
            raise ValueError("estimate_chirps must be positive.")
        if self.preamble_len <= 0:
            raise ValueError("preamble_len must be positive.")
        if self.tau_step <= 0 or self.beta_step <= 0:
            raise ValueError("coarse search steps must be positive.")
        if self.fine_tau_step <= 0 or self.fine_beta_step <= 0:
            raise ValueError("fine search steps must be positive.")
        if self.zeta_span < 0 or self.zeta_step <= 0:
            raise ValueError("zeta search parameters are invalid.")
        if self.frequency_chunk <= 0:
            raise ValueError("frequency_chunk must be positive.")


@dataclass(frozen=True)
class InitialStateEstimate:
    """一个检测事件对应的前导码相干同步估计结果。"""

    event_index: int
    coarse_start_sample: int
    reference_bin: int
    signed_reference_bin: int
    estimate_chirps: int
    tau0_chip: float
    tau0_sample: float
    beta_bin: float
    cfo_hz: float
    zeta: float
    payload_sto_chip: float
    payload_sto_sample: float
    payload_start_sample: float
    objective: float
    noncoherent_power: float
    coherent_gain_db: float
    mean_abs_z0: float
    coarse_tau0_chip: float
    coarse_beta_bin: float
    coarse_objective: float
    hit_tau_boundary: bool
    hit_beta_boundary: bool


def signed_fft_bin(bin_index: int, fft_len: int) -> int:
    """把循环 FFT bin 转成带符号频偏 bin。"""

    value = int(bin_index) % int(fft_len)
    half = int(fft_len) // 2
    return int(value - int(fft_len) if value > half else value)


def _inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(math.floor((float(stop) - float(start)) / float(step) + 0.5)) + 1
    if count <= 0:
        return np.asarray([], dtype=np.float64)
    values = float(start) + np.arange(count, dtype=np.float64) * float(step)
    return values[values <= float(stop) + abs(float(step)) * 1e-6]


def _build_oversampled_downchirp(detector_config: PreambleDetectorConfig) -> np.ndarray:
    upchirp = build_upchirp(
        detector_config.sf,
        symbol_id=0,
        os_factor=detector_config.os_factor,
    )
    return np.conjugate(upchirp).astype(np.complex64)


def _extract_base_dechirped(
    samples: np.ndarray,
    seed: InitialStateSeed,
    detector_config: PreambleDetectorConfig,
    search_config: InitialStateSearchConfig,
) -> np.ndarray:
    chirp_samples = detector_config.chirp_samples
    total = int(search_config.estimate_chirps * chirp_samples)
    start = int(seed.start_sample)
    end = start + total
    if start < 0 or end > samples.size:
        raise ValueError(
            f"event {seed.event_index} does not have {search_config.estimate_chirps} "
            "complete upchirps after start_sample."
        )
    chirps = np.asarray(samples[start:end], dtype=np.complex64).reshape(
        int(search_config.estimate_chirps),
        chirp_samples,
    )
    downchirp = _build_oversampled_downchirp(detector_config)
    return (chirps * downchirp[np.newaxis, :]).astype(np.complex64)


def _frequency_sums(
    base: np.ndarray,
    f_values: np.ndarray,
    fft_len: int,
    chunk_size: int,
) -> np.ndarray:
    """计算 A_s(f)=sum_n base_s[n] exp(-j2π f n/N)。"""

    n = np.arange(int(fft_len), dtype=np.float64)
    out = np.empty((base.shape[0], f_values.size), dtype=np.complex128)
    for start in range(0, f_values.size, int(chunk_size)):
        stop = min(f_values.size, start + int(chunk_size))
        current_f = f_values[start:stop]
        rot = np.exp(-2j * np.pi * current_f[:, np.newaxis] * n[np.newaxis, :] / fft_len)
        out[:, start:stop] = np.asarray(base, dtype=np.complex128) @ rot.T
    return out


def _coarse_search(
    base: np.ndarray,
    detector_config: PreambleDetectorConfig,
    search_config: InitialStateSearchConfig,
) -> tuple[float, float, float]:
    search_config.validate()
    m_bins = detector_config.n_bins
    fft_len = detector_config.chirp_samples
    tau_min = -0.5 * m_bins if search_config.tau_min is None else float(search_config.tau_min)
    tau_max = 0.5 * m_bins if search_config.tau_max is None else float(search_config.tau_max)
    tau_values = _inclusive_grid(tau_min, tau_max, search_config.tau_step)
    beta_values = _inclusive_grid(search_config.beta_min, search_config.beta_max, search_config.beta_step)
    if tau_values.size == 0 or beta_values.size == 0:
        raise ValueError("coarse search grid is empty.")

    # zeta=0 的粗搜索中 f=beta-tau，先把所有可能频偏的 bin0 相关值预计算出来。
    f_grid = beta_values[:, np.newaxis] - tau_values[np.newaxis, :]
    f_values, inverse = np.unique(np.round(f_grid.reshape(-1), decimals=10), return_inverse=True)
    f_indices = inverse.reshape(beta_values.size, tau_values.size)
    a_s_f = _frequency_sums(base, f_values, fft_len, search_config.frequency_chunk)

    s_index = np.arange(base.shape[0], dtype=np.float64)[:, np.newaxis]
    tau_phase = tau_values / 2.0 + tau_values * tau_values / (2.0 * m_bins)

    best_tau = float(tau_values[0])
    best_beta = float(beta_values[0])
    best_score = -1.0
    for beta_idx, beta in enumerate(beta_values):
        phase = s_index * float(beta) + tau_phase[np.newaxis, :]
        comp = np.exp(-2j * np.pi * phase)
        z = a_s_f[:, f_indices[beta_idx, :]] * comp
        scores = np.abs(np.sum(z, axis=0)) ** 2
        local_idx = int(np.argmax(scores))
        local_score = float(scores[local_idx])
        if local_score > best_score:
            best_score = local_score
            best_tau = float(tau_values[local_idx])
            best_beta = float(beta)

    return best_tau, best_beta, best_score


def _candidate_z0_values(
    base: np.ndarray,
    detector_config: PreambleDetectorConfig,
    tau0: float,
    beta: float,
    zeta: float,
) -> np.ndarray:
    m_bins = detector_config.n_bins
    fft_len = detector_config.chirp_samples
    n = np.arange(fft_len, dtype=np.float64)
    values = []
    for s_idx in range(base.shape[0]):
        tau_s = float(tau0) + float(s_idx) * (m_bins - 1.0) * float(zeta)
        f_s = float(beta) - tau_s
        phi_s = float(s_idx) * float(beta) + tau_s / 2.0 + tau_s * tau_s / (2.0 * m_bins)
        freq_comp = np.exp(-2j * np.pi * f_s * n / fft_len)
        phase_comp = np.exp(-2j * np.pi * phi_s)
        values.append(np.sum(base[s_idx].astype(np.complex128) * freq_comp) * phase_comp)
    return np.asarray(values, dtype=np.complex128)


def _direct_score(
    base: np.ndarray,
    detector_config: PreambleDetectorConfig,
    tau0: float,
    beta: float,
    zeta: float,
) -> tuple[float, float, float, float]:
    z0_values = _candidate_z0_values(base, detector_config, tau0, beta, zeta)
    objective = float(np.abs(np.sum(z0_values)) ** 2)
    noncoherent = float(np.sum(np.abs(z0_values) ** 2))
    coherent_gain_db = 10.0 * math.log10((objective + 1e-30) / (noncoherent + 1e-30))
    mean_abs_z0 = float(np.mean(np.abs(z0_values), dtype=np.float64))
    return objective, noncoherent, coherent_gain_db, mean_abs_z0


def _fine_search(
    base: np.ndarray,
    detector_config: PreambleDetectorConfig,
    search_config: InitialStateSearchConfig,
    coarse_tau: float,
    coarse_beta: float,
) -> tuple[float, float, float, float, float, float, float]:
    tau_values = _inclusive_grid(
        coarse_tau - search_config.fine_tau_radius,
        coarse_tau + search_config.fine_tau_radius,
        search_config.fine_tau_step,
    )
    beta_values = _inclusive_grid(
        coarse_beta - search_config.fine_beta_radius,
        coarse_beta + search_config.fine_beta_radius,
        search_config.fine_beta_step,
    )
    if search_config.zeta_span > 0.0:
        zeta_values = _inclusive_grid(
            -search_config.zeta_span,
            search_config.zeta_span,
            search_config.zeta_step,
        )
    else:
        zeta_values = np.asarray([0.0], dtype=np.float64)

    best = (coarse_tau, coarse_beta, 0.0, -1.0, 0.0, float("nan"), 0.0)
    for zeta in zeta_values:
        for tau in tau_values:
            for beta in beta_values:
                objective, noncoherent, coherent_gain_db, mean_abs_z0 = _direct_score(
                    base,
                    detector_config,
                    float(tau),
                    float(beta),
                    float(zeta),
                )
                if objective > best[3]:
                    best = (
                        float(tau),
                        float(beta),
                        float(zeta),
                        objective,
                        noncoherent,
                        coherent_gain_db,
                        mean_abs_z0,
                    )
    return best


def estimate_initial_state(
    samples: np.ndarray,
    seed: InitialStateSeed,
    detector_config: PreambleDetectorConfig,
    search_config: InitialStateSearchConfig,
) -> InitialStateEstimate:
    """对一个检测事件做前导码相干初始状态估计。"""

    detector_config.validate()
    search_config.validate()
    base = _extract_base_dechirped(samples, seed, detector_config, search_config)
    coarse_tau, coarse_beta, coarse_objective = _coarse_search(
        base,
        detector_config,
        search_config,
    )
    tau0, beta, zeta, objective, noncoherent, coherent_gain_db, mean_abs_z0 = _fine_search(
        base,
        detector_config,
        search_config,
        coarse_tau,
        coarse_beta,
    )

    m_bins = detector_config.n_bins
    os_factor = detector_config.os_factor
    s_payload = float(search_config.preamble_len) + 4.25
    payload_sto_chip = tau0 + s_payload * (m_bins - 1.0) * zeta
    payload_sto_sample = payload_sto_chip * os_factor
    payload_start_sample = (
        float(seed.start_sample)
        + s_payload * detector_config.chirp_samples
        + payload_sto_sample
    )
    cfo_hz = beta * float(detector_config.bw) / float(m_bins)
    tau_min = -0.5 * m_bins if search_config.tau_min is None else float(search_config.tau_min)
    tau_max = 0.5 * m_bins if search_config.tau_max is None else float(search_config.tau_max)
    hit_tau_boundary = (
        abs(tau0 - tau_min) <= search_config.fine_tau_step
        or abs(tau0 - tau_max) <= search_config.fine_tau_step
    )
    hit_beta_boundary = (
        abs(beta - search_config.beta_min) <= search_config.fine_beta_step
        or abs(beta - search_config.beta_max) <= search_config.fine_beta_step
    )

    return InitialStateEstimate(
        event_index=int(seed.event_index),
        coarse_start_sample=int(seed.start_sample),
        reference_bin=int(seed.reference_bin),
        signed_reference_bin=signed_fft_bin(seed.reference_bin, detector_config.chirp_samples),
        estimate_chirps=int(search_config.estimate_chirps),
        tau0_chip=float(tau0),
        tau0_sample=float(tau0 * os_factor),
        beta_bin=float(beta),
        cfo_hz=float(cfo_hz),
        zeta=float(zeta),
        payload_sto_chip=float(payload_sto_chip),
        payload_sto_sample=float(payload_sto_sample),
        payload_start_sample=float(payload_start_sample),
        objective=float(objective),
        noncoherent_power=float(noncoherent),
        coherent_gain_db=float(coherent_gain_db),
        mean_abs_z0=float(mean_abs_z0),
        coarse_tau0_chip=float(coarse_tau),
        coarse_beta_bin=float(coarse_beta),
        coarse_objective=float(coarse_objective),
        hit_tau_boundary=bool(hit_tau_boundary),
        hit_beta_boundary=bool(hit_beta_boundary),
    )


def load_complex64_file(path: Path) -> np.memmap:
    """读取 GNU Radio complex64 IQ 文件。"""

    size_bytes = path.stat().st_size
    if size_bytes % np.dtype(np.complex64).itemsize != 0:
        raise ValueError(f"{path} is not a raw complex64 file.")
    return np.memmap(path, dtype=np.complex64, mode="r")
