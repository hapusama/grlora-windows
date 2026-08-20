"""Savaux 论文的 oversampled LoRa 解调 baseline。

本模块只实现论文 "A Low-Complexity Demodulation for Oversampled LoRa Signal"
里给出的解调规则：

* 把一个过采样 LoRa symbol 按 OSR 拆成多个 branch；
* 对每个 branch 使用论文 Eq. (34)-(36) 的专用 DFT；
* 按 Eq. (37) 的相位项合并各 branch；
* 对合并后的 periodogram 取最大值作为 raw FFT bin。

它刻意不使用本项目已有的 offset coherence、packet phase line、Top-L lock、
payload prior 或 CRC-guided selection，方便作为干净的论文 baseline。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal

import numpy as np

from ...chirp import build_upchirp, bin_to_grlora_symbol, signed_fft_bin


CfoCorrectionMode = Literal["none", "symbol", "continuous"]


@dataclass(frozen=True)
class _WrappedTailKernel:
    """Cached chirp-convolution constants for the wrapped-tail DFT."""

    forward_chirp: np.ndarray
    reverse_chirp_fft: np.ndarray
    fft_length: int


@dataclass(frozen=True)
class _SavauxKernel:
    """Constants shared by all symbols for one SF/OSR pair."""

    branch_weights: np.ndarray
    wrap_phases: np.ndarray


@dataclass(frozen=True)
class PaperOversampledDemodResult:
    """单个 symbol 的论文 baseline 解调结果。

    `combined_spectrum` 是 Eq. (37) 合并后的频谱；`branch_spectra` 保存
    每个 OSR branch 的 Eq. (34) 结果，便于后续诊断。
    """

    raw_fft_bin: int
    signed_fft_bin: int
    symbol_value: int
    peak_real: float
    peak_imag: float
    peak_amp: float
    peak_power: float
    peak_phase: float
    peak_margin_db: float
    total_power: float
    combined_spectrum: np.ndarray
    branch_spectra: tuple[np.ndarray, ...]
    os_factor: int
    cfo_correction_mode: str
    cfo_common_phase_rad: float


def _validate_os_factor(os_factor: int) -> int:
    """检查 OSR / os_factor 是否为正整数。"""

    value = int(os_factor)
    if value <= 0:
        raise ValueError(f"os_factor must be positive, got {os_factor}")
    return value


@lru_cache(maxsize=512)
def _oversampled_downchirp(
    sf: int,
    os_factor: int,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
) -> np.ndarray:
    """构造一个完整过采样 symbol 长度的 downchirp。

    论文推导默认接收端已经同步。这里保留可选 CFO 参数，只是为了让同一个
    论文 metric 能在本项目 header-first 已估计 CFO 的数据上公平评估；
    这不是新的增强算法。
    """

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    n = np.arange(n_bins * os_value, dtype=np.float64)
    # 用带整数 CFO 偏移的 reference upchirp 生成本地 downchirp；
    # 小数 CFO 用额外相位旋转补偿。
    reference = build_upchirp(sf=sf, symbol_id=int(cfo_int), os_factor=os_value)
    frac_cfo = np.exp(-2j * np.pi * float(cfo_frac) * n / float(n_bins * os_value))
    result = (np.conjugate(reference) * frac_cfo).astype(np.complex64)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=32)
def _branch_dft_matrix(sf: int, os_factor: int, branch_index: int) -> np.ndarray:
    """返回论文 Eq. (36) 的 branch 矩阵 F^(q,ro)。

    q=0 时它就是普通 DFT，调用方直接用 FFT。
    q>0 时，第 k 行会把 wrap 后缀 p >= N-k 乘上 exp(j*2*pi*q/ro)，
    对应 Eq. (35)-(36) 中 Heaviside 阶跃项带来的 branch-specific 相位。
    """

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    q = int(branch_index)
    if not (0 <= q < os_value):
        raise ValueError(f"branch_index must be in [0, {os_value}), got {branch_index}")

    k = np.arange(n_bins, dtype=np.float64)[:, None]
    p = np.arange(n_bins, dtype=np.float64)[None, :]
    kernel = np.exp(-2j * np.pi * k * p / float(n_bins))
    if q == 0:
        return kernel.astype(np.complex64)

    phase = np.ones((n_bins, n_bins), dtype=np.complex64)
    # 对候选 bin k 来说，LoRa chirp wrap 发生在 p >= N-k。
    # 论文的 branch DFT 与普通 DFT 唯一区别就是这段 suffix 的常相位。
    tail = (k > 0) & (p >= (float(n_bins) - k))
    phase[tail] = np.complex64(np.exp(2j * np.pi * q / float(os_value)))
    return (kernel * phase).astype(np.complex64)


@lru_cache(maxsize=16)
def _branch_combination_weights(n_bins: int, os_factor: int) -> np.ndarray:
    length = int(n_bins)
    os_value = _validate_os_factor(os_factor)
    bins = np.arange(length, dtype=np.float64)
    branches = np.arange(os_value, dtype=np.float64)[:, np.newaxis]
    weights = np.exp(
        -2j * np.pi * branches * bins[np.newaxis, :] / float(length * os_value)
    )
    weights.setflags(write=False)
    return weights


@lru_cache(maxsize=16)
def _savaux_kernel(sf: int, os_factor: int) -> _SavauxKernel:
    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    wrap_phases = np.exp(
        2j * np.pi * np.arange(os_value, dtype=np.float64) / float(os_value)
    )
    wrap_phases.setflags(write=False)
    return _SavauxKernel(
        branch_weights=_branch_combination_weights(n_bins, os_value),
        wrap_phases=wrap_phases,
    )


@lru_cache(maxsize=16)
def _wrapped_tail_kernel(n_bins: int) -> _WrappedTailKernel:
    length = int(n_bins)
    if length <= 0:
        raise ValueError("n_bins must be positive")
    indexes = np.arange(length, dtype=np.float64)
    forward_chirp = np.exp(1j * np.pi * indexes * indexes / float(length))
    reverse_chirp = np.exp(-1j * np.pi * indexes * indexes / float(length))
    fft_length = 1 << int((2 * length - 1).bit_length())
    reverse_chirp_fft = np.fft.fft(reverse_chirp, fft_length)
    forward_chirp.setflags(write=False)
    reverse_chirp_fft.setflags(write=False)
    return _WrappedTailKernel(forward_chirp, reverse_chirp_fft, fft_length)


def _wrapped_tail_dft_batch(branches: np.ndarray) -> np.ndarray:
    """Evaluate Eq. (36)'s triangular wrapped-tail DFT by convolution."""

    values = np.asarray(branches, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError("branches must have shape (n_bins, branch_count)")
    n_bins, branch_count = map(int, values.shape)
    if n_bins <= 0 or branch_count <= 0:
        raise ValueError("branches must be non-empty")
    kernel = _wrapped_tail_kernel(n_bins)
    lhs = np.zeros((n_bins, branch_count), dtype=np.complex128)
    lhs[1:] = values[:0:-1] * kernel.forward_chirp[1:, np.newaxis]
    convolution = np.fft.ifft(
        np.fft.fft(lhs, kernel.fft_length, axis=0)
        * kernel.reverse_chirp_fft[:, np.newaxis],
        axis=0,
    )[:n_bins]
    tail = kernel.forward_chirp[:, np.newaxis] * convolution
    tail[0] = 0.0
    return tail.astype(np.complex64)


def _paper_branch_spectrum(
    dechirped_branch: np.ndarray,
    sf: int,
    os_factor: int,
    branch_index: int,
) -> np.ndarray:
    """计算一个 decimated branch 的 Y^(q,ro)[k]。

    输入 branch 已经完成 dechirp；这里只负责论文定义的 branch DFT。
    """

    n_bins = 1 << int(sf)
    branch = np.asarray(dechirped_branch, dtype=np.complex64)
    if branch.size != n_bins:
        raise ValueError(f"branch has {branch.size} samples, expected {n_bins}")
    q = int(branch_index)
    os_value = _validate_os_factor(os_factor)
    if not (0 <= q < os_value):
        raise ValueError(f"branch_index must be in [0, {os_value}), got {branch_index}")
    spectrum = np.fft.fft(branch)
    if q > 0:
        wrap_phase = _savaux_kernel(sf, os_value).wrap_phases[q]
        tail = _wrapped_tail_dft_batch(branch[:, np.newaxis])[:, 0]
        spectrum = spectrum + (wrap_phase - 1.0) * tail
    return (spectrum / math.sqrt(float(n_bins))).astype(np.complex64)


def combine_paper_branch_spectra(
    branch_spectra: tuple[np.ndarray, ...] | list[np.ndarray],
    os_factor: int,
) -> np.ndarray:
    """使用论文 Eq. (37) 合并所有 branch 频谱。"""

    spectra = [np.asarray(item, dtype=np.complex64) for item in branch_spectra]
    if not spectra:
        raise ValueError("at least one branch spectrum is required")
    n_bins = int(spectra[0].size)
    if any(item.size != n_bins for item in spectra):
        raise ValueError("all branch spectra must have the same length")
    os_value = _validate_os_factor(os_factor)
    if len(spectra) != os_value:
        raise ValueError(f"got {len(spectra)} branch spectra, expected {os_value}")

    combined = np.zeros(n_bins, dtype=np.complex128)
    weights = _branch_combination_weights(n_bins, os_value)
    for q, spectrum in enumerate(spectra):
        # Eq. (37): 每个 branch q 的同一个候选 k 需要乘
        # exp(-j*2*pi*q*k/(N*R)) 后再相干相加。
        combined += weights[q] * spectrum.astype(np.complex128)
    return combined.astype(np.complex64)


def paper_oversampled_spectrum(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "none",
) -> tuple[np.ndarray, tuple[np.ndarray, ...], float]:
    """对一个过采样 symbol 计算论文 baseline 的合并频谱。

    `cfo_correction_mode='none'` 是最纯的论文假设。
    `symbol` / `continuous` 只是先用接收端估计的 CFO 做预补偿，再执行
    完全相同的论文 metric，方便在本项目同步后的真实 capture 上比较。
    """

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    start = int(start_sample)
    stop = start + n_bins * os_value
    if start < 0 or stop > np.asarray(samples).size:
        raise ValueError(f"symbol at {start_sample} exceeds input sample range")

    mode = str(cfo_correction_mode)
    if mode not in {"none", "symbol", "continuous"}:
        raise ValueError(f"unknown CFO correction mode: {cfo_correction_mode}")

    # 保留完整 OSR symbol，不先抽中心点；后面才按 q::R 拆 branch。
    symbol = np.asarray(samples[start:stop], dtype=np.complex64)
    cfo_common_phase_rad = 0.0
    use_cfo_int = int(cfo_int) if mode in {"symbol", "continuous"} else 0
    use_cfo_frac = float(cfo_frac) if mode in {"symbol", "continuous"} else 0.0
    if mode == "continuous":
        if header_start_sample is None:
            raise ValueError("header_start_sample is required for continuous CFO correction")
        # gr-lora_sdr 的 CFO-aware downchirp 处理 symbol 内部频率偏移；
        # continuous 模式额外消掉从 header 起点累计到当前 symbol 的公共相位。
        cfo_total = float(cfo_int) + float(cfo_frac)
        relative_chip_start = float(start - int(header_start_sample)) / float(os_value)
        cfo_common_phase_rad = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
        symbol = (symbol * np.exp(-1j * cfo_common_phase_rad)).astype(np.complex64)

    downchirp = _oversampled_downchirp(
        sf=sf,
        os_factor=os_value,
        cfo_int=use_cfo_int,
        cfo_frac=use_cfo_frac,
    )
    # 第一步：整段过采样 symbol 先 dechirp。
    dechirped = (symbol * downchirp).astype(np.complex64)

    branch_spectra = tuple(
        _paper_branch_spectrum(
            # 第二步：按 OSR 拆成 branch q，即 n = R*p + q。
            dechirped_branch=dechirped[q::os_value],
            sf=sf,
            os_factor=os_value,
            branch_index=q,
        )
        for q in range(os_value)
    )
    return (
        combine_paper_branch_spectra(branch_spectra, os_factor=os_value),
        branch_spectra,
        cfo_common_phase_rad,
    )


def demod_paper_oversampled_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    is_header: bool = False,
    ldro: bool = False,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "none",
) -> PaperOversampledDemodResult:
    """使用论文-only oversampled metric 解调一个 LoRa symbol。"""

    combined, branches, cfo_common_phase_rad = paper_oversampled_spectrum(
        samples=samples,
        start_sample=start_sample,
        sf=sf,
        os_factor=os_factor,
        cfo_int=cfo_int,
        cfo_frac=cfo_frac,
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    # 第三步：在 Eq. (37) 合并频谱上做 ML argmax，即取最大 periodogram。
    power = np.abs(combined).astype(np.float64) ** 2
    raw_bin = int(np.argmax(power))
    peak = complex(combined[raw_bin])
    peak_power = float(power[raw_bin])
    second_power = float(np.partition(power, -2)[-2]) if power.size > 1 else 0.0
    total_power = float(np.sum(power, dtype=np.float64))
    # raw FFT bin 再映射成 gr-lora_sdr hard symbol value，方便后续 payload codec/CRC 评估。
    symbol_value = bin_to_grlora_symbol(raw_bin, sf=sf, is_header=is_header, ldro=ldro)

    return PaperOversampledDemodResult(
        raw_fft_bin=raw_bin,
        signed_fft_bin=signed_fft_bin(raw_bin, 1 << int(sf)),
        symbol_value=int(symbol_value),
        peak_real=float(peak.real),
        peak_imag=float(peak.imag),
        peak_amp=float(abs(peak)),
        peak_power=peak_power,
        peak_phase=float(math.atan2(peak.imag, peak.real)),
        peak_margin_db=float(10.0 * math.log10((peak_power + 1e-30) / (second_power + 1e-30))),
        total_power=total_power,
        combined_spectrum=combined,
        branch_spectra=branches,
        os_factor=int(os_factor),
        cfo_correction_mode=str(cfo_correction_mode),
        cfo_common_phase_rad=float(cfo_common_phase_rad),
    )
