# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0
"""Signal-processing helpers for preamble FFT features and plots.

Feature export and plotting share the same chirp/dechirp primitives, but the
plot path intentionally keeps raw FFT bin positions so peak-bin jitter remains
visible.
"""

import math

import numpy as np

from .constants import EPS_POWER


def build_upchirp(sf, symbol_id=0):
    n_bins = 1 << sf
    n = np.arange(n_bins, dtype=np.float64)
    n_fold = n_bins - int(symbol_id)
    chirp = np.empty(n_bins, dtype=np.complex64)

    first = n < n_fold
    chirp[first] = np.exp(
        2.0j * np.pi * (n[first] * n[first] / (2.0 * n_bins) + (symbol_id / n_bins - 0.5) * n[first])
    )
    chirp[~first] = np.exp(
        2.0j * np.pi * (n[~first] * n[~first] / (2.0 * n_bins) + (symbol_id / n_bins - 1.5) * n[~first])
    )
    return chirp.astype(np.complex64, copy=False)


def normalize_magnitude(magnitude, mode):
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return magnitude.astype(np.float32, copy=False)
    if mode == "max":
        denom = float(np.max(magnitude))
    elif mode == "sum":
        denom = float(np.sum(magnitude))
    elif mode == "l2":
        denom = float(np.sqrt(np.sum(magnitude * magnitude)))
    else:
        raise ValueError(f"unsupported normalization mode: {mode}")
    if denom <= 0.0 or not np.isfinite(denom):
        return np.zeros_like(magnitude, dtype=np.float32)
    return (magnitude / denom).astype(np.float32, copy=False)


def db10(value):
    value = float(value)
    if value <= 0.0 or not np.isfinite(value):
        return float("nan")
    return 10.0 * math.log10(value)


def mean_power_db(samples):
    if samples.size == 0:
        return float("nan")
    power = float(np.mean(np.abs(samples) ** 2))
    return db10(max(power, EPS_POWER))


def resolve_ldro(sf, bw, ldro_mode):
    if int(ldro_mode) == 0:
        return 0
    if int(ldro_mode) == 1:
        return 1
    return 1 if ((1 << int(sf)) / float(bw)) >= 0.016 else 0


def lora_payload_symbol_count(sf, bw, cr, payload_len, has_crc, impl_head, ldro_mode):
    sf = int(sf)
    cr = int(cr)
    payload_len = max(0, int(payload_len))
    crc = 1 if has_crc else 0
    ih = 1 if impl_head else 0
    de = resolve_ldro(sf, bw, ldro_mode)
    denominator = 4 * max(1, sf - 2 * de)
    numerator = 8 * payload_len - 4 * sf + 28 + 16 * crc - 20 * ih
    coded_blocks = max(math.ceil(numerator / denominator), 0)
    return 8 + coded_blocks * (cr + 4)


def estimate_packet_ranges(iq_size, frame, args):
    # frame_sync 给出的是前导码+sync word+SFD 的范围；包尾需要结合 header 中的
    # payload_len/cr/crc/ldro 通过 LoRa airtime 公式估算出来。
    sf = int(frame["sf"])
    bw = float(frame["bw"])
    samples_per_symbol = int(frame["samples_per_symbol"])
    pay_len = int(frame.get("pay_len", args.pay_len))
    if pay_len < 0:
        pay_len = int(args.pay_len)
    cr = int(frame.get("cr", args.cr))
    if cr < 1:
        cr = int(args.cr)
    has_crc = bool(int(frame.get("crc", int(args.has_crc))))
    ldro_mode = int(frame.get("ldro_mode", args.ldro_mode))
    payload_symbols = lora_payload_symbol_count(
        sf,
        bw,
        cr,
        pay_len,
        has_crc,
        args.impl_head,
        ldro_mode,
    )
    packet_symbols = float(frame["preamble_len"]) + 4.25 + float(payload_symbols)
    packet_start = max(0, int(frame["start_sample"]))
    preamble_end = max(packet_start, int(frame["end_sample"]))
    packet_end = packet_start + int(math.ceil(packet_symbols * samples_per_symbol))
    packet_end = max(preamble_end, min(int(iq_size), packet_end))
    return {
        "packet_start_sample": packet_start,
        "packet_end_sample": packet_end,
    }


def circular_peak_width_bins(magnitude, peak_bin, threshold_db=-3.0):
    # 计算 dechirp 后 FFT 主峰宽度：从主峰向左右查找低于阈值的位置。
    # 默认 threshold_db=-3，表示宽度为主峰幅度下降 3 dB 处的 circular FFT bin 宽度。
    magnitude = np.asarray(magnitude, dtype=np.float64)
    n_bins = int(magnitude.size)
    if n_bins == 0:
        return float("nan")
    peak_bin = int(peak_bin)
    peak_amp = float(magnitude[peak_bin])
    if peak_amp <= 0.0 or not np.isfinite(peak_amp):
        return 0.0

    threshold = peak_amp * (10.0 ** (float(threshold_db) / 20.0))
    above = magnitude >= threshold
    if np.all(above):
        return float(n_bins)
    if not above[peak_bin]:
        return 0.0

    tripled = np.concatenate([magnitude, magnitude, magnitude])
    center = peak_bin + n_bins
    left = center
    # FFT bin 是环形的，所以把频谱复制三份，在中间那份上向左右扩展。
    while left > center - n_bins and tripled[left - 1] >= threshold:
        left -= 1
    right = center
    while right < center + n_bins and tripled[right + 1] >= threshold:
        right += 1

    # 在阈值交点处做线性插值，避免宽度只能是整数 bin。
    left_below = float(tripled[left - 1])
    left_above = float(tripled[left])
    if left_above == left_below:
        left_cross = float(left)
    else:
        left_cross = (left - 1) + (threshold - left_below) / (left_above - left_below)

    right_above = float(tripled[right])
    right_below = float(tripled[right + 1])
    if right_below == right_above:
        right_cross = float(right)
    else:
        right_cross = right + (threshold - right_above) / (right_below - right_above)

    return max(0.0, float(right_cross - left_cross))


def symbol_peak_to_residual_db_from_magnitude(magnitude, peak_bin):
    """Return dechirp FFT peak energy divided by residual-bin energy in dB."""
    energy = np.asarray(magnitude, dtype=np.float64) ** 2
    total_energy = float(np.sum(energy))
    if total_energy <= 0.0:
        return float("nan")
    peak_energy = float(energy[int(peak_bin)])
    residual_energy = total_energy - peak_energy
    return db10(peak_energy / max(residual_energy, EPS_POWER))


def symbol_plan(preamble_len):
    """Return the preamble upchirp symbols analyzed for each packet."""
    preamble_len = max(0, int(preamble_len))
    return [("preamble_upchirp", "up") for _ in range(preamble_len)]


def analyze_frame(iq, frame, args):
    # 对每个检测到的数据包逐个分析前导码符号：
    # 1) 切出一个 symbol 的 IQ；2) 下采样到 2**SF 点；
    # 3) 乘理想 downchirp 完成 dechirp；4) FFT 后取主峰幅度和 -3 dB 宽度。
    sf = int(frame["sf"])
    n_bins = 1 << sf
    os_factor = int(round(float(frame["sample_rate"]) / float(frame["bw"])))
    os_factor = max(1, os_factor)
    samples_per_symbol = int(frame["samples_per_symbol"])
    downsample_phase = args.downsample_phase
    if downsample_phase is None:
        downsample_phase = os_factor // 2
    downsample_phase = int(np.clip(downsample_phase, 0, os_factor - 1))

    upchirp = build_upchirp(sf, 0)
    downchirp = np.conj(upchirp)
    fft_n = args.nfft if args.nfft else n_bins
    plan = symbol_plan(int(frame["preamble_len"]))

    magnitudes = np.zeros((len(plan), fft_n), dtype=np.float32)
    peak_bins = np.zeros(len(plan), dtype=np.int32)
    peak_width_bins = np.zeros(len(plan), dtype=np.float32)
    symbol_peak_to_residual_db = np.zeros(len(plan), dtype=np.float32)
    is_preamble = np.zeros(len(plan), dtype=bool)

    frame_start = int(frame["start_sample"])
    for symbol_index, (symbol_type, chirp_kind) in enumerate(plan):
        symbol_start = frame_start + symbol_index * samples_per_symbol
        is_preamble[symbol_index] = symbol_type == "preamble_upchirp"

        decim_start = symbol_start + downsample_phase
        decim_end = decim_start + samples_per_symbol
        samples = iq[decim_start:decim_end:os_factor]
        if samples.size < n_bins:
            padded = np.zeros(n_bins, dtype=np.complex64)
            padded[: samples.size] = samples
            samples = padded
        elif samples.size > n_bins:
            samples = samples[:n_bins]

        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64, copy=False)
        ref = downchirp if chirp_kind == "up" else upchirp
        spectrum = np.fft.fft(samples * ref, n=fft_n)
        mag = np.abs(spectrum).astype(np.float32)

        # 主峰所在 bin，用来对齐每个前导码符号的局部幅度谱。
        peak_bins[symbol_index] = int(np.argmax(mag))

        # 前导码主峰宽度：以主峰幅度为基准，计算下降 args.peak_width_db dB 后的宽度。
        peak_width_bins[symbol_index] = float(
            circular_peak_width_bins(mag, peak_bins[symbol_index], args.peak_width_db)
        )
        symbol_peak_to_residual_db[symbol_index] = float(
            symbol_peak_to_residual_db_from_magnitude(mag, peak_bins[symbol_index])
        )
        magnitudes[symbol_index, :] = normalize_magnitude(mag, args.normalize)

    return {
        "magnitudes": magnitudes,
        "peak_bins": peak_bins,
        "peak_width_bins": peak_width_bins,
        "symbol_peak_to_residual_db": symbol_peak_to_residual_db,
        "is_preamble": is_preamble,
    }


def peak_spectrum_offsets(args):
    half_width = max(0, int(args.peak_spectrum_half_width))
    return np.arange(-half_width, half_width + 1, dtype=np.int32)


def average_preamble_peak_features(analysis, args):
    """Average preamble dechirp spectra after aligning each symbol on its main peak."""
    mask = analysis["is_preamble"]
    magnitudes = analysis["magnitudes"][mask]
    peak_bins = analysis["peak_bins"][mask]
    peak_width_bins = analysis["peak_width_bins"][mask]
    offsets = peak_spectrum_offsets(args)

    if magnitudes.size == 0:
        return {
            "preamble_peak_spectrum": np.full(offsets.shape, np.nan, dtype=np.float32),
            "preamble_peak_width_bins_avg": float("nan"),
        }

    fft_n = int(magnitudes.shape[1])
    local_spectra = np.zeros((magnitudes.shape[0], offsets.size), dtype=np.float32)
    for row_index, peak_bin in enumerate(peak_bins):
        bins = (int(peak_bin) + offsets) % fft_n
        local_spectra[row_index, :] = magnitudes[row_index, bins]

    return {
        "preamble_peak_spectrum": np.nanmean(local_spectra, axis=0).astype(np.float32, copy=False),
        "preamble_peak_width_bins_avg": float(np.nanmean(peak_width_bins)),
    }


def compute_packet_average_metrics(analysis, ranges, iq, args):
    """计算每包保留的 IQ 功率、前导码主峰集中度和局部频谱特征。"""
    packet_iq = np.asarray(iq[ranges["packet_start_sample"]:ranges["packet_end_sample"]], dtype=np.complex64)
    mask = analysis["is_preamble"]
    peak_to_residual_values = analysis["symbol_peak_to_residual_db"][mask]
    peak_features = average_preamble_peak_features(analysis, args)

    # 平均 IQ 功率：这里是 USRP 复数样本功率的 dB 值，不是经过射频链路标定后的 dBm。
    packet_power_db = mean_power_db(packet_iq)

    # 前导码主峰集中度：dechirp FFT 主峰能量 / 其余 bin 残余能量，不等价于传统信道 SNR。
    preamble_peak_to_residual_db = (
        float(np.nanmean(peak_to_residual_values)) if peak_to_residual_values.size else float("nan")
    )
    return {
        "packet_avg_power_db": packet_power_db,
        "preamble_peak_to_residual_db": preamble_peak_to_residual_db,
        **peak_features,
    }


def full_fft_offsets(n_bins):
    return np.arange(-(int(n_bins) // 2), int(n_bins) // 2, dtype=np.int32)


def fft_bin_to_shifted_offset(fft_bin, n_bins):
    fft_bin = int(fft_bin) % int(n_bins)
    half = int(n_bins) // 2
    return int(fft_bin if fft_bin < half else fft_bin - int(n_bins))


def packet_preamble_dechirp_fft_amplitude(iq, frame, args):
    """Return one packet's complete raw preamble dechirp FFT magnitude."""
    sf = int(frame["sf"])
    n_bins = 1 << sf
    os_factor = int(round(float(frame["sample_rate"]) / float(frame["bw"])))
    os_factor = max(1, os_factor)
    samples_per_symbol = int(frame["samples_per_symbol"])
    preamble_len = int(frame["preamble_len"])
    downsample_phase = args.downsample_phase
    if downsample_phase is None:
        downsample_phase = os_factor // 2
    downsample_phase = int(np.clip(downsample_phase, 0, os_factor - 1))

    downchirp = np.conj(build_upchirp(sf, 0))
    frame_start = int(frame["start_sample"])
    raw_magnitudes = []
    peak_bins = []

    for symbol_index in range(max(0, preamble_len)):
        symbol_start = frame_start + symbol_index * samples_per_symbol
        decim_start = symbol_start + downsample_phase
        decim_end = decim_start + samples_per_symbol
        samples = iq[decim_start:decim_end:os_factor]
        if samples.size == 0:
            continue
        if samples.size < n_bins:
            padded = np.zeros(n_bins, dtype=np.complex64)
            padded[: samples.size] = samples
            samples = padded
        elif samples.size > n_bins:
            samples = samples[:n_bins]

        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64, copy=False)
        magnitude = np.abs(np.fft.fft(samples * downchirp, n=n_bins)).astype(np.float32)
        peak_bin = int(np.argmax(magnitude))
        peak_amp = float(magnitude[peak_bin])
        if peak_amp <= 0.0 or not np.isfinite(peak_amp):
            continue

        peak_bins.append(peak_bin)
        raw_magnitudes.append(magnitude)

    if not raw_magnitudes:
        return None

    packet_magnitude = np.mean(np.vstack(raw_magnitudes), axis=0)
    peak_amp = float(np.max(packet_magnitude))
    if peak_amp <= 0.0 or not np.isfinite(peak_amp):
        return None

    packet_peak_bin = int(np.argmax(packet_magnitude))
    spectrum = np.fft.fftshift(np.nan_to_num(packet_magnitude, nan=0.0, posinf=0.0, neginf=0.0))
    return {
        "x_offsets": full_fft_offsets(n_bins),
        "spectrum": spectrum.astype(np.float32, copy=False),
        "n_bins": n_bins,
        "sf": sf,
        "symbol_count": len(raw_magnitudes),
        "peak_bins": np.asarray(peak_bins, dtype=np.int32),
        "packet_peak_bin": packet_peak_bin,
        "packet_peak_offset": fft_bin_to_shifted_offset(packet_peak_bin, n_bins),
        "packet_peak_abs": peak_amp,
    }
