#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lora_detect.py
==============
基于纯 NumPy 的 LoRa 离线前导码检测与符号解调脚本。

不依赖 GNU Radio 的 frame_sync 模块，而是使用滑动窗口 + FFT-peak 能量的方法
在 USRP 采集的基带 IQ 数据中搜索 LoRa 前导码，并输出检测到的帧位置、
信噪比(SNR)、整数频偏(k_hat)以及 Net ID / Downchirp 的解调结果。

适用场景
--------
- 需要快速定位 IQ 文件中是否存在 LoRa 数据包；
- 对 gr-lora_sdr 的 frame_sync 做对比验证；
- 采集数据已经过数字下变频(DDC)到基带，无需再次混频。

数据格式
--------
默认读取 GNU Radio / UHD 的 `fc32` 格式：即交错的 float32 (I, Q) 二进制流。
如果文件是 `sc16` (16-bit 整数)，请先用 UHD 工具或修改本脚本里的 `read_iq()`
函数进行格式转换。
"""

import numpy as np
import argparse
import sys

# Windows 终端默认 GBK，强制 stdout 用 utf-8 避免中文乱码
if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding='utf-8')


def build_upchirp(id, sf, os_factor=1):
    """
    生成标准 LoRa upchirp（参考 gr-lora_sdr 的 build_upchirp）。

    Parameters
    ----------
    id : int
        符号调制值（未调制前导码用 0）。
    sf : int
        扩频因子。
    os_factor : int
        过采样因子 = 采样率 / 带宽。

    Returns
    -------
    chirp : np.ndarray, dtype=complex64
        长度为 (2**sf * os_factor) 的复数序列。
    """
    N = 1 << sf
    n_fold = N * os_factor - id * os_factor
    chirp = np.zeros(N * os_factor, dtype=np.complex64)
    for n in range(N * os_factor):
        if n < n_fold:
            phase = 2.0 * np.pi * (
                n * n / (2.0 * N) / (os_factor ** 2)
                + (id / N - 0.5) * n / os_factor
            )
        else:
            phase = 2.0 * np.pi * (
                n * n / (2.0 * N) / (os_factor ** 2)
                + (id / N - 1.5) * n / os_factor
            )
        chirp[n] = np.exp(1j * phase)
    return chirp


def build_ref_chirps(sf, os_factor=1):
    """
    生成未调制的参考 upchirp 和 downchirp。
    downchirp 是 upchirp 的复共轭。
    """
    up = build_upchirp(0, sf, os_factor)
    down = np.conj(up)
    return up, down


def read_iq(path):
    """
    从二进制文件读取 fc32（gr_complex，即交错 float32 I/Q）格式的 IQ 数据。
    如果文件是其他格式（如 sc16），请在此函数内添加转换逻辑。
    """
    raw = np.fromfile(path, dtype=np.complex64)
    return raw


def mod_diff(a, b, N):
    """
    计算两个 bin 索引在循环边界下的最小差异（支持 N/2 折叠）。
    用于判断连续窗口的 peak_bin 是否"一致"。
    """
    d = a - b
    return (d + N // 2) % N - N // 2


def demod_symbol(samples, ref_chirp):
    """
    对给定样本进行 LoRa 符号解调（dechirp + FFT + argmax）。

    Parameters
    ----------
    samples : np.ndarray
        长度应等于一个符号的样本数。
    ref_chirp : np.ndarray
        参考 downchirp（解调 payload/netid 时用）或 upchirp（解调 downchirp 时用）。

    Returns
    -------
    peak_bin : int
        FFT 峰值所在的 bin 索引，对应符号值。
    snr_db : float
        基于峰值与总能量之差估算的 SNR（dB）。
    """
    dechirped = samples * np.conj(ref_chirp)
    fft_res = np.fft.fft(dechirped)
    fft_mag = np.abs(fft_res)
    peak_bin = int(np.argmax(fft_mag))
    sig_en = fft_mag[peak_bin]
    noise_en = np.sum(fft_mag) - sig_en + 1e-12
    snr_db = 10.0 * np.log10(sig_en / noise_en)
    return peak_bin, snr_db


def detect_preamble_sliding(iq, sf, bw, fs, step_factor=4, threshold=0.02, min_preamble=6):
    """
    滑动窗口检测前导码（改进版：先做 downsample，再在 downsampled 流上滑动）。

    核心思路
    --------
    1. 先把 IQ 按 os_factor 做 downsample（取中间相位），恢复为每个符号 N=2^sf 个样本。
    2. 在 downsampled 序列上以 `N // step_factor` 的步长滑动 N 长度窗口。
    3. 对每个窗口做 dechirp（N 点 FFT），计算 peak SNR。
    4. 前导码是一串连续 upchirp，因此真实信号所在区域会连续多个窗口都呈现高 SNR。
       由于滑动步长可能不与符号边界对齐，peak_bin 可能出现 N/4 的循环跳变，因此
       本实现**仅依据 SNR 的连续性**进行判断，不再强制要求 peak_bin 不变。

    Parameters
    ----------
    iq : np.ndarray
        输入复数基带序列。
    sf, bw, fs : int/float
        扩频因子、带宽(Hz)、采样率(Hz)。
    step_factor : int
        滑动步进 = N // step_factor（在 downsampled 域）。默认 4。
    threshold : float
        峰值能量与剩余能量的比值（线性 SNR）阈值。信号与噪声通常相差 1~2 个数量级，
        对于弱信号可尝试 0.01~0.02，强信号可用 0.05 以上。
    min_preamble : int
        最少需要连续多少个窗口满足 SNR 阈值。
        一个符号在 step_factor=4 时约覆盖 4 个窗口，min_preamble=6 对应约 1.5 个符号。

    Returns
    -------
    frames : list[dict]
        每个元素包含检测到的帧信息。
    upchirp, downchirp : np.ndarray
        长度为 N 的参考 chirp，供后续解调复用。
    """
    os_factor = int(round(fs / bw))
    N = 1 << sf
    symbol_samples = N * os_factor
    step = N // step_factor  # downsampled 域的滑动步长

    # 生成参考 chirp（N 点，无过采样）
    upchirp, downchirp = build_ref_chirps(sf, os_factor=1)

    # Downsample：取 offset = os_factor//2 的相位
    offset = os_factor // 2
    iq_down = iq[offset::os_factor]

    n_windows = (len(iq_down) - N) // step + 1
    peak_bins = np.zeros(n_windows, dtype=int)
    peak_snr = np.zeros(n_windows, dtype=float)

    print(f"[INFO] os_factor={os_factor}, N={N}, downsampled_len={len(iq_down)}, "
          f"step={step}, total_windows={n_windows}")

    for i in range(n_windows):
        window = iq_down[i * step:i * step + N]
        dechirped = window * np.conj(downchirp)
        fft_mag = np.abs(np.fft.fft(dechirped))

        peak_bin = int(np.argmax(fft_mag))
        sig_en = fft_mag[peak_bin]
        noise_en = np.sum(fft_mag) - sig_en + 1e-12
        snr = sig_en / noise_en

        peak_bins[i] = peak_bin
        peak_snr[i] = snr

    print(f"[DEBUG] max peak_snr={np.max(peak_snr):.6f}, threshold={threshold}")

    # 仅依据 SNR 连续性找连续段
    frames = []
    i = 0
    while i < len(peak_snr):
        if peak_snr[i] < threshold:
            i += 1
            continue

        j = i + 1
        while j < len(peak_snr) and peak_snr[j] >= threshold:
            j += 1

        count = j - i
        if count >= min_preamble:
            mid_win = (i + j) // 2
            # 将 downsampled 窗口索引转回原始样本索引
            start_sample = mid_win * step * os_factor + offset
            k_hat = int(np.round(np.median(peak_bins[i:j])))
            mean_snr_db = 10.0 * np.log10(np.mean(peak_snr[i:j]))

            frames.append({
                'start_sample': start_sample,
                'peak_snr_db': mean_snr_db,
                'k_hat': k_hat,
                'preamble_count': count,
                'win_start': i,
                'win_end': j
            })
        i = j

    return frames, upchirp, downchirp


def fine_align_and_demod(iq, frame_start, k_hat, sf, bw, fs, upchirp, downchirp, preamble_len=8):
    """
    在粗略检测到的前导码位置附近，做精细对齐，并尝试解调 Net ID 和 Downchirp。

    由于传入的 upchirp/downchirp 已经是 N 点（downsampled 后），我们需要先把原始 IQ
    的对应切片按 os_factor downsample，再进行 N 点 FFT 解调。
    """
    os_factor = int(round(fs / bw))
    N = 1 << sf
    symbol_samples_orig = N * os_factor  # 原始流上一个符号的样本数
    offset_ds = os_factor // 2           # downsample 时取的相位偏移

    def _downsample(seg):
        """从原始 symbol 长度切片中提取 N 个样本。"""
        return seg[offset_ds::os_factor]

    def _demod_orig(seg, ref):
        """对原始长度切片先 downsample 再用 demod_symbol 解调。"""
        return demod_symbol(_downsample(seg), ref)

    # NetID1 的理论起始位置（粗略）
    netid1_coarse = frame_start + preamble_len * symbol_samples_orig

    # 搜索范围：±1/4 原始符号，步长为 os_factor（即 downsampled 域一个样本）
    search_range = symbol_samples_orig // 4
    best_offset = 0
    best_score = -1.0

    for off in range(-search_range, search_range + 1, os_factor):
        pos = netid1_coarse + off
        if pos < 0 or pos + symbol_samples_orig > len(iq):
            continue
        segment = iq[pos:pos + symbol_samples_orig]
        dechirped = _downsample(segment) * np.conj(downchirp)
        score = np.max(np.abs(np.fft.fft(dechirped)))
        if score > best_score:
            best_score = score
            best_offset = off

    aligned = netid1_coarse + best_offset

    results = {}
    if aligned + symbol_samples_orig <= len(iq):
        results['netid1'], results['snr_netid1'] = _demod_orig(
            iq[aligned:aligned + symbol_samples_orig], downchirp)
    if aligned + 2 * symbol_samples_orig <= len(iq):
        results['netid2'], results['snr_netid2'] = _demod_orig(
            iq[aligned + symbol_samples_orig:aligned + 2 * symbol_samples_orig], downchirp)
    if aligned + 3 * symbol_samples_orig <= len(iq):
        results['down1'], results['snr_down1'] = _demod_orig(
            iq[aligned + 2 * symbol_samples_orig:aligned + 3 * symbol_samples_orig], upchirp)
    if aligned + 4 * symbol_samples_orig <= len(iq):
        results['down2'], results['snr_down2'] = _demod_orig(
            iq[aligned + 3 * symbol_samples_orig:aligned + 4 * symbol_samples_orig], upchirp)

    results['aligned_netid1'] = aligned
    return results


def main():
    parser = argparse.ArgumentParser(
        description="基于滑动窗口 FFT-peak 的 LoRa 离线数据包检测"
    )
    parser.add_argument("-f", "--file", required=True, help="输入 IQ 文件路径（fc32 格式）")
    parser.add_argument("--sf", type=int, default=11, help="扩频因子（默认 11）")
    parser.add_argument("--bw", type=float, default=125e3, help="带宽 Hz（默认 125000）")
    parser.add_argument("--fs", "--samp-rate", dest="fs", type=float, default=500e3,
                        help="采样率 Hz（默认 500000）")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="SNR 峰值阈值（线性倍数，默认 5.0，约 7 dB）")
    parser.add_argument("--min-preamble", type=int, default=8,
                        help="最少连续满足条件的窗口数（默认 8，约 2 个符号）")
    parser.add_argument("--preamble-len", type=int, default=16,
                        help="发射端前导码长度（默认 16）")
    args = parser.parse_args()

    # 读取数据
    iq = read_iq(args.file)
    print(f"[INFO] 读取 {len(iq)} 个样本，时长 {len(iq) / args.fs:.3f} s")

    if len(iq) == 0:
        print("[ERROR] 文件为空或格式不匹配")
        sys.exit(1)

    # 前导码检测
    frames, upchirp, downchirp = detect_preamble_sliding(
        iq, args.sf, args.bw, args.fs,
        step_factor=4,
        threshold=args.threshold,
        min_preamble=args.min_preamble
    )

    if not frames:
        print("[WARN] 未检测到任何前导码。建议尝试：")
        print("       1) 降低 --threshold（如 3.0 或 2.0）")
        print("       2) 检查 SF/BW/FS 参数是否与发射端一致")
        print("       3) 确认 IQ 文件格式为 fc32（float32 交错 I/Q）")
        sys.exit(0)

    print(f"\n[INFO] 共检测到 {len(frames)} 个候选帧")

    # 对每个候选帧做精细对齐和 NetID/Downchirp 解调
    for idx, frame in enumerate(frames):
        print(f"\n========== 帧 #{idx + 1} ==========")
        print(f"  粗略起始样本: {frame['start_sample']}")
        print(f"  粗略起始时间: {frame['start_sample'] / args.fs:.4f} s")
        print(f"  平均 SNR:     {frame['peak_snr_db']:.2f} dB")
        print(f"  k_hat (CFO整数+STO整数): {frame['k_hat']}")
        print(f"  连续窗口数:   {frame['preamble_count']} (约 {frame['preamble_count'] / 4:.1f} 个符号)")

        # 精细对齐与验证
        demod = fine_align_and_demod(
            iq, frame['start_sample'], frame['k_hat'],
            args.sf, args.bw, args.fs,
            upchirp, downchirp,
            preamble_len=args.preamble_len
        )

        if 'netid1' in demod:
            print(f"  NetID1:       {demod['netid1']:4d}  (SNR {demod['snr_netid1']:.2f} dB)")
        if 'netid2' in demod:
            print(f"  NetID2:       {demod['netid2']:4d}  (SNR {demod['snr_netid2']:.2f} dB)")
        if 'down1' in demod:
            print(f"  Downchirp1:   {demod['down1']:4d}  (SNR {demod['snr_down1']:.2f} dB)")
        if 'down2' in demod:
            print(f"  Downchirp2:   {demod['down2']:4d}  (SNR {demod['snr_down2']:.2f} dB)")
        if 'aligned_netid1' in demod:
            print(f"  对齐后 NetID1 位置: {demod['aligned_netid1']}")

        # 尝试解调前几个 payload 符号（仅作参考）
        os_factor = int(round(args.fs / args.bw))
        symbol_samples_orig = (1 << args.sf) * os_factor
        offset_ds = os_factor // 2
        payload_start = demod.get('aligned_netid1', frame['start_sample']) + 4 * symbol_samples_orig + symbol_samples_orig // 4
        print(f"  前 3 个 payload 符号解调:")
        for s in range(3):
            pos = payload_start + s * symbol_samples_orig
            if pos + symbol_samples_orig > len(iq):
                print(f"    Symbol {s}: 样本超出文件范围")
                break
            segment = iq[pos:pos + symbol_samples_orig]
            bin_val, snr = demod_symbol(segment[offset_ds::os_factor], downchirp)
            print(f"    Symbol {s}: bin={bin_val:4d}, SNR={snr:.2f} dB")


if __name__ == "__main__":
    main()
