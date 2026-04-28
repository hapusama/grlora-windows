#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# Offline LoRa packet RSSI/SNR and preamble FFT feature exporter for fc32/cfile IQ captures.
# conda run --no-capture-output -n gr-lora python gr-lora_sdr/examples/lora_file_preamble_fft.py --all-bin --input-dir gr-lora_sdr/data/USRP_IQ --samp-rate 500000 --center-freq 487.7e6 --sync-word 0x34 --crc-mode 0 --no-throttle --no-print-header --print-payload none
# In --all-bin mode, SF/TP/preamble length are parsed from each file name:
# experiment_corridor_position_sf_tp_preamble.bin

import csv
import copy
import gc
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from argparse import ArgumentParser, SUPPRESS
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pmt
from gnuradio import blocks
from gnuradio import gr
from gnuradio.eng_arg import eng_float

import gnuradio.lora_sdr as lora_sdr


RSSI_OFFSET_LF = -164.0
RSSI_OFFSET_HF = -157.0
RF_MID_BAND_THRESH = 525000000.0
SX1276_SNR_STEP_DB = 0.25
SX1276_SNR_REG_MIN = -128
SX1276_SNR_REG_MAX = 127
EPS_POWER = 1e-30
WINDOWS_ACCESS_VIOLATION = 0xC0000005


@contextmanager
def suppress_native_output(enabled=True):
    """Temporarily hide GNU Radio/C++ stdout/stderr noise during one detector run."""
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def prepare_file_source_path(input_file):
    """Return an ASCII-only hardlink path for GNU Radio file_source on Windows.

    GNU Radio's C++ file_source uses fopen on Windows, which can fail on paths
    containing Chinese characters. Python can still read those paths, so only
    the GNU Radio input side is redirected through a temporary hardlink.
    """
    source_path = Path(input_file).resolve()
    staging_dir = Path(__file__).resolve().parent / "_file_source_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()
    staged_path = staging_dir / f"{digest}{source_path.suffix.lower()}"
    if staged_path.exists():
        try:
            if staged_path.samefile(source_path):
                return str(staged_path)
        except OSError:
            pass
        staged_path.unlink()

    try:
        os.link(source_path, staged_path)
    except OSError as exc:
        raise RuntimeError(
            f"failed to create ASCII hardlink for GNU Radio file_source: {source_path}"
        ) from exc
    return str(staged_path)


def cleanup_file_source_path(capture_args):
    staged_path = getattr(capture_args, "file_source_path", "")
    if not staged_path:
        return
    try:
        Path(staged_path).unlink(missing_ok=True)
    except OSError:
        pass


class preamble_metadata_sink(gr.basic_block):
    """Collect frame_sync preamble messages.

    frame_sync 负责在 IQ 中检测包同步位置。这里收集的是每个包的
    preamble/sync/SFD 对齐样本范围，以及 frame_sync 估计的 SNR/CFO/STO/SFO。
    注意：frame_sync 自带的 frame_count 只是内部检测计数，不作为最终输出字段。
    """

    def __init__(self):
        gr.basic_block.__init__(
            self,
            name="preamble_metadata_sink",
            in_sig=None,
            out_sig=None,
        )
        self.frames = []
        self._lock = threading.Lock()
        self.message_port_register_in(pmt.intern("preamble"))
        self.set_msg_handler(pmt.intern("preamble"), self.handle_preamble)

    def _dict_value(self, msg, key, default=None):
        value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
        if pmt.is_null(value):
            return default
        return pmt.to_python(value)

    def handle_preamble(self, msg):
        if not pmt.is_dict(msg):
            print("[preamble_fft] ignored non-dict preamble message")
            return

        start_sample = self._dict_value(msg, "start_sample", None)
        end_sample = self._dict_value(msg, "end_sample", None)
        if start_sample is None or end_sample is None:
            print("[preamble_fft] ignored message without sample range")
            return

        frame = {
            "frame_count": int(self._dict_value(msg, "frame_count", 0)),
            "sf": int(self._dict_value(msg, "sf", 7)),
            "bw": float(self._dict_value(msg, "bw", 125000)),
            "sample_rate": float(self._dict_value(msg, "sample_rate", 125000)),
            "samples_per_symbol": int(self._dict_value(msg, "samples_per_symbol", 1 << int(self._dict_value(msg, "sf", 7)))),
            "preamble_len": int(self._dict_value(msg, "preamble_len", 8)),
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "n_samples": int(self._dict_value(msg, "n_samples", int(end_sample) - int(start_sample))),
            "n_symbols": float(self._dict_value(msg, "n_symbols", 0.0)),
            "snr_db": self._dict_value(msg, "snr_db", np.nan),
            "cfo": self._dict_value(msg, "cfo", np.nan),
            "sto": self._dict_value(msg, "sto", np.nan),
            "sfo": self._dict_value(msg, "sfo", np.nan),
            "netid1": int(self._dict_value(msg, "netid1", -1)),
            "netid2": int(self._dict_value(msg, "netid2", -1)),
        }
        with self._lock:
            self.frames.append(frame)


class header_metadata_sink(gr.basic_block):
    """Collect decoded PHY header messages in detection order.

    header_decoder 解出 payload 长度、编码率、CRC 标志等 PHY header 信息。
    这些信息后面用于估算 packet_end_sample，也就是一个包大概到哪里结束。
    """

    def __init__(self):
        gr.basic_block.__init__(
            self,
            name="header_metadata_sink",
            in_sig=None,
            out_sig=None,
        )
        self.headers = []
        self._lock = threading.Lock()
        self.message_port_register_in(pmt.intern("frame_info"))
        self.set_msg_handler(pmt.intern("frame_info"), self.handle_frame_info)

    def _dict_value(self, msg, key, default=None):
        value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
        if pmt.is_null(value):
            return default
        return pmt.to_python(value)

    def handle_frame_info(self, msg):
        if not pmt.is_dict(msg):
            return

        header = {
            "cr": int(self._dict_value(msg, "cr", -1)),
            "pay_len": int(self._dict_value(msg, "pay_len", -1)),
            "crc": int(self._dict_value(msg, "crc", 0)),
            "ldro_mode": int(self._dict_value(msg, "ldro_mode", 2)),
            "header_err": int(self._dict_value(msg, "err", 1)),
        }
        with self._lock:
            self.headers.append(header)


def payload_msg_to_bytes(msg):
    """Convert crc_verif payload PMT into raw bytes."""
    if hasattr(pmt, "is_u8vector") and pmt.is_u8vector(msg):
        return bytes(pmt.u8vector_elements(msg))
    if hasattr(pmt, "is_blob") and pmt.is_blob(msg):
        return bytes(pmt.blob_data(msg))
    if pmt.is_symbol(msg):
        try:
            return pmt.symbol_to_string(msg).encode("latin-1", errors="ignore")
        except UnicodeDecodeError:
            return b""

    try:
        payload = pmt.to_python(msg)
    except UnicodeDecodeError:
        return b""
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode("latin-1", errors="ignore")
    return b""


def extract_payload_packet_number(payload):
    if len(payload) >= 8 and (payload[0] & 0xE0) in (0x40, 0x60, 0x80, 0xA0):
        # Branch4 LoRaWAN-like PHYPayload: MHDR|DevAddr|FCtrl|FCnt|...
        return int(payload[6]) + (int(payload[7]) << 8)
    if len(payload) >= 2:
        return int(payload[0]) + (int(payload[1]) << 8)
    return ""


def print_payload_mode(mode):
    return {"none": 0, "ascii": 1, "hex": 2}.get(str(mode).lower(), 0)


def int_or_default(value, default=-1):
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class payload_metadata_sink(gr.basic_block):
    """Collect decoded payload packet numbers in detection order."""

    def __init__(self):
        gr.basic_block.__init__(
            self,
            name="payload_metadata_sink",
            in_sig=None,
            out_sig=None,
        )
        self.payloads = []
        self._lock = threading.Lock()
        self.message_port_register_in(pmt.intern("payload"))
        self.set_msg_handler(pmt.intern("payload"), self.handle_payload)

    def handle_payload(self, msg):
        payload = payload_msg_to_bytes(msg)
        packet_number = extract_payload_packet_number(payload)

        with self._lock:
            self.payloads.append(
                {
                    "header_packet_counter": packet_number,
                    "payload_packet_number": packet_number,
                    "decoded_payload_len": len(payload),
                }
            )


class lora_file_preamble_fft_rx(gr.top_block):
    """Run the same file RX chain far enough to obtain frame_sync preamble ranges."""

    def __init__(self, args):
        gr.top_block.__init__(self, "LoRa File Preamble FFT", catch_exceptions=True)

        self.input_file = args.input_file
        self.file_source_path = args.file_source_path
        self.sf = args.sf
        self.bw = args.bw
        self.samp_rate = args.samp_rate
        self.cr = args.cr
        self.pay_len = args.pay_len
        self.has_crc = args.has_crc
        self.impl_head = args.impl_head
        self.soft_decoding = args.soft_decoding
        self.center_freq = args.center_freq
        self.sync_word = args.sync_word
        self.ldro_mode = args.ldro_mode
        self.preamble_len = args.preamble_len

        os_factor = int(round(float(self.samp_rate) / float(self.bw)))
        min_buf = int(np.ceil(os_factor * ((1 << self.sf) + 2)))

        self.file_source = blocks.file_source(
            gr.sizeof_gr_complex,
            self.file_source_path,
            False,
            0,
            0,
        )
        self.file_source.set_min_output_buffer(min_buf)

        self.frame_sync = lora_sdr.frame_sync(
            int(self.center_freq),
            int(self.bw),
            self.sf,
            self.impl_head,
            [self.sync_word],
            os_factor,
            int(self.preamble_len),
        )
        self.fft_demod = lora_sdr.fft_demod(self.soft_decoding, True)
        self.gray_mapping = lora_sdr.gray_mapping(self.soft_decoding)
        self.deinterleaver = lora_sdr.deinterleaver(self.soft_decoding)
        self.hamming_dec = lora_sdr.hamming_dec(self.soft_decoding)
        self.header_decoder = lora_sdr.header_decoder(
            self.impl_head,
            self.cr,
            self.pay_len,
            self.has_crc,
            self.ldro_mode,
            args.print_header,
        )
        self.require_valid_payload = bool(args.require_valid_payload)
        self.metadata_sink = preamble_metadata_sink()
        self.header_sink = header_metadata_sink()
        if self.require_valid_payload:
            self.dewhitening = lora_sdr.dewhitening()
            crc_mode = lora_sdr.Crc_mode.SX1276 if args.crc_mode == 1 else lora_sdr.Crc_mode.GRLORA
            self.crc_verif = lora_sdr.crc_verif(
                print_payload_mode(args.print_payload),
                True,
                crc_mode,
            )
            self.payload_bytes_null_sink = blocks.null_sink(gr.sizeof_char)
            self.crc_valid_sink = blocks.vector_sink_b()
            self.payload_sink = payload_metadata_sink()
        else:
            self.payload_null_sink = blocks.null_sink(gr.sizeof_char)

        if args.throttle:
            self.throttle = blocks.throttle(gr.sizeof_gr_complex, self.samp_rate, True)
            self.throttle.set_min_output_buffer(min_buf)
            self.connect((self.file_source, 0), (self.throttle, 0))
            self.connect((self.throttle, 0), (self.frame_sync, 0))
        else:
            self.connect((self.file_source, 0), (self.frame_sync, 0))

        self.connect((self.frame_sync, 0), (self.fft_demod, 0))
        self.connect((self.fft_demod, 0), (self.gray_mapping, 0))
        self.connect((self.gray_mapping, 0), (self.deinterleaver, 0))
        self.connect((self.deinterleaver, 0), (self.hamming_dec, 0))
        self.connect((self.hamming_dec, 0), (self.header_decoder, 0))
        if self.require_valid_payload:
            self.connect((self.header_decoder, 0), (self.dewhitening, 0))
            self.connect((self.dewhitening, 0), (self.crc_verif, 0))
            self.connect((self.crc_verif, 0), (self.payload_bytes_null_sink, 0))
            self.connect((self.crc_verif, 1), (self.crc_valid_sink, 0))
            self.msg_connect((self.crc_verif, "msg"), (self.payload_sink, "payload"))
        else:
            self.connect((self.header_decoder, 0), (self.payload_null_sink, 0))
        self.msg_connect((self.header_decoder, "frame_info"), (self.frame_sync, "frame_info"))
        self.msg_connect((self.header_decoder, "frame_info"), (self.header_sink, "frame_info"))
        self.msg_connect((self.frame_sync, "preamble"), (self.metadata_sink, "preamble"))


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


def sx1276_rssi_offset(center_freq):
    return RSSI_OFFSET_HF if float(center_freq) > RF_MID_BAND_THRESH else RSSI_OFFSET_LF


def sx1276_snr_reg_value(snr_db):
    """Convert an SNR estimate to the SX1276 RegPktSnrValue quarter-dB format."""
    if not np.isfinite(float(snr_db)):
        return None
    reg = int(round(float(snr_db) / SX1276_SNR_STEP_DB))
    return int(np.clip(reg, SX1276_SNR_REG_MIN, SX1276_SNR_REG_MAX))


def sx1276_style_snr_db(snr_db):
    reg = sx1276_snr_reg_value(snr_db)
    if reg is None:
        return float("nan")
    return float(reg) * SX1276_SNR_STEP_DB


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


def symbol_snr_db_from_magnitude(magnitude, peak_bin):
    energy = np.asarray(magnitude, dtype=np.float64) ** 2
    total_energy = float(np.sum(energy))
    if total_energy <= 0.0:
        return float("nan")
    peak_energy = float(energy[int(peak_bin)])
    noise_energy = total_energy - peak_energy
    return db10(peak_energy / max(noise_energy, EPS_POWER))


def symbol_plan(part, preamble_len):
    """Return the symbols analyzed around the preamble/header boundary."""
    preamble_len = max(0, int(preamble_len))
    plan = [("preamble_upchirp", "up") for _ in range(preamble_len)]
    if part == "phy-header":
        plan.extend(
            [
                ("sync_word_0", "up"),
                ("sync_word_1", "up"),
                ("sfd_downchirp_0", "down"),
                ("sfd_downchirp_1", "down"),
            ]
        )
    return plan


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

    cfo = float(frame["cfo"]) if np.isfinite(float(frame["cfo"])) else 0.0
    upchirp = build_upchirp(sf, 0)
    downchirp = np.conj(upchirp)
    fft_n = args.nfft if args.nfft else n_bins
    plan = symbol_plan(args.part, int(frame["preamble_len"]))

    magnitudes = np.zeros((len(plan), fft_n), dtype=np.float32)
    peak_bins = np.zeros(len(plan), dtype=np.int32)
    peak_width_bins = np.zeros(len(plan), dtype=np.float32)
    symbol_snr_db = np.zeros(len(plan), dtype=np.float32)
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
        if args.cfo_correct:
            n = np.arange(n_bins, dtype=np.float32)
            samples = samples * np.exp(-2.0j * np.pi * cfo * n / n_bins).astype(np.complex64)

        ref = downchirp if chirp_kind == "up" else upchirp
        spectrum = np.fft.fft(samples * ref, n=fft_n)
        mag = np.abs(spectrum).astype(np.float32)

        # 主峰所在 bin，用来对齐每个前导码符号的局部幅度谱。
        peak_bins[symbol_index] = int(np.argmax(mag))

        # 前导码主峰宽度：以主峰幅度为基准，计算下降 args.peak_width_db dB 后的宽度。
        peak_width_bins[symbol_index] = float(
            circular_peak_width_bins(mag, peak_bins[symbol_index], args.peak_width_db)
        )
        symbol_snr_db[symbol_index] = float(symbol_snr_db_from_magnitude(mag, peak_bins[symbol_index]))
        magnitudes[symbol_index, :] = normalize_magnitude(mag, args.normalize)

    return {
        "magnitudes": magnitudes,
        "peak_bins": peak_bins,
        "peak_width_bins": peak_width_bins,
        "symbol_snr_db": symbol_snr_db,
        "is_preamble": is_preamble,
    }


def fmt_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(value):
        return ""
    return f"{value:.9g}"


def write_dict_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_capture_metadata_value(value):
    match = re.search(r"-?\d+", str(value))
    if match is None:
        return value
    return int(match.group(0))


def parse_capture_metadata(path):
    parts = re.split(r"[_-]", Path(path).stem)
    keys = ["experiment_id", "corridor_id", "position_id", "sf", "tx_power_dbm", "preamble_len"]
    metadata = {
        "lab_name": "",
        "lab_path": "",
        "lab_note": "",
        "experiment_id": "",
        "corridor_id": "",
        "position_id": "",
        "tx_power_dbm": "",
        "filename_sf": "",
        "filename_tx_power_dbm": "",
        "filename_preamble_len": "",
    }
    if len(parts) >= 6:
        for key, value in zip(keys, parts[:6]):
            parsed = parse_capture_metadata_value(value)
            if key == "sf":
                metadata["filename_sf"] = parsed
            elif key == "tx_power_dbm":
                metadata["tx_power_dbm"] = parsed
                metadata["filename_tx_power_dbm"] = parsed
            elif key == "preamble_len":
                metadata["filename_preamble_len"] = parsed
            else:
                metadata[key] = parsed
    return metadata


def read_text_file(path):
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return Path(path).read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    return Path(path).read_text(errors="ignore").strip()


def parse_lab_note_overrides(note_text):
    """从 lab 文件夹的补充说明中解析会影响解码参数的修正项。"""
    overrides = {}
    if not note_text:
        return overrides

    sf_match = re.search(r"sf\s*其实是\s*(\d+)", note_text, flags=re.IGNORECASE)
    if sf_match is None:
        sf_match = re.search(r"(?:sf|spreading\s*factor)\s*(?:=|是|为|:)\s*(\d+)", note_text, flags=re.IGNORECASE)
    if sf_match is not None:
        overrides["sf"] = int(sf_match.group(1))

    return overrides


def load_lab_metadata(lab_dir):
    """读取一个 lab 文件夹的补充说明，并生成会写入输出结果的 lab 元数据。"""
    lab_dir = Path(lab_dir)
    note_paths = sorted(lab_dir.rglob("补充.txt"))
    note_text = "\n".join(read_text_file(path) for path in note_paths)
    overrides = parse_lab_note_overrides(note_text)
    return {
        "lab_name": lab_dir.name,
        "lab_path": str(lab_dir),
        "lab_note": note_text,
        "lab_note_path": ";".join(str(path) for path in note_paths),
        "override_sf": overrides.get("sf", ""),
    }


def default_output_dir():
    return Path(__file__).resolve().parent / "preamble_fft"


def all_bin_root_output_dir(args, input_dir):
    """Use the lab/input folder itself for root-level --all-bin unless the user chose an output dir."""
    output_dir = Path(args.output_dir)
    if output_dir == default_output_dir():
        return Path(input_dir)
    return output_dir


def discover_lab_jobs(args):
    """按 USRP_IQ 下的 lab 子文件夹分组；每组单独输出到自己的 lab 文件夹。"""
    input_dir = Path(args.input_dir)
    jobs = []
    for lab_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        input_files = sorted(lab_dir.rglob("*.bin"))
        if not input_files:
            continue
        jobs.append(
            {
                "label": lab_dir.name,
                "input_files": input_files,
                "output_dir": lab_dir,
                "lab_metadata": load_lab_metadata(lab_dir),
            }
        )

    root_input_files = sorted(input_dir.glob("*.bin"))
    if jobs:
        if root_input_files:
            print(
                f"[preamble_fft] found {len(root_input_files)} root-level .bin file(s) in {input_dir}; "
                "skip them because lab-folder mode is active"
            )
        return jobs

    if root_input_files:
        # 兼容旧目录结构：如果没有任何 lab 子文件夹，就退回到根目录 .bin 批处理。
        return [
            {
                "label": input_dir.name,
                "input_files": root_input_files,
                "output_dir": all_bin_root_output_dir(args, input_dir),
                "lab_metadata": {
                    "lab_name": input_dir.name,
                    "lab_path": str(input_dir),
                    "lab_note": "",
                    "lab_note_path": "",
                    "override_sf": "",
                },
            }
        ]

    return []


def resolve_capture_args(base_args, input_file, lab_metadata=None, prepare_source=True):
    args = copy.copy(base_args)
    args.input_file = str(input_file)
    args.file_source_path = prepare_file_source_path(input_file) if prepare_source else ""
    metadata = parse_capture_metadata(input_file)
    if lab_metadata:
        metadata.update(
            {
                "lab_name": lab_metadata.get("lab_name", ""),
                "lab_path": lab_metadata.get("lab_path", ""),
                "lab_note": lab_metadata.get("lab_note", ""),
                "lab_note_path": lab_metadata.get("lab_note_path", ""),
                "lab_note_override_sf": lab_metadata.get("override_sf", ""),
            }
        )
    args.capture_metadata = metadata
    filename_sf = metadata.get("filename_sf", "")
    filename_preamble_len = metadata.get("filename_preamble_len", "")
    override_sf = lab_metadata.get("override_sf", "") if lab_metadata else ""
    args.sf = int(
        base_args.sf
        if base_args.sf is not None
        else (override_sf if override_sf != "" else (filename_sf if filename_sf != "" else 7))
    )
    args.preamble_len = int(
        base_args.preamble_len
        if base_args.preamble_len is not None
        else (filename_preamble_len if filename_preamble_len != "" else 16)
    )
    args.capture_metadata["resolved_sf"] = args.sf
    args.capture_metadata["resolved_preamble_len"] = args.preamble_len
    return args


def merge_frame_and_header_metadata(frames, headers, payloads, capture_args):
    metadata = capture_args.capture_metadata
    require_valid_payload = bool(getattr(capture_args, "require_valid_payload", False))
    merged = []
    for index, frame in enumerate(frames):
        payload = payloads[index] if index < len(payloads) else None
        if require_valid_payload and not (payload and payload.get("crc_valid", False)):
            continue

        item = dict(frame)
        if index < len(headers):
            item.update(headers[index])
        else:
            item.update(
                {
                    "cr": int(capture_args.cr),
                    "pay_len": int(capture_args.pay_len),
                    "crc": int(capture_args.has_crc),
                    "ldro_mode": int(capture_args.ldro_mode),
                    "header_err": -1,
                }
            )
        if payload:
            item.update(payload)
        else:
            item.update(
                {
                    "header_packet_counter": "",
                    "payload_packet_number": "",
                    "decoded_payload_len": 0,
                }
            )
        item.update(metadata)
        item["input_file"] = str(capture_args.input_file)
        item["file_name"] = Path(capture_args.input_file).name
        item["file_stem"] = Path(capture_args.input_file).stem
        item["packet_index_in_file"] = index
        item["global_packet_id"] = ""
        merged.append(item)
    return merged


def json_safe(value):
    """Convert numpy/Path values into plain JSON-compatible Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def run_detector_once(capture_args, current_tb=None):
    """Run GNU Radio detector for one IQ file and return copied metadata lists."""
    tb = None
    try:
        with suppress_native_output(getattr(capture_args, "quiet_gnuradio", True)):
            tb = lora_file_preamble_fft_rx(capture_args)
            if current_tb is not None:
                current_tb[0] = tb
            tb.start()
            tb.wait()
            if current_tb is not None:
                current_tb[0] = None

            frames = sorted(tb.metadata_sink.frames, key=lambda item: (item["start_sample"], item["frame_count"]))
            headers = list(tb.header_sink.headers)
            payloads = []
            if getattr(capture_args, "require_valid_payload", False):
                crc_valid_flags = [bool(item) for item in tb.crc_valid_sink.data()]
                for index, payload in enumerate(tb.payload_sink.payloads):
                    item = dict(payload)
                    item["crc_valid"] = crc_valid_flags[index] if index < len(crc_valid_flags) else False
                    payloads.append(item)
        return frames, headers, payloads
    finally:
        if current_tb is not None:
            current_tb[0] = None
        cleanup_file_source_path(capture_args)
        tb = None
        gc.collect()


def write_detector_json(capture_args, output_path):
    """Hidden worker mode: detect one file and write metadata as JSON for the parent process."""
    frames, headers, payloads = run_detector_once(capture_args)
    data = {
        "frames": frames,
        "headers": headers,
        "payloads": payloads,
    }
    Path(output_path).write_text(json.dumps(json_safe(data), ensure_ascii=False), encoding="utf-8")


def child_detector_command(capture_args, json_path):
    """Build a one-file worker command using the same Python executable."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "-f",
        str(capture_args.input_file),
        "--detect-only-json",
        str(json_path),
        "--sf",
        str(int(capture_args.sf)),
        "--preamble-len",
        str(int(capture_args.preamble_len)),
        "--bw",
        str(float(capture_args.bw)),
        "--samp-rate",
        str(float(capture_args.samp_rate)),
        "--cr",
        str(int(capture_args.cr)),
        "--pay-len",
        str(int(capture_args.pay_len)),
        "--center-freq",
        str(float(capture_args.center_freq)),
        "--sync-word",
        hex(int(capture_args.sync_word)),
        "--ldro-mode",
        str(int(capture_args.ldro_mode)),
        "--crc-mode",
        str(int(capture_args.crc_mode)),
        "--no-print-header",
        "--print-payload",
        str(capture_args.print_payload),
    ]
    cmd.append("--has-crc" if capture_args.has_crc else "--no-crc")
    cmd.append("--soft-decoding" if capture_args.soft_decoding else "--hard-decoding")
    if capture_args.impl_head:
        cmd.append("--impl-head")
    cmd.append("--throttle" if capture_args.throttle else "--no-throttle")
    if capture_args.cfo_correct:
        cmd.append("--cfo-correct")
    if getattr(capture_args, "require_valid_payload", False):
        cmd.append("--require-valid-payload")
    if capture_args.downsample_phase is not None:
        cmd.extend(["--downsample-phase", str(int(capture_args.downsample_phase))])
    if capture_args.nfft:
        cmd.extend(["--nfft", str(int(capture_args.nfft))])
    if not getattr(capture_args, "quiet_gnuradio", True):
        cmd.append("--show-gnuradio-log")
    return cmd


def tail_lines(text, limit=40):
    lines = text.splitlines()
    return "\n".join(lines[-limit:])


def format_returncode(returncode):
    if returncode == WINDOWS_ACCESS_VIOLATION:
        return f"{returncode} (0x{returncode:08X}, Windows access violation)"
    if returncode < 0:
        return f"{returncode} (signal {-returncode})"
    return str(returncode)


def run_detector_isolated(capture_args):
    """Run one IQ file in a fresh child process to avoid GNU Radio resource buildup in batch mode."""
    attempts = max(1, int(capture_args.worker_retries) + 1)
    last_result = None
    for attempt in range(1, attempts + 1):
        with tempfile.TemporaryDirectory(prefix="lora_preamble_detect_") as temp_dir:
            json_path = Path(temp_dir) / "detected.json"
            result = subprocess.run(
                child_detector_command(capture_args, json_path),
                cwd=str(Path.cwd()),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            last_result = result
            if result.returncode == 0 and json_path.exists():
                data = json.loads(json_path.read_text(encoding="utf-8"))
                return data.get("frames", []), data.get("headers", []), data.get("payloads", [])

            if attempt < attempts:
                print(
                    f"[preamble_fft] worker failed for {capture_args.input_file} "
                    f"(exit={format_returncode(result.returncode)}), retry {attempt}/{attempts - 1}"
                )

    if last_result is None:
        return [], [], []

    if last_result.returncode == 0:
        print(f"[preamble_fft] worker produced no JSON for {capture_args.input_file}")
    else:
        print(
            f"[preamble_fft] worker failed for {capture_args.input_file} "
            f"(exit={format_returncode(last_result.returncode)})"
        )
    combined = "\n".join(part for part in (last_result.stdout, last_result.stderr) if part)
    if combined and int(capture_args.worker_log_lines) > 0:
        print(tail_lines(combined, int(capture_args.worker_log_lines)))
    elif combined:
        print("[preamble_fft] worker log suppressed; use --worker-log-lines 40 to show GNU Radio details")
    return [], [], []


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
    """计算每包保留的 RSSI/SNR 和前导码主峰局部频谱特征。"""
    packet_iq = np.asarray(iq[ranges["packet_start_sample"]:ranges["packet_end_sample"]], dtype=np.complex64)
    mask = analysis["is_preamble"]
    symbol_snrs = analysis["symbol_snr_db"][mask]
    peak_features = average_preamble_peak_features(analysis, args)

    # 平均 RSSI：先计算整个包 IQ 的平均功率，再套用 SX1276 LF/HF offset 风格转换。
    packet_power_db = mean_power_db(packet_iq)
    packet_rssi = sx1276_rssi_offset(args.center_freq) + packet_power_db if np.isfinite(packet_power_db) else float("nan")

    # 平均 SNR：先对前导码 dechirp FFT SNR 取平均，再按 SX1276 RegPktSnrValue 的 0.25 dB 口径量化。
    packet_snr_raw = float(np.nanmean(symbol_snrs)) if symbol_snrs.size else float("nan")
    packet_snr_reg = sx1276_snr_reg_value(packet_snr_raw)
    packet_snr = sx1276_style_snr_db(packet_snr_raw)
    return {
        "packet_avg_rssi_dbm": packet_rssi,
        "packet_avg_snr_db": packet_snr,
        "packet_avg_snr_reg_value": packet_snr_reg if packet_snr_reg is not None else "",
        **peak_features,
    }


def build_packet_row(frame, metrics, args):
    """生成唯一 CSV 的一行：一个数据包一行，只保留核心平均特征和必要索引。"""
    row = {
        "file_name": frame["file_name"],
        "lab_name": frame.get("lab_name", ""),
        "experiment_id": frame.get("experiment_id", ""),
        "corridor_id": frame.get("corridor_id", ""),
        "position_id": frame.get("position_id", ""),
        "tx_power_dbm": frame.get("tx_power_dbm", ""),
        "filename_sf": frame.get("filename_sf", ""),
        "filename_tx_power_dbm": frame.get("filename_tx_power_dbm", ""),
        "filename_preamble_len": frame.get("filename_preamble_len", ""),
        "header_packet_counter": frame.get("header_packet_counter", frame.get("payload_packet_number", "")),
        "packet_avg_rssi_dbm": fmt_float(metrics["packet_avg_rssi_dbm"]),
        "packet_avg_snr_db": fmt_float(metrics["packet_avg_snr_db"]),
        "packet_avg_snr_reg_value": metrics["packet_avg_snr_reg_value"],
        "preamble_peak_width_3db_bins_avg": fmt_float(metrics["preamble_peak_width_bins_avg"]),
    }
    for offset, value in zip(peak_spectrum_offsets(args), metrics["preamble_peak_spectrum"]):
        row[f"preamble_peak_mag_bin_{int(offset):+d}"] = fmt_float(value)
    return row


def save_results(args, capture_results):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 旧版本会生成这些明细 CSV；现在输出收敛后主动清掉，避免 Excel 里误看旧文件。
    for stale_name in ("rssi_samples_5ms.csv", "preamble_symbol_features.csv", "position_summary.csv"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    packet_rows = []
    npz_frames = []
    packet_avg_rssi_dbm = []
    packet_avg_snr_db = []
    packet_avg_snr_reg_value = []
    preamble_peak_spectrum = []
    preamble_peak_width_bins_avg = []

    for capture_args, frames in capture_results:
        iq_path = Path(capture_args.input_file)
        iq = np.memmap(iq_path, dtype=np.complex64, mode="r")
        for frame in frames:
            analysis = analyze_frame(iq, frame, capture_args)
            ranges = estimate_packet_ranges(iq.size, frame, capture_args)
            metrics = compute_packet_average_metrics(analysis, ranges, iq, capture_args)
            npz_frames.append(frame)
            packet_rows.append(build_packet_row(frame, metrics, capture_args))
            packet_avg_rssi_dbm.append(metrics["packet_avg_rssi_dbm"])
            packet_avg_snr_db.append(metrics["packet_avg_snr_db"])
            packet_avg_snr_reg_value.append(
                int(metrics["packet_avg_snr_reg_value"]) if metrics["packet_avg_snr_reg_value"] != "" else -9999
            )
            preamble_peak_spectrum.append(metrics["preamble_peak_spectrum"])
            preamble_peak_width_bins_avg.append(metrics["preamble_peak_width_bins_avg"])

    packet_path = output_dir / "packet_features.csv"

    # packet_features.csv：Excel 友好版，一行代表一个数据包，只保留平均 RSSI/SNR、
    # FHDR 包计数、文件名元数据和主峰附近平均幅度谱。
    packet_fields = [
        "file_name", "lab_name",
        "experiment_id", "corridor_id", "position_id", "tx_power_dbm",
        "filename_sf", "filename_tx_power_dbm", "filename_preamble_len", "header_packet_counter",
        "packet_avg_rssi_dbm", "packet_avg_snr_db", "packet_avg_snr_reg_value",
        "preamble_peak_width_3db_bins_avg",
    ]
    packet_fields.extend(f"preamble_peak_mag_bin_{int(offset):+d}" for offset in peak_spectrum_offsets(args))

    write_dict_csv(packet_path, packet_rows, packet_fields)

    npz_path = output_dir / "preamble_features.npz"
    # preamble_features.npz：Python 分析版。每个下标对应一个数据包，字段和 CSV 保持同一口径。
    # preamble_peak_spectrum[i] 是第 i 个包的主峰对齐平均局部幅度谱。
    np.savez_compressed(
        npz_path,
        file_names=np.asarray([f["file_name"] for f in npz_frames]),
        lab_names=np.asarray([f.get("lab_name", "") for f in npz_frames]),
        position_labels=np.asarray([str(f.get("position_id", "")) for f in npz_frames]),
        experiment_id=np.asarray(
            [int_or_default(f.get("experiment_id", ""), -1) for f in npz_frames],
            dtype=np.int32,
        ),
        corridor_id=np.asarray(
            [int_or_default(f.get("corridor_id", ""), -1) for f in npz_frames],
            dtype=np.int32,
        ),
        position_id=np.asarray(
            [int_or_default(f.get("position_id", ""), -1) for f in npz_frames],
            dtype=np.int32,
        ),
        tx_power_dbm=np.asarray(
            [int_or_default(f.get("tx_power_dbm", ""), -9999) for f in npz_frames],
            dtype=np.int32,
        ),
        filename_tx_power_dbm=np.asarray(
            [int_or_default(f.get("filename_tx_power_dbm", ""), -9999) for f in npz_frames],
            dtype=np.int32,
        ),
        filename_sf=np.asarray(
            [int_or_default(f.get("filename_sf", ""), -1) for f in npz_frames],
            dtype=np.int32,
        ),
        filename_preamble_len=np.asarray(
            [int_or_default(f.get("filename_preamble_len", ""), -1) for f in npz_frames],
            dtype=np.int32,
        ),
        header_packet_counter=np.asarray(
            [
                int_or_default(f.get("header_packet_counter", f.get("payload_packet_number", "")), -1)
                for f in npz_frames
            ],
            dtype=np.int32,
        ),
        packet_avg_rssi_dbm=np.asarray(packet_avg_rssi_dbm, dtype=np.float32),
        packet_avg_snr_db=np.asarray(packet_avg_snr_db, dtype=np.float32),
        packet_avg_snr_reg_value=np.asarray(packet_avg_snr_reg_value, dtype=np.int16),
        preamble_peak_width_3db_bins_avg=np.asarray(preamble_peak_width_bins_avg, dtype=np.float32),
        preamble_peak_spectrum_offsets=peak_spectrum_offsets(args),
        preamble_peak_spectrum=np.asarray(preamble_peak_spectrum, dtype=np.float32),
        normalization=args.normalize,
        part=args.part,
        peak_width_db=args.peak_width_db,
        peak_spectrum_half_width=int(args.peak_spectrum_half_width),
    )
    return {
        "npz": npz_path,
        "packet": packet_path,
        "packet_count": len(packet_rows),
    }


def build_arg_parser():
    parser = ArgumentParser(
        description="Export LoRa packet RSSI/SNR and preamble dechirp FFT features from fc32/cfile IQ captures."
    )
    parser.add_argument("-f", "--input-file", type=str, default=None, help="Input fc32/cfile IQ file.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "USRP_IQ"),
        help="Directory containing lab subdirectories with USRP IQ .bin files for --all-bin.",
    )
    parser.add_argument(
        "--all-bin",
        action="store_true",
        default=False,
        help="Process each lab subdirectory under --input-dir separately and save outputs into that lab directory.",
    )
    parser.add_argument(
        "--sf",
        "--spreading-factor",
        type=int,
        default=None,
        help="LoRa spreading factor. In --all-bin mode, omit this to use each file name's SF field.",
    )
    parser.add_argument("--bw", "--bandwidth", type=eng_float, default=125e3, help="LoRa bandwidth in Hz.")
    parser.add_argument("--samp-rate", type=eng_float, default=500e3, help="IQ sample rate in Hz.")
    parser.add_argument("--cr", "--coding-rate", type=int, default=1, help="Coding rate index, 1..4.")
    parser.add_argument("--pay-len", type=int, default=255, help="Payload length for implicit header mode.")
    parser.add_argument("--has-crc", action="store_true", default=True, help="Packet has payload CRC.")
    parser.add_argument("--no-crc", action="store_false", dest="has_crc", help="Packet has no payload CRC.")
    parser.add_argument("--impl-head", action="store_true", default=False, help="Use implicit header mode.")
    parser.add_argument("--soft-decoding", action="store_true", default=True, help="Use soft decoding.")
    parser.add_argument("--hard-decoding", action="store_false", dest="soft_decoding", help="Use hard decoding.")
    parser.add_argument("--center-freq", type=eng_float, default=487.7e6, help="RF center frequency used by frame_sync SFO estimation and SX1276-style RSSI offset.")
    parser.add_argument("--sync-word", type=lambda x: int(x, 0), default=0x34, help="LoRa sync word, decimal or 0x hex.")
    parser.add_argument("--ldro-mode", type=int, default=2, help="LDRO mode: 0 disabled, 1 enabled, 2 auto.")
    parser.add_argument(
        "--preamble-len",
        type=int,
        default=None,
        help="Expected preamble upchirp count. In --all-bin mode, omit this to use each file name's preamble field.",
    )
    parser.add_argument("--crc-mode", type=int, choices=[0, 1], default=0, help="CRC mode used only with --require-valid-payload: 0 GRLORA, 1 SX1276.")
    parser.add_argument(
        "--print-header",
        action="store_true",
        dest="print_header",
        help="Print decoded PHY header information. This is enabled by default.",
    )
    parser.add_argument(
        "--no-print-header",
        action="store_false",
        dest="print_header",
        help="Do not print decoded PHY header information.",
    )
    parser.add_argument(
        "--print-payload",
        choices=["none", "ascii", "hex"],
        default="ascii",
        help="Payload print format used only with --require-valid-payload.",
    )
    parser.add_argument(
        "--require-valid-payload",
        "--with-fcnt",
        action="store_true",
        default=False,
        help="Decode payloads, keep only packets with CRC-valid payloads, and fill FCnt when parseable. Disabled by default.",
    )
    parser.add_argument(
        "--throttle",
        action="store_true",
        dest="throttle",
        help="Throttle file playback to sample rate. This is enabled by default.",
    )
    parser.add_argument(
        "--no-throttle",
        action="store_false",
        dest="throttle",
        help="Process the file as fast as GNU Radio can schedule it.",
    )
    parser.set_defaults(throttle=True)
    parser.set_defaults(print_header=True)

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir()),
        help="Output directory for single-file mode, or fallback output directory when --all-bin has no lab subdirectories.",
    )
    parser.add_argument(
        "--part",
        choices=["preamble", "phy-header"],
        default="preamble",
        help="Analyze only preamble upchirps, or preamble + sync word + two full SFD downchirps.",
    )
    parser.add_argument(
        "--normalize",
        choices=["max", "sum", "l2", "none"],
        default="max",
        help="FFT amplitude normalization mode.",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=0,
        help="FFT length. Default is 2**SF, matching frame_sync.",
    )
    parser.add_argument(
        "--downsample-phase",
        type=int,
        default=None,
        help="Oversampled symbol decimation phase. Default is os_factor//2, matching frame_sync.",
    )
    parser.add_argument(
        "--cfo-correct",
        action="store_true",
        default=False,
        help="Apply frame_sync CFO estimate before FFT.",
    )
    parser.add_argument(
        "--peak-width-db",
        type=float,
        default=-3.0,
        help="Peak-width threshold relative to peak amplitude in dB. Default -3 dB.",
    )
    parser.add_argument(
        "--peak-spectrum-half-width",
        type=int,
        default=8,
        help="Number of FFT bins to keep on each side of the aligned preamble main peak.",
    )
    parser.add_argument(
        "--isolated-workers",
        action="store_true",
        default=False,
        help="In --all-bin mode, run each .bin in a child process so native GNU Radio crashes only skip that file.",
    )
    parser.add_argument(
        "--show-gnuradio-log",
        action="store_false",
        dest="quiet_gnuradio",
        help="Show GNU Radio/C++ stdout and stderr during direct detector runs. Hidden by default.",
    )
    parser.set_defaults(quiet_gnuradio=True)
    parser.add_argument(
        "--worker-retries",
        type=int,
        default=1,
        help="Extra retries for each --isolated-workers child process after a native crash or missing JSON.",
    )
    parser.add_argument(
        "--worker-log-lines",
        type=int,
        default=0,
        help="Print this many tail lines from a failed --isolated-workers child process. Default suppresses noisy C++ logs.",
    )
    # 已收敛输出：不再支持 per-frame 明细 CSV 和未使用的 RSSI 采样间隔参数。
    parser.add_argument(
        "--detect-only-json",
        type=str,
        default=None,
        help=SUPPRESS,
    )
    return parser


def main():
    # 解析命令行参数，决定处理单个文件，还是按 USRP_IQ 下的 lab 文件夹批量处理。
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.detect_only_json:
        if not args.input_file:
            parser.error("--detect-only-json requires -f/--input-file")
        capture_args = resolve_capture_args(args, Path(args.input_file), None)
        write_detector_json(capture_args, args.detect_only_json)
        return 0

    if args.all_bin:
        jobs = discover_lab_jobs(args)
        if not jobs:
            parser.error(f"no .bin files found in {Path(args.input_dir)} or its lab subdirectories")
    else:
        if not args.input_file:
            parser.error("either -f/--input-file or --all-bin is required")
        jobs = [
            {
                "label": Path(args.input_file).stem,
                "input_files": [Path(args.input_file)],
                "output_dir": Path(args.output_dir),
                "lab_metadata": None,
            }
        ]

    # 用列表保存当前 GNU Radio top_block，便于 Ctrl+C 时安全停止。
    current_tb = [None]

    def sig_handler(sig=None, frame=None):
        if current_tb[0] is not None:
            current_tb[0].stop()
            current_tb[0].wait()
        sys.exit(130)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    total_packets = 0
    successful_jobs = 0
    for job in jobs:
        # 每个 lab 文件夹作为一个独立任务：单独读取其中所有 bin，输出也写回该 lab 文件夹。
        job_args = copy.copy(args)
        job_args.output_dir = str(job["output_dir"])
        lab_metadata = job.get("lab_metadata")
        print(
            f"[preamble_fft] ===== job {job['label']}: "
            f"{len(job['input_files'])} file(s), output -> {job_args.output_dir} ====="
        )
        if lab_metadata and lab_metadata.get("lab_note"):
            print(f"[preamble_fft] lab note: {lab_metadata['lab_note']}")
        if lab_metadata and lab_metadata.get("override_sf") != "":
            print(f"[preamble_fft] lab note overrides SF to {lab_metadata['override_sf']}")

        capture_results = []
        for input_file in job["input_files"]:
            # 根据文件名补全实验编号、位置编号、SF、发射功率、前导码长度等信息。
            # 如果 lab 的补充.txt 写明了修正参数，会在 resolve_capture_args 里覆盖文件名参数。
            # 子进程隔离模式下，父进程只调度，不提前创建 GNU Radio file_source 的 hardlink。
            use_isolated_worker = bool(args.all_bin and args.isolated_workers)
            capture_args = resolve_capture_args(
                job_args,
                input_file,
                lab_metadata,
                prepare_source=not use_isolated_worker,
            )
            print(
                f"[preamble_fft] processing {input_file} "
                f"(sf={capture_args.sf}, preamble_len={capture_args.preamble_len})"
            )

            # 跑 gr-lora_sdr 接收链，收集每个数据包的前导码范围和 PHY header 信息。
            if use_isolated_worker:
                frames, headers, payloads = run_detector_isolated(capture_args)
            else:
                frames, headers, payloads = run_detector_once(capture_args, current_tb)

            # 合并包范围和 header；本脚本不再解 payload，包号字段保持为空。
            if not frames:
                print(f"[preamble_fft] no valid preamble ranges for {input_file}")
                continue
            merged_frames = merge_frame_and_header_metadata(frames, headers, payloads, capture_args)
            if capture_args.require_valid_payload:
                dropped = len(frames) - len(merged_frames)
                print(
                    f"[preamble_fft] kept {len(merged_frames)}/{len(frames)} "
                    f"CRC-valid payload packet(s) from {input_file.name}"
                    + (f"; dropped {dropped}" if dropped else "")
                )
                if not merged_frames:
                    continue
            capture_results.append((capture_args, merged_frames))
            print(f"[preamble_fft] collected {len(merged_frames)} packet(s) from {input_file.name}")

        if not capture_results:
            print(f"[preamble_fft] no valid packets for job {job['label']}")
            continue

        # 基于该 lab 内收集到的包位置重新读取 IQ，写出该 lab 自己的 CSV 和 NPZ。
        outputs = save_results(job_args, capture_results)
        total_packets += outputs["packet_count"]
        successful_jobs += 1
        print(f"[preamble_fft] wrote {outputs['packet_count']} packet(s) -> {outputs['packet']}, {outputs['npz']}")

    if successful_jobs == 0:
        print("[preamble_fft] no valid packets were published by frame_sync/header_decoder")
        return 1
    print(f"[preamble_fft] finished {successful_jobs} job(s), {total_packets} packet(s) total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
