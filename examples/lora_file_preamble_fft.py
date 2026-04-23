#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# Offline LoRa preamble FFT magnitude exporter for fc32/cfile IQ captures.
#
# Example:
# python .\gr-lora_sdr\examples\lora_file_preamble_fft.py `
#   -f .\gr-lora_sdr\data\USRP_IQ\0_0_0_10_6_16.bin `
#   --sf 10 --bw 125000 --samp-rate 500000 --cr 1 `
#   --center-freq 487.7e6 --sync-word 0x34 --preamble-len 16 `
#   --ldro-mode 2 --crc-mode 0 --write-csv
# python .\gr-lora_sdr\examples\lora_file_preamble_fft.py -f .\gr-lora_sdr\data\USRP_IQ\0_0_0_10_6_16.bin --sf 10 --bw 125000 --samp-rate 500000 --cr 1 --center-freq 487.7e6 --sync-word 0x34 --preamble-len 16 --ldro-mode 2 --crc-mode 0 --write-csv


import csv
import signal
import sys
import threading
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pmt
from gnuradio import blocks
from gnuradio import gr
from gnuradio.eng_arg import eng_float

import gnuradio.lora_sdr as lora_sdr


class preamble_metadata_sink(gr.basic_block):
    """Collect frame_sync preamble messages."""

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
        print(
            f"[preamble_fft] detected frame {frame['frame_count']} "
            f"samples {frame['start_sample']}:{frame['end_sample']}"
        )


class lora_file_preamble_fft_rx(gr.top_block):
    """Run the same file RX chain far enough to obtain frame_sync preamble ranges."""

    def __init__(self, args):
        gr.top_block.__init__(self, "LoRa File Preamble FFT", catch_exceptions=True)

        self.input_file = args.input_file
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
            self.input_file,
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
        self.dewhitening = lora_sdr.dewhitening()
        crc_mode = lora_sdr.Crc_mode.SX1276 if args.crc_mode == 1 else lora_sdr.Crc_mode.GRLORA
        print_payload = {
            "none": 0,
            "ascii": 1,
            "hex": 2,
        }[args.print_payload]
        self.crc_verif = lora_sdr.crc_verif(
            print_payload,
            False,
            crc_mode,
        )
        self.metadata_sink = preamble_metadata_sink()

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
        self.connect((self.header_decoder, 0), (self.dewhitening, 0))
        self.connect((self.dewhitening, 0), (self.crc_verif, 0))

        self.msg_connect((self.header_decoder, "frame_info"), (self.frame_sync, "frame_info"))
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


def symbol_plan(part, preamble_len):
    if part == "preamble":
        return [("preamble_upchirp", "up")] * preamble_len
    if part == "phy-header":
        return (
            [("preamble_upchirp", "up")] * preamble_len
            + [("sync_word", "up")] * 2
            + [("sfd_downchirp", "down")] * 2
        )
    raise ValueError(f"unsupported part: {part}")


def analyze_frame(iq, frame, args):
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
    raw_peak = np.zeros(len(plan), dtype=np.float32)
    peak_bins = np.zeros(len(plan), dtype=np.int32)
    symbol_starts = np.zeros(len(plan), dtype=np.uint64)
    symbol_types = []

    frame_start = int(frame["start_sample"])
    for symbol_index, (symbol_type, chirp_kind) in enumerate(plan):
        symbol_start = frame_start + symbol_index * samples_per_symbol
        symbol_starts[symbol_index] = symbol_start
        symbol_types.append(symbol_type)

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
        peak_bins[symbol_index] = int(np.argmax(mag))
        raw_peak[symbol_index] = float(mag[peak_bins[symbol_index]])
        magnitudes[symbol_index, :] = normalize_magnitude(mag, args.normalize)

    return {
        "magnitudes": magnitudes,
        "raw_peak": raw_peak,
        "peak_bins": peak_bins,
        "symbol_starts": symbol_starts,
        "symbol_types": np.asarray(symbol_types),
        "downsample_phase": downsample_phase,
        "os_factor": os_factor,
        "fft_n": fft_n,
    }


def write_summary_csv(path, frames, analyses):
    fieldnames = [
        "frame_count",
        "start_sample",
        "end_sample",
        "sf",
        "bw",
        "sample_rate",
        "samples_per_symbol",
        "preamble_len",
        "snr_db",
        "cfo",
        "sto",
        "sfo",
        "netid1",
        "netid2",
        "symbol_count",
        "downsample_phase",
        "os_factor",
        "fft_n",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame, analysis in zip(frames, analyses):
            row = {key: frame.get(key, "") for key in fieldnames}
            row["symbol_count"] = int(analysis["magnitudes"].shape[0])
            row["downsample_phase"] = int(analysis["downsample_phase"])
            row["os_factor"] = int(analysis["os_factor"])
            row["fft_n"] = int(analysis["fft_n"])
            writer.writerow(row)


def write_frame_csv(path, frame, analysis):
    magnitudes = analysis["magnitudes"]
    bin_columns = [f"bin_{i}" for i in range(magnitudes.shape[1])]
    fieldnames = [
        "frame_count",
        "symbol_index",
        "symbol_type",
        "symbol_start_sample",
        "peak_bin",
        "peak_raw_mag",
    ] + bin_columns

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for symbol_index in range(magnitudes.shape[0]):
            row = [
                frame["frame_count"],
                symbol_index,
                analysis["symbol_types"][symbol_index],
                int(analysis["symbol_starts"][symbol_index]),
                int(analysis["peak_bins"][symbol_index]),
                f"{float(analysis['raw_peak'][symbol_index]):.9g}",
            ]
            row.extend(f"{float(v):.9g}" for v in magnitudes[symbol_index])
            writer.writerow(row)


def save_results(args, frames):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iq_path = Path(args.input_file)
    iq = np.memmap(iq_path, dtype=np.complex64, mode="r")

    analyses = []
    for frame in frames:
        analysis = analyze_frame(iq, frame, args)
        analyses.append(analysis)
        if args.write_csv:
            csv_path = output_dir / f"frame_{int(frame['frame_count']):03d}_preamble_fft.csv"
            write_frame_csv(csv_path, frame, analysis)

    magnitudes = np.stack([a["magnitudes"] for a in analyses], axis=0)
    peak_bins = np.stack([a["peak_bins"] for a in analyses], axis=0)
    raw_peak = np.stack([a["raw_peak"] for a in analyses], axis=0)
    symbol_starts = np.stack([a["symbol_starts"] for a in analyses], axis=0)

    npz_path = output_dir / "preamble_fft_magnitudes.npz"
    np.savez_compressed(
        npz_path,
        magnitudes=magnitudes,
        peak_bins=peak_bins,
        raw_peak=raw_peak,
        symbol_starts=symbol_starts,
        frame_counts=np.asarray([f["frame_count"] for f in frames], dtype=np.int32),
        frame_start_samples=np.asarray([f["start_sample"] for f in frames], dtype=np.uint64),
        frame_end_samples=np.asarray([f["end_sample"] for f in frames], dtype=np.uint64),
        snr_db=np.asarray([f["snr_db"] for f in frames], dtype=np.float32),
        cfo=np.asarray([f["cfo"] for f in frames], dtype=np.float32),
        sto=np.asarray([f["sto"] for f in frames], dtype=np.float32),
        sfo=np.asarray([f["sfo"] for f in frames], dtype=np.float32),
        netid1=np.asarray([f["netid1"] for f in frames], dtype=np.int32),
        netid2=np.asarray([f["netid2"] for f in frames], dtype=np.int32),
        symbol_types=analyses[0]["symbol_types"],
        normalization=args.normalize,
        part=args.part,
    )

    summary_path = output_dir / "preamble_fft_summary.csv"
    write_summary_csv(summary_path, frames, analyses)
    return npz_path, summary_path


def build_arg_parser():
    parser = ArgumentParser(
        description="Export normalized FFT magnitudes for each detected LoRa packet preamble symbol."
    )
    parser.add_argument("-f", "--input-file", type=str, required=True, help="Input fc32/cfile IQ file.")
    parser.add_argument("--sf", "--spreading-factor", type=int, default=7, help="LoRa spreading factor.")
    parser.add_argument("--bw", "--bandwidth", type=eng_float, default=125e3, help="LoRa bandwidth in Hz.")
    parser.add_argument("--samp-rate", type=eng_float, default=500e3, help="IQ sample rate in Hz.")
    parser.add_argument("--cr", "--coding-rate", type=int, default=1, help="Coding rate index, 1..4.")
    parser.add_argument("--pay-len", type=int, default=255, help="Payload length for implicit header mode.")
    parser.add_argument("--has-crc", action="store_true", default=True, help="Packet has payload CRC.")
    parser.add_argument("--no-crc", action="store_false", dest="has_crc", help="Packet has no payload CRC.")
    parser.add_argument("--impl-head", action="store_true", default=False, help="Use implicit header mode.")
    parser.add_argument("--soft-decoding", action="store_true", default=True, help="Use soft decoding.")
    parser.add_argument("--hard-decoding", action="store_false", dest="soft_decoding", help="Use hard decoding.")
    parser.add_argument("--center-freq", type=eng_float, default=868.1e6, help="RF center frequency used by frame_sync SFO estimation.")
    parser.add_argument("--sync-word", type=lambda x: int(x, 0), default=0x34, help="LoRa sync word, decimal or 0x hex.")
    parser.add_argument("--ldro-mode", type=int, default=2, help="LDRO mode: 0 disabled, 1 enabled, 2 auto.")
    parser.add_argument("--preamble-len", type=int, default=16, help="Expected preamble upchirp count.")
    parser.add_argument("--crc-mode", type=int, choices=[0, 1], default=0, help="CRC mode: 0 GRLORA, 1 SX1276.")
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
        help="Payload print format used by crc_verif. Default matches lora_file_RX.py.",
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
        default=str(Path(__file__).resolve().parent / "preamble_fft"),
        help="Output directory.",
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
        "--write-csv",
        action="store_true",
        default=False,
        help="Also write one wide CSV matrix per frame. NPZ and summary CSV are always written.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    tb = lora_file_preamble_fft_rx(args)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(130)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.wait()

    frames = sorted(tb.metadata_sink.frames, key=lambda item: (item["start_sample"], item["frame_count"]))
    if not frames:
        print("[preamble_fft] no valid preamble ranges were published by frame_sync/header_decoder")
        return 1

    npz_path, summary_path = save_results(args, frames)
    print(f"[preamble_fft] wrote {len(frames)} frame(s)")
    print(f"[preamble_fft] npz: {npz_path}")
    print(f"[preamble_fft] summary: {summary_path}")
    if args.write_csv:
        print(f"[preamble_fft] per-frame CSV files are in: {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
