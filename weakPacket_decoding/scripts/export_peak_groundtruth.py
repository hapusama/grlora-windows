#!/usr/bin/env python3
"""导出 gr-lora_sdr fft_demod 内部的 peak 级 groundtruth。"""
# D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\export_peak_groundtruth.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin -o gr-lora_sdr\weakPacket_decoding\data\peak_groundtruth\0_0_0_10_14_8_peak_gt.csv --sf 10 --bw 125000 --samp-rate 500000 --cr 1 --center-freq 487.7e6 --sync-word 0x34 --preamble-len 8 --ldro-mode 2 --crc-mode 0
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WEAK_ROOT = Path(__file__).resolve().parents[1]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from noisy_iq.detector import cleanup_file_source_path, prepare_file_source_path


def parse_int_auto(value: str) -> int:
    return int(value, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "运行 gr-lora_sdr 接收链，并监听 fft_demod 的 peak_candidates "
            "消息端口，生成符号/peak 级 groundtruth CSV。"
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="raw complex64 IQ 文件。")
    parser.add_argument("-o", "--output", type=Path, required=True, help="输出 peak groundtruth CSV。")
    parser.add_argument("--sf", "--spreading-factor", type=int, default=7, help="LoRa SF，默认 7。")
    parser.add_argument("--bw", "--bandwidth", type=float, default=125000.0, help="LoRa 带宽 Hz，默认 125000。")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ 采样率 Hz，默认 500000。")
    parser.add_argument("--cr", "--coding-rate", type=int, default=1, help="编码率索引，默认 1。")
    parser.add_argument("--pay-len", type=int, default=255, help="隐式头 fallback payload 长度。")
    parser.add_argument("--has-crc", action="store_true", default=True, help="发送端带 PHY CRC，默认开启。")
    parser.add_argument("--no-crc", action="store_false", dest="has_crc", help="发送端不带 PHY CRC。")
    parser.add_argument("--impl-head", action="store_true", default=False, help="隐式头模式。")
    parser.add_argument("--soft-decoding", action="store_true", default=True, help="用 soft decoding 链路，默认开启。")
    parser.add_argument("--hard-decoding", action="store_false", dest="soft_decoding", help="使用 hard decoding 链路。")
    parser.add_argument("--center-freq", type=float, default=868.1e6, help="RF 中心频率，用于 SFO 估计，默认 868.1e6。")
    parser.add_argument("--sync-word", type=parse_int_auto, default=0x34, help="同步字，默认 0x34。")
    parser.add_argument("--preamble-len", type=int, default=16, help="前导码长度/同步触发参数，默认 16。")
    parser.add_argument("--ldro-mode", type=int, default=2, help="LDRO：0 关，1 开，2 自动。")
    parser.add_argument("--crc-mode", type=int, choices=[0, 1], default=0, help="0=GRLORA，1=SX1276。")
    parser.add_argument("--print-header", action="store_true", default=False, help="打印 header_decoder 信息。")
    parser.add_argument("--max-log-approx", action="store_true", default=True, help="soft LLR 使用 max-log 近似，默认开启。")
    parser.add_argument("--no-max-log-approx", action="store_false", dest="max_log_approx", help="soft LLR 不使用 max-log 近似。")
    return parser.parse_args()


def _pmt_value(pmt, msg, key: str, default: Any = None) -> Any:
    value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
    if pmt.is_null(value):
        return default
    try:
        return pmt.to_python(value)
    except Exception:
        return default


def _u16vector(pmt, value) -> list[int]:
    if hasattr(pmt, "is_u16vector") and pmt.is_u16vector(value):
        return [int(item) for item in pmt.u16vector_elements(value)]
    return []


def _f32vector(pmt, value) -> list[float]:
    if hasattr(pmt, "is_f32vector") and pmt.is_f32vector(value):
        return [float(item) for item in pmt.f32vector_elements(value)]
    return []


def _c32vector(pmt, value) -> dict[str, list[float]]:
    if hasattr(pmt, "is_c32vector") and pmt.is_c32vector(value):
        values = list(pmt.c32vector_elements(value))
        return {
            "real": [float(item.real) for item in values],
            "imag": [float(item.imag) for item in values],
        }
    return {"real": [], "imag": []}


def export_peak_groundtruth(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from gnuradio import blocks, gr
        import gnuradio.lora_sdr as lora_sdr
        import pmt
    except ImportError as exc:
        raise RuntimeError(
            "需要在包含 GNU Radio 和 gr-lora_sdr 的 conda 环境中运行，"
            "例如 conda activate gr-lora。"
        ) from exc

    class PeakCandidateSink(gr.basic_block):
        """收集 fft_demod 发布的逐符号 Top-K peak 候选。"""

        def __init__(self):
            gr.basic_block.__init__(self, name="peak_candidate_sink", in_sig=None, out_sig=None)
            self.records: list[dict[str, Any]] = []
            self._lock = threading.Lock()
            self.message_port_register_in(pmt.intern("peak_candidates"))
            self.set_msg_handler(pmt.intern("peak_candidates"), self.handle_peak_candidates)

        def handle_peak_candidates(self, msg):
            if not pmt.is_dict(msg):
                return
            bins_pmt = pmt.dict_ref(msg, pmt.intern("candidate_bins"), pmt.PMT_NIL)
            symbols_pmt = pmt.dict_ref(msg, pmt.intern("candidate_symbols"), pmt.PMT_NIL)
            values_pmt = pmt.dict_ref(msg, pmt.intern("candidate_values"), pmt.PMT_NIL)
            powers_pmt = pmt.dict_ref(msg, pmt.intern("candidate_powers"), pmt.PMT_NIL)
            phases_pmt = pmt.dict_ref(msg, pmt.intern("candidate_phases"), pmt.PMT_NIL)
            record = {
                "frame_count": int(_pmt_value(pmt, msg, "frame_count", -1)),
                "symbol_index": int(_pmt_value(pmt, msg, "symbol_index", -1)),
                "is_header": bool(_pmt_value(pmt, msg, "is_header", False)),
                "sf": int(_pmt_value(pmt, msg, "sf", args.sf)),
                "cr": int(_pmt_value(pmt, msg, "cr", args.cr)),
                "ldro": bool(_pmt_value(pmt, msg, "ldro", False)),
                "samples_per_symbol": int(_pmt_value(pmt, msg, "samples_per_symbol", 1 << args.sf)),
                "top_k": int(_pmt_value(pmt, msg, "top_k", 0)),
                "hard_bin": int(_pmt_value(pmt, msg, "hard_bin", -1)),
                "hard_symbol": int(_pmt_value(pmt, msg, "hard_symbol", -1)),
                "confidence_db": float(_pmt_value(pmt, msg, "confidence_db", float("nan"))),
                "total_power": float(_pmt_value(pmt, msg, "total_power", float("nan"))),
                "noise_power_est": float(_pmt_value(pmt, msg, "noise_power_est", float("nan"))),
                "cfo_int": int(_pmt_value(pmt, msg, "cfo_int", 0)),
                "cfo_frac": float(_pmt_value(pmt, msg, "cfo_frac", 0.0)),
                "candidate_bins": _u16vector(pmt, bins_pmt),
                "candidate_symbols": _u16vector(pmt, symbols_pmt),
                "candidate_values": _c32vector(pmt, values_pmt),
                "candidate_powers": _f32vector(pmt, powers_pmt),
                "candidate_phases": _f32vector(pmt, phases_pmt),
            }
            with self._lock:
                self.records.append(record)

    class PeakGroundtruthTopBlock(gr.top_block):
        """完整接收链，只额外监听 fft_demod 的 peak 候选消息。"""

        def __init__(self, file_source_path: Path):
            gr.top_block.__init__(self, "LoRa Peak Groundtruth Exporter", catch_exceptions=True)
            os_factor = int(round(float(args.samp_rate) / float(args.bw)))
            min_buf = int(os_factor * ((1 << int(args.sf)) + 2))

            self.file_source = blocks.file_source(gr.sizeof_gr_complex, str(file_source_path), False, 0, 0)
            self.file_source.set_min_output_buffer(min_buf)
            self.frame_sync = lora_sdr.frame_sync(
                int(args.center_freq),
                int(args.bw),
                int(args.sf),
                bool(args.impl_head),
                [int(args.sync_word)],
                os_factor,
                int(args.preamble_len),
            )
            self.fft_demod = lora_sdr.fft_demod(bool(args.soft_decoding), bool(args.max_log_approx))
            self.gray_mapping = lora_sdr.gray_mapping(bool(args.soft_decoding))
            self.deinterleaver = lora_sdr.deinterleaver(bool(args.soft_decoding))
            self.hamming_dec = lora_sdr.hamming_dec(bool(args.soft_decoding))
            self.header_decoder = lora_sdr.header_decoder(
                bool(args.impl_head),
                int(args.cr),
                int(args.pay_len),
                bool(args.has_crc),
                int(args.ldro_mode),
                bool(args.print_header),
            )
            self.dewhitening = lora_sdr.dewhitening()
            crc_mode = lora_sdr.Crc_mode.SX1276 if int(args.crc_mode) == 1 else lora_sdr.Crc_mode.GRLORA
            self.crc_verif = lora_sdr.crc_verif(0, False, crc_mode)
            self.peak_sink = PeakCandidateSink()

            self.connect((self.file_source, 0), (self.frame_sync, 0))
            self.connect((self.frame_sync, 0), (self.fft_demod, 0))
            self.connect((self.fft_demod, 0), (self.gray_mapping, 0))
            self.connect((self.gray_mapping, 0), (self.deinterleaver, 0))
            self.connect((self.deinterleaver, 0), (self.hamming_dec, 0))
            self.connect((self.hamming_dec, 0), (self.header_decoder, 0))
            self.connect((self.header_decoder, 0), (self.dewhitening, 0))
            self.connect((self.dewhitening, 0), (self.crc_verif, 0))
            self.msg_connect((self.header_decoder, "frame_info"), (self.frame_sync, "frame_info"))
            self.msg_connect((self.fft_demod, "peak_candidates"), (self.peak_sink, "peak_candidates"))

    staged_path = prepare_file_source_path(args.input)
    try:
        tb = PeakGroundtruthTopBlock(staged_path)
        tb.start()
        tb.wait()
        records = list(tb.peak_sink.records)
    finally:
        cleanup_file_source_path(staged_path)

    records.sort(key=lambda item: (item["frame_count"], item["symbol_index"]))
    return {
        "input_file": str(Path(args.input).resolve()),
        "format": "gr-lora_sdr fft_demod peak_candidates",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
        "record_count": len(records),
        "records": records,
    }


def write_peak_csv(path: Path, payload: dict[str, Any]) -> None:
    """把嵌套的 Top-K peak 记录展开成便于表格查看的 CSV。"""
    records = payload["records"]
    max_top_k = max((int(record.get("top_k", 0)) for record in records), default=0)
    fixed_fields = [
        "input_file",
        "frame_count",
        "symbol_index",
        "is_header",
        "sf",
        "cr",
        "ldro",
        "samples_per_symbol",
        "top_k",
        "hard_bin",
        "hard_symbol",
        "confidence_db",
        "total_power",
        "noise_power_est",
        "cfo_int",
        "cfo_frac",
    ]
    candidate_fields = []
    for rank in range(1, max_top_k + 1):
        candidate_fields.extend(
            [
                f"top{rank}_bin",
                f"top{rank}_symbol",
                f"top{rank}_real",
                f"top{rank}_imag",
                f"top{rank}_power",
                f"top{rank}_phase",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fixed_fields + candidate_fields)
        writer.writeheader()
        for record in records:
            row = {
                "input_file": payload["input_file"],
                "frame_count": record["frame_count"],
                "symbol_index": record["symbol_index"],
                "is_header": int(bool(record["is_header"])),
                "sf": record["sf"],
                "cr": record["cr"],
                "ldro": int(bool(record["ldro"])),
                "samples_per_symbol": record["samples_per_symbol"],
                "top_k": record["top_k"],
                "hard_bin": record["hard_bin"],
                "hard_symbol": record["hard_symbol"],
                "confidence_db": record["confidence_db"],
                "total_power": record["total_power"],
                "noise_power_est": record["noise_power_est"],
                "cfo_int": record["cfo_int"],
                "cfo_frac": record["cfo_frac"],
            }
            values = record.get("candidate_values", {})
            real_values = values.get("real", [])
            imag_values = values.get("imag", [])
            bins = record.get("candidate_bins", [])
            symbols = record.get("candidate_symbols", [])
            powers = record.get("candidate_powers", [])
            phases = record.get("candidate_phases", [])
            for index in range(max_top_k):
                rank = index + 1
                row[f"top{rank}_bin"] = bins[index] if index < len(bins) else ""
                row[f"top{rank}_symbol"] = symbols[index] if index < len(symbols) else ""
                row[f"top{rank}_real"] = real_values[index] if index < len(real_values) else ""
                row[f"top{rank}_imag"] = imag_values[index] if index < len(imag_values) else ""
                row[f"top{rank}_power"] = powers[index] if index < len(powers) else ""
                row[f"top{rank}_phase"] = phases[index] if index < len(phases) else ""
            writer.writerow(row)
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    payload = export_peak_groundtruth(args)
    write_peak_csv(args.output, payload)
    print(f"records={payload['record_count']}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
