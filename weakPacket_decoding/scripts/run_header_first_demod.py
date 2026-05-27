#!/usr/bin/env python3
"""从 gr-lora 风格 framesync 候选开始，执行 header-first FFT demod。"""
# D:\mysoft2\miniconda3\envs\gr-lora\python.exe weakPacket_decoding\scripts\run_header_first_demod.py -i data\USRP_IQ\0_0_0_10_14_16.bin -s weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_16_sync_chain_stft.csv -o weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_16_header_first_symbols.csv --frames-output weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_16_header_first_frames.csv --sf 10 --bw 125000 --samp-rate 500000 --ldro-mode 2
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[1]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.header_first_demod import (  # noqa: E402
    HeaderDecodeResult,
    SymbolDemodResult,
    decode_explicit_header,
    demod_symbol_sequence,
)
from weak_decoder.preamble_detector import load_complex64_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 run_weak_sync_chain.py 输出的 gr-lora framesync 候选，"
            "只对 grlora_framesync_valid==1 的 frame 做 header-first FFT demod。"
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="raw complex64 IQ 文件。")
    parser.add_argument("-s", "--sync-csv", type=Path, required=True, help="run_weak_sync_chain.py 输出的 sync_chain CSV。")
    parser.add_argument("-o", "--output", type=Path, required=True, help="逐 symbol FFT peak 输出 CSV。")
    parser.add_argument("--frames-output", type=Path, default=None, help="可选：逐 frame header 解码摘要 CSV。")
    parser.add_argument("--sf", type=int, default=10, help="LoRa SF，默认 10。")
    parser.add_argument("--bw", type=float, default=125000.0, help="LoRa BW Hz，默认 125000。")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ 采样率 Hz，默认 500000。")
    parser.add_argument("--ldro-mode", type=int, default=2, help="LDRO 模式：0 关，1 开，2 自动，默认 2。")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少个有效 framesync 候选。")
    parser.add_argument(
        "--include-invalid-header",
        action="store_true",
        default=False,
        help="header checksum 失败时仍保留该 frame 的 8 个 header symbol 输出。",
    )
    return parser.parse_args()


def _to_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return int(default)
    return int(float(value))


def _to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return float(default)
    return float(value)


def _valid_flag(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "0")).strip() in {"1", "true", "True"}


def _header_start_sample(row: dict[str, str], os_factor: int) -> int:
    """优先使用 gr-lora 精同步后的 data 起点；这个位置就是 PHY header 第 0 个 symbol 起点。"""

    for key in ("header_start_sample", "grlora_fine_payload_start_sample"):
        if row.get(key, "") != "":
            return _to_int(row, key)
    if row.get("grlora_synced_payload_start_sample", "") != "":
        synced_start = _to_int(row, "grlora_synced_payload_start_sample")
        cfo_int = _to_int(row, "grlora_cfo_int_est")
        netid_offset = _to_int(row, "grlora_netid_offset") if _valid_flag(row, "grlora_netid_valid") else 0
        sto_correction = _to_int(row, "grlora_payload_sto_sample_correction")
        return int(synced_start + int(os_factor) * cfo_int - int(os_factor) * netid_offset - sto_correction)
    if row.get("located_payload_start_sample", "") != "":
        return _to_int(row, "located_payload_start_sample")
    raise ValueError("sync CSV row does not contain a usable header start sample.")


def load_valid_sync_rows(path: Path, max_frames: int | None, os_factor: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if _valid_flag(row, "grlora_framesync_valid")]
    rows.sort(key=lambda item: _header_start_sample(item, os_factor=os_factor))
    if max_frames is not None:
        rows = rows[: int(max_frames)]
    return rows


def _join_ints(values: tuple[int, ...] | list[int]) -> str:
    return " ".join(str(int(item)) for item in values)


def frame_summary_row(
    source_row: dict[str, str],
    frame_index: int,
    header_start_sample: int,
    header: HeaderDecodeResult | None,
    error: str = "",
) -> dict[str, object]:
    base = {
        "frame_index": int(frame_index),
        "packet_index": source_row.get("packet_index", ""),
        "event_index": source_row.get("event_index", ""),
        "header_start_sample": int(header_start_sample),
        "source_grlora_framesync_valid": int(_valid_flag(source_row, "grlora_framesync_valid")),
        "source_grlora_netid_valid": source_row.get("grlora_netid_valid", ""),
        "source_grlora_cfo_int": source_row.get("grlora_cfo_int_est", ""),
        "source_grlora_cfo_frac": source_row.get("grlora_cfo_frac_est", ""),
        "source_grlora_payload_sto_frac": source_row.get("grlora_payload_sto_frac_est", ""),
        "source_grlora_sfo_hat": source_row.get("grlora_sfo_hat", ""),
        "error": error,
    }
    if header is None:
        base.update(
            {
                "header_valid": 0,
                "payload_len": "",
                "cr": "",
                "has_crc": "",
                "ldro": "",
                "payload_symbol_count": "",
                "total_symbol_count": "",
                "header_checksum_received": "",
                "header_checksum_computed": "",
                "header_error": "",
                "header_nibbles": "",
                "gray_symbols": "",
                "codewords": "",
                "decoded_nibbles": "",
            }
        )
        return base

    base.update(
        {
            "header_valid": int(header.header_valid),
            "payload_len": int(header.payload_len),
            "cr": int(header.cr),
            "has_crc": int(header.has_crc),
            "ldro": int(header.ldro),
            "payload_symbol_count": int(header.payload_symbol_count),
            "total_symbol_count": int(header.total_symbol_count),
            "header_checksum_received": int(header.header_checksum_received),
            "header_checksum_computed": int(header.header_checksum_computed),
            "header_error": int(header.header_error),
            "header_nibbles": _join_ints(header.header_nibbles),
            "gray_symbols": _join_ints(header.gray_symbols),
            "codewords": _join_ints(header.codewords),
            "decoded_nibbles": _join_ints(header.decoded_nibbles),
        }
    )
    return base


def symbol_row(
    source_row: dict[str, str],
    frame_index: int,
    header: HeaderDecodeResult | None,
    symbol: SymbolDemodResult,
    sf: int,
    bw: float,
    os_factor: int,
) -> dict[str, object]:
    return {
        "frame_index": int(frame_index),
        "packet_index": source_row.get("packet_index", ""),
        "event_index": source_row.get("event_index", ""),
        "stage": symbol.stage,
        "frame_symbol_index": symbol.frame_symbol_index,
        "stage_symbol_index": symbol.stage_symbol_index,
        "start_sample": symbol.start_sample,
        "sf": int(sf),
        "bw": float(bw),
        "os_factor": int(os_factor),
        "cfo_int": source_row.get("grlora_cfo_int_est", ""),
        "cfo_frac": source_row.get("grlora_cfo_frac_est", ""),
        "sto_frac": source_row.get("grlora_payload_sto_frac_est", ""),
        "sfo_hat": source_row.get("grlora_sfo_hat", ""),
        "sfo_cum_before": symbol.sfo_cum_before,
        "sfo_sample_adjust_after": symbol.sfo_sample_adjust_after,
        "raw_fft_bin": symbol.raw_fft_bin,
        "signed_fft_bin": symbol.signed_fft_bin,
        "symbol_value": symbol.symbol_value,
        "peak_real": symbol.peak_real,
        "peak_imag": symbol.peak_imag,
        "peak_amp": symbol.peak_amp,
        "peak_power": symbol.peak_power,
        "peak_phase": symbol.peak_phase,
        "peak_margin_db": symbol.peak_margin_db,
        "total_power": symbol.total_power,
        "header_valid": "" if header is None else int(header.header_valid),
        "payload_len": "" if header is None else header.payload_len,
        "payload_cr": "" if header is None else header.cr,
        "payload_has_crc": "" if header is None else int(header.has_crc),
        "payload_ldro": "" if header is None else int(header.ldro),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    ratio = float(args.samp_rate) / float(args.bw)
    os_factor = int(round(ratio))
    if os_factor <= 0 or abs(ratio - os_factor) > 1e-6:
        raise ValueError(f"--samp-rate / --bw must be an integer, got {ratio}.")

    samples = load_complex64_file(args.input)
    sync_rows = load_valid_sync_rows(args.sync_csv, args.max_frames, os_factor=os_factor)
    symbol_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []

    for frame_index, row in enumerate(sync_rows):
        frame_sf = _to_int(row, "sf", args.sf)
        frame_bw = _to_float(row, "bw", args.bw)
        frame_os_factor = _to_int(row, "os_factor", os_factor)
        header_start = _header_start_sample(row, os_factor=frame_os_factor)
        cfo_int = _to_int(row, "grlora_cfo_int_est")
        cfo_frac = _to_float(row, "grlora_cfo_frac_est")
        sfo_hat = _to_float(row, "grlora_sfo_hat")
        sfo_cum_initial = _to_float(row, "grlora_sfo_cum_initial")

        try:
            header_symbols = demod_symbol_sequence(
                samples=samples,
                header_start_sample=header_start,
                sf=frame_sf,
                os_factor=frame_os_factor,
                cfo_int=cfo_int,
                cfo_frac=cfo_frac,
                sfo_hat=sfo_hat,
                sfo_cum_initial=sfo_cum_initial,
                header_count=8,
                payload_count=0,
                payload_ldro=False,
            )
            header = decode_explicit_header(
                [item.symbol_value for item in header_symbols],
                sf=frame_sf,
                bw=frame_bw,
                ldro_mode=args.ldro_mode,
            )
            frame_rows.append(frame_summary_row(row, frame_index, header_start, header))
            if header.header_valid:
                symbols = demod_symbol_sequence(
                    samples=samples,
                    header_start_sample=header_start,
                    sf=frame_sf,
                    os_factor=frame_os_factor,
                    cfo_int=cfo_int,
                    cfo_frac=cfo_frac,
                    sfo_hat=sfo_hat,
                    sfo_cum_initial=sfo_cum_initial,
                    header_count=8,
                    payload_count=header.payload_symbol_count,
                    payload_ldro=header.ldro,
                )
                for symbol in symbols:
                    symbol_rows.append(symbol_row(row, frame_index, header, symbol, frame_sf, frame_bw, frame_os_factor))
            elif args.include_invalid_header:
                for symbol in header_symbols:
                    symbol_rows.append(symbol_row(row, frame_index, header, symbol, frame_sf, frame_bw, frame_os_factor))
        except Exception as exc:
            frame_rows.append(frame_summary_row(row, frame_index, header_start, None, error=f"{type(exc).__name__}: {exc}"))

    symbol_fields = [
        "frame_index",
        "packet_index",
        "event_index",
        "stage",
        "frame_symbol_index",
        "stage_symbol_index",
        "start_sample",
        "sf",
        "bw",
        "os_factor",
        "cfo_int",
        "cfo_frac",
        "sto_frac",
        "sfo_hat",
        "sfo_cum_before",
        "sfo_sample_adjust_after",
        "raw_fft_bin",
        "signed_fft_bin",
        "symbol_value",
        "peak_real",
        "peak_imag",
        "peak_amp",
        "peak_power",
        "peak_phase",
        "peak_margin_db",
        "total_power",
        "header_valid",
        "payload_len",
        "payload_cr",
        "payload_has_crc",
        "payload_ldro",
    ]
    frame_fields = [
        "frame_index",
        "packet_index",
        "event_index",
        "header_start_sample",
        "source_grlora_framesync_valid",
        "source_grlora_netid_valid",
        "source_grlora_cfo_int",
        "source_grlora_cfo_frac",
        "source_grlora_payload_sto_frac",
        "source_grlora_sfo_hat",
        "header_valid",
        "payload_len",
        "cr",
        "has_crc",
        "ldro",
        "payload_symbol_count",
        "total_symbol_count",
        "header_checksum_received",
        "header_checksum_computed",
        "header_error",
        "header_nibbles",
        "gray_symbols",
        "codewords",
        "decoded_nibbles",
        "error",
    ]
    write_csv(args.output, symbol_rows, symbol_fields)
    frames_output = args.frames_output or args.output.with_name(args.output.stem + "_frames.csv")
    write_csv(frames_output, frame_rows, frame_fields)

    valid_headers = sum(int(row.get("header_valid", 0)) for row in frame_rows)
    payload_rows = sum(1 for row in symbol_rows if row.get("stage") == "payload")
    print(f"framesync_candidates={len(sync_rows)}")
    print(f"header_valid={valid_headers}/{len(frame_rows)}")
    print(f"symbol_rows={len(symbol_rows)}")
    print(f"payload_rows={payload_rows}")
    print(f"wrote={args.output}")
    print(f"wrote_frames={frames_output}")


if __name__ == "__main__":
    main()
