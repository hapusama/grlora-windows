#!/usr/bin/env python3
"""Evaluate blind payload decoder on low-SNR LoRa packets.

This script tests the block-wise beam search decoder against low-SNR IQ,
comparing it to naive argmax decoding. It reports:
  1. Argmax payload SER
  2. Blind decoder payload SER
  3. CRC-valid packet recovery rate
  4. Complexity metrics (candidates, beam width, runtime)
  5. GT bin candidate coverage statistics

Input:
  - Low SNR IQ .bin file
  - Header-first symbol CSV with GT bins
  - Decoded header parameters

Output:
  - Per-packet results CSV
  - Summary statistics JSON
  - Comparison plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.blind_payload_decoder import (  # noqa: E402
    blind_decode_payload,
    compute_symbol_confidence,
)
from weak_decoder.chirp import build_downchirp  # noqa: E402
from weak_decoder.phase_guided_demod import (  # noqa: E402
    PhaseLine,
    extract_header_anchors,
    fit_phase_line,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate blind payload decoder on low-SNR packets"
    )
    parser.add_argument(
        "-i", "--input-iq",
        type=Path,
        required=True,
        help="Low SNR complex64 IQ .bin file",
    )
    parser.add_argument(
        "-g", "--gt-csv",
        type=Path,
        required=True,
        help="Header-first symbol CSV with GT payload bins",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output CSV path for per-packet results",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional summary JSON output path",
    )
    parser.add_argument(
        "--packet",
        type=int,
        help="Optional: only process specified packet_index",
    )

    # Decoder parameters
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for high/low classification (default: 0.6)",
    )
    parser.add_argument(
        "--amplitude-topk",
        type=int,
        default=8,
        help="Amplitude topK for candidate set (default: 8)",
    )
    parser.add_argument(
        "--argmax-window",
        type=int,
        default=4,
        help="Argmax ±W window for candidate set (default: 4)",
    )
    parser.add_argument(
        "--phase-topk",
        type=int,
        default=8,
        help="Phase-residual topK for candidate set (default: 8)",
    )
    parser.add_argument(
        "--combined-topk",
        type=int,
        default=16,
        help="Combined topK for candidate set (default: 16)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=10,
        help="Beam search width (default: 10)",
    )
    parser.add_argument(
        "--max-block-candidates",
        type=int,
        default=100,
        help="Max candidates per block (default: 100)",
    )
    parser.add_argument(
        "--phase-weight",
        type=float,
        default=0.85,
        help="Phase weight for scoring (default: 0.85)",
    )
    parser.add_argument(
        "--cfo-correction-mode",
        choices=("symbol", "continuous"),
        default="continuous",
        help="CFO correction mode (default: continuous)",
    )

    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    if value == "":
        return int(default)
    return int(float(value))


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = str(row.get(key, "")).strip()
    if value == "":
        return float(default)
    return float(value)


def load_packet_data(csv_path: Path, packet_filter: int | None) -> dict[int, dict]:
    """Load packet data from header-first CSV."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    packets = {}

    for row in rows:
        packet_idx = _int(row, "packet_index", -1)
        if packet_filter is not None and packet_idx != packet_filter:
            continue

        if packet_idx not in packets:
            packets[packet_idx] = {
                "header_symbols": [],
                "payload_symbols": [],
                "header_start_sample": None,
                "sf": _int(row, "sf", 10),
                "os_factor": _int(row, "os_factor", 4),
                "cfo_int": _int(row, "cfo_int", 0),
                "cfo_frac": _float(row, "cfo_frac", 0.0),
                "preamble_len": _float(row, "preamble_len", 8.0),
                "payload_len": _int(row, "payload_len", 0),
                "cr": _int(row, "cr", 1),
                "has_crc": bool(_int(row, "has_crc", 0)),
                "header_valid": bool(_int(row, "header_valid", 0)),
            }

        stage = row.get("stage", "")

        if stage == "header":
            stage_idx = _int(row, "stage_symbol_index", -1)
            if stage_idx == 0:
                packets[packet_idx]["header_start_sample"] = _int(row, "start_sample")
            if 0 <= stage_idx < 8:
                packets[packet_idx]["header_symbols"].append(_int(row, "symbol_value", 0))

        elif stage == "payload":
            packets[packet_idx]["payload_symbols"].append({
                "payload_symbol_index": _int(row, "stage_symbol_index", -1),
                "start_sample": _int(row, "start_sample"),
                "gt_bin": _int(row, "raw_fft_bin", -1),
            })

    # Filter valid packets
    valid_packets = {}
    for pkt_idx, pkt_data in packets.items():
        if pkt_data["header_valid"] and len(pkt_data["header_symbols"]) == 8:
            pkt_data["payload_symbols"].sort(key=lambda s: s["payload_symbol_index"])
            valid_packets[pkt_idx] = pkt_data

    return valid_packets


def _symbol_sample_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    return int(start_sample) + int(os_value // 2) + os_value * np.arange(n_bins, dtype=np.int64)


def extract_fft(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    downchirp: np.ndarray,
    cfo_total: float,
    header_start_sample: int,
    cfo_correction_mode: str,
) -> np.ndarray:
    """Extract FFT spectrum for one symbol."""
    n_bins = 1 << int(sf)
    indexes = _symbol_sample_indexes(start_sample, sf, os_factor)

    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(f"Symbol at {start_sample} exceeds IQ range")

    symbol = np.asarray(samples[indexes], dtype=np.complex64)

    if str(cfo_correction_mode) == "continuous":
        relative_chip_start = float(start_sample - header_start_sample) / float(os_factor)
        cfo_phase = float(2.0 * math.pi * float(cfo_total) * relative_chip_start / n_bins)
        symbol = (symbol * np.exp(-1j * cfo_phase)).astype(np.complex64)

    dechirped = (symbol * downchirp).astype(np.complex64)
    return np.fft.fft(dechirped).astype(np.complex64)


def compute_argmax_decoding(
    spectra: list[np.ndarray],
    gt_bins: list[int],
) -> tuple[list[int], float]:
    """Compute argmax decoding and SER."""
    argmax_bins = [int(np.argmax(np.abs(spec) ** 2)) for spec in spectra]
    errors = sum(1 for pred, gt in zip(argmax_bins, gt_bins) if pred != gt)
    ser = errors / len(gt_bins) if gt_bins else 0.0
    return argmax_bins, ser


def evaluate_packet(
    samples: np.ndarray,
    packet_data: dict,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate blind decoder on one packet."""
    sf = packet_data["sf"]
    cr = packet_data["cr"]
    n_bins = 1 << sf

    cfo_total = float(packet_data["cfo_int"]) + float(packet_data["cfo_frac"])
    downchirp = build_downchirp(sf, cfo_int=packet_data["cfo_int"], cfo_frac=packet_data["cfo_frac"])

    header_start = packet_data["header_start_sample"]

    # Extract header-based phase line
    header_abs, header_phases = extract_header_anchors(
        samples,
        header_start,
        sf,
        packet_data["os_factor"],
        packet_data["cfo_int"],
        packet_data["cfo_frac"],
        args.cfo_correction_mode,
        packet_data["preamble_len"],
        tuple(packet_data["header_symbols"]),
    )

    phase_line = PhaseLine()
    if header_abs.size >= 2:
        phase_line = fit_phase_line(header_abs, header_phases)

    # Extract payload FFT spectra
    spectra = []
    gt_bins = []

    for sym_data in packet_data["payload_symbols"]:
        try:
            spec = extract_fft(
                samples,
                sym_data["start_sample"],
                sf,
                packet_data["os_factor"],
                downchirp,
                cfo_total,
                header_start,
                args.cfo_correction_mode,
            )
            spectra.append(spec)
            gt_bins.append(sym_data["gt_bin"])
        except ValueError as e:
            print(f"  Warning: skipping symbol: {e}")
            continue

    if not spectra:
        return {"error": "No valid spectra"}

    # Argmax baseline
    argmax_bins, argmax_ser = compute_argmax_decoding(spectra, gt_bins)

    # Blind decoder
    payload_symbol_start_abs = packet_data["preamble_len"] + 12.25

    start_time = time.time()
    blind_result = blind_decode_payload(
        spectra=spectra,
        phase_line=phase_line,
        payload_symbol_start_abs_index=payload_symbol_start_abs,
        sf=sf,
        cr=cr,
        payload_len=packet_data["payload_len"],
        has_crc=packet_data["has_crc"],
        confidence_threshold=args.confidence_threshold,
        amplitude_topk=args.amplitude_topk,
        argmax_window=args.argmax_window,
        phase_topk=args.phase_topk,
        combined_topk=args.combined_topk,
        beam_width=args.beam_width,
        max_block_candidates=args.max_block_candidates,
        phase_weight=args.phase_weight,
    )
    decode_time = time.time() - start_time

    # Compute blind decoder SER
    if blind_result.success and len(blind_result.symbol_bins) == len(gt_bins):
        blind_errors = sum(1 for pred, gt in zip(blind_result.symbol_bins, gt_bins) if pred != gt)
        blind_ser = blind_errors / len(gt_bins)
    else:
        blind_ser = 1.0

    return {
        "n_symbols": len(spectra),
        "argmax_ser": argmax_ser,
        "blind_ser": blind_ser,
        "crc_valid": blind_result.crc_valid,
        "beam_rank": blind_result.beam_rank,
        "cumulative_score": blind_result.cumulative_score,
        "n_valid_hamming": blind_result.n_valid_hamming,
        "n_high_confidence": blind_result.n_high_confidence,
        "n_low_confidence": blind_result.n_low_confidence,
        "avg_candidates_per_symbol": blind_result.avg_candidates_per_symbol,
        "n_blocks": blind_result.n_blocks,
        "decode_time_ms": decode_time * 1000,
        "phase_line_slope_pi": phase_line.slope_pi,
        "phase_line_r2": phase_line.fit_r2,
    }


def main() -> int:
    args = parse_args()

    print(f"Loading IQ from {args.input_iq}...")
    samples = np.fromfile(args.input_iq, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"Empty IQ file: {args.input_iq}")

    print(f"Loading packet data from {args.gt_csv}...")
    packets = load_packet_data(args.gt_csv, args.packet)
    print(f"  Loaded {len(packets)} valid packets")

    # Evaluate each packet
    results = []
    for pkt_idx in sorted(packets.keys()):
        print(f"Processing packet {pkt_idx}...")
        pkt_data = packets[pkt_idx]

        result = evaluate_packet(samples, pkt_data, args)

        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        result["packet_index"] = pkt_idx
        result["payload_len"] = pkt_data["payload_len"]
        result["cr"] = pkt_data["cr"]
        result["has_crc"] = pkt_data["has_crc"]

        results.append(result)

        print(f"  Argmax SER: {result['argmax_ser']:.3f}")
        print(f"  Blind SER: {result['blind_ser']:.3f}")
        print(f"  CRC valid: {result['crc_valid']}")
        print(f"  High/Low confidence: {result['n_high_confidence']}/{result['n_low_confidence']}")
        print(f"  Avg candidates: {result['avg_candidates_per_symbol']:.1f}")
        print(f"  Decode time: {result['decode_time_ms']:.1f} ms")

    if not results:
        print("No packets processed successfully")
        return 1

    # Write CSV
    print(f"\nWriting results to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Compute summary
    summary = {
        "n_packets": len(results),
        "argmax_mean_ser": float(np.mean([r["argmax_ser"] for r in results])),
        "blind_mean_ser": float(np.mean([r["blind_ser"] for r in results])),
        "crc_valid_rate": float(np.mean([r["crc_valid"] for r in results])),
        "perfect_decode_rate": float(np.mean([r["blind_ser"] == 0.0 for r in results])),
        "avg_decode_time_ms": float(np.mean([r["decode_time_ms"] for r in results])),
        "avg_high_confidence": float(np.mean([r["n_high_confidence"] for r in results])),
        "avg_low_confidence": float(np.mean([r["n_low_confidence"] for r in results])),
        "avg_candidates_per_symbol": float(np.mean([r["avg_candidates_per_symbol"] for r in results])),
        "parameters": {
            "confidence_threshold": args.confidence_threshold,
            "amplitude_topk": args.amplitude_topk,
            "argmax_window": args.argmax_window,
            "phase_topk": args.phase_topk,
            "combined_topk": args.combined_topk,
            "beam_width": args.beam_width,
            "max_block_candidates": args.max_block_candidates,
            "phase_weight": args.phase_weight,
        },
    }

    if args.summary_json:
        print(f"Writing summary to {args.summary_json}...")
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Print summary
    print("\n=== Summary ===")
    print(f"Packets processed: {summary['n_packets']}")
    print(f"Argmax mean SER: {summary['argmax_mean_ser']:.3f}")
    print(f"Blind decoder mean SER: {summary['blind_mean_ser']:.3f}")
    print(f"SER improvement: {summary['argmax_mean_ser'] - summary['blind_mean_ser']:.3f}")
    print(f"CRC-valid rate: {summary['crc_valid_rate']:.3f}")
    print(f"Perfect decode rate: {summary['perfect_decode_rate']:.3f}")
    print(f"Avg decode time: {summary['avg_decode_time_ms']:.1f} ms")
    print(f"Avg candidates/symbol: {summary['avg_candidates_per_symbol']:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
