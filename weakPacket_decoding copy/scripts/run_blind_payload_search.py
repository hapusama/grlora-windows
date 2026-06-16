#!/usr/bin/env python3
"""Run blind payload decoder on low-SNR packets.

This script integrates the blind payload search into the existing phase-guided
demod pipeline:

  1. Header-first decoding (may use header rescue if enabled)
  2. Header-based phase line fitting
  3. Extract payload FFT evidence
  4. Per-symbol confidence analysis
  5. Blind candidate search with LoRa code constraints
  6. CRC validation

Comparison modes:
  - argmax: baseline single-symbol argmax
  - phase_guided: existing Phase-MAP with fixed high-confidence symbols
  - blind_search: new block-wise beam search over candidate sets

Outputs per-packet results with SER, CRC-valid recovery rate, and complexity metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.blind_payload_search import (  # noqa: E402
    BlindSearchConfig,
    blind_payload_search,
)
from weak_decoder.chirp import build_downchirp, bin_to_grlora_symbol  # noqa: E402
from weak_decoder.header_first_demod import decode_explicit_header  # noqa: E402
from weak_decoder.payload_codec import decode_payload_symbols  # noqa: E402
from weak_decoder.phase_guided_demod import (  # noqa: E402
    PhaseGuidedPayloadConfig,
    extract_header_anchors,
    extract_single_fft,
    fit_phase_line,
    phase_guided_rescue_header,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run blind payload decoder on low-SNR packets"
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
        help="Header-first symbol CSV with GT bins for evaluation",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output CSV path for per-packet results",
    )
    parser.add_argument(
        "--mode",
        choices=["argmax", "phase_guided", "blind_search", "all"],
        default="blind_search",
        help="Decoder mode (default: blind_search)",
    )
    parser.add_argument(
        "--packet",
        type=int,
        help="Optional: only process specified packet_index",
    )
    # Blind search parameters
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for candidate search (default: 0.5)",
    )
    parser.add_argument(
        "--max-candidates-per-symbol",
        type=int,
        default=4,
        help="Max candidates per low-confidence symbol (default: 4)",
    )
    parser.add_argument(
        "--block-beam-width",
        type=int,
        default=64,
        help="Beam width for block-level search (default: 64)",
    )
    parser.add_argument(
        "--global-beam-width",
        type=int,
        default=16,
        help="Beam width for cross-block search (default: 16)",
    )
    # Header rescue
    parser.add_argument(
        "--enable-header-rescue",
        action="store_true",
        help="Enable phase-guided header rescue for failed header decoding",
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


def load_packet_metadata(csv_path: Path, packet_filter: int | None) -> dict[int, dict[str, Any]]:
    """Load per-packet metadata and GT symbols from header-first CSV."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    packets: dict[int, dict[str, Any]] = {}

    for row in rows:
        packet_idx = _int(row, "packet_index", -1)
        if packet_filter is not None and packet_idx != packet_filter:
            continue

        if packet_idx not in packets:
            packets[packet_idx] = {
                "packet_index": packet_idx,
                "header_start_sample": -1,
                "sf": 10,
                "os_factor": 4,
                "cfo_int": 0,
                "cfo_frac": 0.0,
                "sfo_hat": 0.0,
                "preamble_len": 8.0,
                "header_symbol_values": [],
                "payload_symbols": [],
                "gt_bins": {},
            }

        stage = row.get("stage", "")

        if stage == "header":
            stage_idx = _int(row, "stage_symbol_index", -1)
            if stage_idx == 0:
                packets[packet_idx]["header_start_sample"] = _int(row, "start_sample")
            if 0 <= stage_idx < 8:
                packets[packet_idx]["header_symbol_values"].append(_int(row, "symbol_value", 0))

            # Update packet-level params
            packets[packet_idx].update({
                "sf": _int(row, "sf", 10),
                "os_factor": _int(row, "os_factor", 4),
                "cfo_int": _int(row, "cfo_int", 0),
                "cfo_frac": _float(row, "cfo_frac", 0.0),
                "sfo_hat": _float(row, "sfo_hat", 0.0),
                "preamble_len": _float(row, "preamble_len", 8.0),
            })

        elif stage == "payload" and _int(row, "header_valid", 0) == 1:
            payload_idx = _int(row, "stage_symbol_index", -1)
            gt_bin = _int(row, "raw_fft_bin", -1)

            packets[packet_idx]["payload_symbols"].append({
                "payload_symbol_index": payload_idx,
                "start_sample": _int(row, "start_sample"),
                "gt_bin": gt_bin,
            })
            packets[packet_idx]["gt_bins"][payload_idx] = gt_bin

            # Get header decode params
            if "payload_len" not in packets[packet_idx]:
                packets[packet_idx].update({
                    "payload_len": _int(row, "payload_len", 10),
                    "cr": _int(row, "cr", 1),
                    "has_crc": bool(_int(row, "has_crc", 1)),
                    "ldro": bool(_int(row, "ldro", 0)),
                })

    # Filter packets with complete data
    valid_packets = {
        k: v for k, v in packets.items()
        if len(v["header_symbol_values"]) == 8 and len(v["payload_symbols"]) > 0
    }

    return valid_packets


def run_argmax_baseline(
    samples: np.ndarray,
    packet_meta: dict[str, Any],
) -> dict[str, Any]:
    """Run argmax baseline decoder."""
    sf = packet_meta["sf"]
    n_bins = 1 << sf
    cfo_total = float(packet_meta["cfo_int"]) + float(packet_meta["cfo_frac"])
    downchirp = build_downchirp(sf, cfo_int=packet_meta["cfo_int"], cfo_frac=packet_meta["cfo_frac"])

    argmax_symbols = []
    argmax_bins = []

    for sym_info in packet_meta["payload_symbols"]:
        try:
            spectrum = extract_single_fft(
                samples,
                sym_info["start_sample"],
                sf,
                packet_meta["os_factor"],
                downchirp,
                cfo_total,
                packet_meta["header_start_sample"],
                "continuous",
            )
            power = np.abs(spectrum) ** 2
            argmax_bin = int(np.argmax(power))
            symbol_value = bin_to_grlora_symbol(argmax_bin, sf=sf, is_header=False, ldro=packet_meta["ldro"])

            argmax_symbols.append(symbol_value)
            argmax_bins.append(argmax_bin)
        except Exception:
            break

    # Compute SER
    gt_bins = packet_meta["gt_bins"]
    correct = sum(1 for i, b in enumerate(argmax_bins) if gt_bins.get(i, -1) == b)
    ser = 1.0 - (correct / len(argmax_bins)) if argmax_bins else 1.0

    # Try to decode
    crc_valid = False
    try:
        result = decode_payload_symbols(
            argmax_symbols,
            sf=sf,
            cr=packet_meta["cr"],
            ldro=packet_meta["ldro"],
            payload_len=packet_meta["payload_len"],
            has_crc=packet_meta["has_crc"],
            crc_mode="grlora",
        )
        crc_valid = result.crc_valid
    except Exception:
        pass

    return {
        "method": "argmax",
        "symbol_count": len(argmax_symbols),
        "ser": ser,
        "crc_valid": int(crc_valid),
        "candidates_evaluated": 0,
    }


def run_blind_search(
    samples: np.ndarray,
    packet_meta: dict[str, Any],
    config: BlindSearchConfig,
) -> dict[str, Any]:
    """Run blind payload search."""
    sf = packet_meta["sf"]
    n_bins = 1 << sf
    cfo_total = float(packet_meta["cfo_int"]) + float(packet_meta["cfo_frac"])
    downchirp = build_downchirp(sf, cfo_int=packet_meta["cfo_int"], cfo_frac=packet_meta["cfo_frac"])

    # Build header-based phase line
    header_abs, header_phases = extract_header_anchors(
        samples,
        packet_meta["header_start_sample"],
        sf,
        packet_meta["os_factor"],
        packet_meta["cfo_int"],
        packet_meta["cfo_frac"],
        "continuous",
        packet_meta["preamble_len"],
        packet_meta["header_symbol_values"],
    )

    phase_line = None
    if header_abs.size >= 2:
        phase_line = fit_phase_line(header_abs, header_phases)

    if phase_line is None or phase_line.anchor_count < 2:
        return {
            "method": "blind_search",
            "error": "insufficient_anchors",
            "ser": 1.0,
            "crc_valid": 0,
        }

    # Extract payload FFT evidence
    payload_spectra = []
    payload_abs_indices = []

    for sym_info in packet_meta["payload_symbols"]:
        try:
            spectrum = extract_single_fft(
                samples,
                sym_info["start_sample"],
                sf,
                packet_meta["os_factor"],
                downchirp,
                cfo_total,
                packet_meta["header_start_sample"],
                "continuous",
            )
            payload_spectra.append(spectrum)

            abs_idx = float(packet_meta["preamble_len"]) + 12.25 + float(sym_info["payload_symbol_index"])
            payload_abs_indices.append(abs_idx)
        except Exception:
            break

    if not payload_spectra:
        return {
            "method": "blind_search",
            "error": "no_payload_symbols",
            "ser": 1.0,
            "crc_valid": 0,
        }

    # Run blind search
    candidates = blind_payload_search(
        payload_spectra,
        payload_abs_indices,
        phase_line,
        sf,
        packet_meta["cr"],
        packet_meta["ldro"],
        packet_meta["payload_len"],
        packet_meta["has_crc"],
        config,
    )

    if not candidates:
        return {
            "method": "blind_search",
            "error": "no_candidates",
            "ser": 1.0,
            "crc_valid": 0,
            "candidates_evaluated": 0,
        }

    # Use best candidate
    best = candidates[0]

    # Compute SER
    gt_bins = packet_meta["gt_bins"]
    correct = sum(1 for i, b in enumerate(best.raw_bins) if gt_bins.get(i, -1) == b)
    ser = 1.0 - (correct / len(best.raw_bins)) if best.raw_bins else 1.0

    return {
        "method": "blind_search",
        "symbol_count": len(best.symbol_values),
        "ser": ser,
        "crc_valid": int(best.crc_valid),
        "candidates_evaluated": len(candidates),
        "total_score": best.total_score,
        "hamming_valid_ratio": best.hamming_valid_ratio,
    }


def main() -> int:
    args = parse_args()

    # Load data
    print(f"Loading IQ from {args.input_iq}...")
    samples = np.fromfile(args.input_iq, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"Empty IQ file: {args.input_iq}")

    print(f"Loading packet metadata from {args.gt_csv}...")
    packets = load_packet_metadata(args.gt_csv, args.packet)
    print(f"  Loaded {len(packets)} packets")

    # Setup config
    blind_config = BlindSearchConfig(
        confidence_threshold=args.confidence_threshold,
        max_candidates_per_symbol=args.max_candidates_per_symbol,
        block_beam_width=args.block_beam_width,
        global_beam_width=args.global_beam_width,
    )

    # Process packets
    results = []

    for packet_idx in sorted(packets.keys()):
        packet_meta = packets[packet_idx]
        print(f"\nProcessing packet {packet_idx} ({len(packet_meta['payload_symbols'])} symbols)...")

        if args.mode in ["argmax", "all"]:
            print("  Running argmax baseline...")
            result = run_argmax_baseline(samples, packet_meta)
            result["packet_index"] = packet_idx
            results.append(result)
            print(f"    SER={result['ser']:.3f}, CRC={result['crc_valid']}")

        if args.mode in ["blind_search", "all"]:
            print("  Running blind search...")
            result = run_blind_search(samples, packet_meta, blind_config)
            result["packet_index"] = packet_idx
            results.append(result)
            if "error" in result:
                print(f"    Error: {result['error']}")
            else:
                print(f"    SER={result['ser']:.3f}, CRC={result['crc_valid']}, "
                      f"candidates={result['candidates_evaluated']}")

    # Write results
    print(f"\nWriting results to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if results:
        with args.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    # Summary
    print("\n=== Summary ===")
    by_method = {}
    for r in results:
        method = r["method"]
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(r)

    for method, method_results in by_method.items():
        valid_results = [r for r in method_results if "error" not in r]
        if not valid_results:
            continue

        avg_ser = sum(r["ser"] for r in valid_results) / len(valid_results)
        crc_valid_count = sum(r["crc_valid"] for r in valid_results)
        crc_recovery_rate = crc_valid_count / len(valid_results)

        print(f"\n{method}:")
        print(f"  Packets: {len(valid_results)}")
        print(f"  Avg SER: {avg_ser:.3f}")
        print(f"  CRC-valid recovery: {crc_recovery_rate:.3f} ({crc_valid_count}/{len(valid_results)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
