#!/usr/bin/env python3
"""GT bin candidate coverage analysis for low SNR LoRa payload symbols.

This experiment validates the hypothesis that at low SNR, even when argmax fails,
the true GT bin remains in candidate sets constructed from:
  1. Amplitude topK
  2. Argmax ±W window
  3. Phase-residual topK (relative to header-based phase line)
  4. Amplitude + phase combined topK

Input:
  - Low SNR IQ (with known SNR)
  - Header-first symbol CSV with GT bins
  - Decoded header parameters (payload_len, CR, CRC, LDRO)

Output:
  - Per-symbol candidate coverage statistics
  - GT bin rank distribution
  - Circular distance distribution (when argmax is wrong)
  - Coverage by candidate construction method
  - Per-packet summary

This validates whether single-packet blind search over candidate sets is feasible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_downchirp, signed_fft_bin  # noqa: E402
from weak_decoder.phase_guided_demod import (  # noqa: E402
    PhaseLine,
    build_candidate_set,
    extract_header_anchors,
    extract_preamble_anchors,
    fit_phase_line,
    score_candidates_phase_guided,
)


@dataclass(frozen=True)
class SymbolGroundTruth:
    """Ground truth for one payload symbol from header-first CSV."""
    packet_index: int
    payload_symbol_index: int
    start_sample: int
    header_start_sample: int
    sf: int
    os_factor: int
    cfo_int: int
    cfo_frac: float
    sfo_hat: float
    preamble_len: float
    gt_raw_fft_bin: int
    header_symbol_values: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GT bin candidate coverage at low SNR"
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
        help="Header-first symbol CSV with GT payload bins (header_valid=1)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output CSV path for per-symbol candidate coverage statistics",
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
    parser.add_argument(
        "--target-snr-db",
        type=float,
        help="Target SNR (dB) for metadata (not enforced, just recorded)",
    )
    # Candidate set construction parameters
    parser.add_argument(
        "--amplitude-topk",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
        help="Amplitude topK values to test",
    )
    parser.add_argument(
        "--argmax-window",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 8],
        help="Argmax ±W window sizes to test",
    )
    parser.add_argument(
        "--phase-topk",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="Phase-residual topK values to test",
    )
    parser.add_argument(
        "--combined-topk",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
        help="Amplitude+phase combined topK values to test",
    )
    parser.add_argument(
        "--phase-weight",
        type=float,
        default=0.85,
        help="Phase weight for combined scoring (default: 0.85)",
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


def load_gt_symbols(csv_path: Path, packet_filter: int | None) -> list[SymbolGroundTruth]:
    """Load GT payload symbols from header-first CSV."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # First pass: collect header symbols per packet
    header_symbols_by_packet: dict[int, list[int]] = defaultdict(list)
    header_start_by_packet: dict[int, int] = {}

    for row in rows:
        if row.get("stage") != "header":
            continue
        packet_idx = _int(row, "packet_index", -1)
        if packet_filter is not None and packet_idx != packet_filter:
            continue
        stage_sym_idx = _int(row, "stage_symbol_index", -1)
        if stage_sym_idx == 0:
            header_start_by_packet[packet_idx] = _int(row, "start_sample")
        if 0 <= stage_sym_idx < 8:
            header_symbols_by_packet[packet_idx].append(_int(row, "symbol_value", 0))

    # Second pass: collect payload symbols
    symbols: list[SymbolGroundTruth] = []
    for row in rows:
        if row.get("stage") != "payload":
            continue
        if _int(row, "header_valid", 0) != 1:
            continue
        packet_idx = _int(row, "packet_index", -1)
        if packet_filter is not None and packet_idx != packet_filter:
            continue

        header_start = header_start_by_packet.get(packet_idx)
        if header_start is None:
            # Estimate from frame_symbol_index
            frame_sym_idx = _int(row, "frame_symbol_index")
            samples_per_symbol = (1 << _int(row, "sf")) * _int(row, "os_factor")
            header_start = _int(row, "start_sample") - frame_sym_idx * samples_per_symbol

        header_syms = tuple(header_symbols_by_packet.get(packet_idx, []))
        if len(header_syms) != 8:
            continue  # Skip if we don't have all 8 header symbols

        symbols.append(
            SymbolGroundTruth(
                packet_index=packet_idx,
                payload_symbol_index=_int(row, "stage_symbol_index", -1),
                start_sample=_int(row, "start_sample"),
                header_start_sample=header_start,
                sf=_int(row, "sf", 10),
                os_factor=_int(row, "os_factor", 4),
                cfo_int=_int(row, "cfo_int", 0),
                cfo_frac=_float(row, "cfo_frac", 0.0),
                sfo_hat=_float(row, "sfo_hat", 0.0),
                preamble_len=_float(row, "preamble_len", 8.0),
                gt_raw_fft_bin=_int(row, "raw_fft_bin", -1),
                header_symbol_values=header_syms,
            )
        )

    symbols.sort(key=lambda s: (s.packet_index, s.payload_symbol_index))
    if not symbols:
        raise ValueError("No valid GT payload symbols found (need stage=payload, header_valid=1)")
    return symbols


def _symbol_sample_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    return int(start_sample) + int(os_value // 2) + os_value * np.arange(n_bins, dtype=np.int64)


def extract_single_fft(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    downchirp: np.ndarray,
    cfo_total: float,
    header_start_sample: int,
    cfo_correction_mode: str,
) -> np.ndarray:
    """Extract complex FFT spectrum for one symbol."""
    n_bins = 1 << int(sf)
    indexes = _symbol_sample_indexes(start_sample, sf, os_factor)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(f"Symbol at start_sample={start_sample} exceeds IQ range")

    symbol = np.asarray(samples[indexes], dtype=np.complex64)

    if str(cfo_correction_mode) == "continuous":
        relative_chip_start = float(start_sample - header_start_sample) / float(os_factor)
        cfo_phase = float(2.0 * math.pi * float(cfo_total) * relative_chip_start / n_bins)
        symbol = (symbol * np.exp(-1j * cfo_phase)).astype(np.complex64)

    dechirped = (symbol * downchirp).astype(np.complex64)
    return np.fft.fft(dechirped).astype(np.complex64)


def circular_distance(bin1: int, bin2: int, n_bins: int) -> int:
    """Compute minimum circular distance between two bins."""
    dist = abs(int(bin1) - int(bin2))
    return min(dist, n_bins - dist)


def analyze_symbol_candidates(
    spectrum: np.ndarray,
    gt_bin: int,
    predicted_phase_rad: float,
    phase_weight: float,
    amplitude_topk_values: list[int],
    argmax_window_values: list[int],
    phase_topk_values: list[int],
    combined_topk_values: list[int],
) -> dict[str, Any]:
    """Analyze candidate coverage for one symbol."""
    n_bins = spectrum.size
    power = np.abs(spectrum) ** 2
    argmax_bin = int(np.argmax(power))

    # GT bin statistics
    gt_power = float(power[gt_bin])
    sorted_indices = np.argsort(power)[::-1]
    gt_rank = int(np.where(sorted_indices == gt_bin)[0][0]) + 1
    is_argmax_correct = int(argmax_bin == gt_bin)

    # Circular distance when argmax is wrong
    circ_dist = circular_distance(gt_bin, argmax_bin, n_bins) if not is_argmax_correct else 0

    # Phase analysis
    gt_phase = float(np.angle(spectrum[gt_bin]))
    phase_residual = float(np.angle(np.exp(1j * (gt_phase - predicted_phase_rad))))
    phase_score = float(np.cos(phase_residual))

    # Argmax bin phase for comparison
    argmax_phase = float(np.angle(spectrum[argmax_bin]))
    argmax_phase_residual = float(np.angle(np.exp(1j * (argmax_phase - predicted_phase_rad))))
    argmax_phase_score = float(np.cos(argmax_phase_residual))

    # Amplitude rankings
    amp_ranks = {int(idx): rank + 1 for rank, idx in enumerate(sorted_indices)}

    # Phase-based scoring for all bins
    phases = np.angle(spectrum)
    phase_residuals = np.angle(np.exp(1j * (phases - predicted_phase_rad)))
    phase_scores = np.cos(phase_residuals)
    phase_sorted_indices = np.argsort(phase_scores)[::-1]
    phase_ranks = {int(idx): rank + 1 for rank, idx in enumerate(phase_sorted_indices)}

    # Combined scoring (amplitude + phase)
    amp_factors = power / (float(np.max(power)) + 1e-30)
    combined_scores = phase_weight * phase_scores + (1.0 - phase_weight) * amp_factors
    combined_sorted_indices = np.argsort(combined_scores)[::-1]
    combined_ranks = {int(idx): rank + 1 for rank, idx in enumerate(combined_sorted_indices)}

    # Coverage analysis
    coverage = {
        "gt_bin": gt_bin,
        "gt_rank_by_amp": gt_rank,
        "gt_rank_by_phase": phase_ranks.get(gt_bin, n_bins + 1),
        "gt_rank_by_combined": combined_ranks.get(gt_bin, n_bins + 1),
        "argmax_bin": argmax_bin,
        "is_argmax_correct": is_argmax_correct,
        "circular_distance": circ_dist,
        "gt_phase_residual": phase_residual,
        "gt_phase_score": phase_score,
        "argmax_phase_residual": argmax_phase_residual,
        "argmax_phase_score": argmax_phase_score,
        "gt_power": gt_power,
        "argmax_power": float(power[argmax_bin]),
        "total_power": float(np.sum(power)),
        "gt_energy_ratio": gt_power / float(np.sum(power)),
    }

    # Test amplitude topK coverage
    for k in amplitude_topk_values:
        coverage[f"in_amp_top{k}"] = int(gt_rank <= k)

    # Test argmax ±W window coverage
    for w in argmax_window_values:
        in_window = circular_distance(gt_bin, argmax_bin, n_bins) <= w
        coverage[f"in_argmax_window_{w}"] = int(in_window)

    # Test phase topK coverage
    for k in phase_topk_values:
        coverage[f"in_phase_top{k}"] = int(phase_ranks.get(gt_bin, n_bins + 1) <= k)

    # Test combined topK coverage
    for k in combined_topk_values:
        coverage[f"in_combined_top{k}"] = int(combined_ranks.get(gt_bin, n_bins + 1) <= k)

    return coverage


def process_packet_symbols(
    samples: np.ndarray,
    symbols: list[SymbolGroundTruth],
    phase_weight: float,
    amplitude_topk_values: list[int],
    argmax_window_values: list[int],
    phase_topk_values: list[int],
    combined_topk_values: list[int],
    cfo_correction_mode: str,
) -> list[dict[str, Any]]:
    """Process all symbols in a packet and analyze candidate coverage."""
    if not symbols:
        return []

    first = symbols[0]
    sf = first.sf
    n_bins = 1 << sf
    cfo_total = float(first.cfo_int) + float(first.cfo_frac)
    downchirp = build_downchirp(sf, cfo_int=first.cfo_int, cfo_frac=first.cfo_frac)

    # Build header-based phase line
    header_abs, header_phases = extract_header_anchors(
        samples,
        first.header_start_sample,
        sf,
        first.os_factor,
        first.cfo_int,
        first.cfo_frac,
        cfo_correction_mode,
        first.preamble_len,
        first.header_symbol_values,
    )

    phase_line = PhaseLine()
    if header_abs.size >= 2:
        phase_line = fit_phase_line(header_abs, header_phases)

    # Process each payload symbol
    results = []
    chirp_samples = n_bins * first.os_factor

    for sym in symbols:
        try:
            spectrum = extract_single_fft(
                samples,
                sym.start_sample,
                sym.sf,
                sym.os_factor,
                downchirp,
                cfo_total,
                sym.header_start_sample,
                cfo_correction_mode,
            )
        except ValueError as e:
            print(f"  Warning: skipping symbol at {sym.start_sample}: {e}")
            continue

        # Predict phase for this symbol
        abs_symbol_index = float(sym.preamble_len) + 12.25 + float(sym.payload_symbol_index)
        predicted_phase = phase_line.predict(abs_symbol_index)

        # Analyze candidates
        coverage = analyze_symbol_candidates(
            spectrum,
            sym.gt_raw_fft_bin,
            predicted_phase,
            phase_weight,
            amplitude_topk_values,
            argmax_window_values,
            phase_topk_values,
            combined_topk_values,
        )

        # Add metadata
        result = {
            "packet_index": sym.packet_index,
            "payload_symbol_index": sym.payload_symbol_index,
            "start_sample": sym.start_sample,
            "sf": sym.sf,
            "abs_symbol_index": abs_symbol_index,
            "predicted_phase": predicted_phase,
            "phase_line_slope_pi": phase_line.slope_pi,
            "phase_line_r2": phase_line.fit_r2,
            **coverage,
        }
        results.append(result)

    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write results to CSV."""
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compute_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Compute summary statistics across all symbols."""
    if not rows:
        return {}

    n_symbols = len(rows)
    n_packets = len({r["packet_index"] for r in rows})

    # Argmax accuracy
    argmax_correct = sum(r["is_argmax_correct"] for r in rows)
    argmax_accuracy = argmax_correct / n_symbols

    # GT bin rank distribution
    ranks = [r["gt_rank_by_amp"] for r in rows]

    # Circular distance distribution (only for errors)
    error_rows = [r for r in rows if not r["is_argmax_correct"]]
    circ_dists = [r["circular_distance"] for r in error_rows] if error_rows else []

    # Coverage rates by method
    coverage_rates = {}

    # Amplitude topK
    for k in args.amplitude_topk:
        key = f"in_amp_top{k}"
        if key in rows[0]:
            coverage_rates[key] = sum(r[key] for r in rows) / n_symbols

    # Argmax window
    for w in args.argmax_window:
        key = f"in_argmax_window_{w}"
        if key in rows[0]:
            coverage_rates[key] = sum(r[key] for r in rows) / n_symbols

    # Phase topK
    for k in args.phase_topk:
        key = f"in_phase_top{k}"
        if key in rows[0]:
            coverage_rates[key] = sum(r[key] for r in rows) / n_symbols

    # Combined topK
    for k in args.combined_topk:
        key = f"in_combined_top{k}"
        if key in rows[0]:
            coverage_rates[key] = sum(r[key] for r in rows) / n_symbols

    summary = {
        "n_symbols": n_symbols,
        "n_packets": n_packets,
        "argmax_accuracy": argmax_accuracy,
        "gt_rank_mean": float(np.mean(ranks)),
        "gt_rank_median": float(np.median(ranks)),
        "gt_rank_p95": float(np.percentile(ranks, 95)),
        "gt_rank_max": int(np.max(ranks)),
        "circular_distance_mean": float(np.mean(circ_dists)) if circ_dists else 0.0,
        "circular_distance_median": float(np.median(circ_dists)) if circ_dists else 0.0,
        "circular_distance_p95": float(np.percentile(circ_dists, 95)) if circ_dists else 0.0,
        "coverage_rates": coverage_rates,
        "parameters": {
            "phase_weight": args.phase_weight,
            "amplitude_topk": args.amplitude_topk,
            "argmax_window": args.argmax_window,
            "phase_topk": args.phase_topk,
            "combined_topk": args.combined_topk,
        },
    }

    return summary


def main() -> int:
    args = parse_args()

    # Load data
    print(f"Loading IQ from {args.input_iq}...")
    samples = np.fromfile(args.input_iq, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"Empty IQ file: {args.input_iq}")

    print(f"Loading GT symbols from {args.gt_csv}...")
    symbols = load_gt_symbols(args.gt_csv, args.packet)
    print(f"  Loaded {len(symbols)} payload symbols from {len({s.packet_index for s in symbols})} packets")

    # Group by packet
    by_packet: dict[int, list[SymbolGroundTruth]] = defaultdict(list)
    for sym in symbols:
        by_packet[sym.packet_index].append(sym)

    # Process each packet
    all_results = []
    for packet_idx in sorted(by_packet.keys()):
        packet_symbols = by_packet[packet_idx]
        print(f"Processing packet {packet_idx} ({len(packet_symbols)} symbols)...")

        results = process_packet_symbols(
            samples,
            packet_symbols,
            args.phase_weight,
            args.amplitude_topk,
            args.argmax_window,
            args.phase_topk,
            args.combined_topk,
            args.cfo_correction_mode,
        )
        all_results.extend(results)

    # Write outputs
    print(f"Writing results to {args.output}...")
    write_csv(args.output, all_results)

    # Compute and write summary
    summary = compute_summary(all_results, args)
    if args.summary_json:
        print(f"Writing summary to {args.summary_json}...")
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Print key findings
    print("\n=== Key Findings ===")
    print(f"Argmax accuracy: {summary['argmax_accuracy']:.3f}")
    print(f"GT bin rank: mean={summary['gt_rank_mean']:.1f}, "
          f"median={summary['gt_rank_median']:.0f}, "
          f"p95={summary['gt_rank_p95']:.0f}")

    print("\nCandidate coverage rates:")
    for method, rate in sorted(summary["coverage_rates"].items()):
        print(f"  {method}: {rate:.3f}")

    if summary['argmax_accuracy'] < 1.0:
        print(f"\nWhen argmax is wrong:")
        print(f"  Circular distance: mean={summary['circular_distance_mean']:.1f}, "
              f"median={summary['circular_distance_median']:.0f}, "
              f"p95={summary['circular_distance_p95']:.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
