#!/usr/bin/env python3
"""Evaluate phase-aware bin metrics for low-SNR candidate recall.

This script does not decode payload bytes.  It answers a narrower question:
given full FFT evidence after weak sync + header-first, can a phase-aware bin
score place the true raw FFT bin into Top-L more often than amplitude argmax?
GT raw bins are used only for this offline metric evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_two_stage_weak_decoder import extract_fft, load_packets  # noqa: E402
from weak_decoder.chirp import build_downchirp  # noqa: E402
from weak_decoder.phase_guided_demod import (  # noqa: E402
    PhaseLine,
    estimate_preamble_phase_profile,
    extract_header_anchors,
    extract_single_dechirped,
    fit_phase_line,
    score_candidates_preamble_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate phase-aware Top-L bin recall.")
    parser.add_argument("-i", "--input-iq", type=Path, required=True)
    parser.add_argument("-s", "--symbol-csv", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--amplitude-floor-db", type=float, default=24.0)
    parser.add_argument(
        "--line-source",
        choices=("header", "payload-argmax"),
        default="header",
        help="phase line source used by the phase-aware bin metric",
    )
    parser.add_argument("--anchor-top-fraction", type=float, default=0.60)
    parser.add_argument("--anchor-min-margin-db", type=float, default=0.0)
    parser.add_argument("--anchor-trim-frac", type=float, default=0.25)
    parser.add_argument(
        "--phase-weights",
        default="0,0.1,0.2,0.35,0.5,0.75,1.0,1.5,2.0",
        help="comma-separated weights for score=amp+w*cos(phase residual)",
    )
    parser.add_argument(
        "--profile-weights",
        default="0",
        help="comma-separated weights for preamble-profile corrected FFT score",
    )
    parser.add_argument(
        "--top-l",
        default="1,4,8,16,32,64,128",
        help="comma-separated Top-L recall cutoffs",
    )
    return parser.parse_args()


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(float(item.strip())) for item in str(text).split(",") if item.strip()]


def _rank_of(scores: np.ndarray, target_bin: int) -> int:
    target = int(target_bin)
    if target < 0 or target >= scores.size:
        return 0
    order = np.argsort(scores)[::-1]
    hit = np.where(order == target)[0]
    return int(hit[0] + 1) if hit.size else 0


def _packet_phase_line(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> PhaseLine:
    header_abs, header_phases = extract_header_anchors(
        samples=samples,
        header_start_sample=int(packet["header_start_sample"]),
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        cfo_correction_mode=str(args.cfo_correction_mode),
        preamble_len=float(packet.get("preamble_len", args.preamble_len)),
        header_symbol_values=tuple(packet["header_symbols"]),
    )
    if header_abs.size >= 2:
        return fit_phase_line(header_abs, header_phases)
    return PhaseLine()


def _payload_argmax_phase_line(
    packet: dict[str, Any],
    spectra: list[np.ndarray],
    args: argparse.Namespace,
) -> PhaseLine:
    anchors: list[tuple[float, float, float]] = []
    for symbol_index, spectrum in enumerate(spectra):
        spec = np.asarray(spectrum, dtype=np.complex64)
        power = np.abs(spec).astype(np.float64) ** 2
        if power.size < 2:
            continue
        order = np.argsort(power)[::-1]
        top1 = int(order[0])
        top2 = int(order[1])
        margin_db = 10.0 * math.log10(
            float(power[top1] + 1e-30) / float(power[top2] + 1e-30)
        )
        if margin_db < float(args.anchor_min_margin_db):
            continue
        abs_index = float(packet.get("preamble_len", args.preamble_len)) + 12.25 + float(symbol_index)
        anchors.append((abs_index, float(np.angle(spec[top1])), float(margin_db)))

    if len(anchors) < 2:
        return PhaseLine()
    anchors.sort(key=lambda item: item[2], reverse=True)
    keep_count = max(2, int(math.ceil(len(anchors) * max(0.05, min(1.0, float(args.anchor_top_fraction))))))
    kept = anchors[:keep_count]
    x = np.asarray([item[0] for item in kept], dtype=np.float64)
    phases = np.unwrap(np.asarray([item[1] for item in kept], dtype=np.float64))
    weights = np.asarray([max(0.1, item[2]) for item in kept], dtype=np.float64)
    return fit_phase_line(
        x,
        phases,
        trim_frac=max(0.0, min(0.45, float(args.anchor_trim_frac))),
        weights=weights,
    )


def _payload_evidence(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    sf = int(packet["sf"])
    cfo_total = float(packet["cfo_int"]) + float(packet["cfo_frac"])
    downchirp = build_downchirp(sf, cfo_int=packet["cfo_int"], cfo_frac=packet["cfo_frac"])
    spectra: list[np.ndarray] = []
    dechirped_symbols: list[np.ndarray] = []
    gt_bins: list[int] = []
    for symbol in packet["payload_symbols"]:
        try:
            dechirped = extract_single_dechirped(
                samples=samples,
                start_sample=int(symbol["start_sample"]),
                sf=sf,
                os_factor=int(packet["os_factor"]),
                downchirp=downchirp,
                cfo_total=cfo_total,
                header_start_sample=int(packet["header_start_sample"]),
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
            spectra.append(
                np.fft.fft(np.asarray(dechirped, dtype=np.complex64)).astype(np.complex64)
            )
            dechirped_symbols.append(np.asarray(dechirped, dtype=np.complex64))
            gt_bins.append(int(symbol.get("gt_bin", -1)))
        except ValueError:
            continue
    return spectra, dechirped_symbols, gt_bins


def evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    phase_weights: list[float],
    profile_weights: list[float],
) -> list[dict[str, Any]]:
    spectra, dechirped_symbols, gt_bins = _payload_evidence(samples, packet, args)
    if str(args.line_source) == "payload-argmax":
        phase_line = _payload_argmax_phase_line(packet, spectra, args)
    else:
        phase_line = _packet_phase_line(samples, packet, args)
    profile, profile_quality, profile_anchors = estimate_preamble_phase_profile(
        samples=samples,
        fine_payload_start_sample=int(packet["header_start_sample"]),
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        preamble_len=float(packet.get("preamble_len", args.preamble_len)),
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        cfo_correction_mode=str(args.cfo_correction_mode),
    )
    floor_db = max(1.0, float(args.amplitude_floor_db))
    rows: list[dict[str, Any]] = []
    all_bins = np.arange(1 << int(packet["sf"]), dtype=np.int64)
    for symbol_index, (spectrum, dechirped, gt_bin) in enumerate(
        zip(spectra, dechirped_symbols, gt_bins)
    ):
        spec = np.asarray(spectrum, dtype=np.complex64)
        power = np.abs(spec).astype(np.float64) ** 2
        max_power = float(np.max(power)) if power.size else 0.0
        rel_db = 10.0 * np.log10((power + 1e-30) / (max_power + 1e-30))
        amp_score = np.maximum(rel_db, -floor_db) / floor_db

        predicted = phase_line.predict(float(packet.get("preamble_len", args.preamble_len)) + 12.25 + symbol_index)
        residual = np.angle(np.exp(1j * (np.angle(spec) - float(predicted))))
        phase_score = np.cos(residual)
        profile_score = score_candidates_preamble_profile(
            dechirped,
            all_bins,
            profile,
        )

        amp_rank = _rank_of(power, int(gt_bin))
        for weight in phase_weights:
            for profile_weight in profile_weights:
                combined = (
                    amp_score
                    + float(weight) * phase_score
                    + float(profile_weight) * profile_score
                )
                rows.append(
                    {
                        "packet_index": int(packet["packet_index"]),
                        "payload_symbol_index": int(symbol_index),
                        "phase_weight": float(weight),
                        "profile_weight": float(profile_weight),
                        "gt_raw_fft_bin": int(gt_bin),
                        "amp_rank": int(amp_rank),
                        "phase_metric_rank": int(_rank_of(combined, int(gt_bin))),
                        "phase_line_slope_pi": float(phase_line.slope_pi),
                        "phase_line_r2": float(phase_line.fit_r2) if math.isfinite(phase_line.fit_r2) else "",
                        "phase_line_anchor_count": int(phase_line.anchor_count),
                        "profile_quality": float(profile_quality),
                        "profile_anchors": int(profile_anchors),
                        "line_source": str(args.line_source),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_summary(rows: list[dict[str, Any]], top_l: list[int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (float(row["phase_weight"]), float(row.get("profile_weight", 0.0))),
            [],
        ).append(row)

    summaries: list[dict[str, Any]] = []
    for (weight, profile_weight), items in sorted(grouped.items()):
        amp_ranks = [int(item["amp_rank"]) for item in items if int(item["amp_rank"]) > 0]
        phase_ranks = [
            int(item["phase_metric_rank"])
            for item in items
            if int(item["phase_metric_rank"]) > 0
        ]
        out: dict[str, Any] = {
            "phase_weight": float(weight),
            "profile_weight": float(profile_weight),
            "symbol_count": int(len(items)),
            "amp_mean_rank": float(np.mean(amp_ranks)) if amp_ranks else 0.0,
            "phase_mean_rank": float(np.mean(phase_ranks)) if phase_ranks else 0.0,
        }
        for cutoff in top_l:
            out[f"amp_top{cutoff}_recall"] = float(
                np.mean([rank <= cutoff for rank in amp_ranks])
            ) if amp_ranks else 0.0
            out[f"phase_top{cutoff}_recall"] = float(
                np.mean([rank <= cutoff for rank in phase_ranks])
            ) if phase_ranks else 0.0
        summaries.append(out)
    return summaries


def main() -> int:
    args = parse_args()
    samples = np.fromfile(args.input_iq, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {args.input_iq}")
    packets = load_packets(args.symbol_csv, args.packet)
    if not packets:
        raise SystemExit("no header-valid packets found in symbol CSV")

    phase_weights = _parse_float_list(args.phase_weights)
    profile_weights = _parse_float_list(args.profile_weights)
    top_l = _parse_int_list(args.top_l)
    rows: list[dict[str, Any]] = []
    for packet_index in sorted(packets):
        print(f"evaluate packet {packet_index}", flush=True)
        rows.extend(
            evaluate_packet(
                samples,
                packets[packet_index],
                args,
                phase_weights,
                profile_weights,
            )
        )

    write_csv(args.output, rows)
    summary = build_summary(rows, top_l)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
