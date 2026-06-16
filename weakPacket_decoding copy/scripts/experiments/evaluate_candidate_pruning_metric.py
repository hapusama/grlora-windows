#!/usr/bin/env python3
"""Evaluate lightweight phase-aware candidate pruning metrics.

This is a first-stage experiment only: it ranks raw FFT bins and measures
GT-bin Recall@L against energy baselines.  GT bins from the clean header-first
CSV are used only for offline evaluation.
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

from run_two_stage_weak_decoder import (  # noqa: E402
    extract_fft,
    extract_multi_offset_fft_evidence,
    load_packets,
)
from weak_decoder.candidate_pruning import (  # noqa: E402
    PhaseGatedMetricConfig,
    PhaseTrend,
    estimate_payload_residual_offset,
    fit_early_payload_phase_trend,
    phase_gated_scores,
    phase_rescue_scores,
    rank_of,
    top_bins,
)
from weak_decoder.chirp import build_downchirp  # noqa: E402
from weak_decoder.phase_guided_demod import (  # noqa: E402
    PhaseLine,
    extract_header_anchors,
    fit_phase_line,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate phase-gated candidate-pruning Recall@L."
    )
    parser.add_argument("-i", "--input-iq", type=Path, required=True)
    parser.add_argument("-s", "--symbol-csv", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="do not print JSON summary")
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--top-l", default="1,2,4,8,16,32,64")
    parser.add_argument("--phase-bonus-weights", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--phase-gate-width-pi", default="0.333333,0.5")
    parser.add_argument(
        "--phase-evidence",
        choices=("center", "multi-offset"),
        default="multi-offset",
        help="energy evidence used by the phase-gated score",
    )
    parser.add_argument("--energy-preselect-count", type=int, default=128)
    parser.add_argument("--energy-preselect-factor", type=int, default=4)
    parser.add_argument("--noise-floor-percentile", type=float, default=50.0)
    parser.add_argument("--amp-gate-floor-db", type=float, default=-18.0)
    parser.add_argument("--amp-gate-temp-db", type=float, default=4.0)
    parser.add_argument("--rel-db-floor", type=float, default=-30.0)
    parser.add_argument("--early-anchor-symbols", type=int, default=8)
    parser.add_argument(
        "--phase-trend-source",
        choices=("header", "header-offset", "early-payload"),
        default="early-payload",
        help="phase predictor used by the pruning metric",
    )
    parser.add_argument("--early-anchor-min-margin-db", type=float, default=0.0)
    parser.add_argument("--early-anchor-trim-frac", type=float, default=0.25)
    parser.add_argument(
        "--rescue-counts",
        default="0,1,2,4",
        help="exact Top-L rescue lanes: keep L-r energy bins and add r phase bins",
    )
    return parser.parse_args()


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(float(item.strip())) for item in str(text).split(",") if item.strip()]


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


def _payload_spectra(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[int], list[int]]:
    sf = int(packet["sf"])
    cfo_total = float(packet["cfo_int"]) + float(packet["cfo_frac"])
    downchirp = build_downchirp(sf, cfo_int=packet["cfo_int"], cfo_frac=packet["cfo_frac"])
    center_spectra: list[np.ndarray] = []
    multi_spectra: list[np.ndarray] = []
    abs_indices: list[float] = []
    gt_bins: list[int] = []
    payload_symbol_indices: list[int] = []
    for symbol in packet["payload_symbols"]:
        try:
            center = extract_fft(
                samples=samples,
                start_sample=int(symbol["start_sample"]),
                sf=sf,
                os_factor=int(packet["os_factor"]),
                downchirp=downchirp,
                cfo_total=cfo_total,
                header_start_sample=int(packet["header_start_sample"]),
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
            multi = extract_multi_offset_fft_evidence(
                samples=samples,
                start_sample=int(symbol["start_sample"]),
                sf=sf,
                os_factor=int(packet["os_factor"]),
                downchirp=downchirp,
                cfo_total=cfo_total,
                header_start_sample=int(packet["header_start_sample"]),
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
        except ValueError:
            continue
        payload_idx = int(symbol.get("payload_symbol_index", len(payload_symbol_indices)))
        center_spectra.append(center)
        multi_spectra.append(multi)
        abs_indices.append(float(packet.get("preamble_len", args.preamble_len)) + 12.25 + float(payload_idx))
        gt_bins.append(int(symbol.get("gt_bin", -1)))
        payload_symbol_indices.append(payload_idx)
    return center_spectra, multi_spectra, abs_indices, gt_bins, payload_symbol_indices


def evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    top_l_values: list[int],
    phase_weights: list[float],
    gate_widths_pi: list[float],
    rescue_counts: list[int],
) -> list[dict[str, Any]]:
    center_spectra, multi_spectra, abs_indices, gt_bins, payload_symbol_indices = _payload_spectra(
        samples,
        packet,
        args,
    )
    if not center_spectra:
        return []

    phase_line = _packet_phase_line(samples, packet, args)
    center_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in center_spectra]
    multi_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
    evidence_powers = multi_powers if str(args.phase_evidence) == "multi-offset" else center_powers
    if str(args.phase_trend_source) == "early-payload":
        trend = fit_early_payload_phase_trend(
            center_spectra=center_spectra,
            evidence_powers=evidence_powers,
            abs_indices=abs_indices,
            fallback_line=phase_line,
            max_symbols=int(args.early_anchor_symbols),
            min_margin_db=float(args.early_anchor_min_margin_db),
            trim_frac=float(args.early_anchor_trim_frac),
        )
    else:
        offset = 0.0
        offset_std = float("inf")
        offset_count = 0
        source = "header"
        if str(args.phase_trend_source) == "header-offset":
            offset, offset_std, offset_count = estimate_payload_residual_offset(
                center_spectra=center_spectra,
                evidence_powers=evidence_powers,
                abs_indices=abs_indices,
                phase_line=phase_line,
                max_symbols=int(args.early_anchor_symbols),
                min_margin_db=float(args.early_anchor_min_margin_db),
            )
            source = "header-offset"
        trend = PhaseTrend(
            line=phase_line,
            residual_offset_rad=offset,
            residual_std_rad=offset_std,
            early_anchor_count=offset_count,
            source=source,
        )
    max_top_l = max(top_l_values) if top_l_values else 1

    rows: list[dict[str, Any]] = []
    for symbol_pos, (center, center_power, multi_power, evidence_power, abs_idx, gt_bin) in enumerate(
        zip(center_spectra, center_powers, multi_powers, evidence_powers, abs_indices, gt_bins)
    ):
        center_rank = rank_of(center_power, gt_bin)
        multi_rank = rank_of(multi_power, gt_bin)
        for phase_weight in phase_weights:
            for gate_width_pi in gate_widths_pi:
                cfg = PhaseGatedMetricConfig(
                    phase_bonus_weight=float(phase_weight),
                    phase_gate_width_rad=float(gate_width_pi) * math.pi,
                    energy_preselect_count=int(args.energy_preselect_count),
                    energy_preselect_factor=int(args.energy_preselect_factor),
                    noise_floor_percentile=float(args.noise_floor_percentile),
                    rel_db_floor=float(args.rel_db_floor),
                    amp_gate_floor_db=float(args.amp_gate_floor_db),
                    amp_gate_temp_db=float(args.amp_gate_temp_db),
                )
                gated_scores = phase_gated_scores(
                    center_spectrum=center,
                    evidence_power=evidence_power,
                    predicted_phase_rad=trend.predict(abs_idx),
                    phase_trend_quality=trend.quality,
                    max_top_l=max_top_l,
                    config=cfg,
                )
                phase_rank = rank_of(gated_scores, gt_bin)
                row: dict[str, Any] = {
                    "packet_index": int(packet["packet_index"]),
                    "payload_symbol_index": int(payload_symbol_indices[symbol_pos]),
                    "gt_raw_fft_bin": int(gt_bin),
                    "phase_bonus_weight": float(phase_weight),
                    "phase_gate_width_pi": float(gate_width_pi),
                    "phase_evidence": str(args.phase_evidence),
                    "center_rank": int(center_rank),
                    "multi_offset_rank": int(multi_rank),
                    "phase_gated_rank": int(phase_rank),
                    "phase_line_slope_pi": float(phase_line.slope_pi),
                    "phase_line_r2": float(phase_line.fit_r2) if math.isfinite(phase_line.fit_r2) else "",
                    "trend_source": str(trend.source),
                    "trend_slope_pi": float(trend.line.slope_pi),
                    "trend_r2": float(trend.line.fit_r2) if math.isfinite(trend.line.fit_r2) else "",
                    "phase_line_anchor_count": int(phase_line.anchor_count),
                    "residual_offset_pi": float(trend.residual_offset_rad / math.pi),
                    "residual_std_pi": (
                        float(trend.residual_std_rad / math.pi)
                        if math.isfinite(trend.residual_std_rad)
                        else ""
                    ),
                    "early_anchor_count": int(trend.early_anchor_count),
                    "phase_trend_quality": float(trend.quality),
                }

                for cutoff in top_l_values:
                    center_hit = int(0 < center_rank <= int(cutoff))
                    multi_hit = int(0 < multi_rank <= int(cutoff))
                    phase_hit = int(0 < phase_rank <= int(cutoff))
                    row[f"center_top{cutoff}_hit"] = center_hit
                    row[f"multi_top{cutoff}_hit"] = multi_hit
                    row[f"phase_gated_top{cutoff}_hit"] = phase_hit
                    row[f"phase_rescue_top{cutoff}"] = int((not multi_hit) and phase_hit)
                    row[f"phase_damage_top{cutoff}"] = int(multi_hit and (not phase_hit))

                    for rescue_count in rescue_counts:
                        if rescue_count <= 0 or rescue_count >= cutoff:
                            continue
                        rescue_scores = phase_rescue_scores(
                            energy_scores=evidence_power,
                            phase_scores=gated_scores,
                            top_l=int(cutoff),
                            rescue_count=int(rescue_count),
                        )
                        rescue_set = set(int(v) for v in top_bins(rescue_scores, int(cutoff)))
                        row[f"rescue_r{rescue_count}_top{cutoff}_hit"] = int(gt_bin in rescue_set)
                rows.append(row)
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


def build_summary(rows: list[dict[str, Any]], top_l_values: list[int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                float(row["phase_bonus_weight"]),
                float(row["phase_gate_width_pi"]),
                str(row["phase_evidence"]),
            ),
            [],
        ).append(row)

    summary: list[dict[str, Any]] = []
    for (phase_weight, gate_width_pi, phase_evidence), items in sorted(grouped.items()):
        out: dict[str, Any] = {
            "phase_bonus_weight": float(phase_weight),
            "phase_gate_width_pi": float(gate_width_pi),
            "phase_evidence": phase_evidence,
            "symbol_count": int(len(items)),
            "packet_count": int(len({int(item["packet_index"]) for item in items})),
            "center_mean_rank": float(np.mean([int(item["center_rank"]) for item in items])),
            "multi_offset_mean_rank": float(np.mean([int(item["multi_offset_rank"]) for item in items])),
            "phase_gated_mean_rank": float(np.mean([int(item["phase_gated_rank"]) for item in items])),
            "mean_phase_trend_quality": float(np.mean([float(item["phase_trend_quality"]) for item in items])),
            "mean_early_anchor_count": float(np.mean([int(item["early_anchor_count"]) for item in items])),
            "trend_sources": ",".join(sorted({str(item.get("trend_source", "")) for item in items})),
        }
        for cutoff in top_l_values:
            center = [int(item.get(f"center_top{cutoff}_hit", 0)) for item in items]
            multi = [int(item.get(f"multi_top{cutoff}_hit", 0)) for item in items]
            phase = [int(item.get(f"phase_gated_top{cutoff}_hit", 0)) for item in items]
            rescue = [int(item.get(f"phase_rescue_top{cutoff}", 0)) for item in items]
            damage = [int(item.get(f"phase_damage_top{cutoff}", 0)) for item in items]
            out[f"center_recall@{cutoff}"] = float(np.mean(center))
            out[f"multi_offset_recall@{cutoff}"] = float(np.mean(multi))
            out[f"phase_gated_recall@{cutoff}"] = float(np.mean(phase))
            out[f"gain_vs_center@{cutoff}"] = out[f"phase_gated_recall@{cutoff}"] - out[f"center_recall@{cutoff}"]
            out[f"gain_vs_multi_offset@{cutoff}"] = out[f"phase_gated_recall@{cutoff}"] - out[f"multi_offset_recall@{cutoff}"]
            out[f"phase_rescue_count@{cutoff}"] = int(np.sum(rescue))
            out[f"phase_damage_count@{cutoff}"] = int(np.sum(damage))
            out[f"net_rescue@{cutoff}"] = int(np.sum(rescue) - np.sum(damage))
            for key in items[0]:
                if key.startswith("rescue_r") and key.endswith(f"_top{cutoff}_hit"):
                    label = key.replace("_hit", "")
                    vals = [int(item.get(key, 0)) for item in items]
                    out[f"{label}_recall"] = float(np.mean(vals))
        summary.append(out)
    return summary


def main() -> int:
    args = parse_args()
    samples = np.fromfile(args.input_iq, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {args.input_iq}")

    packets = load_packets(args.symbol_csv, args.packet)
    if not packets:
        raise SystemExit("no header-valid packets found in symbol CSV")

    top_l_values = _parse_int_list(args.top_l)
    phase_weights = _parse_float_list(args.phase_bonus_weights)
    gate_widths_pi = _parse_float_list(args.phase_gate_width_pi)
    rescue_counts = _parse_int_list(args.rescue_counts)

    rows: list[dict[str, Any]] = []
    for packet_index in sorted(packets):
        if not args.quiet:
            print(f"evaluate packet {packet_index}", flush=True)
        rows.extend(
            evaluate_packet(
                samples=samples,
                packet=packets[packet_index],
                args=args,
                top_l_values=top_l_values,
                phase_weights=phase_weights,
                gate_widths_pi=gate_widths_pi,
                rescue_counts=rescue_counts,
            )
        )

    if not rows:
        raise SystemExit("no payload rows evaluated")
    write_csv(args.output, rows)
    summary = build_summary(rows, top_l_values)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if not args.quiet:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
