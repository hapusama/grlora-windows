#!/usr/bin/env python3
"""Small in-folder sweep for phase_line path selector variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


THIS = Path(__file__).resolve()
PHASE_LINE_DIR = THIS.parents[1]
WEAK_ROOT = THIS.parents[3]
EXPERIMENT_DIR = WEAK_ROOT / "scripts" / "experiments"
PHASE_EXPERIMENT_DIR = EXPERIMENT_DIR / "phase_line"
for path in (str(WEAK_ROOT), str(EXPERIMENT_DIR), str(PHASE_EXPERIMENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_phase_line_threshold_sweep import _dataset_paths, _load_metadata  # noqa: E402
from run_symbol_phase_two_stage import _extract_payload_spectra_with_coherence, _ser  # noqa: E402
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.phase_line import PhasePathSelectorConfig, select_phase_viterbi_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep phase_line selector variants.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snrs", nargs="+", type=float, default=[-24.0, -25.0])
    parser.add_argument("--top-l", type=int, default=24)
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PHASE_LINE_DIR / "_eval" / "phase_path_variant_sweep")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--signal-reference-power", type=float, default=None)
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
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


def _avg(rows: Sequence[dict[str, Any]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else 0.0


def _candidate_recall(result: Any, gt_bins: Sequence[int]) -> float:
    hits = 0
    total = 0
    for idx, ev in enumerate(result.evidences):
        if idx >= len(gt_bins):
            continue
        gt = int(gt_bins[idx])
        if gt < 0:
            continue
        total += 1
        hits += int(gt in {int(v) for v in ev.top_bins})
    return float(hits / max(1, total))


def _variants(top_l: int) -> list[tuple[str, PhasePathSelectorConfig]]:
    base = PhasePathSelectorConfig(
        top_l=int(top_l),
        phase_order=1,
        energy_weight=0.35,
        coherence_weight=0.40,
        rank_weight=0.0,
        first_order_weight=0.18,
        second_order_weight=0.0,
    )
    return [
        ("current", base),
        ("no_hard_anchor", replace(base, hard_anchor_top_k=0)),
        ("hard_top2", replace(base, hard_anchor_top_k=2)),
        ("hard_top3", replace(base, hard_anchor_top_k=3)),
        (
            "hard_soft2_m05",
            replace(
                base,
                hard_anchor_soft_top_k=2,
                hard_anchor_soft_max_margin_db=0.5,
                hard_anchor_soft_max_peak_to_median_db=99.0,
            ),
        ),
        (
            "hard_soft2_m07",
            replace(
                base,
                hard_anchor_soft_top_k=2,
                hard_anchor_soft_max_margin_db=0.7,
                hard_anchor_soft_max_peak_to_median_db=99.0,
            ),
        ),
        (
            "hard_soft3_m05",
            replace(
                base,
                hard_anchor_soft_top_k=3,
                hard_anchor_soft_max_margin_db=0.5,
                hard_anchor_soft_max_peak_to_median_db=99.0,
            ),
        ),
        (
            "hard_soft2_p625",
            replace(
                base,
                hard_anchor_soft_top_k=2,
                hard_anchor_soft_max_margin_db=99.0,
                hard_anchor_soft_max_peak_to_median_db=6.25,
            ),
        ),
        (
            "hard_soft3_p63",
            replace(
                base,
                hard_anchor_soft_top_k=3,
                hard_anchor_soft_max_margin_db=99.0,
                hard_anchor_soft_max_peak_to_median_db=6.30,
            ),
        ),
        (
            "hard_strict",
            replace(
                base,
                hard_anchor_top_k=1,
                hard_anchor_margin_db=1.0,
                hard_anchor_peak_to_median_db=7.0,
                hard_anchor_min_coherence=0.90,
            ),
        ),
        (
            "hard_loose_coh075",
            replace(
                base,
                hard_anchor_top_k=1,
                hard_anchor_peak_to_median_db=5.5,
                hard_anchor_min_coherence=0.75,
                sliding_window_min_anchors=3,
            ),
        ),
        ("no_sliding", replace(base, sliding_window_refine_enabled=False)),
        ("top2_no_sliding", replace(base, hard_anchor_top_k=2, sliding_window_refine_enabled=False)),
        ("no_anchor_bias", replace(base, anchor_phase_bias_weight=0.0, anchor_slope_weight=0.0)),
        (
            "more_coherence",
            replace(base, energy_weight=0.25, coherence_weight=0.55, first_order_weight=0.14),
        ),
        (
            "more_energy",
            replace(base, energy_weight=0.50, coherence_weight=0.25, first_order_weight=0.16),
        ),
        (
            "weaker_phase",
            replace(base, first_order_weight=0.10, anchor_slope_weight=0.02),
        ),
        (
            "stronger_phase",
            replace(base, first_order_weight=0.28, anchor_slope_weight=0.08),
        ),
        (
            "first0_anchor_slope005",
            replace(base, first_order_weight=0.0, anchor_slope_weight=0.05),
        ),
        (
            "first0_anchor_slope010",
            replace(base, first_order_weight=0.0, anchor_slope_weight=0.10),
        ),
        (
            "first005_anchor_slope010",
            replace(base, first_order_weight=0.05, anchor_slope_weight=0.10),
        ),
        (
            "first010_anchor_relaxed",
            replace(base, first_order_weight=0.10, anchor_slope_weight=0.08, anchor_slope_max_rmse_pi=0.35),
        ),
        (
            "coh_rescue16_gain005",
            replace(base, phase_proposal_coherence_rescue_count=16, phase_proposal_coherence_min_gain=0.05),
        ),
        (
            "hard_anchor_phase_proposal",
            replace(base, phase_proposal_use_hard_anchors=True),
        ),
        (
            "extra_coh4_gain005",
            replace(base, extra_coherence_candidates=4, extra_coherence_min_gain=0.05),
        ),
        (
            "extra_coh8_gain005",
            replace(base, extra_coherence_candidates=8, extra_coherence_min_gain=0.05),
        ),
        (
            "extra_coh8_gain010",
            replace(base, extra_coherence_candidates=8, extra_coherence_min_gain=0.10),
        ),
        (
            "extra_coh12_gain010",
            replace(base, extra_coherence_candidates=12, extra_coherence_min_gain=0.10),
        ),
        (
            "coh_rescue20_gain000",
            replace(base, phase_proposal_coherence_rescue_count=20, phase_proposal_coherence_min_gain=0.0),
        ),
        (
            "coh_rescue8_gain005",
            replace(base, phase_proposal_coherence_rescue_count=8, phase_proposal_coherence_min_gain=0.05),
        ),
        (
            "proposal_off",
            replace(base, phase_proposal_enabled=False),
        ),
        (
            "consensus_on",
            replace(
                base,
                phase_proposal_consensus_enabled=True,
                phase_proposal_consensus_min_score=0.20,
            ),
        ),
        (
            "top28_current",
            replace(base, top_l=28),
        ),
        (
            "top32_current",
            replace(base, top_l=32),
        ),
        (
            "top40_current",
            replace(base, top_l=40),
        ),
        (
            "adaptive_top32_anchor8",
            replace(base, adaptive_top_l_high=32, adaptive_top_l_anchor_threshold=8),
        ),
        (
            "adaptive_top32_anchor10",
            replace(base, adaptive_top_l_high=32, adaptive_top_l_anchor_threshold=10),
        ),
        (
            "adaptive_top32_anchor12",
            replace(base, adaptive_top_l_high=32, adaptive_top_l_anchor_threshold=12),
        ),
        (
            "adaptive_top28_anchor10",
            replace(base, adaptive_top_l_high=28, adaptive_top_l_anchor_threshold=10),
        ),
        (
            "adaptive_top40_anchor10",
            replace(base, adaptive_top_l_high=40, adaptive_top_l_anchor_threshold=10),
        ),
        (
            "adaptive_top48_anchor10",
            replace(base, adaptive_top_l_high=48, adaptive_top_l_anchor_threshold=10),
        ),
    ]


def main() -> int:
    args = parse_args()
    paths = _dataset_paths(str(args.dataset))
    metadata = _load_metadata(paths, str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    packets = load_packets(paths["symbols"], args.packet)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {paths['iq']}")
    if args.signal_reference_power is not None:
        signal_power = float(args.signal_reference_power)
    elif "signal_reference_power" in metadata:
        signal_power = float(metadata["signal_reference_power"])
    else:
        signal_power = float(np.mean(np.abs(samples).astype(np.float64) ** 2))

    base_seed = int(metadata.get("seed", 42))
    rng = np.random.default_rng(base_seed)
    unit_noise = (
        rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        + 1j * rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    ).astype(np.complex64)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for snr_db in args.snrs:
        noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
        noisy = (samples + math.sqrt(noise_power / 2.0) * unit_noise).astype(np.complex64, copy=False)
        spectra_by_packet: dict[int, tuple[list[np.ndarray], list[np.ndarray], list[float], list[int], list[np.ndarray], int, bool]] = {}
        for packet_index in sorted(packets):
            packet = packets[packet_index]
            center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
                noisy,
                packet,
                SimpleNamespace(
                    cfo_correction_mode=str(args.cfo_correction_mode),
                    preamble_len=float(args.preamble_len),
                ),
            )
            spectra_by_packet[int(packet_index)] = (
                center_spectra,
                multi_spectra,
                abs_indices,
                gt_bins,
                coherences,
                int(packet["sf"]),
                bool(packet["ldro"]),
            )

        for name, config in _variants(int(args.top_l)):
            rows_for_variant: list[dict[str, Any]] = []
            for packet_index, payload in spectra_by_packet.items():
                center_spectra, multi_spectra, abs_indices, gt_bins, coherences, sf, ldro = payload
                evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
                result = select_phase_viterbi_path(
                    center_spectra=center_spectra,
                    evidence_powers=evidence_powers,
                    abs_indices=abs_indices,
                    config=config,
                    offset_coherences=coherences,
                )
                raw_ser, symbol_ser, compared = _ser(result.selected_raw_bins, gt_bins, sf=sf, ldro=ldro)
                row = {
                    "dataset": args.dataset,
                    "target_snr_db": float(snr_db),
                    "variant": name,
                    "packet_index": int(packet_index),
                    "raw_ser": float(raw_ser),
                    "symbol_ser": float(symbol_ser),
                    "gt_compared_symbols": int(compared),
                    "candidate_recall": float(_candidate_recall(result, gt_bins)),
                    "locked_count": int(result.locked_count),
                    "mean_phase_score": float(result.mean_phase_score),
                    "mean_amp_score": float(result.mean_amp_score),
                    "beam_final_size": int(result.beam_final_size),
                }
                packet_rows.append(row)
                rows_for_variant.append(row)
            summary = {
                "dataset": args.dataset,
                "target_snr_db": float(snr_db),
                "variant": name,
                "packet_count": int(len(rows_for_variant)),
                "symbol_ser": _avg(rows_for_variant, "symbol_ser"),
                "raw_ser": _avg(rows_for_variant, "raw_ser"),
                "candidate_recall": _avg(rows_for_variant, "candidate_recall"),
                "mean_locked_count": _avg(rows_for_variant, "locked_count"),
                "mean_phase_score": _avg(rows_for_variant, "mean_phase_score"),
                "mean_amp_score": _avg(rows_for_variant, "mean_amp_score"),
            }
            summary_rows.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    _write_csv(out_dir / "packet_variant_sweep.csv", packet_rows)
    _write_csv(out_dir / "snr_variant_sweep.csv", summary_rows)
    (out_dir / "snr_variant_sweep.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={out_dir / 'snr_variant_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
