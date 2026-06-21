#!/usr/bin/env python3
"""Packet-level diagnostics for phase_path variant gating.

This script is intentionally kept under phase_line/_eval.  It reuses the
existing experiment loaders, then records packet-local signals that are
available without GT at runtime alongside GT-only SER/recall diagnostics.
"""

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
from weak_decoder.phase_line.selector import (  # noqa: E402
    _anchor_phase_line_rmse_pi,
    _is_hard_anchor,
    _is_high_confidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose phase_path packet-level gating signals.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snrs", nargs="+", type=float, default=[-24.0, -25.0])
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PHASE_LINE_DIR / "_eval" / "variant_gate_diagnostic")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
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


def _mean_evidence(rows: Sequence[Any], attr: str) -> float:
    values = [float(getattr(ev, attr)) for ev in rows if math.isfinite(float(getattr(ev, attr)))]
    return float(np.mean(values)) if values else 0.0


def _variants() -> list[tuple[str, PhasePathSelectorConfig]]:
    base = PhasePathSelectorConfig(
        top_l=24,
        phase_order=1,
        energy_weight=0.35,
        coherence_weight=0.40,
        rank_weight=0.0,
        first_order_weight=0.18,
        second_order_weight=0.0,
    )
    return [
        ("current", base),
        ("hard_prop", replace(base, phase_proposal_use_hard_anchors=True)),
        ("top28", replace(base, top_l=28)),
        ("top32", replace(base, top_l=32)),
        ("top40", replace(base, top_l=40)),
        ("no_sliding", replace(base, sliding_window_refine_enabled=False)),
        ("more_coh", replace(base, energy_weight=0.25, coherence_weight=0.55, first_order_weight=0.14)),
        ("proposal_off", replace(base, phase_proposal_enabled=False)),
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
    variants = _variants()

    for snr_db in args.snrs:
        noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
        noisy = (samples + math.sqrt(noise_power / 2.0) * unit_noise).astype(np.complex64, copy=False)
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
            evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
            variant_results: dict[str, Any] = {}
            row: dict[str, Any] = {
                "dataset": str(args.dataset),
                "target_snr_db": float(snr_db),
                "packet_index": int(packet_index),
            }
            for name, config in variants:
                result = select_phase_viterbi_path(
                    center_spectra=center_spectra,
                    evidence_powers=evidence_powers,
                    abs_indices=abs_indices,
                    config=config,
                    offset_coherences=coherences,
                )
                variant_results[name] = result
                _raw_ser, symbol_ser, compared = _ser(result.selected_raw_bins, gt_bins, sf=int(packet["sf"]), ldro=bool(packet["ldro"]))
                row[f"{name}_ser"] = float(symbol_ser)
                row[f"{name}_recall"] = float(_candidate_recall(result, gt_bins))
                row[f"{name}_phase"] = float(result.mean_phase_score)
                row[f"{name}_amp"] = float(result.mean_amp_score)
                row[f"{name}_line_rmse_pi"] = float(result.phase_line.fit_rmse_pi)
                row["gt_compared_symbols"] = int(compared)

            current = variant_results["current"]
            evidences = tuple(current.evidences)
            hard_mask = tuple(bool(_is_hard_anchor(ev, variants[0][1])) for ev in evidences)
            high_mask = tuple(bool(_is_high_confidence(ev, variants[0][1])) for ev in evidences)
            row["hard_count"] = int(sum(hard_mask))
            row["high_count"] = int(sum(high_mask))
            row["anchor_rmse_pi"] = float(_anchor_phase_line_rmse_pi(evidences, hard_mask))
            row["mean_margin_db"] = float(_mean_evidence(evidences, "top1_margin_db"))
            row["mean_peak_to_median_db"] = float(_mean_evidence(evidences, "top1_peak_to_median_db"))
            row["mean_top1_coherence"] = float(_mean_evidence(evidences, "top1_coherence_score"))
            for name, _config in variants:
                if name == "current":
                    continue
                row[f"delta_{name}"] = float(row[f"{name}_ser"] - row["current_ser"])
            packet_rows.append(row)

        snr_rows = [row for row in packet_rows if float(row["target_snr_db"]) == float(snr_db)]
        summary: dict[str, Any] = {
            "dataset": str(args.dataset),
            "target_snr_db": float(snr_db),
            "packet_count": int(len(snr_rows)),
            "mean_hard_count": _avg(snr_rows, "hard_count"),
            "mean_anchor_rmse_pi": _avg(snr_rows, "anchor_rmse_pi"),
        }
        for name, _config in variants:
            summary[f"{name}_ser"] = _avg(snr_rows, f"{name}_ser")
            summary[f"{name}_recall"] = _avg(snr_rows, f"{name}_recall")
        summary_rows.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    _write_csv(out_dir / "packet_variant_gate.csv", packet_rows)
    _write_csv(out_dir / "snr_variant_gate.csv", summary_rows)
    print(f"wrote={out_dir / 'packet_variant_gate.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
