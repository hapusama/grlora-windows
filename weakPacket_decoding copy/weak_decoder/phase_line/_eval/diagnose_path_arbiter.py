#!/usr/bin/env python3
"""Diagnose packet-level arbitration between phase-path selectors.

The main selector intentionally returns a single path.  This diagnostic checks
whether packet-local, non-GT features can safely choose between that path and a
small set of alternative internal selectors.
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
from run_phase_path_ablation import _build_v3_config  # noqa: E402
from run_symbol_phase_two_stage import _extract_payload_spectra_with_coherence, _ser  # noqa: E402
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.candidate_pruning import wrap_phase  # noqa: E402
from weak_decoder.phase_line import PhasePathSelectorConfig, select_phase_viterbi_path  # noqa: E402
from weak_decoder.symbol_phase_two_stage import select_symbol_bins_two_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose phase-path arbitration.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snrs", nargs="+", type=float, default=[-24.0, -25.0])
    parser.add_argument("--output-dir", type=Path, default=PHASE_LINE_DIR / "_eval" / "path_arbiter_diagnostic")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
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


def _path_phase_stats(center_spectra: Sequence[np.ndarray], selected: Sequence[int]) -> dict[str, float]:
    phases: list[float] = []
    for spectrum, raw_bin in zip(center_spectra, selected):
        b = int(raw_bin)
        spec = np.asarray(spectrum)
        if b < 0 or b >= spec.size:
            continue
        phases.append(float(np.angle(spec[b])))
    if len(phases) < 2:
        return {"first_abs_pi": 0.0, "second_abs_pi": 0.0}
    r1 = [abs(float(wrap_phase(phases[idx] - phases[idx - 1]))) / math.pi for idx in range(1, len(phases))]
    if len(phases) < 3:
        r2: list[float] = []
    else:
        r2 = [
            abs(float(wrap_phase(phases[idx] - 2.0 * phases[idx - 1] + phases[idx - 2]))) / math.pi
            for idx in range(2, len(phases))
        ]
    return {
        "first_abs_pi": float(np.mean(r1)) if r1 else 0.0,
        "second_abs_pi": float(np.mean(r2)) if r2 else 0.0,
    }


def _mean_selected_score(result: Any, selected: Sequence[int], attr: str) -> float:
    vals: list[float] = []
    for ev, raw_bin in zip(result.evidences, selected):
        b = int(raw_bin)
        if attr == "energy":
            if 0 <= b < ev.evidence_power.size:
                max_power = float(np.max(ev.evidence_power)) if ev.evidence_power.size else 0.0
                vals.append(float(ev.evidence_power[b] / (max_power + 1e-30)) if max_power > 0 else 0.0)
        elif attr == "coherence":
            if ev.offset_coherence is not None and 0 <= b < ev.offset_coherence.size:
                vals.append(float(ev.offset_coherence[b]))
    return float(np.mean(vals)) if vals else 0.0


def _avg(rows: Sequence[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else 0.0


def main() -> int:
    args = parse_args()
    paths = _dataset_paths(str(args.dataset))
    metadata = _load_metadata(paths, str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    packets = load_packets(paths["symbols"], None)
    signal_power = float(metadata.get("signal_reference_power", np.mean(np.abs(samples).astype(np.float64) ** 2)))
    rng = np.random.default_rng(int(metadata.get("seed", 42)))
    unit_noise = (
        rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        + 1j * rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    ).astype(np.complex64)

    current_cfg = PhasePathSelectorConfig()
    second_cfg = replace(current_cfg, phase_order=2, first_order_weight=0.0, second_order_weight=0.05)
    no_proposal_cfg = replace(current_cfg, phase_proposal_enabled=False)
    strong_cfg = replace(current_cfg, first_order_weight=0.24, anchor_slope_weight=0.08)
    v3_cfg = _build_v3_config(24)

    packet_rows: list[dict[str, Any]] = []
    for snr_db in args.snrs:
        noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
        noisy = (samples + math.sqrt(noise_power / 2.0) * unit_noise).astype(np.complex64, copy=False)
        for packet_index in sorted(packets):
            packet = packets[packet_index]
            center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
                noisy,
                packet,
                SimpleNamespace(cfo_correction_mode=str(args.cfo_correction_mode), preamble_len=float(args.preamble_len)),
            )
            evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
            results: dict[str, Any] = {
                "current": select_phase_viterbi_path(center_spectra, evidence_powers, abs_indices, current_cfg, offset_coherences=coherences),
                "second": select_phase_viterbi_path(center_spectra, evidence_powers, abs_indices, second_cfg, offset_coherences=coherences),
                "no_proposal": select_phase_viterbi_path(center_spectra, evidence_powers, abs_indices, no_proposal_cfg, offset_coherences=coherences),
                "strong": select_phase_viterbi_path(center_spectra, evidence_powers, abs_indices, strong_cfg, offset_coherences=coherences),
                "v3": select_symbol_bins_two_stage(
                    center_spectra=center_spectra,
                    evidence_powers=evidence_powers,
                    abs_indices=abs_indices,
                    config=v3_cfg,
                    offset_coherences=coherences,
                ),
            }
            row: dict[str, Any] = {
                "dataset": str(args.dataset),
                "target_snr_db": float(snr_db),
                "packet_index": int(packet_index),
            }
            for name, result in results.items():
                _raw, ser, _compared = _ser(result.selected_raw_bins, gt_bins, sf=int(packet["sf"]), ldro=bool(packet["ldro"]))
                stats = _path_phase_stats(center_spectra, result.selected_raw_bins)
                row[f"{name}_ser"] = float(ser)
                row[f"{name}_mean_phase"] = float(result.mean_phase_score)
                row[f"{name}_mean_amp"] = float(result.mean_amp_score)
                row[f"{name}_mean_energy"] = _mean_selected_score(result, result.selected_raw_bins, "energy")
                row[f"{name}_mean_coherence"] = _mean_selected_score(result, result.selected_raw_bins, "coherence")
                row[f"{name}_first_abs_pi"] = stats["first_abs_pi"]
                row[f"{name}_second_abs_pi"] = stats["second_abs_pi"]
                row[f"{name}_locked_count"] = int(result.locked_count)
                row[f"{name}_trajectory_score"] = float(result.trajectory_score)
            row["oracle_best_ser"] = min(float(row[f"{name}_ser"]) for name in results)
            packet_rows.append(row)

    # Simple gate probes.  These are GT-evaluated diagnostics only.
    gates: list[tuple[str, Any]] = [
        ("always_current", lambda row: "current"),
        ("v3_when_current_first_gt_035", lambda row: "v3" if float(row["current_first_abs_pi"]) > 0.35 else "current"),
        ("v3_when_current_second_gt_042", lambda row: "v3" if float(row["current_second_abs_pi"]) > 0.42 else "current"),
        ("second_when_current_second_lt_024", lambda row: "second" if float(row["current_second_abs_pi"]) < 0.24 else "current"),
        ("second_when_current_first_lt_018", lambda row: "second" if float(row["current_first_abs_pi"]) < 0.18 else "current"),
        ("no_prop_when_amp_gt_092", lambda row: "no_proposal" if float(row["current_mean_amp"]) > 0.92 else "current"),
        ("v3_when_locked_ge_14_first_gt_035", lambda row: "v3" if int(row["current_locked_count"]) >= 14 and float(row["current_first_abs_pi"]) > 0.35 else "current"),
    ]
    summary_rows: list[dict[str, Any]] = []
    for gate_name, chooser in gates:
        for snr_db in args.snrs:
            rows = [row for row in packet_rows if float(row["target_snr_db"]) == float(snr_db)]
            selected = []
            for row in rows:
                name = str(chooser(row))
                selected.append({**row, "selected_method": name, "selected_ser": float(row[f"{name}_ser"])})
            summary_rows.append(
                {
                    "gate": gate_name,
                    "target_snr_db": float(snr_db),
                    "symbol_ser": _avg(selected, "selected_ser"),
                    "oracle_best_ser": _avg(selected, "oracle_best_ser"),
                }
            )

    out_dir = Path(args.output_dir)
    _write_csv(out_dir / "packet_path_arbiter.csv", packet_rows)
    _write_csv(out_dir / "gate_summary.csv", summary_rows)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
