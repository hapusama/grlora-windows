#!/usr/bin/env python3
"""Deterministic random search for PhasePathSelectorConfig weights."""

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
    parser = argparse.ArgumentParser(description="Search phase_path config variants.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snrs", nargs="+", type=float, default=[-24.0, -25.0])
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--output-dir", type=Path, default=PHASE_LINE_DIR / "_eval" / "phase_path_random_search")
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


def _config_dict(config: PhasePathSelectorConfig) -> dict[str, Any]:
    keep = (
        "energy_weight",
        "coherence_weight",
        "first_order_weight",
        "first_order_scale_pi",
        "anchor_phase_bias_weight",
        "anchor_slope_weight",
        "anchor_slope_max_rmse_pi",
        "sliding_window_radius",
        "sliding_window_min_anchors",
        "sliding_window_phase_weight",
        "sliding_window_energy_weight",
        "sliding_window_coherence_weight",
        "sliding_window_min_gain",
        "phase_proposal_coherence_rescue_count",
        "phase_proposal_coherence_min_gain",
        "adaptive_top_l_mid",
        "adaptive_top_l_mid_anchor_threshold",
        "adaptive_top_l_mid_max_anchor_rmse_pi",
    )
    return {name: getattr(config, name) for name in keep}


def _variants(rng: np.random.Generator, trials: int) -> list[tuple[str, PhasePathSelectorConfig]]:
    base = PhasePathSelectorConfig()
    variants: list[tuple[str, PhasePathSelectorConfig]] = [
        ("current", base),
        ("no_sliding", replace(base, sliding_window_refine_enabled=False)),
        ("phase_strong", replace(base, first_order_weight=0.24, anchor_slope_weight=0.08)),
        ("phase_weak", replace(base, first_order_weight=0.12, anchor_slope_weight=0.04)),
        ("coh16_gain005", replace(base, phase_proposal_coherence_rescue_count=16, phase_proposal_coherence_min_gain=0.05)),
    ]
    for idx in range(int(trials)):
        ew = float(rng.uniform(0.22, 0.48))
        cw = float(rng.uniform(0.25, 0.58))
        fw = float(rng.uniform(0.10, 0.30))
        variants.append(
            (
                f"rnd_{idx:03d}",
                replace(
                    base,
                    energy_weight=ew,
                    coherence_weight=cw,
                    first_order_weight=fw,
                    first_order_scale_pi=float(rng.uniform(0.55, 0.95)),
                    anchor_phase_bias_weight=float(rng.uniform(0.0, 0.07)),
                    anchor_slope_weight=float(rng.uniform(0.02, 0.10)),
                    anchor_slope_max_rmse_pi=float(rng.uniform(0.20, 0.45)),
                    sliding_window_radius=int(rng.choice([4, 5, 6, 7, 8])),
                    sliding_window_min_anchors=int(rng.choice([3, 4, 5])),
                    sliding_window_phase_weight=float(rng.uniform(0.05, 0.30)),
                    sliding_window_energy_weight=float(rng.uniform(0.20, 0.50)),
                    sliding_window_coherence_weight=float(rng.uniform(0.30, 0.65)),
                    sliding_window_min_gain=float(rng.uniform(0.0, 0.04)),
                    phase_proposal_coherence_rescue_count=int(rng.choice([8, 12, 16])),
                    phase_proposal_coherence_min_gain=float(rng.choice([0.05, 0.10, 0.15])),
                    adaptive_top_l_mid=int(rng.choice([24, 28, 32])),
                    adaptive_top_l_mid_anchor_threshold=int(rng.choice([7, 8, 9])),
                    adaptive_top_l_mid_max_anchor_rmse_pi=float(rng.choice([0.25, 0.30, 0.35])),
                ),
            )
        )
    return variants


def main() -> int:
    args = parse_args()
    paths = _dataset_paths(str(args.dataset))
    metadata = _load_metadata(paths, str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    packets = load_packets(paths["symbols"], None)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {paths['iq']}")
    signal_power = float(metadata.get("signal_reference_power", np.mean(np.abs(samples).astype(np.float64) ** 2)))
    noise_rng = np.random.default_rng(int(metadata.get("seed", 42)))
    unit_noise = (
        noise_rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        + 1j * noise_rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    ).astype(np.complex64)

    spectra_by_snr: dict[float, list[tuple[list[np.ndarray], list[np.ndarray], list[float], list[int], list[np.ndarray], int, bool]]] = {}
    for snr_db in args.snrs:
        noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
        noisy = (samples + math.sqrt(noise_power / 2.0) * unit_noise).astype(np.complex64, copy=False)
        items = []
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
            items.append((center_spectra, evidence_powers, abs_indices, gt_bins, coherences, int(packet["sf"]), bool(packet["ldro"])))
        spectra_by_snr[float(snr_db)] = items

    variants = _variants(np.random.default_rng(int(args.seed)), int(args.trials))
    rows: list[dict[str, Any]] = []
    for name, config in variants:
        rec: dict[str, Any] = {"variant": name}
        rec.update(_config_dict(config))
        total_score = 0.0
        worst = 0.0
        for snr_db in args.snrs:
            ser_values = []
            recall_values = []
            for center_spectra, evidence_powers, abs_indices, gt_bins, coherences, sf, ldro in spectra_by_snr[float(snr_db)]:
                result = select_phase_viterbi_path(
                    center_spectra=center_spectra,
                    evidence_powers=evidence_powers,
                    abs_indices=abs_indices,
                    config=config,
                    offset_coherences=coherences,
                )
                _raw, ser, _compared = _ser(result.selected_raw_bins, gt_bins, sf=sf, ldro=ldro)
                ser_values.append(float(ser))
                hits = 0
                total = 0
                for idx, ev in enumerate(result.evidences):
                    if idx >= len(gt_bins):
                        continue
                    total += 1
                    hits += int(int(gt_bins[idx]) in {int(v) for v in ev.top_bins})
                recall_values.append(float(hits / max(1, total)))
            mean_ser = float(np.mean(ser_values)) if ser_values else 1.0
            rec[f"ser_{snr_db:g}"] = mean_ser
            rec[f"recall_{snr_db:g}"] = float(np.mean(recall_values)) if recall_values else 0.0
            total_score += mean_ser
            worst = max(worst, mean_ser)
        rec["mean_ser"] = float(total_score / max(1, len(args.snrs)))
        rec["worst_ser"] = float(worst)
        rows.append(rec)
        print(json.dumps({"variant": name, "mean_ser": rec["mean_ser"], "worst_ser": rec["worst_ser"]}, ensure_ascii=False), flush=True)

    rows.sort(key=lambda row: (float(row["worst_ser"]), float(row["mean_ser"])))
    out_dir = Path(args.output_dir)
    _write_csv(out_dir / "random_search_results.csv", rows)
    (out_dir / "random_search_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best={json.dumps(rows[:5], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
