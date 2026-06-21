"""Sweep a selected-path local phase rescue refiner.

The refiner is diagnostic/prototype code.  It starts from the current
``select_phase_viterbi_path`` result, fits a local phase line from the selected
path around each symbol, and optionally switches to a wider energy-preselected
candidate when phase/energy/coherence agree.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
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
from weak_decoder.candidate_pruning import top_bins, wrap_phase  # noqa: E402
from weak_decoder.phase_line import PhasePathSelectorConfig, select_phase_viterbi_path  # noqa: E402
from weak_decoder.phase_line.selector import _is_hard_anchor  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    preselect: int
    radius: int
    min_anchors: int
    max_rmse_pi: float
    phase_weight: float
    energy_weight: float
    coherence_weight: float
    phase_scale_pi: float
    min_gain: float
    max_drop_from_current_db: float
    max_drop_from_top1_db: float
    skip_hard_anchors: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep path-rescue refinement variants.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snrs", nargs="+", type=float, default=[-24.0, -25.0])
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PHASE_LINE_DIR / "_eval" / "path_rescue_refine_sweep")
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


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(10.0 * math.log10((float(numerator) + 1e-30) / (float(denominator) + 1e-30)))


def _avg(rows: Sequence[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else 0.0


def _local_path_prediction(
    center_spectra: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    selected: Sequence[int],
    target_idx: int,
    radius: int,
    min_anchors: int,
) -> tuple[bool, float, float]:
    idx = int(target_idx)
    lo = max(0, idx - max(1, int(radius)))
    hi = min(len(center_spectra), idx + max(1, int(radius)) + 1)
    rows: list[tuple[float, float, float]] = []
    for j in range(lo, hi):
        if j == idx or j >= len(selected) or j >= len(abs_indices):
            continue
        spec = np.asarray(center_spectra[j])
        b = int(selected[j])
        if b < 0 or b >= spec.size:
            continue
        distance = abs(j - idx)
        weight = math.exp(-0.5 * (float(distance) / max(1e-6, 0.5 * float(radius))) ** 2)
        rows.append((float(abs_indices[j]), float(np.angle(spec[b])), float(weight)))
    if len(rows) < max(2, int(min_anchors)):
        return False, 0.0, float("nan")
    rows.sort(key=lambda item: item[0])
    xs = np.asarray([item[0] for item in rows], dtype=np.float64)
    phases = np.unwrap(np.asarray([item[1] for item in rows], dtype=np.float64))
    weights = np.asarray([item[2] for item in rows], dtype=np.float64)
    target_abs = float(abs_indices[idx])
    try:
        coef = np.polyfit(xs - target_abs, phases, deg=1, w=weights)
    except np.linalg.LinAlgError:
        return False, 0.0, float("nan")
    fitted = np.polyval(coef, xs - target_abs)
    rmse_pi = float(math.sqrt(float(np.average((phases - fitted) ** 2, weights=weights))) / math.pi)
    return True, float(np.polyval(coef, 0.0)), float(rmse_pi)


def _score_bin(
    spectrum: np.ndarray,
    power: np.ndarray,
    coherence: np.ndarray | None,
    raw_bin: int,
    predicted: float,
    variant: Variant,
) -> tuple[float, float, float, float]:
    b = int(raw_bin)
    if b < 0 or b >= spectrum.size or b >= power.size:
        return -float("inf"), 0.0, 0.0, 0.0
    phase_scale = max(1e-6, float(variant.phase_scale_pi) * math.pi)
    residual = float(wrap_phase(float(np.angle(spectrum[b])) - float(predicted)))
    phase = float(math.exp(-((residual / phase_scale) ** 2)))
    max_power = float(np.max(power)) if power.size else 0.0
    energy = float(power[b] / (max_power + 1e-30)) if max_power > 0.0 else 0.0
    coh = 0.0
    if coherence is not None and b < coherence.size:
        coh = float(max(0.0, min(1.0, coherence[b])))
    score = (
        float(variant.phase_weight) * phase
        + float(variant.energy_weight) * energy
        + float(variant.coherence_weight) * coh
    )
    return float(score), float(phase), float(energy), float(coh)


def _refine_selected(
    result: Any,
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    coherences: Sequence[np.ndarray],
    variant: Variant,
    config: PhasePathSelectorConfig,
) -> tuple[tuple[int, ...], int]:
    selected = [int(v) for v in result.selected_raw_bins]
    if not selected:
        return tuple(selected), 0
    base = list(selected)
    changes = 0
    for idx, ev in enumerate(result.evidences):
        if idx >= len(base) or idx >= len(center_spectra) or idx >= len(evidence_powers):
            continue
        if bool(variant.skip_hard_anchors) and _is_hard_anchor(ev, config):
            continue
        has_pred, predicted, rmse_pi = _local_path_prediction(
            center_spectra,
            abs_indices,
            base,
            idx,
            int(variant.radius),
            int(variant.min_anchors),
        )
        if not has_pred or (math.isfinite(rmse_pi) and rmse_pi > float(variant.max_rmse_pi)):
            continue
        spectrum = np.asarray(center_spectra[idx])
        power = np.asarray(evidence_powers[idx], dtype=np.float64)
        coherence = np.asarray(coherences[idx], dtype=np.float64) if idx < len(coherences) else None
        current = int(base[idx])
        current_score, _cur_phase, _cur_energy, _cur_coh = _score_bin(
            spectrum,
            power,
            coherence,
            current,
            predicted,
            variant,
        )
        top1 = int(ev.top1_bin)
        top1_power = float(power[top1]) if 0 <= top1 < power.size else float(np.max(power)) if power.size else 0.0
        current_power = float(power[current]) if 0 <= current < power.size else 0.0
        best_bin = current
        best_score = current_score
        for raw_bin in top_bins(power, min(int(variant.preselect), power.size)):
            b = int(raw_bin)
            if b == current:
                continue
            if _db_ratio(float(power[b]), top1_power) < -float(variant.max_drop_from_top1_db):
                continue
            if _db_ratio(float(power[b]), current_power) < -float(variant.max_drop_from_current_db):
                continue
            score, _phase, _energy, _coh = _score_bin(spectrum, power, coherence, b, predicted, variant)
            if score > best_score:
                best_score = score
                best_bin = b
        if best_bin != current and best_score >= current_score + float(variant.min_gain):
            selected[idx] = int(best_bin)
            changes += 1
    return tuple(selected), int(changes)


def _variants() -> tuple[Variant, ...]:
    out: list[Variant] = []
    for preselect in (128, 256, 512):
        for min_gain in (0.05, 0.10, 0.15):
            out.append(
                Variant(
                    name=f"p{preselect}_g{str(min_gain).replace('.', 'p')}_balanced",
                    preselect=preselect,
                    radius=6,
                    min_anchors=4,
                    max_rmse_pi=0.22,
                    phase_weight=0.55,
                    energy_weight=0.25,
                    coherence_weight=0.20,
                    phase_scale_pi=0.35,
                    min_gain=min_gain,
                    max_drop_from_current_db=12.0,
                    max_drop_from_top1_db=24.0,
                    skip_hard_anchors=True,
                )
            )
            out.append(
                Variant(
                    name=f"p{preselect}_g{str(min_gain).replace('.', 'p')}_phase",
                    preselect=preselect,
                    radius=6,
                    min_anchors=4,
                    max_rmse_pi=0.18,
                    phase_weight=0.70,
                    energy_weight=0.18,
                    coherence_weight=0.12,
                    phase_scale_pi=0.30,
                    min_gain=min_gain,
                    max_drop_from_current_db=9.0,
                    max_drop_from_top1_db=24.0,
                    skip_hard_anchors=True,
                )
            )
    out.append(
        Variant(
            name="p256_g0p10_no_hard_skip",
            preselect=256,
            radius=6,
            min_anchors=4,
            max_rmse_pi=0.22,
            phase_weight=0.55,
            energy_weight=0.25,
            coherence_weight=0.20,
            phase_scale_pi=0.35,
            min_gain=0.10,
            max_drop_from_current_db=12.0,
            max_drop_from_top1_db=24.0,
            skip_hard_anchors=False,
        )
    )
    return tuple(out)


def main() -> int:
    args = parse_args()
    paths = _dataset_paths(str(args.dataset))
    metadata = _load_metadata(paths, str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    packets = load_packets(paths["symbols"], args.packet)
    signal_power = float(metadata.get("signal_reference_power", np.mean(np.abs(samples).astype(np.float64) ** 2)))
    rng = np.random.default_rng(int(metadata.get("seed", 42)))
    unit_noise = (
        rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        + 1j * rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    ).astype(np.complex64)
    config = PhasePathSelectorConfig()
    variants = _variants()

    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for snr_db in args.snrs:
        noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
        noisy = (samples + math.sqrt(noise_power / 2.0) * unit_noise).astype(np.complex64, copy=False)
        items: list[tuple[int, Any, list[np.ndarray], list[np.ndarray], list[float], list[int], list[np.ndarray], int, bool]] = []
        for packet_index in sorted(packets):
            packet = packets[packet_index]
            center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
                noisy,
                packet,
                SimpleNamespace(cfo_correction_mode=str(args.cfo_correction_mode), preamble_len=float(args.preamble_len)),
            )
            evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
            result = select_phase_viterbi_path(
                center_spectra=center_spectra,
                evidence_powers=evidence_powers,
                abs_indices=abs_indices,
                config=config,
                offset_coherences=coherences,
            )
            items.append(
                (
                    int(packet_index),
                    result,
                    center_spectra,
                    evidence_powers,
                    abs_indices,
                    gt_bins,
                    coherences,
                    int(packet["sf"]),
                    bool(packet["ldro"]),
                )
            )

        current_rows: list[dict[str, Any]] = []
        for packet_index, result, _center, _powers, _abs, gt_bins, _coh, sf, ldro in items:
            _raw, ser, compared = _ser(result.selected_raw_bins, gt_bins, sf=sf, ldro=ldro)
            current_rows.append({"symbol_ser": float(ser), "compared": int(compared), "changes": 0})
            packet_rows.append(
                {
                    "dataset": str(args.dataset),
                    "target_snr_db": float(snr_db),
                    "variant": "current",
                    "packet_index": int(packet_index),
                    "symbol_ser": float(ser),
                    "changes": 0,
                }
            )
        summary_rows.append(
            {
                "dataset": str(args.dataset),
                "target_snr_db": float(snr_db),
                "variant": "current",
                "symbol_ser": _avg(current_rows, "symbol_ser"),
                "mean_changes": 0.0,
            }
        )

        for variant in variants:
            rows_for_variant: list[dict[str, Any]] = []
            for packet_index, result, center_spectra, evidence_powers, abs_indices, gt_bins, coherences, sf, ldro in items:
                refined, changes = _refine_selected(
                    result,
                    center_spectra,
                    evidence_powers,
                    abs_indices,
                    coherences,
                    variant,
                    config,
                )
                _raw, ser, compared = _ser(refined, gt_bins, sf=sf, ldro=ldro)
                row = {
                    "dataset": str(args.dataset),
                    "target_snr_db": float(snr_db),
                    "variant": str(variant.name),
                    "packet_index": int(packet_index),
                    "symbol_ser": float(ser),
                    "changes": int(changes),
                    "compared": int(compared),
                }
                packet_rows.append(row)
                rows_for_variant.append(row)
            summary = {
                "dataset": str(args.dataset),
                "target_snr_db": float(snr_db),
                "variant": str(variant.name),
                "symbol_ser": _avg(rows_for_variant, "symbol_ser"),
                "mean_changes": _avg(rows_for_variant, "changes"),
            }
            summary_rows.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    out_dir = Path(args.output_dir)
    _write_csv(out_dir / "packet_path_rescue_refine.csv", packet_rows)
    _write_csv(out_dir / "snr_path_rescue_refine.csv", summary_rows)
    (out_dir / "snr_path_rescue_refine.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={out_dir / 'snr_path_rescue_refine.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
