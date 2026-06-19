#!/usr/bin/env python3
"""SNR-threshold sweep for symbol-level phase two-stage decoding.

This script evaluates the standard single-center-offset FFT argmax baseline,
the multi-offset argmax baseline, and the current symbol-level phase two-stage
selector under the same synthetic AWGN realizations.  CRC is checked only after
hard symbol decisions; it is not used to choose bins.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_symbol_phase_two_stage import (  # noqa: E402
    _argmax_bins,
    _decode_selected,
    _extract_payload_spectra_with_coherence,
    _extract_payload_spectra,
    _false_locks,
    _candidate_recall,
    _packet_phase_line,
    _ser,
    build_config,
)
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.symbol_phase_two_stage import select_symbol_bins_two_stage  # noqa: E402


DEFAULT_DATASETS = (
    "0_0_0_10_14_8",
    "0_0_0_10_14_16",
    "0_0_0_10_14_32",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SNR thresholds for center FFT, multi-offset, and phase two-stage."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snr-start", type=float, default=-16.0)
    parser.add_argument("--snr-stop", type=float, default=-26.0)
    parser.add_argument("--snr-step", type=float, default=-1.0)
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "symbol_phase_threshold_sweep")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--independent-noise", action="store_true")

    parser.add_argument("--top-l-low-confidence", type=int, default=24)
    parser.add_argument("--lock-margin-db", type=float, default=1.5)
    parser.add_argument("--lock-peak-to-median-db", type=float, default=5.0)
    parser.add_argument("--lock-phase-score", type=float, default=0.35)
    parser.add_argument("--min-locked-for-line", type=int, default=4)
    parser.add_argument("--line-trim-frac", type=float, default=0.25)
    parser.add_argument("--phase-model", choices=("linear", "quadratic"), default="linear")
    parser.add_argument("--selection-mode", choices=("override", "smooth", "coherence", "window", "window_guarded"), default="coherence")
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--trajectory-rmse-scale-pi", type=float, default=0.30)
    parser.add_argument("--trajectory-phase-weight", type=float, default=0.20)
    parser.add_argument("--trajectory-line-weight", type=float, default=0.00)
    parser.add_argument("--trajectory-amp-weight", type=float, default=0.80)
    parser.add_argument("--trajectory-profile-weight", type=float, default=0.00)
    parser.add_argument("--phase-override-min-gain", type=float, default=0.15)
    parser.add_argument("--phase-override-max-drop-db", type=float, default=0.60)
    parser.add_argument("--phase-override-score-margin", type=float, default=0.06)
    parser.add_argument("--phase-override-min-line-anchors", type=int, default=8)
    parser.add_argument("--phase-override-max-line-rmse-pi", type=float, default=0.25)
    parser.add_argument("--coherence-weight", type=float, default=0.0)
    parser.add_argument("--coherence-candidate-top-l", type=int, default=0)
    parser.add_argument("--lock-min-coherence", type=float, default=0.0)
    parser.add_argument("--smooth-phase-weight", type=float, default=0.05)
    parser.add_argument("--smooth-amp-weight", type=float, default=0.50)
    parser.add_argument("--smooth-coherence-weight", type=float, default=0.90)
    parser.add_argument("--smooth-slope-penalty", type=float, default=0.05)
    parser.add_argument("--smooth-curvature-penalty", type=float, default=0.10)
    parser.add_argument("--smooth-max-energy-drop-db", type=float, default=20.0)
    parser.add_argument("--smooth-min-line-anchors", type=int, default=4)
    parser.add_argument("--smooth-min-locked-ratio", type=float, default=0.0)
    parser.add_argument("--smooth-max-line-rmse-pi", type=float, default=float("inf"))
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--window-degree", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--window-phase-weight", type=float, default=0.05)
    parser.add_argument("--window-amp-weight", type=float, default=0.50)
    parser.add_argument("--window-coherence-weight", type=float, default=0.90)
    parser.add_argument("--window-slope-weight", type=float, default=0.00)
    parser.add_argument("--window-curvature-weight", type=float, default=0.00)
    parser.add_argument("--window-phase-scale-pi", type=float, default=0.25)
    parser.add_argument("--window-slope-scale-pi", type=float, default=0.45)
    parser.add_argument("--window-curvature-scale-pi", type=float, default=0.25)
    parser.add_argument("--window-recent-decay", type=float, default=0.75)
    parser.add_argument("--window-anchor-span", type=float, default=8.0)
    parser.add_argument("--window-anchor-min", type=int, default=2)
    parser.add_argument("--window-anchor-max-rmse-pi", type=float, default=0.40)
    parser.add_argument("--window-min-locked-ratio", type=float, default=0.10)
    parser.add_argument("--window-guard-min-phase-gain", type=float, default=0.10)
    parser.add_argument("--window-guard-max-energy-drop-db", type=float, default=0.75)
    parser.add_argument("--window-guard-max-coherence-drop", type=float, default=0.08)
    return parser.parse_args()


def _snr_values(start: float, stop: float, step: float) -> list[float]:
    if abs(float(step)) <= 1e-12:
        raise ValueError("snr-step must be nonzero")
    values: list[float] = []
    current = float(start)
    if float(step) < 0:
        while current >= float(stop) - 1e-9:
            values.append(round(current, 6))
            current += float(step)
    else:
        while current <= float(stop) + 1e-9:
            values.append(round(current, 6))
            current += float(step)
    return values


def _dataset_paths(dataset: str) -> dict[str, Path]:
    base = WEAK_ROOT / "data" / "low_snr_gt_bin" / dataset
    return {
        "iq": WEAK_ROOT.parent / "data" / "USRP_IQ" / f"{dataset}.bin",
        "symbols": WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / f"{dataset}_header_first_symbols.csv",
        "metadata": base / f"{dataset}_low_snr_gt_bin_metadata.json",
    }


def _avg(rows: Sequence[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def _prefix_decode(prefix: str, packet: dict[str, Any], raw_bins: Sequence[int], args: argparse.Namespace) -> dict[str, Any]:
    decoded = _decode_selected(packet, raw_bins, args)
    return {
        f"{prefix}_crc_valid": int(decoded.get("crc_valid", 0)),
        f"{prefix}_payload_hex": decoded.get("payload_hex", ""),
        f"{prefix}_decode_error": decoded.get("decode_error", ""),
    }


def _evaluate_packet_methods(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    config,
) -> dict[str, Any]:
    center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
        samples, packet, args
    )
    evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
    header_line = _packet_phase_line(samples, packet, args)
    result = select_symbol_bins_two_stage(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=config,
        fallback_line=header_line,
        offset_coherences=coherences,
    )

    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    center_bins = _argmax_bins(center_spectra)
    multi_bins = _argmax_bins(multi_spectra)
    selected_bins = tuple(result.selected_raw_bins)

    center_raw_ser, center_symbol_ser, compared = _ser(center_bins, gt_bins, sf=sf, ldro=ldro)
    multi_raw_ser, multi_symbol_ser, _ = _ser(multi_bins, gt_bins, sf=sf, ldro=ldro)
    selected_raw_ser, selected_symbol_ser, _ = _ser(selected_bins, gt_bins, sf=sf, ldro=ldro)
    candidate_hits, candidate_total = _candidate_recall(result, gt_bins)
    false_locks, locked_with_gt = _false_locks(result, gt_bins)
    selected_coherences: list[float] = []
    for idx, raw_bin in enumerate(selected_bins):
        if idx < len(coherences):
            b = int(raw_bin)
            if 0 <= b < coherences[idx].size:
                selected_coherences.append(float(coherences[idx][b]))

    row: dict[str, Any] = {
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "payload_len": int(packet["payload_len"]),
        "symbol_count": int(len(selected_bins)),
        "gt_compared_symbols": int(compared),
        "center_raw_ser": float(center_raw_ser),
        "center_symbol_ser": float(center_symbol_ser),
        "center_symbol_accuracy": float(1.0 - center_symbol_ser),
        "multi_raw_ser": float(multi_raw_ser),
        "multi_symbol_ser": float(multi_symbol_ser),
        "multi_symbol_accuracy": float(1.0 - multi_symbol_ser),
        "selected_raw_ser": float(selected_raw_ser),
        "selected_symbol_ser": float(selected_symbol_ser),
        "selected_symbol_accuracy": float(1.0 - selected_symbol_ser),
        "locked_symbol_count": int(result.locked_count),
        "uncertain_symbol_count": int(result.uncertain_count),
        "locked_ratio": float(result.locked_count / max(1, len(selected_bins))),
        "false_lock_count": int(false_locks),
        "locked_with_gt_count": int(locked_with_gt),
        "false_lock_rate": float(false_locks / max(1, locked_with_gt)),
        "uncertain_candidate_hits": int(candidate_hits),
        "uncertain_candidate_total": int(candidate_total),
        "uncertain_candidate_recall": float(candidate_hits / max(1, candidate_total)),
        "mean_selected_offset_coherence": float(np.mean(selected_coherences)) if selected_coherences else 0.0,
        "phase_line_r2": float(result.phase_line.fit_r2) if math.isfinite(result.phase_line.fit_r2) else "",
        "phase_line_rmse_pi": float(result.phase_line.fit_rmse_pi) if math.isfinite(result.phase_line.fit_rmse_pi) else "",
        "phase_line_anchor_count": int(result.phase_line.anchor_count),
    }
    row.update(_prefix_decode("center", packet, center_bins, args))
    row.update(_prefix_decode("multi", packet, multi_bins, args))
    row.update(_prefix_decode("selected", packet, selected_bins, args))
    return row


def _summarize(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "packet_count": int(len(rows)),
        "center_symbol_ser": _avg(rows, "center_symbol_ser"),
        "center_symbol_accuracy": _avg(rows, "center_symbol_accuracy"),
        "center_crc_valid_rate": _avg(rows, "center_crc_valid"),
        "multi_symbol_ser": _avg(rows, "multi_symbol_ser"),
        "multi_symbol_accuracy": _avg(rows, "multi_symbol_accuracy"),
        "multi_crc_valid_rate": _avg(rows, "multi_crc_valid"),
        "selected_symbol_ser": _avg(rows, "selected_symbol_ser"),
        "selected_symbol_accuracy": _avg(rows, "selected_symbol_accuracy"),
        "selected_crc_valid_rate": _avg(rows, "selected_crc_valid"),
        "selected_ser_gain_vs_center": _avg(rows, "center_symbol_ser") - _avg(rows, "selected_symbol_ser"),
        "selected_crc_gain_vs_center": _avg(rows, "selected_crc_valid") - _avg(rows, "center_crc_valid"),
        "selected_ser_gain_vs_multi": _avg(rows, "multi_symbol_ser") - _avg(rows, "selected_symbol_ser"),
        "selected_crc_gain_vs_multi": _avg(rows, "selected_crc_valid") - _avg(rows, "multi_crc_valid"),
        "mean_locked_ratio": _avg(rows, "locked_ratio"),
        "mean_false_lock_rate": _avg(rows, "false_lock_rate"),
        "mean_uncertain_candidate_recall": _avg(rows, "uncertain_candidate_recall"),
        "mean_selected_offset_coherence": _avg(rows, "mean_selected_offset_coherence"),
        "mean_phase_line_r2": _avg(rows, "phase_line_r2"),
        "mean_phase_line_rmse_pi": _avg(rows, "phase_line_rmse_pi"),
    }


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


def _threshold_from_curve(
    curve: Sequence[dict[str, Any]],
    metric_key: str,
    target: float,
    higher_is_better: bool,
) -> tuple[float | None, str]:
    points: list[tuple[float, float]] = []
    for row in curve:
        try:
            snr = float(row["target_snr_db"])
            value = float(row[metric_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(snr) and math.isfinite(value):
            points.append((snr, value))
    if not points:
        return None, "missing"
    points.sort(key=lambda item: item[0], reverse=True)

    def passes(value: float) -> bool:
        return value >= float(target) if higher_is_better else value <= float(target)

    pass_flags = [passes(value) for _snr, value in points]
    if all(pass_flags):
        return min(snr for snr, _value in points), "all_pass"
    if not any(pass_flags):
        return None, "all_fail"

    for idx in range(len(points) - 1):
        snr_hi, value_hi = points[idx]
        snr_lo, value_lo = points[idx + 1]
        if passes(value_hi) and not passes(value_lo):
            if abs(value_lo - value_hi) <= 1e-12:
                return snr_hi, "flat_crossing"
            ratio = (float(target) - value_hi) / (value_lo - value_hi)
            ratio = max(0.0, min(1.0, float(ratio)))
            return float(snr_hi + ratio * (snr_lo - snr_hi)), "interpolated"
    return None, "non_monotonic"


def _threshold_tables(summary_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = (
        ("SER<=10%", "symbol_ser", 0.10, False),
        ("accuracy>=90%", "symbol_accuracy", 0.90, True),
        ("CRC/PRR>=90%", "crc_valid_rate", 0.90, True),
        ("CRC/PRR>=80%", "crc_valid_rate", 0.80, True),
        ("CRC/PRR>=50%", "crc_valid_rate", 0.50, True),
    )
    methods = ("center", "multi", "selected")
    datasets = sorted({str(row["dataset"]) for row in summary_rows})
    thresholds: list[dict[str, Any]] = []
    for dataset in datasets:
        curve = [row for row in summary_rows if str(row["dataset"]) == dataset]
        for method in methods:
            for metric_name, suffix, target, higher_is_better in specs:
                key = f"{method}_{suffix}"
                threshold, status = _threshold_from_curve(curve, key, target, higher_is_better)
                thresholds.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric_name,
                        "threshold_snr_db": "" if threshold is None else float(threshold),
                        "status": status,
                    }
                )

    gains: list[dict[str, Any]] = []
    by_key = {
        (row["dataset"], row["method"], row["metric"]): row
        for row in thresholds
    }
    comparisons = (
        ("selected", "center"),
        ("multi", "center"),
        ("selected", "multi"),
    )
    for dataset in datasets:
        for method, baseline in comparisons:
            for metric_name, _suffix, _target, _higher in specs:
                method_row = by_key[(dataset, method, metric_name)]
                base_row = by_key[(dataset, baseline, metric_name)]
                try:
                    method_thr = float(method_row["threshold_snr_db"])
                    base_thr = float(base_row["threshold_snr_db"])
                    gain = base_thr - method_thr
                    status = "ok"
                except (TypeError, ValueError):
                    method_thr = float("nan")
                    base_thr = float("nan")
                    gain = float("nan")
                    status = "missing_threshold"
                gains.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "baseline": baseline,
                        "metric": metric_name,
                        "method_threshold_snr_db": "" if not math.isfinite(method_thr) else method_thr,
                        "baseline_threshold_snr_db": "" if not math.isfinite(base_thr) else base_thr,
                        "gain_db": "" if not math.isfinite(gain) else gain,
                        "status": status,
                    }
                )
    return thresholds, gains


def main() -> int:
    args = parse_args()
    config = build_config(args)
    snrs = _snr_values(args.snr_start, args.snr_stop, args.snr_step)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        paths = _dataset_paths(str(dataset))
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        signal_power = float(metadata["signal_reference_power"])
        base_seed = int(metadata.get("seed", 42))
        samples = np.fromfile(paths["iq"], dtype=np.complex64)
        if samples.size == 0:
            raise ValueError(f"empty IQ file: {paths['iq']}")
        packets = load_packets(paths["symbols"], None)
        if not packets:
            raise ValueError(f"no packets loaded: {paths['symbols']}")

        unit_noise: np.ndarray | None = None
        if not bool(args.independent_noise):
            rng = np.random.default_rng(base_seed)
            noise_i = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
            noise_q = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
            unit_noise = (noise_i + 1j * noise_q).astype(np.complex64)

        for step_index, snr_db in enumerate(snrs):
            noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
            if unit_noise is None:
                rng = np.random.default_rng(base_seed + step_index)
                sigma = math.sqrt(float(noise_power) / 2.0)
                noise_i = rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                noise_q = rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                noisy = (samples + (noise_i + 1j * noise_q)).astype(np.complex64, copy=False)
            else:
                sigma = math.sqrt(float(noise_power) / 2.0)
                noisy = (samples + sigma * unit_noise).astype(np.complex64, copy=False)

            rows: list[dict[str, Any]] = []
            for packet_index in sorted(packets):
                row = _evaluate_packet_methods(noisy, packets[packet_index], args, config)
                row["dataset"] = str(dataset)
                row["target_snr_db"] = float(snr_db)
                rows.append(row)
                packet_rows.append(row)
            summary = _summarize(rows, str(dataset), float(snr_db))
            summary_rows.append(summary)
            print(
                f"{dataset} snr={snr_db:>6.1f} "
                f"center_ser={summary['center_symbol_ser']:.3f} center_crc={summary['center_crc_valid_rate']:.3f} "
                f"selected_ser={summary['selected_symbol_ser']:.3f} selected_crc={summary['selected_crc_valid_rate']:.3f}",
                flush=True,
            )

    for snr_db in snrs:
        group = [row for row in summary_rows if abs(float(row["target_snr_db"]) - float(snr_db)) < 1e-9]
        aggregate = {"dataset": "mean_of_datasets", "target_snr_db": float(snr_db), "packet_count": sum(int(row["packet_count"]) for row in group)}
        for key in summary_rows[0]:
            if key in aggregate or key in ("dataset", "target_snr_db", "packet_count"):
                continue
            aggregate[key] = float(np.mean([float(row[key]) for row in group]))
        summary_rows.append(aggregate)

    threshold_rows, gain_rows = _threshold_tables(summary_rows)
    _write_csv(out_dir / "per_packet_metrics.csv", packet_rows)
    _write_csv(out_dir / "snr_curve_summary.csv", summary_rows)
    _write_csv(out_dir / "threshold_table.csv", threshold_rows)
    _write_csv(out_dir / "gain_table.csv", gain_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "snr_values": snrs,
                "datasets": list(args.datasets),
                "summary_rows": summary_rows,
                "threshold_rows": threshold_rows,
                "gain_rows": gain_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote={out_dir / 'snr_curve_summary.csv'}")
    print(f"wrote={out_dir / 'threshold_table.csv'}")
    print(f"wrote={out_dir / 'gain_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
