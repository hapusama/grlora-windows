#!/usr/bin/env python3
"""Evaluate direct phase-line selection under synthetic AWGN."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


THIS = Path(__file__).resolve()
WEAK_ROOT = THIS.parents[2]
EXPERIMENT_DIR = WEAK_ROOT / "scripts" / "experiments"
for path in (str(WEAK_ROOT), str(EXPERIMENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_symbol_phase_threshold_sweep import _dataset_paths, _snr_values  # noqa: E402
from run_symbol_phase_two_stage import (  # noqa: E402
    _argmax_bins,
    _decode_selected,
    _extract_payload_spectra_with_coherence,
    _packet_phase_line,
    _ser,
)
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.pure_phaseline import PurePhaseLineConfig, select_pure_phase_line_bins  # noqa: E402
from weak_decoder.pure_phaseline import select_pure_phase_path_bins  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pure phase-line FFT-bin selection.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--snrs", nargs="+", type=float, default=None)
    parser.add_argument("--snr-start", type=float, default=-22.0)
    parser.add_argument("--snr-stop", type=float, default=-26.0)
    parser.add_argument("--snr-step", type=float, default=-1.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "baseline_comparison" / "pure_phaseline",
    )
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--independent-noise", action="store_true")
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--spectrum-source", choices=("center", "multi"), default="center")
    parser.add_argument("--line-source", choices=("header", "payload_argmax", "payload_consensus"), default="payload_consensus")
    parser.add_argument("--selection-mode", choices=("line", "path"), default="path")
    parser.add_argument("--candidate-mode", choices=("all", "energy_floor", "top_l"), default="top_l")
    parser.add_argument("--energy-floor-db", type=float, default=24.0)
    parser.add_argument("--top-l", type=int, default=32)
    parser.add_argument("--phase-weight", type=float, default=1.0)
    parser.add_argument("--amplitude-weight", type=float, default=0.05)
    parser.add_argument("--path-transition-weight", type=float, default=0.65)
    parser.add_argument("--path-amplitude-weight", type=float, default=0.35)
    parser.add_argument("--path-line-weight", type=float, default=0.05)
    parser.add_argument("--path-slope-span-pi", type=float, default=0.30)
    parser.add_argument("--path-slope-steps", type=int, default=13)
    parser.add_argument("--refine-iterations", type=int, default=1)
    parser.add_argument("--trim-frac", type=float, default=0.20)
    parser.add_argument("--fallback-anchor-fraction", type=float, default=0.50)
    parser.add_argument("--fallback-min-margin-db", type=float, default=0.0)
    parser.add_argument("--consensus-slope-steps", type=int, default=41)
    parser.add_argument("--consensus-intercept-steps", type=int, default=96)
    parser.add_argument("--consensus-slope-span-pi", type=float, default=0.25)
    parser.add_argument("--consensus-phase-weight", type=float, default=0.75)
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


def _load_metadata(paths: dict[str, Path], dataset: str) -> dict[str, Any]:
    if paths["metadata"].exists():
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))
    low_snr_root = WEAK_ROOT / "data" / "low_snr_gt_bin"
    for path in sorted(low_snr_root.glob(f"{dataset}*/*_metadata.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "signal_reference_power" in metadata:
            return metadata
    return {}


def _payload_reference_power(samples: np.ndarray, packets: dict[int, dict[str, Any]]) -> tuple[float, int]:
    total_power = 0.0
    total_count = 0
    for packet in packets.values():
        symbol_len = (1 << int(packet["sf"])) * int(packet["os_factor"])
        for symbol in packet["payload_symbols"]:
            start = int(symbol["start_sample"])
            stop = start + symbol_len
            if start < 0 or stop > samples.size:
                continue
            chunk = np.asarray(samples[start:stop], dtype=np.complex64)
            total_power += float(np.sum(np.abs(chunk).astype(np.float64) ** 2))
            total_count += int(chunk.size)
    if total_count <= 0:
        return float(np.mean(np.abs(samples).astype(np.float64) ** 2)), int(samples.size)
    return float(total_power / total_count), int(total_count)


def _pure_config(args: argparse.Namespace) -> PurePhaseLineConfig:
    return PurePhaseLineConfig(
        candidate_mode=str(args.candidate_mode),
        energy_floor_db=float(args.energy_floor_db),
        top_l=int(args.top_l),
        phase_weight=float(args.phase_weight),
        amplitude_weight=float(args.amplitude_weight),
        path_transition_weight=float(args.path_transition_weight),
        path_amplitude_weight=float(args.path_amplitude_weight),
        path_line_weight=float(args.path_line_weight),
        path_slope_span_pi=float(args.path_slope_span_pi),
        path_slope_steps=int(args.path_slope_steps),
        refine_iterations=int(args.refine_iterations),
        trim_frac=float(args.trim_frac),
        fallback_line_source="consensus" if str(args.line_source) == "payload_consensus" else "argmax",
        fallback_anchor_fraction=float(args.fallback_anchor_fraction),
        fallback_min_margin_db=float(args.fallback_min_margin_db),
        consensus_slope_steps=int(args.consensus_slope_steps),
        consensus_intercept_steps=int(args.consensus_intercept_steps),
        consensus_slope_span_pi=float(args.consensus_slope_span_pi),
        consensus_phase_weight=float(args.consensus_phase_weight),
    )


def _evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    config: PurePhaseLineConfig,
) -> dict[str, Any]:
    center_spectra, multi_spectra, abs_indices, gt_bins, _coherences = _extract_payload_spectra_with_coherence(
        samples,
        packet,
        args,
    )
    spectra = multi_spectra if str(args.spectrum_source) == "multi" else center_spectra
    header_line = _packet_phase_line(samples, packet, args)
    reference_line = None if str(args.line_source) == "payload_argmax" else header_line
    selector = select_pure_phase_path_bins if str(args.selection_mode) == "path" else select_pure_phase_line_bins
    result = selector(spectra=spectra, abs_indices=abs_indices, reference_line=reference_line, config=config)

    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    center_bins = _argmax_bins(center_spectra)
    multi_bins = _argmax_bins(multi_spectra)
    pure_bins = tuple(int(v) for v in result.selected_raw_bins)
    center_raw_ser, center_symbol_ser, compared = _ser(center_bins, gt_bins, sf=sf, ldro=ldro)
    multi_raw_ser, multi_symbol_ser, _ = _ser(multi_bins, gt_bins, sf=sf, ldro=ldro)
    pure_raw_ser, pure_symbol_ser, _ = _ser(pure_bins, gt_bins, sf=sf, ldro=ldro)
    decoded = _decode_selected(packet, pure_bins, args)

    return {
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "payload_len": int(packet["payload_len"]),
        "cr": int(packet["cr"]),
        "has_crc": int(bool(packet["has_crc"])),
        "ldro": int(ldro),
        "symbol_count": int(len(pure_bins)),
        "gt_compared_symbols": int(compared),
        "center_raw_ser": float(center_raw_ser),
        "center_symbol_ser": float(center_symbol_ser),
        "multi_raw_ser": float(multi_raw_ser),
        "multi_symbol_ser": float(multi_symbol_ser),
        "pure_raw_ser": float(pure_raw_ser),
        "pure_symbol_ser": float(pure_symbol_ser),
        "pure_gain_vs_center": float(center_symbol_ser - pure_symbol_ser),
        "pure_gain_vs_multi": float(multi_symbol_ser - pure_symbol_ser),
        "pure_crc_valid": int(decoded.get("crc_valid", 0)),
        "pure_payload_hex": decoded.get("payload_hex", ""),
        "pure_decode_error": decoded.get("decode_error", ""),
        "mean_phase_score": float(result.mean_phase_score),
        "mean_amp_score": float(result.mean_amp_score),
        "mean_candidate_count": float(result.mean_candidate_count),
        "mean_selected_energy_drop_db": float(result.mean_selected_energy_drop_db),
        "phase_line_slope_pi": float(result.phase_line.slope_pi),
        "phase_line_r2": float(result.phase_line.fit_r2) if math.isfinite(result.phase_line.fit_r2) else "",
        "phase_line_rmse_pi": float(result.phase_line.fit_rmse_pi) if math.isfinite(result.phase_line.fit_rmse_pi) else "",
        "phase_line_anchor_count": int(result.phase_line.anchor_count),
        "reference_line_slope_pi": float(result.reference_line.slope_pi),
        "reference_line_anchor_count": int(result.reference_line.anchor_count),
        "refinement_count": int(result.refinement_count),
        "selection_mode": str(args.selection_mode),
        "error": result.error,
    }


def _summarize(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float, seed: int) -> dict[str, Any]:
    return {
        "dataset": str(dataset),
        "target_snr_db": float(snr_db),
        "noise_seed": int(seed),
        "packet_count": int(len(rows)),
        "center_symbol_ser": _avg(rows, "center_symbol_ser"),
        "multi_symbol_ser": _avg(rows, "multi_symbol_ser"),
        "pure_symbol_ser": _avg(rows, "pure_symbol_ser"),
        "pure_gain_vs_center": _avg(rows, "pure_gain_vs_center"),
        "pure_gain_vs_multi": _avg(rows, "pure_gain_vs_multi"),
        "pure_crc_valid_rate": _avg(rows, "pure_crc_valid"),
        "mean_phase_score": _avg(rows, "mean_phase_score"),
        "mean_amp_score": _avg(rows, "mean_amp_score"),
        "mean_candidate_count": _avg(rows, "mean_candidate_count"),
        "mean_selected_energy_drop_db": _avg(rows, "mean_selected_energy_drop_db"),
        "mean_phase_line_r2": _avg(rows, "phase_line_r2"),
        "mean_phase_line_rmse_pi": _avg(rows, "phase_line_rmse_pi"),
    }


def _aggregate_by_snr(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in summary_rows:
        groups.setdefault(float(row["target_snr_db"]), []).append(row)
    out: list[dict[str, Any]] = []
    for snr_db, group in sorted(groups.items(), reverse=True):
        out.append(
            {
                "dataset": "mean",
                "target_snr_db": float(snr_db),
                "noise_seed": "mean",
                "packet_count": int(sum(int(row["packet_count"]) for row in group)),
                "center_symbol_ser": _avg(group, "center_symbol_ser"),
                "multi_symbol_ser": _avg(group, "multi_symbol_ser"),
                "pure_symbol_ser": _avg(group, "pure_symbol_ser"),
                "pure_gain_vs_center": _avg(group, "pure_gain_vs_center"),
                "pure_gain_vs_multi": _avg(group, "pure_gain_vs_multi"),
                "pure_crc_valid_rate": _avg(group, "pure_crc_valid_rate"),
                "mean_phase_score": _avg(group, "mean_phase_score"),
                "mean_amp_score": _avg(group, "mean_amp_score"),
                "mean_candidate_count": _avg(group, "mean_candidate_count"),
                "mean_selected_energy_drop_db": _avg(group, "mean_selected_energy_drop_db"),
                "mean_phase_line_r2": _avg(group, "mean_phase_line_r2"),
                "mean_phase_line_rmse_pi": _avg(group, "mean_phase_line_rmse_pi"),
            }
        )
    return out


def _dataset_seeds(args: argparse.Namespace, metadata: dict[str, Any], dataset_index: int) -> list[int]:
    if args.seeds:
        return [int(v) for v in args.seeds]
    return [int(metadata.get("seed", 42)) + int(dataset_index) * 100000]


def main() -> int:
    args = parse_args()
    datasets = [str(v) for v in (args.datasets if args.datasets is not None else [args.dataset])]
    snrs = [float(v) for v in args.snrs] if args.snrs is not None else _snr_values(
        float(args.snr_start),
        float(args.snr_stop),
        float(args.snr_step),
    )
    config = _pure_config(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(datasets):
        paths = _dataset_paths(dataset)
        metadata = _load_metadata(paths, dataset)
        samples = np.fromfile(paths["iq"], dtype=np.complex64)
        packets = load_packets(paths["symbols"], args.packet)
        if samples.size == 0:
            raise ValueError(f"empty IQ file: {paths['iq']}")
        if not packets:
            raise ValueError(f"no packets loaded: {paths['symbols']}")
        if args.signal_reference_power is not None:
            signal_power = float(args.signal_reference_power)
        elif "signal_reference_power" in metadata:
            signal_power = float(metadata["signal_reference_power"])
        else:
            signal_power, ref_count = _payload_reference_power(samples, packets)
            print(f"{dataset}: using payload reference power {signal_power:.6g} from {ref_count} samples", flush=True)

        seeds = _dataset_seeds(args, metadata, dataset_index)
        manifests.append(
            {
                "dataset": dataset,
                "input_iq": str(paths["iq"]),
                "symbol_csv": str(paths["symbols"]),
                "signal_reference_power": float(signal_power),
                "seeds": seeds,
                "packet_count": int(len(packets)),
            }
        )
        for seed in seeds:
            unit_noise: np.ndarray | None = None
            if not bool(args.independent_noise):
                rng = np.random.default_rng(int(seed))
                unit_noise = (
                    rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
                    + 1j * rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
                ).astype(np.complex64)
            for step_index, snr_db in enumerate(snrs):
                noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
                sigma = math.sqrt(float(noise_power) / 2.0)
                if unit_noise is None:
                    rng = np.random.default_rng(int(seed) + int(step_index))
                    noise = (
                        rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                        + 1j * rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                    ).astype(np.complex64)
                    noisy = (samples + noise).astype(np.complex64, copy=False)
                else:
                    noisy = (samples + sigma * unit_noise).astype(np.complex64, copy=False)

                rows: list[dict[str, Any]] = []
                for packet_index in sorted(packets):
                    row = _evaluate_packet(noisy, packets[packet_index], args, config)
                    row["dataset"] = dataset
                    row["target_snr_db"] = float(snr_db)
                    row["noise_seed"] = int(seed)
                    rows.append(row)
                    packet_rows.append(row)
                summary = _summarize(rows, dataset, float(snr_db), int(seed))
                summary_rows.append(summary)
                print(
                    f"{dataset} seed={seed} snr={snr_db:>6.1f} "
                    f"center={summary['center_symbol_ser']:.3f} "
                    f"multi={summary['multi_symbol_ser']:.3f} "
                    f"pure={summary['pure_symbol_ser']:.3f} "
                    f"cand={summary['mean_candidate_count']:.1f} "
                    f"phase={summary['mean_phase_score']:.3f}",
                    flush=True,
                )

    aggregate_rows = _aggregate_by_snr(summary_rows)
    _write_csv(out_dir / "per_packet_metrics.csv", packet_rows)
    _write_csv(out_dir / "snr_summary.csv", summary_rows + aggregate_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "datasets": datasets,
                "snrs": snrs,
                "config": config.__dict__,
                "spectrum_source": str(args.spectrum_source),
                "line_source": str(args.line_source),
                "manifests": manifests,
                "summary_rows": summary_rows,
                "aggregate_rows": aggregate_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote={out_dir / 'snr_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
