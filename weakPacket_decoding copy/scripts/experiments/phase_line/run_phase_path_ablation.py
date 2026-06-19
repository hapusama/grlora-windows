#!/usr/bin/env python3
"""Ablate payload-only phase-smooth path decoding variants.

This script tests the methods proposed in the phase-line handoff:

1. Evidence-only center FFT argmax.
2. Evidence-only multi-offset argmax.
3. Evidence-only Top-L oracle recall.
4. Existing v3/current selector.
5. Existing local phase-line rerank.
6. First-order phase DP.
7. Second-order phase DP.
8. Second-order phase DP with high-confidence soft anchor.

CRC is reported only as a final verification signal; it is not used in any
candidate search.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
WEAK_ROOT = SCRIPT_DIR.parents[2]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_symbol_phase_threshold_sweep import (  # noqa: E402
    DEFAULT_DATASETS,
    _avg,
    _dataset_paths,
    _snr_values,
    _write_csv,
)
from run_symbol_phase_two_stage import (  # noqa: E402
    _argmax_bins,
    _candidate_recall,
    _decode_selected,
    _extract_payload_spectra_with_coherence,
    _packet_phase_line,
    _ser,
)
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.candidate_pruning import wrap_phase  # noqa: E402
from weak_decoder.phase_line import (  # noqa: E402
    PhaseLineSelectorConfig,
    PhasePathSelectorConfig,
    select_phase_smooth_path,
    select_phase_viterbi_path,
)
from weak_decoder.symbol_phase_two_stage import SymbolPhaseConfig, select_symbol_bins_two_stage  # noqa: E402


METHODS = (
    "center",
    "multi",
    "v3",
    "phase_line",
    "phase_dp_first",
    "phase_dp_first_header",
    "phase_dp_second",
    "phase_dp_second_anchor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-smooth path-decoding ablations.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snr-start", type=float, default=-22.0)
    parser.add_argument("--snr-stop", type=float, default=-24.0)
    parser.add_argument("--snr-step", type=float, default=-1.0)
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "phase_path_ablation")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--independent-noise", action="store_true")
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--top-l", type=int, default=24)
    parser.add_argument("--dp-max-energy-drop-db", type=float, default=24.0)
    parser.add_argument("--dp-energy-weight", type=float, default=0.35)
    parser.add_argument("--dp-coherence-weight", type=float, default=0.05)
    parser.add_argument("--dp-rank-weight", type=float, default=0.05)
    parser.add_argument("--first-lambda", type=float, default=0.18)
    parser.add_argument("--first-scale-pi", type=float, default=0.75)
    parser.add_argument("--second-lambda", type=float, default=0.45)
    parser.add_argument("--second-first-lambda", type=float, default=0.00)
    parser.add_argument("--second-scale-pi", type=float, default=0.35)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--anchor-bonus", type=float, default=0.12)
    parser.add_argument("--anchor-top-k", type=int, default=0)
    parser.add_argument("--header-slope-weight", type=float, default=0.08)
    parser.add_argument("--header-slope-scale-pi", type=float, default=0.50)
    parser.add_argument("--header-slope-span", type=int, default=4)
    parser.add_argument("--high-conf-margin-db", type=float, default=3.0)
    parser.add_argument("--high-conf-peak-db", type=float, default=7.0)
    parser.add_argument("--high-conf-min-coherence", type=float, default=0.0)
    parser.add_argument("--random-paths", type=int, default=64)
    return parser.parse_args()


def _load_metadata(paths: dict[str, Path], dataset: str) -> dict[str, Any]:
    if paths["metadata"].exists():
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))
    low_snr_root = WEAK_ROOT / "data" / "low_snr_gt_bin"
    matches = sorted(low_snr_root.glob(f"{dataset}*/*_metadata.json"))
    for path in matches:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "signal_reference_power" in metadata:
            print(f"{dataset}: using metadata fallback {path}", flush=True)
            return metadata
    return {}


def _build_v3_config(top_l: int) -> SymbolPhaseConfig:
    return SymbolPhaseConfig(
        top_l_low_confidence=int(top_l),
        lock_margin_db=1.5,
        lock_peak_to_median_db=5.0,
        lock_phase_score=0.35,
        min_locked_for_line=4,
        line_trim_frac=0.25,
        phase_model="linear",
        selection_mode="coherence",
        beam_width=128,
        trajectory_rmse_scale_pi=0.30,
        phase_weight=0.20,
        line_weight=0.00,
        amp_weight=0.80,
        profile_weight=0.00,
        phase_override_min_gain=0.15,
        phase_override_max_drop_db=0.60,
        phase_override_score_margin=0.06,
        phase_override_min_line_anchors=8,
        phase_override_max_line_rmse_pi=0.25,
        coherence_weight=0.0,
        coherence_candidate_top_l=0,
        lock_min_coherence=0.0,
        smooth_phase_weight=0.05,
        smooth_amp_weight=0.50,
        smooth_coherence_weight=0.90,
        smooth_slope_penalty=0.05,
        smooth_curvature_penalty=0.10,
        smooth_max_energy_drop_db=20.0,
        smooth_min_line_anchors=4,
        smooth_min_locked_ratio=0.0,
        smooth_max_line_rmse_pi=float("inf"),
        window_size=5,
        window_degree=1,
        window_phase_weight=0.05,
        window_amp_weight=0.50,
        window_coherence_weight=0.90,
        window_slope_weight=0.00,
        window_curvature_weight=0.00,
        window_phase_scale_pi=0.25,
        window_slope_scale_pi=0.45,
        window_curvature_scale_pi=0.25,
        window_recent_decay=0.75,
        window_anchor_span=8.0,
        window_anchor_min=2,
        window_anchor_max_rmse_pi=0.40,
        window_min_locked_ratio=0.10,
        window_guard_min_phase_gain=0.10,
        window_guard_max_energy_drop_db=0.75,
        window_guard_max_coherence_drop=0.08,
    )


def _build_phase_line_config(args: argparse.Namespace) -> PhaseLineSelectorConfig:
    return PhaseLineSelectorConfig(
        top_l=int(args.top_l),
        coherence_weight=0.03,
        phase_weight_low_conf=0.82,
        phase_weight_high_conf=0.38,
        energy_weight_low_conf=0.15,
        energy_weight_high_conf=0.57,
        phase_scale_pi=0.28,
        max_energy_drop_db_low_conf=18.0,
        max_energy_drop_db_high_conf=3.0,
    )


def _base_path_config(args: argparse.Namespace, phase_order: int) -> PhasePathSelectorConfig:
    return PhasePathSelectorConfig(
        top_l=int(args.top_l),
        phase_order=int(phase_order),
        energy_weight=float(args.dp_energy_weight),
        coherence_weight=float(args.dp_coherence_weight),
        rank_weight=float(args.dp_rank_weight),
        first_order_weight=float(args.first_lambda if phase_order == 1 else args.second_first_lambda),
        second_order_weight=float(args.second_lambda if phase_order >= 2 else 0.0),
        first_order_scale_pi=float(args.first_scale_pi),
        second_order_scale_pi=float(args.second_scale_pi),
        huber_delta=float(args.huber_delta),
        max_energy_drop_db=float(args.dp_max_energy_drop_db),
        high_confidence_margin_db=float(args.high_conf_margin_db),
        high_confidence_peak_to_median_db=float(args.high_conf_peak_db),
        high_confidence_min_coherence=float(args.high_conf_min_coherence),
        top1_soft_bonus=0.0,
        high_confidence_top_k=0,
    )


def _anchor_path_config(args: argparse.Namespace) -> PhasePathSelectorConfig:
    base = _base_path_config(args, phase_order=2)
    return PhasePathSelectorConfig(
        **{
            **base.__dict__,
            "top1_soft_bonus": float(args.anchor_bonus),
            "high_confidence_top_k": int(args.anchor_top_k),
        }
    )


def _header_path_config(args: argparse.Namespace) -> PhasePathSelectorConfig:
    base = _base_path_config(args, phase_order=1)
    return PhasePathSelectorConfig(
        **{
            **base.__dict__,
            "header_slope_weight": float(args.header_slope_weight),
            "header_slope_scale_pi": float(args.header_slope_scale_pi),
            "header_slope_span": int(args.header_slope_span),
        }
    )


def _prefix_decode(prefix: str, packet: dict[str, Any], raw_bins: Sequence[int], args: argparse.Namespace) -> dict[str, Any]:
    decoded = _decode_selected(packet, raw_bins, args)
    return {
        f"{prefix}_crc_valid": int(decoded.get("crc_valid", 0)),
        f"{prefix}_payload_hex": decoded.get("payload_hex", ""),
        f"{prefix}_decode_error": decoded.get("decode_error", ""),
    }


def _path_phase_metrics(center_spectra: Sequence[np.ndarray], raw_bins: Sequence[int]) -> dict[str, Any]:
    phases: list[float] = []
    for spectrum, raw_bin in zip(center_spectra, raw_bins):
        b = int(raw_bin)
        spec = np.asarray(spectrum)
        if b < 0 or b >= spec.size:
            continue
        phases.append(float(np.angle(spec[b])))
    if len(phases) < 2:
        return {"first_abs_pi": "", "second_abs_pi": ""}
    r1 = [abs(float(wrap_phase(phases[idx] - phases[idx - 1]))) / math.pi for idx in range(1, len(phases))]
    if len(phases) < 3:
        r2: list[float] = []
    else:
        r2 = [
            abs(float(wrap_phase(phases[idx] - 2.0 * phases[idx - 1] + phases[idx - 2]))) / math.pi
            for idx in range(2, len(phases))
        ]
    return {
        "first_abs_pi": float(np.mean(r1)) if r1 else "",
        "second_abs_pi": float(np.mean(r2)) if r2 else "",
    }


def _random_second_smoothness(
    center_spectra: Sequence[np.ndarray],
    evidences,
    rng: np.random.Generator,
    random_paths: int,
) -> dict[str, Any]:
    values: list[float] = []
    for _ in range(max(0, int(random_paths))):
        bins: list[int] = []
        for ev in evidences:
            choices = [int(v) for v in ev.top_bins]
            if not choices:
                choices = [int(ev.top1_bin)]
            bins.append(int(choices[int(rng.integers(0, len(choices)))]))
        metrics = _path_phase_metrics(center_spectra, bins)
        try:
            values.append(float(metrics["second_abs_pi"]))
        except (TypeError, ValueError):
            continue
    if not values:
        return {"random_second_abs_pi_mean": "", "random_second_abs_pi_p10": "", "random_second_abs_pi_p50": ""}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "random_second_abs_pi_mean": float(np.mean(arr)),
        "random_second_abs_pi_p10": float(np.percentile(arr, 10.0)),
        "random_second_abs_pi_p50": float(np.percentile(arr, 50.0)),
    }


def _evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    v3_config: SymbolPhaseConfig,
    phase_line_config: PhaseLineSelectorConfig,
    first_config: PhasePathSelectorConfig,
    header_config: PhasePathSelectorConfig,
    second_config: PhasePathSelectorConfig,
    anchor_config: PhasePathSelectorConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
        samples, packet, args
    )
    evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
    header_line = _packet_phase_line(samples, packet, args)
    v3_result = select_symbol_bins_two_stage(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=v3_config,
        fallback_line=header_line,
        offset_coherences=coherences,
    )
    phase_line_result = select_phase_smooth_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=phase_line_config,
        fallback_line=header_line,
        offset_coherences=coherences,
    )
    first_result = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=first_config,
        offset_coherences=coherences,
    )
    header_result = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=header_config,
        fallback_line=header_line,
        offset_coherences=coherences,
    )
    second_result = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=second_config,
        offset_coherences=coherences,
    )
    anchor_result = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=anchor_config,
        offset_coherences=coherences,
    )

    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    center_bins = _argmax_bins(center_spectra)
    multi_bins = _argmax_bins(multi_spectra)
    method_bins = {
        "center": tuple(center_bins),
        "multi": tuple(multi_bins),
        "v3": tuple(v3_result.selected_raw_bins),
        "phase_line": tuple(phase_line_result.selected_raw_bins),
        "phase_dp_first": tuple(first_result.selected_raw_bins),
        "phase_dp_first_header": tuple(header_result.selected_raw_bins),
        "phase_dp_second": tuple(second_result.selected_raw_bins),
        "phase_dp_second_anchor": tuple(anchor_result.selected_raw_bins),
    }
    _, top_l_total = _candidate_recall(first_result, gt_bins)
    top_l_hits, _ = _candidate_recall(first_result, gt_bins)

    row: dict[str, Any] = {
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "payload_len": int(packet["payload_len"]),
        "symbol_count": int(len(multi_bins)),
        "top_l_candidate_hits": int(top_l_hits),
        "top_l_candidate_total": int(top_l_total),
        "top_l_candidate_recall": float(top_l_hits / max(1, top_l_total)),
        "phase_line_error": phase_line_result.error,
        "phase_dp_first_beam_final_size": int(first_result.beam_final_size),
        "phase_dp_first_header_beam_final_size": int(header_result.beam_final_size),
        "phase_dp_second_beam_final_size": int(second_result.beam_final_size),
        "phase_dp_second_anchor_beam_final_size": int(anchor_result.beam_final_size),
    }
    gt_metrics = _path_phase_metrics(center_spectra, gt_bins)
    row["gt_first_abs_pi"] = gt_metrics["first_abs_pi"]
    row["gt_second_abs_pi"] = gt_metrics["second_abs_pi"]
    row.update(_random_second_smoothness(center_spectra, first_result.evidences, rng, int(args.random_paths)))

    for method, raw_bins in method_bins.items():
        raw_ser, symbol_ser, compared = _ser(raw_bins, gt_bins, sf=sf, ldro=ldro)
        row[f"{method}_raw_ser"] = float(raw_ser)
        row[f"{method}_symbol_ser"] = float(symbol_ser)
        row[f"{method}_symbol_accuracy"] = float(1.0 - symbol_ser)
        row[f"{method}_gt_compared_symbols"] = int(compared)
        metrics = _path_phase_metrics(center_spectra, raw_bins)
        row[f"{method}_first_abs_pi"] = metrics["first_abs_pi"]
        row[f"{method}_second_abs_pi"] = metrics["second_abs_pi"]
        row.update(_prefix_decode(method, packet, raw_bins, args))
    for method in METHODS:
        row[f"{method}_ser_gain_vs_v3"] = float(row["v3_symbol_ser"] - row[f"{method}_symbol_ser"])
    return row


def _summarize(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "packet_count": int(len(rows)),
        "top_l_candidate_recall": _avg(rows, "top_l_candidate_recall"),
        "gt_second_abs_pi": _avg(rows, "gt_second_abs_pi"),
        "random_second_abs_pi_mean": _avg(rows, "random_second_abs_pi_mean"),
        "random_second_abs_pi_p10": _avg(rows, "random_second_abs_pi_p10"),
        "random_second_abs_pi_p50": _avg(rows, "random_second_abs_pi_p50"),
    }
    for method in METHODS:
        out[f"{method}_symbol_ser"] = _avg(rows, f"{method}_symbol_ser")
        out[f"{method}_symbol_accuracy"] = _avg(rows, f"{method}_symbol_accuracy")
        out[f"{method}_crc_valid_rate"] = _avg(rows, f"{method}_crc_valid")
        out[f"{method}_second_abs_pi"] = _avg(rows, f"{method}_second_abs_pi")
        out[f"{method}_ser_gain_vs_v3"] = _avg(rows, f"{method}_ser_gain_vs_v3")
    return out


def main() -> int:
    args = parse_args()
    snrs = _snr_values(args.snr_start, args.snr_stop, args.snr_step)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    v3_config = _build_v3_config(args.top_l)
    phase_line_config = _build_phase_line_config(args)
    first_config = _base_path_config(args, phase_order=1)
    header_config = _header_path_config(args)
    second_config = _base_path_config(args, phase_order=2)
    anchor_config = _anchor_path_config(args)

    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        paths = _dataset_paths(str(dataset))
        samples = np.fromfile(paths["iq"], dtype=np.complex64)
        packets = load_packets(paths["symbols"], None)
        if samples.size == 0:
            raise ValueError(f"empty IQ file: {paths['iq']}")
        if not packets:
            raise ValueError(f"no packets loaded: {paths['symbols']}")
        metadata = _load_metadata(paths, str(dataset))
        if args.signal_reference_power is not None:
            signal_power = float(args.signal_reference_power)
        elif "signal_reference_power" in metadata:
            signal_power = float(metadata["signal_reference_power"])
        else:
            signal_power = float(np.mean(np.abs(samples).astype(np.float64) ** 2))
            print(
                f"{dataset}: metadata missing; using mean clean IQ power as signal_reference_power={signal_power:.6g}",
                flush=True,
            )
        base_seed = int(metadata.get("seed", 42))

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
            smooth_rng = np.random.default_rng(base_seed + 100000 + step_index)

            rows: list[dict[str, Any]] = []
            for packet_index in sorted(packets):
                row = _evaluate_packet(
                    noisy,
                    packets[packet_index],
                    args,
                    v3_config,
                    phase_line_config,
                    first_config,
                    header_config,
                    second_config,
                    anchor_config,
                    smooth_rng,
                )
                row["dataset"] = str(dataset)
                row["target_snr_db"] = float(snr_db)
                rows.append(row)
                packet_rows.append(row)
            summary = _summarize(rows, str(dataset), float(snr_db))
            summary_rows.append(summary)
            print(
                f"{dataset} snr={snr_db:>6.1f} "
                f"topL={summary['top_l_candidate_recall']:.3f} "
                f"v3={summary['v3_symbol_ser']:.3f} "
                f"line={summary['phase_line_symbol_ser']:.3f} "
                f"dp1={summary['phase_dp_first_symbol_ser']:.3f} "
                f"dp1h={summary['phase_dp_first_header_symbol_ser']:.3f} "
                f"dp2={summary['phase_dp_second_symbol_ser']:.3f} "
                f"dp2a={summary['phase_dp_second_anchor_symbol_ser']:.3f}",
                flush=True,
            )

    original_summary_rows = list(summary_rows)
    for snr_db in snrs:
        group = [row for row in original_summary_rows if abs(float(row["target_snr_db"]) - float(snr_db)) < 1e-9]
        if not group:
            continue
        aggregate: dict[str, Any] = {
            "dataset": "mean_of_datasets",
            "target_snr_db": float(snr_db),
            "packet_count": sum(int(row["packet_count"]) for row in group),
        }
        for key in group[0]:
            if key in aggregate or key in ("dataset", "target_snr_db", "packet_count"):
                continue
            aggregate[key] = float(np.mean([float(row[key]) for row in group]))
        summary_rows.append(aggregate)

    _write_csv(out_dir / "per_packet_metrics.csv", packet_rows)
    _write_csv(out_dir / "snr_curve_summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "snr_values": snrs,
                "datasets": list(args.datasets),
                "v3_config": v3_config.__dict__,
                "phase_line_config": phase_line_config.__dict__,
                "first_order_config": first_config.__dict__,
                "first_order_header_config": header_config.__dict__,
                "second_order_config": second_config.__dict__,
                "second_order_anchor_config": anchor_config.__dict__,
                "summary_rows": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote={out_dir / 'snr_curve_summary.csv'}")
    print(f"wrote={out_dir / 'per_packet_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
