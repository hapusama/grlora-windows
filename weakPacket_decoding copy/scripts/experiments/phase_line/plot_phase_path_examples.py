#!/usr/bin/env python3
"""Plot per-packet phase trajectories for phase-line/path experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
WEAK_ROOT = SCRIPT_DIR.parents[2]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_symbol_phase_threshold_sweep import _dataset_paths  # noqa: E402
from run_symbol_phase_two_stage import (  # noqa: E402
    _argmax_bins,
    _decode_selected,
    _extract_payload_spectra_with_coherence,
    _packet_phase_line,
    _ser,
)
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.phase_line import (  # noqa: E402
    PhaseLineSelectorConfig,
    PhasePathSelectorConfig,
    select_phase_smooth_path,
    select_phase_viterbi_path,
)
from weak_decoder.symbol_phase_two_stage import SymbolPhaseConfig, select_symbol_bins_two_stage  # noqa: E402


METHOD_LABELS = {
    "v3": "v3 coherence selector",
    "phase_line": "old local phase-line rerank",
    "phase_dp_first": "first-order phase Viterbi (DP)",
    "phase_dp_first_header": "first-order Viterbi + header slope",
    "phase_dp_second": "second-order phase Viterbi (DP)",
    "phase_dp_second_anchor": "second-order Viterbi + soft anchor",
    "multi": "multi-offset argmax",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot selector/argmax/GT phase paths for one weak packet.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snr-db", type=float, default=-25.0)
    parser.add_argument("--packet", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "phase_path_figures")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--top-l", type=int, default=24)
    parser.add_argument("--dp-max-energy-drop-db", type=float, default=24.0)
    parser.add_argument("--dp-energy-weight", type=float, default=0.25)
    parser.add_argument("--dp-coherence-weight", type=float, default=0.40)
    parser.add_argument("--dp-rank-weight", type=float, default=0.05)
    parser.add_argument("--first-lambda", type=float, default=0.18)
    parser.add_argument("--first-scale-pi", type=float, default=0.75)
    parser.add_argument("--second-lambda", type=float, default=0.05)
    parser.add_argument("--second-first-lambda", type=float, default=0.0)
    parser.add_argument("--second-scale-pi", type=float, default=0.35)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--anchor-bonus", type=float, default=0.05)
    parser.add_argument("--anchor-top-k", type=int, default=0)
    parser.add_argument("--header-slope-weight", type=float, default=0.08)
    parser.add_argument("--header-slope-scale-pi", type=float, default=0.50)
    parser.add_argument("--header-slope-span", type=int, default=4)
    parser.add_argument("--high-conf-margin-db", type=float, default=3.0)
    parser.add_argument("--high-conf-peak-db", type=float, default=7.0)
    parser.add_argument("--high-conf-min-coherence", type=float, default=0.0)
    parser.add_argument("--methods", nargs="+", default=list(METHOD_LABELS))
    return parser.parse_args()


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


def _phase_path(center_spectra: Sequence[np.ndarray], raw_bins: Sequence[int]) -> np.ndarray:
    phases: list[float] = []
    for spectrum, raw_bin in zip(center_spectra, raw_bins):
        b = int(raw_bin)
        spec = np.asarray(spectrum)
        if b < 0 or b >= spec.size:
            phases.append(float("nan"))
        else:
            phases.append(float(np.angle(spec[b])))
    arr = np.asarray(phases, dtype=np.float64)
    valid = np.isfinite(arr)
    if np.count_nonzero(valid) <= 1:
        return arr
    out = arr.copy()
    out[valid] = np.unwrap(out[valid])
    return out


def _phase_path_aligned_to_reference(
    center_spectra: Sequence[np.ndarray],
    raw_bins: Sequence[int],
    reference_phase: Sequence[float],
) -> np.ndarray:
    """Return per-symbol phases on the nearest 2pi branch to a reference path."""

    phases: list[float] = []
    ref = np.asarray(reference_phase, dtype=np.float64)
    for idx, (spectrum, raw_bin) in enumerate(zip(center_spectra, raw_bins)):
        b = int(raw_bin)
        spec = np.asarray(spectrum)
        if b < 0 or b >= spec.size or idx >= ref.size or not math.isfinite(float(ref[idx])):
            phases.append(float("nan"))
            continue
        raw = float(np.angle(spec[b]))
        branch = round((float(ref[idx]) - raw) / (2.0 * math.pi))
        phases.append(float(raw + 2.0 * math.pi * branch))
    return np.asarray(phases, dtype=np.float64)


def _raw_hit_count(pred_bins: Sequence[int], gt_bins: Sequence[int]) -> tuple[int, int]:
    n = min(len(pred_bins), len(gt_bins))
    return int(sum(int(int(pred_bins[idx]) == int(gt_bins[idx])) for idx in range(n))), int(n)


def _linear_r2(y: Sequence[float]) -> float:
    arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(arr)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    yy = arr[valid]
    xx = np.arange(arr.size, dtype=np.float64)[valid]
    coef = np.polyfit(xx, yy, deg=1)
    pred = np.polyval(coef, xx)
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")


def _second_abs_pi(y: Sequence[float]) -> float:
    arr = np.asarray(y, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan")
    return float(np.mean(np.abs(arr[2:] - 2.0 * arr[1:-1] + arr[:-2])) / math.pi)


def _safe_float(value: float) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.3f}"


def _plot_single(
    output: Path,
    dataset: str,
    snr_db: float,
    packet: dict[str, Any],
    method_name: str,
    method_bins: Sequence[int],
    center_bins: Sequence[int],
    gt_bins: Sequence[int],
    center_spectra: Sequence[np.ndarray],
    ser: dict[str, float],
    lock_ratio: float,
) -> None:
    gt_phase = _phase_path(center_spectra, gt_bins)
    center_phase = _phase_path_aligned_to_reference(center_spectra, center_bins, gt_phase)
    method_phase = _phase_path_aligned_to_reference(center_spectra, method_bins, gt_phase)
    if not np.isfinite(gt_phase).any():
        raise ValueError("GT phase path is empty")
    reference = float(gt_phase[np.isfinite(gt_phase)][0])
    x = np.arange(gt_phase.size)
    gt_y = (gt_phase - reference) / math.pi
    center_y = (center_phase - reference) / math.pi
    method_y = (method_phase - reference) / math.pi

    method_r2 = _linear_r2(method_phase)
    gt_r2 = _linear_r2(gt_phase)
    center_r2 = _linear_r2(center_phase)
    method_s2 = _second_abs_pi(method_phase)
    gt_s2 = _second_abs_pi(gt_phase)
    center_s2 = _second_abs_pi(center_phase)
    raw_hits, raw_total = _raw_hit_count(method_bins, gt_bins)

    fig, ax = plt.subplots(figsize=(14.5, 7.6), dpi=130)
    ax.plot(x, method_y, color="#2d70b7", marker="s", markersize=6, linewidth=1.8, alpha=0.34)
    ax.scatter(x, method_y, color="#2d70b7", marker="s", s=38, label="selector bin", zorder=3)
    ax.plot(x, center_y, color="#b86935", marker="^", markersize=6, linewidth=1.5, alpha=0.25)
    ax.scatter(x, center_y, color="#b86935", marker="^", s=42, label="center FFT argmax", zorder=3)
    ax.plot(x, gt_y, color="#2f8b55", marker="o", markersize=7, linewidth=2.0, alpha=0.35)
    ax.scatter(
        x,
        gt_y,
        facecolors="white",
        edgecolors="#2f8b55",
        marker="o",
        s=62,
        linewidths=2.2,
        label="GT bin",
        zorder=4,
    )
    method_label = METHOD_LABELS.get(method_name, method_name)
    title = (
        f"{dataset}  SNR {snr_db:g} dB  packet {int(packet['packet_index'])}  len {int(packet['payload_len'])}\n"
        f"{method_label}: selSER={ser[method_name]:.3f}  argSER={ser['center']:.3f}  "
        f"multiSER={ser['multi']:.3f}  rawHit={raw_hits}/{raw_total}  lock={lock_ratio:.2f}"
    )
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("payload symbol index", fontsize=13)
    ax.set_ylabel("(unwrapped phase - GT[0]) / pi", fontsize=13)
    ax.grid(True, color="#bbbbbb", linewidth=0.8, alpha=0.35)
    ax.legend(loc="lower left", fontsize=12, frameon=False)
    ax.text(
        0.02,
        0.035,
        (
            f"linear R2: selector {_safe_float(method_r2)}  GT {_safe_float(gt_r2)}  "
            f"argmax {_safe_float(center_r2)}\n"
            f"mean |2nd diff|/pi after GT-branch alignment: selector {_safe_float(method_s2)}  "
            f"GT {_safe_float(gt_s2)}  argmax {_safe_float(center_s2)}"
        ),
        transform=ax.transAxes,
        fontsize=12,
        va="bottom",
        ha="left",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_overview(
    output: Path,
    dataset: str,
    snr_db: float,
    packet: dict[str, Any],
    method_bins: dict[str, Sequence[int]],
    center_bins: Sequence[int],
    gt_bins: Sequence[int],
    center_spectra: Sequence[np.ndarray],
    ser: dict[str, float],
) -> None:
    methods = [name for name in ("v3", "phase_line", "phase_dp_first", "phase_dp_first_header", "phase_dp_second", "phase_dp_second_anchor") if name in method_bins]
    gt_phase = _phase_path(center_spectra, gt_bins)
    center_phase = _phase_path_aligned_to_reference(center_spectra, center_bins, gt_phase)
    reference = float(gt_phase[np.isfinite(gt_phase)][0])
    x = np.arange(gt_phase.size)
    gt_y = (gt_phase - reference) / math.pi
    center_y = (center_phase - reference) / math.pi

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2), dpi=130, sharex=True, sharey=True)
    axes_flat = list(axes.ravel())
    for ax, name in zip(axes_flat, methods):
        raw_hits, raw_total = _raw_hit_count(method_bins[name], gt_bins)
        yy = (_phase_path_aligned_to_reference(center_spectra, method_bins[name], gt_phase) - reference) / math.pi
        ax.plot(x, center_y, color="#b86935", marker="^", markersize=4, linewidth=1.0, alpha=0.22)
        ax.plot(x, yy, color="#2d70b7", marker="s", markersize=4.5, linewidth=1.3, alpha=0.72)
        ax.plot(x, gt_y, color="#2f8b55", marker="o", markersize=5, linewidth=1.5, alpha=0.65)
        ax.set_title(f"{METHOD_LABELS.get(name, name)}\nSER={ser[name]:.3f}  rawHit={raw_hits}/{raw_total}", fontsize=11)
        ax.grid(True, color="#bbbbbb", linewidth=0.7, alpha=0.35)
    for ax in axes_flat[len(methods) :]:
        ax.axis("off")
    fig.suptitle(
        f"{dataset}  SNR {snr_db:g} dB  packet {int(packet['packet_index'])}  len {int(packet['payload_len'])}\n"
        "blue=selector, orange=center FFT argmax, green=GT; selector/argmax phases are branch-aligned to GT",
        fontsize=15,
    )
    fig.supxlabel("payload symbol index", fontsize=12)
    fig.supylabel("(unwrapped phase - GT[0]) / pi", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    paths = _dataset_paths(str(args.dataset))
    metadata = _load_metadata(paths, str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    packets = load_packets(paths["symbols"], int(args.packet))
    if int(args.packet) not in packets:
        raise ValueError(f"packet {args.packet} not loaded for dataset {args.dataset}")
    packet = packets[int(args.packet)]
    if args.signal_reference_power is not None:
        signal_power = float(args.signal_reference_power)
    elif "signal_reference_power" in metadata:
        signal_power = float(metadata["signal_reference_power"])
    else:
        signal_power = float(np.mean(np.abs(samples).astype(np.float64) ** 2))
        print(f"{args.dataset}: metadata missing; using mean clean IQ power {signal_power:.6g}", flush=True)
    base_seed = int(metadata.get("seed", 42))
    rng = np.random.default_rng(base_seed)
    noise_i = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    noise_q = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    unit_noise = (noise_i + 1j * noise_q).astype(np.complex64)
    noise_power = signal_power * (10.0 ** (-float(args.snr_db) / 10.0))
    noisy = (samples + math.sqrt(float(noise_power) / 2.0) * unit_noise).astype(np.complex64, copy=False)

    center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
        noisy,
        packet,
        args,
    )
    evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
    header_line = _packet_phase_line(noisy, packet, args)

    v3_result = select_symbol_bins_two_stage(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=_build_v3_config(args.top_l),
        fallback_line=header_line,
        offset_coherences=coherences,
    )
    phase_line_result = select_phase_smooth_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=_build_phase_line_config(args),
        fallback_line=header_line,
        offset_coherences=coherences,
    )
    first_result = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=_base_path_config(args, phase_order=1),
        offset_coherences=coherences,
    )
    first_header_result = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=_header_path_config(args),
        fallback_line=header_line,
        offset_coherences=coherences,
    )
    second_result = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=_base_path_config(args, phase_order=2),
        offset_coherences=coherences,
    )
    second_anchor_result = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=_anchor_path_config(args),
        offset_coherences=coherences,
    )

    center_bins = _argmax_bins(center_spectra)
    multi_bins = _argmax_bins(multi_spectra)
    methods: dict[str, Sequence[int]] = {
        "multi": multi_bins,
        "v3": tuple(v3_result.selected_raw_bins),
        "phase_line": tuple(phase_line_result.selected_raw_bins),
        "phase_dp_first": tuple(first_result.selected_raw_bins),
        "phase_dp_first_header": tuple(first_header_result.selected_raw_bins),
        "phase_dp_second": tuple(second_result.selected_raw_bins),
        "phase_dp_second_anchor": tuple(second_anchor_result.selected_raw_bins),
    }
    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    ser: dict[str, float] = {}
    raw_ser, symbol_ser, _ = _ser(center_bins, gt_bins, sf=sf, ldro=ldro)
    ser["center"] = float(symbol_ser)
    raw_ser, symbol_ser, _ = _ser(multi_bins, gt_bins, sf=sf, ldro=ldro)
    ser["multi"] = float(symbol_ser)
    for name, bins in methods.items():
        _raw_ser, symbol_ser, _compared = _ser(bins, gt_bins, sf=sf, ldro=ldro)
        ser[name] = float(symbol_ser)

    safe_snr = str(args.snr_db).replace("-", "m").replace(".", "p")
    out_dir = Path(args.output_dir) / f"{args.dataset}_snr_{safe_snr}_packet_{int(args.packet)}"
    rows: list[dict[str, Any]] = []
    for name, bins in {"center": center_bins, **methods, "gt": gt_bins}.items():
        self_phase = _phase_path(center_spectra, bins)
        aligned_phase = _phase_path_aligned_to_reference(center_spectra, bins, _phase_path(center_spectra, gt_bins))
        for idx, raw_bin in enumerate(bins):
            rows.append(
                {
                    "path": name,
                    "payload_symbol_index": idx,
                    "raw_bin": int(raw_bin),
                    "phase_pi_self_unwrap": float(self_phase[idx] / math.pi),
                    "phase_pi_gt_aligned": float(aligned_phase[idx] / math.pi),
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "phase_paths.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "payload_symbol_index", "raw_bin", "phase_pi_self_unwrap", "phase_pi_gt_aligned"],
        )
        writer.writeheader()
        writer.writerows(rows)

    lock_ratios = {
        "v3": float(v3_result.locked_count / max(1, len(v3_result.selected_raw_bins))),
        "phase_line": float(phase_line_result.locked_count / max(1, len(phase_line_result.selected_raw_bins))),
        "phase_dp_first": 0.0,
        "phase_dp_first_header": 0.0,
        "phase_dp_second": 0.0,
        "phase_dp_second_anchor": 0.0,
        "multi": 0.0,
    }
    selected_methods = [name for name in args.methods if name in methods]
    for name in selected_methods:
        output = out_dir / f"{name}_phase_path.png"
        _plot_single(
            output=output,
            dataset=str(args.dataset),
            snr_db=float(args.snr_db),
            packet=packet,
            method_name=name,
            method_bins=methods[name],
            center_bins=center_bins,
            gt_bins=gt_bins,
            center_spectra=center_spectra,
            ser=ser,
            lock_ratio=lock_ratios.get(name, 0.0),
        )
        print(f"wrote={output}")
    overview = out_dir / "overview_phase_paths.png"
    _plot_overview(
        output=overview,
        dataset=str(args.dataset),
        snr_db=float(args.snr_db),
        packet=packet,
        method_bins=methods,
        center_bins=center_bins,
        gt_bins=gt_bins,
        center_spectra=center_spectra,
        ser=ser,
    )
    summary = {
        "dataset": str(args.dataset),
        "snr_db": float(args.snr_db),
        "packet": int(args.packet),
        "payload_len": int(packet["payload_len"]),
        "ser": ser,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote={overview}")
    print(f"wrote={out_dir / 'phase_paths.csv'}")
    print(f"wrote={out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
