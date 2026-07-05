#!/usr/bin/env python3
"""Plot packet-level phase-line error diagnostics.

The goal is not to tune another weight.  This script compares the current
selected path against GT bins on the same packet evidence:

* phase smoothness of GT vs selected bins,
* residuals to the selected phase line,
* energy score and rank of GT vs selected bins,
* per-symbol failure rows for persistent-error analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.candidate_pruning import wrap_phase  # noqa: E402
from weak_decoder.phase_guided_demod import fit_phase_line  # noqa: E402
from weak_decoder.baselines.run_ser_comparison import (  # noqa: E402
    _dataset_paths,
    _load_packets,
    _noise_samples,
    _signal_reference_power,
    _write_csv,
)
from weak_decoder.phase_line.savaux_stage1 import (  # noqa: E402
    SavauxStage1Config,
    default_savaux_phase_path_config,
    payload_abs_indices,
)
from weak_decoder.phase_line.variants._legacy_core.selector import _energy_score  # noqa: E402
from weak_decoder.phase_line.variants.island_dp_reconstruction import (  # noqa: E402
    IslandReconstructionConfig,
    select_island_reconstruction_viterbi_path,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction.dual_evidence import (  # noqa: E402
    DualEvidenceFusionConfig,
    build_dual_savaux_stage1_packet_evidence,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction.evaluate_island_dp import (  # noqa: E402
    _branch_residual_sto_from_packet,
)
from weak_decoder.phase_line.variants.v1_one_order_dp import select_phase_viterbi_path  # noqa: E402


METHODS = ("fusion_island", "fusion_v1", "v1", "hard")


def _parse_snr_values(values: Sequence[str]) -> tuple[float, ...]:
    out: list[float] = []
    for value in values:
        if ":" not in value:
            out.append(float(value))
            continue
        start_s, stop_s, step_s = value.split(":", 2)
        start = float(start_s)
        stop = float(stop_s)
        step = float(step_s)
        if step == 0.0:
            raise ValueError("SNR range step must be non-zero")
        cur = start
        if step > 0:
            while cur <= stop + 1e-9:
                out.append(round(cur, 6))
                cur += step
        else:
            while cur >= stop - 1e-9:
                out.append(round(cur, 6))
                cur += step
    return tuple(out)


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(10.0 * math.log10((float(numerator) + 1e-30) / (float(denominator) + 1e-30)))


def _bin_phase(evidence: Any, raw_bin: int) -> float:
    b = int(raw_bin)
    spectrum = np.asarray(evidence.center_spectrum, dtype=np.complex64)
    if b < 0 or b >= spectrum.size:
        return 0.0
    return float(np.angle(spectrum[b]))


def _power_rank(power: np.ndarray, raw_bin: int) -> int:
    b = int(raw_bin)
    values = np.asarray(power, dtype=np.float64)
    if b < 0 or b >= values.size:
        return int(values.size + 1)
    return int(1 + np.count_nonzero(values > float(values[b])))


def _phase_series(evidences: Sequence[Any], bins: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(len(evidences), len(bins))
    xs = np.asarray([float(evidences[idx].abs_symbol_index) for idx in range(count)], dtype=np.float64)
    raw = np.asarray([_bin_phase(evidences[idx], int(bins[idx])) for idx in range(count)], dtype=np.float64)
    order = np.argsort(xs)
    unwrapped = np.empty_like(raw)
    unwrapped[order] = np.unwrap(raw[order])
    return xs, raw, unwrapped


def _line_metrics(evidences: Sequence[Any], bins: Sequence[int]) -> dict[str, Any]:
    xs, raw, unwrapped = _phase_series(evidences, bins)
    if xs.size < 2:
        line = fit_phase_line(xs, unwrapped, trim_frac=0.0)
        return {
            "line": line,
            "line_rmse_pi": float("nan"),
            "diff_std_pi": float("nan"),
            "curvature_median_abs_pi": float("nan"),
        }
    line = fit_phase_line(xs, unwrapped, trim_frac=0.0)
    diffs = np.diff(unwrapped[np.argsort(xs)])
    curv = np.diff(diffs)
    return {
        "line": line,
        "line_rmse_pi": float(line.fit_rmse_pi),
        "diff_std_pi": float(np.std(diffs) / math.pi) if diffs.size else 0.0,
        "curvature_median_abs_pi": float(np.median(np.abs(curv)) / math.pi) if curv.size else 0.0,
    }


def _residual_to_line_pi(phase: float, abs_index: float, line: Any) -> float:
    if int(getattr(line, "anchor_count", 0)) < 2:
        return float("nan")
    return float(wrap_phase(float(phase) - float(line.predict(float(abs_index)))) / math.pi)


def _evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    payload = list(packet["payload_symbols"])
    start_samples = [int(item["start_sample"]) for item in payload]
    gt_bins = tuple(int(item["gt_bin"]) for item in payload)
    residual_sto_chips = [float(item.get("sfo_cum_before", 0.0)) for item in payload]
    branch_residual_sto_chips = _branch_residual_sto_from_packet(packet)
    abs_indices = payload_abs_indices(
        [int(item["payload_symbol_index"]) for item in payload],
        preamble_len=float(packet.get("preamble_len", 8.0)),
    )
    path_config = default_savaux_phase_path_config(top_l=16)
    island_config = IslandReconstructionConfig()
    fusion_config = DualEvidenceFusionConfig(
        mode="product_norm",
        corrected_weight=0.5,
        stage1_top_k=40,
        retain_dechirped_symbols=True,
    )
    dual = build_dual_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        abs_indices=abs_indices,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=int(packet["header_start_sample"]),
        residual_sto_chips=residual_sto_chips,
        stage1_config=SavauxStage1Config(retain_dechirped_symbols=True),
        fusion_config=fusion_config,
    )
    hard_bins = tuple(int(symbol.top1_bin) for symbol in dual.old_stage1.symbols)
    v1 = select_phase_viterbi_path(
        center_spectra=dual.old_stage1.center_spectra,
        evidence_powers=dual.old_stage1.evidence_powers,
        abs_indices=dual.old_stage1.abs_indices,
        config=path_config,
        offset_coherences=dual.old_stage1.branch_phase_agreements,
    )
    fusion_v1 = select_phase_viterbi_path(
        center_spectra=dual.center_spectra,
        evidence_powers=dual.evidence_powers,
        abs_indices=dual.abs_indices,
        config=path_config,
        offset_coherences=dual.branch_phase_agreements,
    )
    fusion_island = select_island_reconstruction_viterbi_path(
        center_spectra=dual.center_spectra,
        evidence_powers=dual.evidence_powers,
        abs_indices=dual.abs_indices,
        config=path_config,
        reconstruction_config=island_config,
        offset_coherences=dual.branch_phase_agreements,
        branch_spectra=dual.branch_spectra,
        dechirped_symbols=dual.dechirped_symbols,
        os_factor=int(packet["os_factor"]),
        residual_sto_chips=residual_sto_chips,
        branch_residual_sto_chips=branch_residual_sto_chips,
    )
    method_bins = {
        "hard": hard_bins,
        "v1": v1.selected_raw_bins,
        "fusion_v1": fusion_v1.selected_raw_bins,
        "fusion_island": fusion_island.selected_raw_bins,
    }
    if method not in method_bins:
        raise ValueError(f"unsupported method {method!r}")
    selected_bins = tuple(int(v) for v in method_bins[method])
    return {
        "gt_bins": gt_bins,
        "hard_bins": hard_bins,
        "fusion_v1_bins": tuple(int(v) for v in fusion_v1.selected_raw_bins),
        "selected_bins": selected_bins,
        "evidences": fusion_island.evidences,
        "locked_mask": tuple(bool(v) for v in fusion_island.locked_mask),
        "phase_result": fusion_island,
    }


def _symbol_rows(
    dataset: str,
    snr_db: float,
    seed: int,
    packet: dict[str, Any],
    result: dict[str, Any],
    method: str,
    selected_line: Any,
    gt_line: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evidences = tuple(result["evidences"])
    gt_bins = tuple(int(v) for v in result["gt_bins"])
    selected_bins = tuple(int(v) for v in result["selected_bins"])
    hard_bins = tuple(int(v) for v in result["hard_bins"])
    fusion_v1_bins = tuple(int(v) for v in result["fusion_v1_bins"])
    locked_mask = tuple(bool(v) for v in result["locked_mask"])
    count = min(len(evidences), len(gt_bins), len(selected_bins))
    for idx in range(count):
        ev = evidences[idx]
        gt = int(gt_bins[idx])
        selected = int(selected_bins[idx])
        hard = int(hard_bins[idx]) if idx < len(hard_bins) else -1
        fusion_v1 = int(fusion_v1_bins[idx]) if idx < len(fusion_v1_bins) else -1
        gt_phase = _bin_phase(ev, gt)
        selected_phase = _bin_phase(ev, selected)
        gt_energy = _energy_score(ev, gt)
        selected_energy = _energy_score(ev, selected)
        gt_power = float(ev.evidence_power[gt]) if 0 <= gt < ev.evidence_power.size else 0.0
        selected_power = float(ev.evidence_power[selected]) if 0 <= selected < ev.evidence_power.size else 0.0
        row = {
            "dataset": dataset,
            "snr_db": float(snr_db),
            "seed": int(seed),
            "packet_index": int(packet["packet_index"]),
            "payload_symbol_index": int(packet["payload_symbols"][idx]["payload_symbol_index"]),
            "method": method,
            "gt_bin": gt,
            "selected_bin": selected,
            "hard_bin": hard,
            "fusion_v1_bin": fusion_v1,
            "selected_correct": int(selected == gt),
            "hard_correct": int(hard == gt),
            "fusion_v1_correct": int(fusion_v1 == gt),
            "is_locked_anchor": int(locked_mask[idx]) if idx < len(locked_mask) else 0,
            "gt_in_top_bins": int(gt in set(int(v) for v in ev.top_bins)),
            "selected_in_top_bins": int(selected in set(int(v) for v in ev.top_bins)),
            "gt_rank": _power_rank(ev.evidence_power, gt),
            "selected_rank": _power_rank(ev.evidence_power, selected),
            "top1_bin": int(ev.top1_bin),
            "top1_margin_db": float(ev.top1_margin_db),
            "top1_peak_to_median_db": float(ev.top1_peak_to_median_db),
            "gt_energy_score": float(gt_energy),
            "selected_energy_score": float(selected_energy),
            "selected_minus_gt_energy_score": float(selected_energy - gt_energy),
            "selected_over_gt_power_db": _db_ratio(selected_power, gt_power),
            "gt_phase_rad": float(gt_phase),
            "selected_phase_rad": float(selected_phase),
            "gt_resid_to_gt_line_pi": _residual_to_line_pi(gt_phase, ev.abs_symbol_index, gt_line),
            "selected_resid_to_selected_line_pi": _residual_to_line_pi(
                selected_phase, ev.abs_symbol_index, selected_line
            ),
            "gt_resid_to_selected_line_pi": _residual_to_line_pi(gt_phase, ev.abs_symbol_index, selected_line),
            "selected_resid_to_gt_line_pi": _residual_to_line_pi(selected_phase, ev.abs_symbol_index, gt_line),
        }
        rows.append(row)
    return rows


def _packet_summary(
    dataset: str,
    snr_db: float,
    seed: int,
    packet: dict[str, Any],
    method: str,
    rows: Sequence[dict[str, Any]],
    selected_metrics: dict[str, Any],
    gt_metrics: dict[str, Any],
) -> dict[str, Any]:
    wrong = [row for row in rows if int(row["selected_correct"]) == 0]
    wrong_count = len(wrong)
    selected_closer = sum(
        abs(float(row["selected_resid_to_selected_line_pi"])) < abs(float(row["gt_resid_to_selected_line_pi"]))
        for row in wrong
    )
    selected_energy_higher = sum(float(row["selected_minus_gt_energy_score"]) > 0.0 for row in wrong)
    gt_missing = sum(int(row["gt_in_top_bins"]) == 0 for row in wrong)
    smoother = "selected"
    if float(gt_metrics["line_rmse_pi"]) + 1e-9 < float(selected_metrics["line_rmse_pi"]):
        smoother = "gt"
    elif abs(float(gt_metrics["line_rmse_pi"]) - float(selected_metrics["line_rmse_pi"])) <= 1e-9:
        smoother = "tie"
    return {
        "dataset": dataset,
        "snr_db": float(snr_db),
        "seed": int(seed),
        "packet_index": int(packet["packet_index"]),
        "method": method,
        "symbol_count": int(len(rows)),
        "selected_err": int(wrong_count),
        "selected_ser": float(wrong_count / max(1, len(rows))),
        "hard_err": int(sum(int(row["hard_correct"]) == 0 for row in rows)),
        "fusion_v1_err": int(sum(int(row["fusion_v1_correct"]) == 0 for row in rows)),
        "selected_line_rmse_pi": float(selected_metrics["line_rmse_pi"]),
        "gt_line_rmse_pi": float(gt_metrics["line_rmse_pi"]),
        "selected_diff_std_pi": float(selected_metrics["diff_std_pi"]),
        "gt_diff_std_pi": float(gt_metrics["diff_std_pi"]),
        "selected_curvature_median_abs_pi": float(selected_metrics["curvature_median_abs_pi"]),
        "gt_curvature_median_abs_pi": float(gt_metrics["curvature_median_abs_pi"]),
        "smoother_by_line_rmse": smoother,
        "wrong_selected_closer_to_selected_line_count": int(selected_closer),
        "wrong_selected_closer_to_selected_line_rate": float(selected_closer / max(1, wrong_count)),
        "wrong_selected_energy_higher_count": int(selected_energy_higher),
        "wrong_selected_energy_higher_rate": float(selected_energy_higher / max(1, wrong_count)),
        "wrong_gt_missing_top_bins_count": int(gt_missing),
        "wrong_gt_missing_top_bins_rate": float(gt_missing / max(1, wrong_count)),
        "wrong_gt_rank_median": float(np.median([float(row["gt_rank"]) for row in wrong])) if wrong else 0.0,
        "wrong_selected_rank_median": float(np.median([float(row["selected_rank"]) for row in wrong])) if wrong else 0.0,
        "wrong_selected_over_gt_power_db_mean": float(
            np.mean([float(row["selected_over_gt_power_db"]) for row in wrong])
        )
        if wrong
        else 0.0,
    }


def _plot_packet(
    path: Path,
    dataset: str,
    snr_db: float,
    seed: int,
    packet: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    selected_metrics: dict[str, Any],
    gt_metrics: dict[str, Any],
    method: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    x = np.asarray([int(row["payload_symbol_index"]) for row in rows], dtype=np.float64)
    wrong = np.asarray([int(row["selected_correct"]) == 0 for row in rows], dtype=bool)
    evid_x = np.asarray([idx for idx, _row in enumerate(rows)], dtype=np.float64)

    # Rebuild unwrapped phase arrays in display order.
    gt_raw = np.asarray([float(row["gt_phase_rad"]) for row in rows], dtype=np.float64)
    selected_raw = np.asarray([float(row["selected_phase_rad"]) for row in rows], dtype=np.float64)
    gt_unwrap = np.unwrap(gt_raw)
    selected_unwrap = np.unwrap(selected_raw)
    selected_line = selected_metrics["line"]
    gt_line = gt_metrics["line"]
    abs_x = np.asarray([12.25 + 8.0 + float(v) for v in x], dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 10.2), dpi=160, sharex=True)
    for ax in axes:
        for sym in x[wrong]:
            ax.axvspan(float(sym) - 0.45, float(sym) + 0.45, color="#d62728", alpha=0.08, linewidth=0)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.36)

    axes[0].plot(x, gt_unwrap / math.pi, color="#1f77b4", linewidth=1.7, marker="o", markersize=4, label="GT phase")
    axes[0].plot(
        x,
        selected_unwrap / math.pi,
        color="#d62728",
        linewidth=1.7,
        marker="x",
        markersize=5,
        label="Selected phase",
    )
    if int(getattr(gt_line, "anchor_count", 0)) >= 2:
        axes[0].plot(x, [gt_line.predict(v) / math.pi for v in abs_x], color="#1f77b4", linestyle=":", label="GT fit")
    if int(getattr(selected_line, "anchor_count", 0)) >= 2:
        axes[0].plot(
            x,
            [selected_line.predict(v) / math.pi for v in abs_x],
            color="#d62728",
            linestyle=":",
            label="Selected fit",
        )
    axes[0].set_ylabel("Unwrapped phase / pi")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(
        x,
        [abs(float(row["gt_resid_to_selected_line_pi"])) for row in rows],
        color="#1f77b4",
        marker="o",
        markersize=4,
        label="GT residual to selected line",
    )
    axes[1].plot(
        x,
        [abs(float(row["selected_resid_to_selected_line_pi"])) for row in rows],
        color="#d62728",
        marker="x",
        markersize=5,
        label="Selected residual to selected line",
    )
    axes[1].set_ylabel("|residual| / pi")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(
        x,
        [float(row["gt_energy_score"]) for row in rows],
        color="#1f77b4",
        marker="o",
        markersize=4,
        label="GT energy score",
    )
    axes[2].plot(
        x,
        [float(row["selected_energy_score"]) for row in rows],
        color="#d62728",
        marker="x",
        markersize=5,
        label="Selected energy score",
    )
    axes[2].set_ylabel("Energy score")
    axes[2].set_ylim(-0.03, 1.05)
    axes[2].legend(loc="best", fontsize=8)

    gap = np.asarray([float(row["selected_over_gt_power_db"]) for row in rows], dtype=np.float64)
    colors = np.where(wrong & (gap > 0.0), "#d62728", np.where(wrong, "#ff9896", "#7f7f7f"))
    axes[3].bar(x, gap, width=0.72, color=colors, alpha=0.82)
    axes[3].axhline(0.0, color="#333333", linewidth=1.0)
    axes[3].set_ylabel("Selected / GT power (dB)")
    axes[3].set_xlabel("Payload symbol index")

    wrong_count = int(np.sum(wrong))
    selected_closer = sum(
        abs(float(row["selected_resid_to_selected_line_pi"])) < abs(float(row["gt_resid_to_selected_line_pi"]))
        for row in rows
        if int(row["selected_correct"]) == 0
    )
    selected_energy_higher = sum(
        float(row["selected_minus_gt_energy_score"]) > 0.0 for row in rows if int(row["selected_correct"]) == 0
    )
    gt_missing = sum(int(row["gt_in_top_bins"]) == 0 for row in rows if int(row["selected_correct"]) == 0)
    fig.suptitle(
        f"{dataset} packet={packet['packet_index']} seed={seed} SNR={snr_db:g} dB method={method} "
        f"errors={wrong_count}/{len(rows)}",
        fontsize=13,
    )
    subtitle = (
        f"line RMSE/pi selected={float(selected_metrics['line_rmse_pi']):.3f}, "
        f"GT={float(gt_metrics['line_rmse_pi']):.3f}; "
        f"wrong smoother-to-selected-line={selected_closer}/{max(1, wrong_count)}, "
        f"wrong energy-favors-selected={selected_energy_higher}/{max(1, wrong_count)}, "
        f"GT missing topL={gt_missing}/{max(1, wrong_count)}"
    )
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _persistent_rows(symbol_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in symbol_rows:
        key = (
            row["dataset"],
            float(row["snr_db"]),
            int(row["packet_index"]),
            int(row["payload_symbol_index"]),
        )
        out = grouped.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "snr_db": float(row["snr_db"]),
                "packet_index": int(row["packet_index"]),
                "payload_symbol_index": int(row["payload_symbol_index"]),
                "trials": 0,
                "wrong_count": 0,
                "gt_missing_top_bins_count": 0,
                "selected_energy_higher_when_wrong_count": 0,
                "selected_smoother_when_wrong_count": 0,
                "mean_gt_rank_when_wrong": [],
                "mean_selected_over_gt_power_db_when_wrong": [],
            },
        )
        out["trials"] += 1
        wrong = int(row["selected_correct"]) == 0
        if wrong:
            out["wrong_count"] += 1
            out["gt_missing_top_bins_count"] += int(int(row["gt_in_top_bins"]) == 0)
            out["selected_energy_higher_when_wrong_count"] += int(float(row["selected_minus_gt_energy_score"]) > 0.0)
            out["selected_smoother_when_wrong_count"] += int(
                abs(float(row["selected_resid_to_selected_line_pi"]))
                < abs(float(row["gt_resid_to_selected_line_pi"]))
            )
            out["mean_gt_rank_when_wrong"].append(float(row["gt_rank"]))
            out["mean_selected_over_gt_power_db_when_wrong"].append(float(row["selected_over_gt_power_db"]))
    rows: list[dict[str, Any]] = []
    for out in grouped.values():
        wrong_count = int(out["wrong_count"])
        row = dict(out)
        row["wrong_rate"] = float(wrong_count / max(1, int(out["trials"])))
        row["gt_missing_top_bins_rate_when_wrong"] = float(
            int(out["gt_missing_top_bins_count"]) / max(1, wrong_count)
        )
        row["selected_energy_higher_rate_when_wrong"] = float(
            int(out["selected_energy_higher_when_wrong_count"]) / max(1, wrong_count)
        )
        row["selected_smoother_rate_when_wrong"] = float(
            int(out["selected_smoother_when_wrong_count"]) / max(1, wrong_count)
        )
        ranks = row.pop("mean_gt_rank_when_wrong")
        gaps = row.pop("mean_selected_over_gt_power_db_when_wrong")
        row["mean_gt_rank_when_wrong"] = float(np.mean(ranks)) if ranks else 0.0
        row["mean_selected_over_gt_power_db_when_wrong"] = float(np.mean(gaps)) if gaps else 0.0
        rows.append(row)
    rows.sort(key=lambda item: (-float(item["wrong_rate"]), int(item["packet_index"]), int(item["payload_symbol_index"])))
    return rows


def _plot_persistent_heatmap(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    packets = sorted({int(row["packet_index"]) for row in rows})
    symbols = sorted({int(row["payload_symbol_index"]) for row in rows})
    p_index = {packet: idx for idx, packet in enumerate(packets)}
    s_index = {symbol: idx for idx, symbol in enumerate(symbols)}
    wrong_rate = np.full((len(packets), len(symbols)), np.nan, dtype=np.float64)
    missing_rate = np.full_like(wrong_rate, np.nan)
    energy_rate = np.full_like(wrong_rate, np.nan)
    for row in rows:
        pi = p_index[int(row["packet_index"])]
        si = s_index[int(row["payload_symbol_index"])]
        wrong_rate[pi, si] = float(row["wrong_rate"])
        missing_rate[pi, si] = float(row["gt_missing_top_bins_rate_when_wrong"])
        energy_rate[pi, si] = float(row["selected_energy_higher_rate_when_wrong"])

    fig, axes = plt.subplots(3, 1, figsize=(13.0, 7.8), dpi=160, sharex=True)
    titles = [
        "Wrong rate across seeds",
        "GT missing from top candidates when wrong",
        "Selected energy higher than GT when wrong",
    ]
    mats = [wrong_rate, missing_rate, energy_rate]
    for ax, title, mat in zip(axes, titles, mats):
        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="magma", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("packet")
        ax.set_yticks(range(len(packets)))
        ax.set_yticklabels([str(v) for v in packets])
        fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    axes[-1].set_xlabel("payload symbol index")
    axes[-1].set_xticks(range(len(symbols)))
    axes[-1].set_xticklabels([str(v) for v in symbols], rotation=90, fontsize=7)
    fig.suptitle("Persistent phase-line error structure", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_packet_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["packet_index"]), []).append(row)
    packets = sorted(grouped)
    ser = [float(np.mean([float(row["selected_ser"]) for row in grouped[p]])) for p in packets]
    selected_rmse = [float(np.mean([float(row["selected_line_rmse_pi"]) for row in grouped[p]])) for p in packets]
    gt_rmse = [float(np.mean([float(row["gt_line_rmse_pi"]) for row in grouped[p]])) for p in packets]
    energy_rate = [
        float(
            sum(int(row["wrong_selected_energy_higher_count"]) for row in grouped[p])
            / max(1, sum(int(row["selected_err"]) for row in grouped[p]))
        )
        for p in packets
    ]
    missing_rate = [
        float(
            sum(int(row["wrong_gt_missing_top_bins_count"]) for row in grouped[p])
            / max(1, sum(int(row["selected_err"]) for row in grouped[p]))
        )
        for p in packets
    ]
    x = np.arange(len(packets), dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), dpi=160, sharex=True)
    axes[0].bar(x, ser, color="#4c78a8")
    axes[0].set_ylabel("SER")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[1].plot(x, selected_rmse, marker="o", label="selected line RMSE/pi", color="#d62728")
    axes[1].plot(x, gt_rmse, marker="o", label="GT line RMSE/pi", color="#1f77b4")
    axes[1].set_ylabel("line RMSE/pi")
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[2].bar(x - 0.18, energy_rate, width=0.36, label="energy favors selected when wrong", color="#e45756")
    axes[2].bar(x + 0.18, missing_rate, width=0.36, label="GT missing topL when wrong", color="#72b7b2")
    axes[2].set_ylabel("rate")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xlabel("packet index")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(v) for v in packets])
    axes[2].legend(loc="best")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("Packet-level error mechanism summary", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="0_0_0_10_14_32")
    parser.add_argument("--snrs", nargs="+", default=["-25"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-packets", type=int, default=10)
    parser.add_argument("--method", choices=METHODS, default="fusion_island")
    parser.add_argument("--signal-reference-mode", choices=("packet", "payload", "header_payload", "whole"), default="packet")
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--plot-all-packets", action="store_true", default=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "phase_line_error_diagnostics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snrs = _parse_snr_values(args.snrs)
    dataset = str(args.dataset)
    iq_path, symbol_path = _dataset_paths(dataset)
    clean = np.fromfile(iq_path, dtype=np.complex64)
    packets = _load_packets(symbol_path)
    if int(args.max_packets) > 0:
        packets = packets[: int(args.max_packets)]
    reference_power, reference_samples, reference_packets = _signal_reference_power(
        clean,
        packets,
        mode=str(args.signal_reference_mode),
        explicit_power=args.signal_reference_power,
    )
    out_dir = Path(args.output_dir).resolve()
    symbol_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    metadata = {
        "dataset": dataset,
        "snrs": list(snrs),
        "seeds": [int(seed) for seed in args.seeds],
        "max_packets": int(args.max_packets),
        "actual_packets": int(len(packets)),
        "method": str(args.method),
        "signal_reference_mode": str(args.signal_reference_mode),
        "signal_reference_power": float(reference_power),
        "signal_reference_sample_count": int(reference_samples),
        "signal_reference_packet_count": int(reference_packets),
        "note": "Phase and energy diagnostics use the same fused evidence as fusion_island unless method=hard/v1.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for seed in args.seeds:
        for snr_db in snrs:
            samples = _noise_samples(clean, float(snr_db), int(seed), reference_power)
            for packet in packets:
                result = _evaluate_packet(samples, packet, str(args.method))
                evidences = tuple(result["evidences"])
                selected_bins = tuple(int(v) for v in result["selected_bins"])
                gt_bins = tuple(int(v) for v in result["gt_bins"])
                selected_metrics = _line_metrics(evidences, selected_bins)
                gt_metrics = _line_metrics(evidences, gt_bins)
                rows = _symbol_rows(
                    dataset,
                    float(snr_db),
                    int(seed),
                    packet,
                    result,
                    str(args.method),
                    selected_metrics["line"],
                    gt_metrics["line"],
                )
                symbol_rows.extend(rows)
                summary = _packet_summary(
                    dataset,
                    float(snr_db),
                    int(seed),
                    packet,
                    str(args.method),
                    rows,
                    selected_metrics,
                    gt_metrics,
                )
                packet_rows.append(summary)
                snr_tag = f"m{abs(float(snr_db)):g}".replace(".", "p")
                plot_path = (
                    out_dir
                    / "plots"
                    / f"snr_{snr_tag}_seed_{int(seed)}"
                    / f"packet_{int(packet['packet_index']):03d}_err_{int(summary['selected_err']):02d}.png"
                )
                _plot_packet(
                    plot_path,
                    dataset,
                    float(snr_db),
                    int(seed),
                    packet,
                    rows,
                    selected_metrics,
                    gt_metrics,
                    str(args.method),
                )
                print(
                    f"{dataset} snr={float(snr_db):g} seed={seed} packet={packet['packet_index']}: "
                    f"err={summary['selected_err']}/{summary['symbol_count']} "
                    f"rmse_sel={summary['selected_line_rmse_pi']:.3f} rmse_gt={summary['gt_line_rmse_pi']:.3f}",
                    flush=True,
                )

    persistent = _persistent_rows(symbol_rows)
    _write_csv(out_dir / "symbol_diagnostics.csv", symbol_rows)
    _write_csv(out_dir / "packet_summary.csv", packet_rows)
    _write_csv(out_dir / "persistent_errors.csv", persistent)
    _plot_persistent_heatmap(out_dir / "persistent_error_heatmap.png", persistent)
    _plot_packet_summary(out_dir / "packet_summary.png", packet_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
