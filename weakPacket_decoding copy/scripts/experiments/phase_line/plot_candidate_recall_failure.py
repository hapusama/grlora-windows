#!/usr/bin/env python3
"""Plot why phase-DP cannot recover when GT bins miss the Top-L set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = SCRIPT_DIR.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot candidate-recall failure diagnostics.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=WEAK_ROOT / "data" / "phase_path_ablation_best_coh040_m22_m26" / "snr_curve_summary.csv",
    )
    parser.add_argument(
        "--packet-symbol-csv",
        type=Path,
        default=WEAK_ROOT
        / "data"
        / "phase_dp_candidate_coverage"
        / "0_0_0_10_14_16_snr_m25_to_m25_packet_10_per_symbol.csv",
    )
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "phase_dp_candidate_coverage_figures")
    return parser.parse_args()


def _read_summary(path: Path, dataset: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("dataset")) != str(dataset):
                continue
            rows.append(
                {
                    "snr": float(row["target_snr_db"]),
                    "top_l_recall": float(row["top_l_candidate_recall"]),
                    "missing_lower_bound": 1.0 - float(row["top_l_candidate_recall"]),
                    "v3_ser": float(row["v3_symbol_ser"]),
                    "dp1_ser": float(row["phase_dp_first_symbol_ser"]),
                    "dp2_ser": float(row["phase_dp_second_symbol_ser"]),
                    "line_ser": float(row["phase_line_symbol_ser"]),
                    "multi_ser": float(row["multi_symbol_ser"]),
                }
            )
    rows.sort(key=lambda item: item["snr"])
    return rows


def _read_packet_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    rows.sort(key=lambda item: int(item["payload_symbol_index"]))
    return rows


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote={path}")


def _plot_recall_and_ser(rows: Sequence[dict[str, float]], output: Path) -> None:
    snr = np.asarray([row["snr"] for row in rows], dtype=np.float64)
    recall = np.asarray([row["top_l_recall"] for row in rows], dtype=np.float64)
    missing = np.asarray([row["missing_lower_bound"] for row in rows], dtype=np.float64)
    v3 = np.asarray([row["v3_ser"] for row in rows], dtype=np.float64)
    dp1 = np.asarray([row["dp1_ser"] for row in rows], dtype=np.float64)
    dp2 = np.asarray([row["dp2_ser"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.0), dpi=140, sharex=True)
    ax = axes[0]
    ax.plot(snr, recall, marker="o", linewidth=2.4, color="#2f8b55", label="GT in Top-24 recall")
    ax.fill_between(snr, recall, 1.0, color="#b14646", alpha=0.16, label="GT missing from candidate set")
    ax.set_ylim(0.50, 1.02)
    ax.set_ylabel("candidate recall")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    ax.set_title("Stage-1 candidate recall collapses in low SNR")

    ax = axes[1]
    ax.fill_between(snr, 0.0, missing, color="#b14646", alpha=0.22, label="unrecoverable lower bound: GT missing")
    ax.plot(snr, missing, marker="x", linewidth=2.0, color="#b14646", label="candidate-missing SER lower bound")
    ax.plot(snr, v3, marker="s", linewidth=2.0, color="#4f6fb5", label="v3 SER")
    ax.plot(snr, dp1, marker="o", linewidth=2.3, color="#2f8b55", label="first-order Viterbi SER")
    ax.plot(snr, dp2, marker="^", linewidth=2.0, color="#a96633", label="second-order Viterbi SER")
    ax.set_ylim(0.0, 0.70)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("symbol error rate")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    _save(fig, output)


def _plot_ser_decomposition(rows: Sequence[dict[str, float]], output: Path) -> None:
    snr_labels = [f"{row['snr']:.0f}" for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.34
    missing = np.asarray([row["missing_lower_bound"] for row in rows], dtype=np.float64)
    dp1 = np.asarray([row["dp1_ser"] for row in rows], dtype=np.float64)
    dp2 = np.asarray([row["dp2_ser"] for row in rows], dtype=np.float64)
    dp1_extra = np.maximum(0.0, dp1 - missing)
    dp2_extra = np.maximum(0.0, dp2 - missing)

    fig, ax = plt.subplots(figsize=(12.8, 5.8), dpi=140)
    ax.bar(x - width / 2, missing, width, color="#b14646", alpha=0.65, label="GT missing from Top-24")
    ax.bar(x - width / 2, dp1_extra, width, bottom=missing, color="#2f8b55", alpha=0.75, label="DP1 selectable miss")
    ax.bar(x + width / 2, missing, width, color="#b14646", alpha=0.30)
    ax.bar(x + width / 2, dp2_extra, width, bottom=missing, color="#a96633", alpha=0.78, label="DP2 selectable miss")
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SER decomposition")
    ax.set_title("A large part of low-SNR SER is already unrecoverable before DP")
    ax.grid(True, axis="y", alpha=0.32)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    for idx, row in enumerate(rows):
        ax.text(idx - width / 2, dp1[idx] + 0.015, f"{dp1[idx]:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(idx + width / 2, dp2[idx] + 0.015, f"{dp2[idx]:.2f}", ha="center", va="bottom", fontsize=9)
    _save(fig, output)


def _packet_label(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "packet"
    first = rows[0]
    packet = str(first.get("packet_index", ""))
    snr = str(first.get("target_snr_db", ""))
    if packet and snr:
        return f"SNR {float(snr):g} dB packet {int(float(packet))}"
    if packet:
        return f"packet {int(float(packet))}"
    return "packet"


def _packet_file_stem(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "packet"
    first = rows[0]
    packet = int(float(first.get("packet_index", 0)))
    snr = float(first.get("target_snr_db", 0.0))
    safe_snr = f"{snr:g}".replace("-", "m").replace(".", "p")
    return f"snr_{safe_snr}_packet_{packet}"


def _plot_packet_status(rows: Sequence[dict[str, Any]], output: Path) -> None:
    idx = np.asarray([int(row["payload_symbol_index"]) for row in rows], dtype=np.int64)
    rank = np.asarray([int(row["gt_stage1_rank"]) if int(row["gt_stage1_rank"]) > 0 else 25 for row in rows], dtype=np.int64)
    missing = np.asarray([int(row["gt_in_stage1_top_l"]) == 0 for row in rows], dtype=bool)
    dp1_hit = np.asarray([int(row["phase_dp_first_hit"]) == 1 for row in rows], dtype=bool)
    dp2_hit = np.asarray([int(row["phase_dp_second_hit"]) == 1 for row in rows], dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(14.2, 8.2), dpi=140, sharex=True, height_ratios=[2.2, 1, 1])
    ax = axes[0]
    colors = np.where(missing, "#b14646", "#2f8b55")
    ax.scatter(idx, rank, c=colors, s=58, edgecolors="black", linewidths=0.3)
    ax.axhline(24.5, color="#b14646", linewidth=1.4, linestyle="--", alpha=0.8)
    ax.set_ylim(25.8, 0.2)
    ax.set_yticks([1, 3, 5, 8, 12, 16, 20, 24, 25])
    ax.set_yticklabels(["1", "3", "5", "8", "12", "16", "20", "24", "missing"])
    ax.set_ylabel("GT rank in Top-24")
    ax.set_title(f"{_packet_label(rows)}: candidate coverage and DP hits")
    ax.grid(True, alpha=0.28)

    for row in rows:
        if int(row["gt_in_stage1_top_l"]) == 0:
            continue
        if int(row["gt_stage1_rank"]) > 12:
            ax.text(
                int(row["payload_symbol_index"]),
                int(row["gt_stage1_rank"]) + 0.35,
                str(row["gt_stage1_rank"]),
                ha="center",
                va="top",
                fontsize=8,
                color="#333333",
            )

    for axis, hit, ylabel, title, color in (
        (axes[1], dp1_hit, "DP1", "first-order Viterbi: green=hit, red=miss", "#2f8b55"),
        (axes[2], dp2_hit, "DP2", "second-order Viterbi: green=hit, red=miss", "#a96633"),
    ):
        y = np.zeros_like(idx, dtype=np.float64)
        axis.scatter(idx[hit], y[hit], color=color, s=72, marker="o", label="hit")
        axis.scatter(idx[~hit], y[~hit], color="#b14646", s=74, marker="x", linewidths=2.0, label="miss")
        for i, row in enumerate(rows):
            if int(row["gt_in_stage1_top_l"]) == 0:
                axis.axvspan(idx[i] - 0.45, idx[i] + 0.45, color="#b14646", alpha=0.10)
        axis.set_yticks([])
        axis.set_ylabel(ylabel, rotation=0, labelpad=24, va="center", fontsize=11)
        axis.set_title(title, loc="left", fontsize=10, pad=4)
        axis.grid(True, axis="x", alpha=0.22)
        axis.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)

    axes[-1].set_xlabel("payload symbol index")
    _save(fig, output)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    summary_rows = _read_summary(Path(args.summary_csv), str(args.dataset))
    packet_rows = _read_packet_rows(Path(args.packet_symbol_csv))
    if not summary_rows:
        raise ValueError(f"no summary rows for dataset={args.dataset}")
    if not packet_rows:
        raise ValueError(f"no packet rows: {args.packet_symbol_csv}")
    _plot_recall_and_ser(summary_rows, out_dir / "snr_topL_recall_vs_ser.png")
    _plot_ser_decomposition(summary_rows, out_dir / "snr_ser_decomposition.png")
    _plot_packet_status(packet_rows, out_dir / f"{_packet_file_stem(packet_rows)}_gt_rank_and_dp_hits.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
