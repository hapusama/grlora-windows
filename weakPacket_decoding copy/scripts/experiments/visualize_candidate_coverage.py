#!/usr/bin/env python3
"""可视化候选集覆盖率分析结果。

从 analyze_gt_bin_candidate_coverage.py 的输出 CSV 生成详细的可视化图表：
1. GT bin amplitude rank 分布（直方图 + CDF）
2. Argmax 错误时的 circular distance 分布
3. 各种候选集构造方法的覆盖率对比（条形图）
4. 覆盖率 vs K/W 参数曲线
5. Per-packet 覆盖率热图
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize candidate coverage analysis results"
    )
    parser.add_argument(
        "-i", "--input-csv",
        type=Path,
        required=True,
        help="Input CSV from analyze_gt_bin_candidate_coverage.py",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150)",
    )
    return parser.parse_args()


def load_results(csv_path: Path) -> list[dict[str, Any]]:
    """Load results from CSV."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Convert numeric fields
            converted = {}
            for key, value in row.items():
                try:
                    if "." in value:
                        converted[key] = float(value)
                    else:
                        converted[key] = int(value)
                except (ValueError, AttributeError):
                    converted[key] = value
            rows.append(converted)
    return rows


def plot_rank_distribution(rows: list[dict[str, Any]], output_dir: Path, dpi: int):
    """绘制 GT bin rank 分布（直方图 + CDF）。"""
    import matplotlib.pyplot as plt

    ranks = np.array([r["gt_rank_by_amp"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=dpi)

    # Histogram
    max_rank = int(np.max(ranks))
    bins = np.arange(1, min(max_rank + 2, 130))
    ax1.hist(ranks, bins=bins, edgecolor="black", alpha=0.7, color="#1f77b4")
    ax1.axvline(np.median(ranks), color="red", linestyle="--", linewidth=2, label=f"Median: {np.median(ranks):.0f}")
    ax1.axvline(np.mean(ranks), color="orange", linestyle="--", linewidth=2, label=f"Mean: {np.mean(ranks):.1f}")
    ax1.set_xlabel("GT Bin Rank (by Amplitude)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"GT Bin Rank Distribution (n={len(ranks)})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # CDF
    sorted_ranks = np.sort(ranks)
    cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    ax2.plot(sorted_ranks, cdf, linewidth=2, color="#1f77b4")
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50th percentile")
    ax2.axhline(0.9, color="orange", linestyle="--", alpha=0.5, label="90th percentile")
    ax2.axhline(0.95, color="green", linestyle="--", alpha=0.5, label="95th percentile")
    ax2.set_xlabel("GT Bin Rank (by Amplitude)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("CDF of GT Bin Rank")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    fig.tight_layout()
    output_path = output_dir / "gt_bin_rank_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_circular_distance(rows: list[dict[str, Any]], output_dir: Path, dpi: int):
    """绘制 argmax 错误时的 circular distance 分布。"""
    import matplotlib.pyplot as plt

    error_rows = [r for r in rows if r["is_argmax_correct"] == 0]
    if not error_rows:
        print("  No argmax errors, skipping circular distance plot")
        return

    dists = np.array([r["circular_distance"] for r in error_rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=dpi)

    # Histogram
    max_dist = int(np.max(dists))
    bins = np.arange(0, max_dist + 2)
    ax1.hist(dists, bins=bins, edgecolor="black", alpha=0.7, color="#d62728")
    ax1.axvline(np.median(dists), color="blue", linestyle="--", linewidth=2, label=f"Median: {np.median(dists):.0f}")
    ax1.set_xlabel("Circular Distance (GT bin to Argmax bin)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Circular Distance When Argmax is Wrong (n={len(error_rows)})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # CDF
    sorted_dists = np.sort(dists)
    cdf = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
    ax2.plot(sorted_dists, cdf, linewidth=2, color="#d62728")
    ax2.set_xlabel("Circular Distance")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("CDF of Circular Distance")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    fig.tight_layout()
    output_path = output_dir / "circular_distance_distribution.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_coverage_comparison(rows: list[dict[str, Any]], output_dir: Path, dpi: int):
    """对比不同候选集构造方法的覆盖率。"""
    import matplotlib.pyplot as plt

    n_symbols = len(rows)

    # Extract coverage metrics
    coverage_metrics = {}
    for key in rows[0].keys():
        if key.startswith("in_amp_top") or key.startswith("in_argmax_window_") or \
           key.startswith("in_phase_top") or key.startswith("in_combined_top"):
            coverage_metrics[key] = sum(r[key] for r in rows) / n_symbols

    # Group by method
    amp_topk = {k: v for k, v in coverage_metrics.items() if k.startswith("in_amp_top")}
    argmax_window = {k: v for k, v in coverage_metrics.items() if k.startswith("in_argmax_window_")}
    phase_topk = {k: v for k, v in coverage_metrics.items() if k.startswith("in_phase_top")}
    combined_topk = {k: v for k, v in coverage_metrics.items() if k.startswith("in_combined_top")}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)
    axes = axes.ravel()

    # Plot each method
    def plot_method(ax, data, title, xlabel):
        if not data:
            return
        labels = list(data.keys())
        values = list(data.values())
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))
        bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("in_", "").replace("_", " ") for l in labels], rotation=45, ha="right")
        ax.set_ylabel("Coverage Rate")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plot_method(axes[0], amp_topk, "Amplitude TopK Coverage", "TopK")
    plot_method(axes[1], argmax_window, "Argmax ±W Window Coverage", "Window Size")
    plot_method(axes[2], phase_topk, "Phase-Residual TopK Coverage", "TopK")
    plot_method(axes[3], combined_topk, "Amplitude+Phase Combined TopK Coverage", "TopK")

    fig.suptitle(f"Candidate Set Coverage Comparison (n={n_symbols} symbols)", fontsize=14, y=0.995)
    fig.tight_layout()
    output_path = output_dir / "coverage_comparison.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_coverage_curves(rows: list[dict[str, Any]], output_dir: Path, dpi: int):
    """绘制覆盖率随 K/W 参数变化的曲线。"""
    import matplotlib.pyplot as plt

    n_symbols = len(rows)

    # Extract parameter values and coverage rates
    def extract_series(prefix, param_extractor):
        series = {}
        for key in rows[0].keys():
            if key.startswith(prefix):
                param = param_extractor(key)
                if param is not None:
                    rate = sum(r[key] for r in rows) / n_symbols
                    series[param] = rate
        return dict(sorted(series.items()))

    amp_topk = extract_series("in_amp_top", lambda k: int(k.replace("in_amp_top", "")))
    argmax_w = extract_series("in_argmax_window_", lambda k: int(k.replace("in_argmax_window_", "")))
    phase_topk = extract_series("in_phase_top", lambda k: int(k.replace("in_phase_top", "")))
    combined_topk = extract_series("in_combined_top", lambda k: int(k.replace("in_combined_top", "")))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)
    axes = axes.ravel()

    def plot_curve(ax, data, title, xlabel, color, marker="o"):
        if not data:
            return
        params = list(data.keys())
        rates = list(data.values())
        ax.plot(params, rates, marker=marker, linewidth=2, markersize=8, color=color, label=title)
        ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="90% coverage")
        ax.axhline(0.95, color="orange", linestyle="--", alpha=0.5, linewidth=1.5, label="95% coverage")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Coverage Rate")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_curve(axes[0], amp_topk, "Amplitude TopK", "K", "#1f77b4")
    plot_curve(axes[1], argmax_w, "Argmax ±W Window", "W", "#ff7f0e")
    plot_curve(axes[2], phase_topk, "Phase-Residual TopK", "K", "#2ca02c")
    plot_curve(axes[3], combined_topk, "Amplitude+Phase TopK", "K", "#d62728")

    fig.suptitle(f"Coverage Rate vs Parameter (n={n_symbols} symbols)", fontsize=14, y=0.995)
    fig.tight_layout()
    output_path = output_dir / "coverage_curves.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_packet_heatmap(rows: list[dict[str, Any]], output_dir: Path, dpi: int):
    """绘制每个 packet 的符号级覆盖率热图。"""
    import matplotlib.pyplot as plt

    # Group by packet
    by_packet = defaultdict(list)
    for r in rows:
        by_packet[r["packet_index"]].append(r)

    if len(by_packet) > 20:
        print(f"  Too many packets ({len(by_packet)}), skipping per-packet heatmap")
        return

    # Select representative coverage metrics
    metrics = ["in_amp_top8", "in_argmax_window_4", "in_phase_top8", "in_combined_top8"]
    metric_labels = ["Amp Top8", "Argmax ±4", "Phase Top8", "Combined Top8"]

    # Check if metrics exist
    available_metrics = [m for m in metrics if m in rows[0]]
    if not available_metrics:
        print("  No standard metrics found, skipping heatmap")
        return

    # Build matrix: packets × symbols × metrics
    packets = sorted(by_packet.keys())
    max_symbols = max(len(syms) for syms in by_packet.values())

    for metric_idx, metric in enumerate(available_metrics):
        matrix = np.full((len(packets), max_symbols), np.nan)
        for i, pkt in enumerate(packets):
            packet_rows = sorted(by_packet[pkt], key=lambda r: r["payload_symbol_index"])
            for j, r in enumerate(packet_rows):
                if j < max_symbols:
                    matrix[i, j] = r.get(metric, 0)

        fig, ax = plt.subplots(figsize=(14, max(6, len(packets) * 0.4)), dpi=dpi)
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")

        ax.set_xlabel("Payload Symbol Index")
        ax.set_ylabel("Packet Index")
        ax.set_title(f"Per-Symbol Coverage: {metric_labels[metric_idx] if metric_idx < len(metric_labels) else metric}")
        ax.set_yticks(range(len(packets)))
        ax.set_yticklabels(packets)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Coverage (1=covered, 0=missed)")

        fig.tight_layout()
        output_path = output_dir / f"per_packet_coverage_{metric}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path}")


def print_summary_stats(rows: list[dict[str, Any]]):
    """打印汇总统计。"""
    n_symbols = len(rows)
    n_packets = len({r["packet_index"] for r in rows})

    argmax_correct = sum(r["is_argmax_correct"] for r in rows)
    argmax_accuracy = argmax_correct / n_symbols

    ranks = np.array([r["gt_rank_by_amp"] for r in rows])

    print("\n=== Summary Statistics ===")
    print(f"Total symbols: {n_symbols}")
    print(f"Total packets: {n_packets}")
    print(f"Argmax accuracy: {argmax_accuracy:.3f}")
    print(f"\nGT bin rank (by amplitude):")
    print(f"  Mean: {np.mean(ranks):.2f}")
    print(f"  Median: {np.median(ranks):.0f}")
    print(f"  90th percentile: {np.percentile(ranks, 90):.0f}")
    print(f"  95th percentile: {np.percentile(ranks, 95):.0f}")
    print(f"  Max: {np.max(ranks):.0f}")

    # Top coverage rates
    print(f"\nKey coverage rates:")
    for key in ["in_amp_top8", "in_argmax_window_4", "in_phase_top8", "in_combined_top8"]:
        if key in rows[0]:
            rate = sum(r[key] for r in rows) / n_symbols
            print(f"  {key}: {rate:.3f}")


def main() -> int:
    args = parse_args()

    print(f"Loading results from {args.input_csv}...")
    rows = load_results(args.input_csv)
    print(f"  Loaded {len(rows)} symbols")

    print_summary_stats(rows)

    print(f"\nGenerating plots...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    plot_rank_distribution(rows, args.output_dir, args.dpi)
    plot_circular_distance(rows, args.output_dir, args.dpi)
    plot_coverage_comparison(rows, args.output_dir, args.dpi)
    plot_coverage_curves(rows, args.output_dir, args.dpi)
    plot_per_packet_heatmap(rows, args.output_dir, args.dpi)

    print(f"\nAll plots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
