#!/usr/bin/env python3
"""Plot candidate-pruning sweep summaries."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


WEAK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_DIR = (
    WEAK_ROOT
    / "data"
    / "candidate_pruning"
    / "sweeps"
    / "validation_20260616_available_matrix"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot candidate-pruning sweep results.")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-l", default="1,8,16,32")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _parse_ints(text: str) -> list[int]:
    return [int(float(item.strip())) for item in str(text).split(",") if item.strip()]


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def _case_label(row: dict[str, str]) -> str:
    preamble = str(row["stem"]).split("_")[-1]
    return f"P{preamble}\n-{int(float(row['snr_db']))} dB"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _main_rows(best_rows: list[dict[str, str]], top_l: int) -> list[dict[str, str]]:
    rows = [
        row
        for row in best_rows
        if _int(row, "top_l") == int(top_l)
        and row.get("trend_source_arg") == "early-payload"
        and row.get("preselect_mode") == "default"
    ]
    return sorted(rows, key=lambda row: (row["stem"], _int(row, "snr_db")))


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.grid": True,
            "grid.color": "#d9dee7",
            "grid.linewidth": 0.65,
            "axes.axisbelow": True,
        }
    )
    return plt


def plot_top_l_recall_bars(
    best_rows: list[dict[str, str]],
    top_l: int,
    out_dir: Path,
    dpi: int,
) -> Path:
    plt = _setup_matplotlib()
    rows = _main_rows(best_rows, top_l)
    labels = [_case_label(row) for row in rows]
    center = [_float(row, "center_recall") for row in rows]
    multi = [_float(row, "multi_offset_recall") for row in rows]
    phase = [_float(row, "phase_gated_recall") for row in rows]

    x = list(range(len(rows)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5.6), dpi=int(dpi))
    ax.bar([v - width for v in x], center, width, label="center FFT", color="#7f8c8d")
    ax.bar(x, multi, width, label="multi-offset", color="#3b82f6")
    ax.bar([v + width for v in x], phase, width, label="phase-gated", color="#f59e0b")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(f"Recall@{top_l}")
    ax.set_title(f"GT-bin Recall@{top_l}: center vs multi-offset vs phase-gated")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncols=3, loc="upper right")
    fig.tight_layout()
    path = out_dir / f"recall_top{top_l}_bars.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_mean_recall_by_top_l(
    best_rows: list[dict[str, str]],
    top_l_values: list[int],
    out_dir: Path,
    dpi: int,
) -> Path:
    plt = _setup_matplotlib()
    means: dict[str, list[float]] = {"center": [], "multi": [], "phase": []}
    for top_l in top_l_values:
        rows = _main_rows(best_rows, top_l)
        means["center"].append(sum(_float(row, "center_recall") for row in rows) / len(rows))
        means["multi"].append(sum(_float(row, "multi_offset_recall") for row in rows) / len(rows))
        means["phase"].append(sum(_float(row, "phase_gated_recall") for row in rows) / len(rows))

    x = list(range(len(top_l_values)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=int(dpi))
    ax.bar([v - width for v in x], means["center"], width, label="center FFT", color="#7f8c8d")
    ax.bar(x, means["multi"], width, label="multi-offset", color="#3b82f6")
    ax.bar([v + width for v in x], means["phase"], width, label="phase-gated", color="#f59e0b")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean recall")
    ax.set_title("Mean Recall Across 12 Dataset/SNR Cases")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{value}" for value in top_l_values])
    ax.legend(ncols=3, loc="upper left")
    for metric, offset in (("center", -width), ("multi", 0.0), ("phase", width)):
        for index, value in enumerate(means[metric]):
            ax.text(index + offset, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out_dir / "mean_recall_by_top_l.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_gain_vs_multi_by_top_l(
    best_rows: list[dict[str, str]],
    top_l_values: list[int],
    out_dir: Path,
    dpi: int,
) -> Path:
    plt = _setup_matplotlib()
    rows_by_top = {top_l: _main_rows(best_rows, top_l) for top_l in top_l_values}
    labels = [_case_label(row) for row in rows_by_top[top_l_values[0]]]
    x = list(range(len(labels)))
    width = min(0.18, 0.72 / max(1, len(top_l_values)))
    offsets = [
        (index - (len(top_l_values) - 1) / 2.0) * width
        for index in range(len(top_l_values))
    ]
    colors = ["#16a34a", "#0891b2", "#8b5cf6", "#f97316"]

    fig, ax = plt.subplots(figsize=(14, 5.8), dpi=int(dpi))
    for index, top_l in enumerate(top_l_values):
        gains = [_float(row, "gain_vs_multi_offset") for row in rows_by_top[top_l]]
        ax.bar(
            [v + offsets[index] for v in x],
            gains,
            width,
            label=f"Top-{top_l}",
            color=colors[index % len(colors)],
        )
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_ylabel("Phase recall - multi-offset recall")
    ax.set_title("Phase-Gated Increment Over Multi-Offset")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncols=len(top_l_values), loc="upper right")
    fig.tight_layout()
    path = out_dir / "gain_vs_multi_by_top_l.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_rescue_damage_totals(
    best_rows: list[dict[str, str]],
    top_l_values: list[int],
    out_dir: Path,
    dpi: int,
) -> Path:
    plt = _setup_matplotlib()
    rescue: list[int] = []
    damage: list[int] = []
    for top_l in top_l_values:
        rows = _main_rows(best_rows, top_l)
        rescue.append(sum(_int(row, "phase_rescue_count") for row in rows))
        damage.append(sum(_int(row, "phase_damage_count") for row in rows))

    x = list(range(len(top_l_values)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=int(dpi))
    ax.bar([v - width / 2 for v in x], rescue, width, label="phase rescue", color="#22c55e")
    ax.bar([v + width / 2 for v in x], damage, width, label="phase damage", color="#ef4444")
    ax.set_ylabel("Symbol count")
    ax.set_title("Total Phase Rescue/Damage Across Main-Line Sweep")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{value}" for value in top_l_values])
    ax.legend(loc="upper right")
    for index, value in enumerate(rescue):
        ax.text(index - width / 2, value + 0.4, str(value), ha="center", va="bottom", fontsize=9)
    for index, value in enumerate(damage):
        ax.text(index + width / 2, value + 0.4, str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = out_dir / "phase_rescue_damage_totals.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_trend_source_comparison(
    best_rows: list[dict[str, str]],
    top_l: int,
    out_dir: Path,
    dpi: int,
) -> Path:
    plt = _setup_matplotlib()
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in best_rows:
        if _int(row, "top_l") != int(top_l) or row.get("preselect_mode") != "default":
            continue
        grouped[(row["stem"], _int(row, "snr_db"))][str(row["trend_source_arg"])] = row
    keys = sorted(grouped, key=lambda item: (item[0], item[1]))
    labels = [
        f"P{key[0].split('_')[-1]}\n-{key[1]} dB"
        for key in keys
    ]
    early = [
        _float(grouped[key].get("early-payload", {}), "phase_gated_recall")
        for key in keys
    ]
    header = [
        _float(grouped[key].get("header-offset", {}), "phase_gated_recall")
        for key in keys
    ]
    x = list(range(len(keys)))
    width = 0.32
    fig, ax = plt.subplots(figsize=(14, 5.6), dpi=int(dpi))
    ax.bar([v - width / 2 for v in x], early, width, label="early-payload", color="#f59e0b")
    ax.bar([v + width / 2 for v in x], header, width, label="header-offset", color="#6366f1")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(f"Best phase Recall@{top_l}")
    ax.set_title(f"Trend Source Comparison at Recall@{top_l}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = out_dir / f"trend_source_comparison_top{top_l}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_gain_heatmap(
    best_rows: list[dict[str, str]],
    top_l: int,
    out_dir: Path,
    dpi: int,
) -> Path:
    plt = _setup_matplotlib()
    rows = _main_rows(best_rows, top_l)
    stems = sorted({row["stem"] for row in rows}, key=lambda stem: int(stem.split("_")[-1]))
    snrs = sorted({_int(row, "snr_db") for row in rows})
    values = []
    by_key = {(row["stem"], _int(row, "snr_db")): row for row in rows}
    for stem in stems:
        values.append([
            _float(by_key[(stem, snr)], "gain_vs_multi_offset")
            for snr in snrs
        ])

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=int(dpi))
    image = ax.imshow(values, cmap="RdYlGn", vmin=-0.02, vmax=0.02, aspect="auto")
    ax.set_title(f"Gain vs Multi-Offset Heatmap, Recall@{top_l}")
    ax.set_xticks(range(len(snrs)))
    ax.set_xticklabels([f"-{snr} dB" for snr in snrs])
    ax.set_yticks(range(len(stems)))
    ax.set_yticklabels([f"P{stem.split('_')[-1]}" for stem in stems])
    for y, row_values in enumerate(values):
        for x, value in enumerate(row_values):
            ax.text(x, y, f"{value:+.4f}", ha="center", va="center", fontsize=9, color="#111827")
    fig.colorbar(image, ax=ax, label="phase - multi")
    fig.tight_layout()
    path = out_dir / f"gain_heatmap_top{top_l}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_index(paths: list[Path], out_dir: Path) -> Path:
    lines = [
        "# Candidate Pruning Plots",
        "",
        "Generated from `best_by_top_l.csv` in the same sweep directory.",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    index_path = out_dir / "PLOTS.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def main() -> int:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    out_dir = args.output_dir.resolve() if args.output_dir else sweep_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_rows = _load_csv(sweep_dir / "best_by_top_l.csv")
    top_l_values = _parse_ints(args.top_l)

    paths: list[Path] = []
    paths.append(plot_mean_recall_by_top_l(best_rows, top_l_values, out_dir, args.dpi))
    paths.append(plot_gain_vs_multi_by_top_l(best_rows, top_l_values, out_dir, args.dpi))
    paths.append(plot_rescue_damage_totals(best_rows, top_l_values, out_dir, args.dpi))
    paths.append(plot_trend_source_comparison(best_rows, 8, out_dir, args.dpi))
    paths.append(plot_gain_heatmap(best_rows, 8, out_dir, args.dpi))
    for top_l in top_l_values:
        paths.append(plot_top_l_recall_bars(best_rows, top_l, out_dir, args.dpi))
    index_path = write_index(paths, out_dir)

    print(f"wrote {len(paths)} plots")
    print(f"index: {index_path}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
