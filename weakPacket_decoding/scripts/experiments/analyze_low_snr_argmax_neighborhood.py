#!/usr/bin/env python3
"""Analyze whether the clean/GT FFT bin stays near the low-SNR argmax bin.

This is a second-pass analysis over CSV files produced by
run_low_snr_gt_bin_experiment.py.  That experiment already recomputes the
low-SNR FFT spectrum and records both:

  * gt_raw_fft_bin: clean/high-SNR payload bin treated as ground truth
  * noisy_argmax_bin: hard-decision FFT argmax in the noisy IQ

Here we measure their circular bin-coordinate distance.  This complements the
existing top-K rank recall metrics: rank tells whether GT remains among strong
peaks, while this script answers whether GT is physically close to the argmax
bin index.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WINDOWS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure circular-bin distance between low-SNR argmax and the "
            "clean GT bin exported by run_low_snr_gt_bin_experiment.py."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more *_low_snr_gt_bin_features_all.csv files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "low_snr_argmax_neighborhood",
        help="Output directory for symbol CSV, summaries, and plots.",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default=",".join(str(value) for value in DEFAULT_WINDOWS),
        help="Comma-separated +/- bin windows for neighborhood hit rates.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Only write CSV outputs.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG DPI.")
    return parser.parse_args()


def parse_windows(text: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one window is required.")
    if values[0] < 0:
        raise ValueError("Window sizes must be non-negative.")
    return values


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    if value == "":
        return int(default)
    return int(float(value))


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = str(row.get(key, "")).strip()
    if value == "":
        return float(default)
    return float(value)


def circular_signed_delta(target_bin: int, reference_bin: int, n_bins: int) -> int:
    """Return target-reference in the interval [-n_bins/2, n_bins/2)."""
    return int((int(target_bin) - int(reference_bin) + n_bins // 2) % n_bins - n_bins // 2)


def source_label(path: Path) -> str:
    parent = path.parent.name
    if parent:
        return parent
    return path.stem


def snr_label(snr_db: float) -> str:
    sign = "m" if float(snr_db) < 0 else "p"
    value = abs(float(snr_db))
    if abs(value - round(value)) < 1e-9:
        text = f"{int(round(value)):02d}"
    else:
        text = f"{value:.1f}".replace(".", "p")
    return f"snr_{sign}{text}dB"


def load_symbol_rows(paths: Iterable[Path], windows: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        label = source_label(resolved)
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                sf = _int(raw, "sf", 10)
                n_bins = 1 << sf
                gt_bin = _int(raw, "gt_raw_fft_bin", -1)
                argmax_bin = _int(raw, "noisy_argmax_bin", -1)
                if not (0 <= gt_bin < n_bins and 0 <= argmax_bin < n_bins):
                    continue
                signed_delta = circular_signed_delta(gt_bin, argmax_bin, n_bins)
                abs_distance = abs(signed_delta)
                out = {
                    "source_dataset": label,
                    "source_csv": str(resolved),
                    "target_snr_db": _float(raw, "target_snr_db"),
                    "packet_index": _int(raw, "packet_index", -1),
                    "event_index": _int(raw, "event_index", -1),
                    "frame_index": _int(raw, "frame_index", -1),
                    "payload_symbol_index": _int(raw, "payload_symbol_index", -1),
                    "sf": sf,
                    "n_bins": n_bins,
                    "gt_raw_fft_bin": gt_bin,
                    "noisy_argmax_bin": argmax_bin,
                    "argmax_to_gt_signed_delta": signed_delta,
                    "argmax_to_gt_abs_distance": abs_distance,
                    "is_argmax_correct": _int(raw, "is_argmax_correct", int(abs_distance == 0)),
                    "gt_bin_rank": _int(raw, "gt_bin_rank", -1),
                    "gt_peak_energy_ratio": _float(raw, "gt_peak_energy_ratio"),
                    "gt_to_argmax_power_db": _float(raw, "gt_to_argmax_power_db"),
                }
                for window in windows:
                    out[f"gt_within_pm{window}_bins"] = int(abs_distance <= int(window))
                rows.append(out)
    if not rows:
        raise ValueError("No usable symbol rows found in the input CSV files.")
    rows.sort(
        key=lambda item: (
            str(item["source_dataset"]),
            float(item["target_snr_db"]),
            int(item["packet_index"]),
            int(item["payload_symbol_index"]),
        )
    )
    return rows


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def summarize_group(rows: list[dict[str, Any]], windows: list[int]) -> dict[str, Any]:
    first = rows[0]
    distances = np.asarray([float(row["argmax_to_gt_abs_distance"]) for row in rows], dtype=np.float64)
    wrong_distances = distances[distances > 0.0]
    ranks = np.asarray([float(row["gt_bin_rank"]) for row in rows if int(row["gt_bin_rank"]) > 0], dtype=np.float64)
    n_bins_values = sorted({int(row["n_bins"]) for row in rows})
    n_bins = n_bins_values[0] if len(n_bins_values) == 1 else float("nan")

    summary: dict[str, Any] = {
        "source_dataset": first["source_dataset"],
        "target_snr_db": float(first["target_snr_db"]),
        "symbol_count": len(rows),
        "packet_count": len({int(row["packet_index"]) for row in rows}),
        "n_bins": n_bins,
        "argmax_correct_rate": float(np.mean(distances == 0.0)),
        "mean_abs_distance_bins": float(np.mean(distances)),
        "median_abs_distance_bins": finite_percentile(distances, 50.0),
        "p75_abs_distance_bins": finite_percentile(distances, 75.0),
        "p90_abs_distance_bins": finite_percentile(distances, 90.0),
        "p95_abs_distance_bins": finite_percentile(distances, 95.0),
        "p99_abs_distance_bins": finite_percentile(distances, 99.0),
        "mean_distance_fraction_of_half_band": float(np.mean(distances / (float(n_bins) / 2.0)))
        if np.isfinite(float(n_bins))
        else float("nan"),
        "gt_rank_mean": float(np.mean(ranks)) if ranks.size else float("nan"),
        "gt_rank_median": float(np.median(ranks)) if ranks.size else float("nan"),
        "gt_top8_recall": float(np.mean(ranks <= 8)) if ranks.size else float("nan"),
        "gt_top16_recall": float(np.mean(ranks <= 16)) if ranks.size else float("nan"),
        "gt_top32_recall": float(np.mean(ranks <= 32)) if ranks.size else float("nan"),
        "wrong_argmax_symbol_count": int(wrong_distances.size),
        "wrong_argmax_mean_abs_distance_bins": float(np.mean(wrong_distances)) if wrong_distances.size else 0.0,
        "wrong_argmax_median_abs_distance_bins": finite_percentile(wrong_distances, 50.0)
        if wrong_distances.size
        else 0.0,
        "wrong_argmax_p90_abs_distance_bins": finite_percentile(wrong_distances, 90.0)
        if wrong_distances.size
        else 0.0,
    }
    for window in windows:
        hit_rate = float(np.mean(distances <= float(window)))
        summary[f"gt_within_pm{window}_bins_rate"] = hit_rate
        summary[f"wrong_argmax_gt_within_pm{window}_bins_rate"] = (
            float(np.mean(wrong_distances <= float(window))) if wrong_distances.size else float("nan")
        )
        if np.isfinite(float(n_bins)):
            summary[f"random_within_pm{window}_bins_rate"] = min(1.0, float(2 * int(window) + 1) / float(n_bins))
    return summary


def build_summaries(rows: list[dict[str, Any]], windows: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_snr: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    by_packet: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        snr_key = (str(row["source_dataset"]), float(row["target_snr_db"]))
        packet_key = (str(row["source_dataset"]), float(row["target_snr_db"]), int(row["packet_index"]))
        by_snr[snr_key].append(row)
        by_packet[packet_key].append(row)

    snr_rows = [summarize_group(items, windows) for _, items in sorted(by_snr.items())]
    packet_rows: list[dict[str, Any]] = []
    for (_, _, packet_index), items in sorted(by_packet.items()):
        item = summarize_group(items, windows)
        item["packet_index"] = int(packet_index)
        item["event_index"] = int(items[0]["event_index"])
        item["frame_index"] = int(items[0]["frame_index"])
        packet_rows.append(item)
    return snr_rows, packet_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_hit_rates(summary_rows: list[dict[str, Any]], windows: list[int], out_path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({str(row["source_dataset"]) for row in summary_rows})
    plot_windows = [window for window in windows if window in {0, 1, 2, 4, 8, 16, 32, 64, 128}]
    if not plot_windows:
        plot_windows = windows[: min(6, len(windows))]

    fig, axes = plt.subplots(len(datasets), 1, figsize=(10.8, 4.2 * len(datasets)), dpi=int(dpi), squeeze=False)
    for axis, dataset in zip(axes.ravel(), datasets):
        items = [row for row in summary_rows if str(row["source_dataset"]) == dataset]
        items.sort(key=lambda row: float(row["target_snr_db"]))
        x = np.asarray([float(row["target_snr_db"]) for row in items], dtype=np.float64)
        for window in plot_windows:
            y = np.asarray([float(row[f"gt_within_pm{window}_bins_rate"]) for row in items], dtype=np.float64)
            axis.plot(x, y, marker="o", linewidth=1.45, label=f"+/-{window}")
        axis.set_title(f"{dataset}: GT-bin neighborhood hit rate around low-SNR argmax")
        axis.set_xlabel("target SNR (dB)")
        axis.set_ylabel("hit rate")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_wrong_argmax_hit_rates(summary_rows: list[dict[str, Any]], windows: list[int], out_path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({str(row["source_dataset"]) for row in summary_rows})
    plot_windows = [window for window in windows if window in {1, 2, 4, 8, 16, 32, 64, 128, 256}]
    if not plot_windows:
        plot_windows = windows[: min(6, len(windows))]

    fig, axes = plt.subplots(len(datasets), 1, figsize=(10.8, 4.2 * len(datasets)), dpi=int(dpi), squeeze=False)
    for axis, dataset in zip(axes.ravel(), datasets):
        items = [row for row in summary_rows if str(row["source_dataset"]) == dataset]
        items.sort(key=lambda row: float(row["target_snr_db"]))
        x = np.asarray([float(row["target_snr_db"]) for row in items], dtype=np.float64)
        for window in plot_windows:
            y = np.asarray(
                [float(row[f"wrong_argmax_gt_within_pm{window}_bins_rate"]) for row in items],
                dtype=np.float64,
            )
            axis.plot(x, y, marker="o", linewidth=1.45, label=f"+/-{window}")
        axis.set_title(f"{dataset}: GT near argmax when argmax is wrong")
        axis.set_xlabel("target SNR (dB)")
        axis.set_ylabel("conditional hit rate")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_distance_boxplot(symbol_rows: list[dict[str, Any]], out_path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in symbol_rows:
        groups[(str(row["source_dataset"]), float(row["target_snr_db"]))].append(float(row["argmax_to_gt_abs_distance"]))

    labels: list[str] = []
    values: list[list[float]] = []
    for (dataset, snr_db), distances in sorted(groups.items()):
        labels.append(f"{dataset}\n{snr_db:g} dB")
        values.append(distances)

    fig, axis = plt.subplots(figsize=(max(10.0, 1.1 * len(labels)), 5.2), dpi=int(dpi))
    axis.boxplot(values, labels=labels, showfliers=False)
    axis.set_title("Circular bin distance from low-SNR argmax to GT bin")
    axis.set_ylabel("abs circular distance (bins)")
    axis.grid(True, axis="y", color="#dddddd", linewidth=0.6)
    axis.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_signed_delta_histograms(symbol_rows: list[dict[str, Any]], out_path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for row in symbol_rows:
        groups[(str(row["source_dataset"]), float(row["target_snr_db"]))].append(int(row["argmax_to_gt_signed_delta"]))

    keys = sorted(groups.keys())
    cols = min(3, len(keys))
    rows_count = int(math.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(4.8 * cols, 3.2 * rows_count), dpi=int(dpi), squeeze=False)
    for axis in axes.ravel():
        axis.set_visible(False)

    for axis, key in zip(axes.ravel(), keys):
        dataset, snr_db = key
        values = np.asarray(groups[key], dtype=np.int32)
        axis.set_visible(True)
        axis.hist(values, bins=64, color="#4c78a8", alpha=0.86)
        axis.axvline(0, color="black", linewidth=0.9)
        axis.set_title(f"{dataset}, {snr_db:g} dB")
        axis.set_xlabel("GT - argmax signed delta (bins)")
        axis.set_ylabel("count")
        axis.grid(True, color="#dddddd", linewidth=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    windows = parse_windows(args.windows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    symbol_rows = load_symbol_rows(args.input, windows)
    summary_by_snr, summary_by_packet = build_summaries(symbol_rows, windows)

    symbol_csv = output_dir / "argmax_neighborhood_symbols.csv"
    snr_csv = output_dir / "argmax_neighborhood_summary_by_snr.csv"
    packet_csv = output_dir / "argmax_neighborhood_summary_by_packet.csv"
    write_csv(symbol_csv, symbol_rows)
    write_csv(snr_csv, summary_by_snr)
    write_csv(packet_csv, summary_by_packet)

    if not args.no_plots:
        plot_hit_rates(summary_by_snr, windows, output_dir / "argmax_neighborhood_hit_rates.png", args.dpi)
        plot_wrong_argmax_hit_rates(
            summary_by_snr,
            windows,
            output_dir / "argmax_neighborhood_wrong_argmax_hit_rates.png",
            args.dpi,
        )
        plot_distance_boxplot(symbol_rows, output_dir / "argmax_neighborhood_abs_distance_boxplot.png", args.dpi)
        plot_signed_delta_histograms(symbol_rows, output_dir / "argmax_neighborhood_signed_delta_histograms.png", args.dpi)

    print(f"symbols={len(symbol_rows)}")
    print(f"summary_rows={len(summary_by_snr)}")
    print(f"wrote_symbols={symbol_csv}")
    print(f"wrote_summary_by_snr={snr_csv}")
    print(f"wrote_summary_by_packet={packet_csv}")
    if not args.no_plots:
        print(f"wrote_plots={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
