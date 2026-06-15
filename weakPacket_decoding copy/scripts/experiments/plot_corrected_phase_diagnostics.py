#!/usr/bin/env python3
"""绘制 corrected payload FFT peak phase 的诊断图。

这个脚本用于回答一个具体问题：

    corrected 后的 wrapped phase 为什么仍然像锯齿？

输入是 run_header_first_demod.py 导出的 *_header_first_symbols.csv。
脚本只读取 header_valid=1 的 payload symbol，对每个 packet 输出四联图：

1. wrapped peak phase；
2. unwrap phase 与一次线性拟合；
3. 去掉线性趋势后的 phase residual；
4. residual 与 raw_fft_bin 的关系。
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot corrected payload FFT peak phase diagnostics from "
            "run_header_first_demod.py symbol CSV."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="header-first symbol CSV，例如 *_header_first_symbols.csv。",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "PNG 输出目录。默认写到 "
            "data/payload_feature_no_offset/plots，便于和 raw/corrected 趋势图放在一起。"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="每个 packet 的拟合统计 CSV。默认写到输出目录 corrected_phase_diagnostics_summary.csv。",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        default=False,
        help="只画图，不写 summary CSV。",
    )
    parser.add_argument(
        "--packet",
        type=int,
        action="append",
        default=None,
        help="只画指定 packet_index。可以重复传多次；默认画全部有效 packet。",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="最多绘制前 N 个 packet，调试大文件时使用。",
    )
    parser.add_argument("--dpi", type=int, default=180, help="输出 PNG 的 DPI。")
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return int(default)
    return int(float(value))


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return float(default)
    return float(value)


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def load_payload_rows(path: Path, packet_filter: set[int] | None) -> list[dict[str, object]]:
    """读取 header 有效的 payload symbol 行。"""

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("stage") != "payload":
                continue
            if _int(raw, "header_valid", 0) != 1:
                continue
            packet_index = _int(raw, "packet_index", -1)
            if packet_filter is not None and packet_index not in packet_filter:
                continue
            peak_power = _float(raw, "peak_power")
            total_power = _float(raw, "total_power")
            rows.append(
                {
                    "frame_index": _int(raw, "frame_index", -1),
                    "packet_index": packet_index,
                    "event_index": _int(raw, "event_index", -1),
                    "payload_symbol_index": _int(raw, "stage_symbol_index", -1),
                    "raw_fft_bin": _int(raw, "raw_fft_bin", -1),
                    "signed_fft_bin": _int(raw, "signed_fft_bin", 0),
                    "symbol_value": _int(raw, "symbol_value", -1),
                    "peak_amp": _float(raw, "peak_amp"),
                    "peak_phase": _float(raw, "peak_phase"),
                    "peak_margin_db": _float(raw, "peak_margin_db"),
                    "peak_power": peak_power,
                    "total_power": total_power,
                    "peak_energy_ratio": float(peak_power / total_power)
                    if np.isfinite(peak_power) and np.isfinite(total_power) and total_power > 0.0
                    else float("nan"),
                    "cfo_int": _int(raw, "cfo_int", 0),
                    "cfo_frac": _float(raw, "cfo_frac"),
                    "sto_frac": _float(raw, "sto_frac"),
                    "sfo_hat": _float(raw, "sfo_hat"),
                    "sfo_cum_before": _float(raw, "sfo_cum_before"),
                    "sfo_sample_adjust_after": _int(raw, "sfo_sample_adjust_after", 0),
                }
            )
    rows.sort(key=lambda item: (int(item["packet_index"]), int(item["payload_symbol_index"])))
    return rows


def group_by_packet(rows: Iterable[dict[str, object]], max_packets: int | None) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["packet_index"])].append(row)
    ordered = dict(sorted(grouped.items()))
    if max_packets is None:
        return ordered
    return dict(list(ordered.items())[: int(max_packets)])


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """返回一次拟合系数、预测值、R^2 和 RMSE。"""

    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    rmse = float(math.sqrt(np.mean((y - pred) ** 2)))
    return coef, pred, r2, rmse


def packet_arrays(packet_rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    """把一组 packet 行转成绘图和统计用数组。"""

    k = np.asarray([int(row["payload_symbol_index"]) for row in packet_rows], dtype=np.float64)
    phase_wrapped = np.asarray([float(row["peak_phase"]) for row in packet_rows], dtype=np.float64)
    phase_unwrap = np.unwrap(phase_wrapped)
    fit_coef, fit_line, fit_r2, fit_rmse = linear_fit(k, phase_unwrap)
    residual = np.angle(np.exp(1j * (phase_unwrap - fit_line)))
    raw_bin = np.asarray([int(row["raw_fft_bin"]) for row in packet_rows], dtype=np.float64)
    bin_coef, bin_fit, bin_r2, bin_rmse = linear_fit(raw_bin, residual)
    return {
        "k": k,
        "phase_wrapped": phase_wrapped,
        "phase_unwrap": phase_unwrap,
        "fit_coef": fit_coef,
        "fit_line": fit_line,
        "fit_r2": np.asarray([fit_r2], dtype=np.float64),
        "fit_rmse": np.asarray([fit_rmse], dtype=np.float64),
        "residual": residual,
        "raw_bin": raw_bin,
        "bin_coef": bin_coef,
        "bin_fit": bin_fit,
        "bin_r2": np.asarray([bin_r2], dtype=np.float64),
        "bin_rmse": np.asarray([bin_rmse], dtype=np.float64),
        "margin": np.asarray([float(row["peak_margin_db"]) for row in packet_rows], dtype=np.float64),
        "energy_ratio": np.asarray([float(row["peak_energy_ratio"]) for row in packet_rows], dtype=np.float64),
        "amp": np.asarray([float(row["peak_amp"]) for row in packet_rows], dtype=np.float64),
    }


def packet_summary(packet_index: int, packet_rows: list[dict[str, object]], arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """整理单个 packet 的 phase 诊断统计量。"""

    first = packet_rows[0]
    phase_wrapped = arrays["phase_wrapped"]
    phase_unwrap = arrays["phase_unwrap"]
    residual = arrays["residual"]
    fit_coef = arrays["fit_coef"]
    bin_coef = arrays["bin_coef"]
    cfo_frac = float(first["cfo_frac"])
    expected_cfo_slope_pi = 2.0 * cfo_frac
    actual_slope_pi = float(fit_coef[0] / math.pi)
    return {
        "packet_index": int(packet_index),
        "event_index": int(first["event_index"]),
        "frame_index": int(first["frame_index"]),
        "payload_symbol_count": len(packet_rows),
        "cfo_int": int(first["cfo_int"]),
        "cfo_frac": cfo_frac,
        "sto_frac": float(first["sto_frac"]),
        "sfo_hat": float(first["sfo_hat"]),
        "phase_slope_rad_per_symbol": float(fit_coef[0]),
        "phase_slope_pi_per_symbol": actual_slope_pi,
        "phase_intercept_pi": float(fit_coef[1] / math.pi),
        "cfo_expected_slope_pi_per_symbol": expected_cfo_slope_pi,
        "slope_over_cfo_expected": float(actual_slope_pi / expected_cfo_slope_pi)
        if abs(expected_cfo_slope_pi) > 1e-12
        else "",
        "linear_fit_r2": float(arrays["fit_r2"][0]),
        "linear_fit_rmse_pi": float(arrays["fit_rmse"][0] / math.pi),
        "wrapped_phase_range_pi": float((np.max(phase_wrapped) - np.min(phase_wrapped)) / math.pi),
        "unwrap_phase_range_pi": float((np.max(phase_unwrap) - np.min(phase_unwrap)) / math.pi),
        "residual_mean_pi": float(np.mean(residual) / math.pi),
        "residual_std_pi": float(np.std(residual) / math.pi),
        "residual_peak_to_peak_pi": float((np.max(residual) - np.min(residual)) / math.pi),
        "residual_bin_slope_pi_per_bin": float(bin_coef[0] / math.pi),
        "residual_bin_fit_r2": float(arrays["bin_r2"][0]),
        "residual_bin_fit_rmse_pi": float(arrays["bin_rmse"][0] / math.pi),
        "peak_margin_mean_db": _finite_mean(arrays["margin"]),
        "peak_energy_ratio_mean": _finite_mean(arrays["energy_ratio"]),
        "peak_amp_mean": _finite_mean(arrays["amp"]),
    }


def plot_packet(
    packet_index: int,
    packet_rows: list[dict[str, object]],
    arrays: dict[str, np.ndarray],
    out_path: Path,
    dpi: int,
) -> None:
    """绘制单包四联诊断图。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = packet_rows[0]
    k = arrays["k"]
    phase_wrapped = arrays["phase_wrapped"]
    phase_unwrap = arrays["phase_unwrap"]
    fit_line = arrays["fit_line"]
    residual = arrays["residual"]
    raw_bin = arrays["raw_bin"]
    margin = arrays["margin"]
    energy_ratio = arrays["energy_ratio"]
    fit_coef = arrays["fit_coef"]
    fit_r2 = float(arrays["fit_r2"][0])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), dpi=int(dpi))
    axes = axes.ravel()

    axes[0].plot(k, phase_wrapped / math.pi, "o-", markersize=3.0, linewidth=1.1)
    axes[0].set_title("wrapped corrected peak phase")
    axes[0].set_ylabel("phase / pi")
    axes[0].set_ylim(-1.05, 1.05)

    axes[1].plot(k, phase_unwrap / math.pi, "o-", markersize=3.0, linewidth=1.1, label="unwrap")
    axes[1].plot(
        k,
        fit_line / math.pi,
        "-",
        linewidth=2.0,
        label=f"linear fit, slope={fit_coef[0] / math.pi:.3f} pi/sym",
    )
    axes[1].set_title(f"unwrap phase: R2={fit_r2:.4f}, cfo_frac={float(first['cfo_frac']):.4f}")
    axes[1].legend()

    axes[2].plot(k, residual / math.pi, "o-", markersize=3.0, linewidth=1.1)
    axes[2].axhline(0.0, color="black", linewidth=0.75)
    axes[2].set_title("phase residual after linear detrend")
    axes[2].set_xlabel("payload symbol index")
    axes[2].set_ylabel("residual / pi")

    scatter = axes[3].scatter(raw_bin, residual / math.pi, c=margin, cmap="viridis", s=34)
    axes[3].set_title(f"residual vs raw_fft_bin, sto_frac={float(first['sto_frac']):.4f}")
    axes[3].set_xlabel("raw_fft_bin")
    axes[3].set_ylabel("residual / pi")
    cbar = fig.colorbar(scatter, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("peak_margin_db")

    for axis in axes:
        axis.grid(True, color="#dddddd", linewidth=0.6)

    title = (
        f"Packet {packet_index} corrected phase diagnostics"
        f" | margin mean={_finite_mean(margin):.2f} dB"
        f" | energy ratio mean={_finite_mean(energy_ratio):.3f}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "packet_index",
        "event_index",
        "frame_index",
        "payload_symbol_count",
        "cfo_int",
        "cfo_frac",
        "sto_frac",
        "sfo_hat",
        "phase_slope_rad_per_symbol",
        "phase_slope_pi_per_symbol",
        "phase_intercept_pi",
        "cfo_expected_slope_pi_per_symbol",
        "slope_over_cfo_expected",
        "linear_fit_r2",
        "linear_fit_rmse_pi",
        "wrapped_phase_range_pi",
        "unwrap_phase_range_pi",
        "residual_mean_pi",
        "residual_std_pi",
        "residual_peak_to_peak_pi",
        "residual_bin_slope_pi_per_bin",
        "residual_bin_fit_r2",
        "residual_bin_fit_rmse_pi",
        "peak_margin_mean_db",
        "peak_energy_ratio_mean",
        "peak_amp_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    packet_filter = set(args.packet) if args.packet is not None else None
    output_dir = args.output_dir or (WEAK_ROOT / "data" / "payload_feature_no_offset" / "plots")
    summary_output = args.summary_output or (output_dir / "corrected_phase_diagnostics_summary.csv")

    rows = load_payload_rows(args.input, packet_filter=packet_filter)
    grouped = group_by_packet(rows, max_packets=args.max_packets)
    if not grouped:
        raise ValueError("没有找到 header_valid=1 的 payload symbol 行。")

    summaries: list[dict[str, object]] = []
    png_count = 0
    for packet_index, packet_rows in grouped.items():
        arrays = packet_arrays(packet_rows)
        summaries.append(packet_summary(packet_index, packet_rows, arrays))
        out_path = output_dir / f"packet_{packet_index:03d}_corrected_phase_diagnostics.png"
        plot_packet(packet_index, packet_rows, arrays, out_path=out_path, dpi=args.dpi)
        png_count += 1

    if not args.no_summary:
        write_summary(summary_output, summaries)

    print(f"packets={len(grouped)}")
    print(f"payload_rows={len(rows)}")
    print(f"wrote_dir={output_dir}")
    print(f"wrote_pngs={png_count}")
    if not args.no_summary:
        print(f"summary={summary_output}")


if __name__ == "__main__":
    main()
