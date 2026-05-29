#!/usr/bin/env python3
"""corrected payload FFT 的 wrong-bin phase/amplitude 对照实验。

目的：
    验证 corrected FFT demod 后观察到的相位/幅度平滑特征，
    是只存在于 selected/正确 bin，还是错误 bin 上也同样存在。

输入：
    1. 原始 complex64 IQ；
    2. run_header_first_demod.py 导出的 *_header_first_symbols.csv。

方法：
    对 header_valid=1 的 payload symbol，按 symbol CSV 里已经校正过的
    start_sample、CFO_int/CFO_frac、os_factor 重新计算 chip-rate FFT 全谱。
    然后不仅取 argmax selected bin，还人为取 selected+offset 和 fixed bin。

输出：
    1. 每个 packet 一张 selected vs wrong bins 四联图；
    2. 每个 packet/candidate 的 phase smoothness、幅度、能量占比、rank 统计；
    3. 每个 symbol/candidate 的细粒度特征 CSV。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[1]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_downchirp, dechirp_fft, positive_mod, signed_fft_bin  # noqa: E402
from weak_decoder.preamble_detector import load_complex64_file  # noqa: E402


@dataclass(frozen=True)
class CandidateSpec:
    """一个要观察的 FFT bin 选择策略。"""

    label: str
    kind: str
    value: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute corrected payload FFT spectra and compare selected bins "
            "with intentionally wrong bins."
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="原始 complex64 IQ bin 文件。")
    parser.add_argument(
        "-s",
        "--symbols-csv",
        type=Path,
        required=True,
        help="run_header_first_demod.py 输出的 *_header_first_symbols.csv。",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认 data/payload_wrong_bin_diagnostics。",
    )
    parser.add_argument("--packet", type=int, action="append", default=None, help="只处理指定 packet_index，可重复。")
    parser.add_argument("--max-packets", type=int, default=None, help="最多处理前 N 个 packet。")
    parser.add_argument(
        "--offset-bins",
        type=str,
        default="1,-1,4,-4,16,64,256",
        help="相对 selected bin 的错误 bin offset 列表，逗号分隔。",
    )
    parser.add_argument(
        "--fixed-bins",
        type=str,
        default="",
        help="固定 raw FFT bin 列表，逗号分隔。为空时使用 fixed-fractions。",
    )
    parser.add_argument(
        "--fixed-fractions",
        type=str,
        default="0,0.25,0.5,0.75",
        help="固定 bin 的 FFT 长度比例，逗号分隔；仅 fixed-bins 为空时生效。",
    )
    parser.add_argument(
        "--plot-wrong-count",
        type=int,
        default=5,
        help="每张图除 selected 外，绘制 R2 最高的前 N 条 wrong-bin 曲线。",
    )
    parser.add_argument("--dpi", type=int, default=180, help="输出 PNG 的 DPI。")
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    if value == "":
        return int(default)
    return int(float(value))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "")).strip()
    if value == "":
        return float(default)
    return float(value)


def _parse_int_list(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(float(item.strip())) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    if not text.strip():
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _finite_median(values: Iterable[float]) -> float:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def load_payload_rows(path: Path, packet_filter: set[int] | None) -> list[dict[str, object]]:
    """读取 corrected header-first CSV 中 header 有效的 payload symbol 行。"""

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
            rows.append(
                {
                    "frame_index": _int(raw, "frame_index", -1),
                    "packet_index": packet_index,
                    "event_index": _int(raw, "event_index", -1),
                    "payload_symbol_index": _int(raw, "stage_symbol_index", -1),
                    "start_sample": _int(raw, "start_sample", 0),
                    "sf": _int(raw, "sf", 10),
                    "os_factor": _int(raw, "os_factor", 1),
                    "cfo_int": _int(raw, "cfo_int", 0),
                    "cfo_frac": _float(raw, "cfo_frac", 0.0),
                    "sto_frac": _float(raw, "sto_frac", 0.0),
                    "sfo_hat": _float(raw, "sfo_hat", 0.0),
                    "csv_raw_fft_bin": _int(raw, "raw_fft_bin", -1),
                    "csv_peak_amp": _float(raw, "peak_amp", float("nan")),
                    "csv_peak_phase": _float(raw, "peak_phase", float("nan")),
                }
            )
    rows.sort(key=lambda item: (int(item["packet_index"]), int(item["payload_symbol_index"])))
    return rows


def group_payload_rows(
    rows: Iterable[dict[str, object]],
    max_packets: int | None,
) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["packet_index"]), []).append(row)
    grouped = dict(sorted(grouped.items()))
    if max_packets is not None:
        grouped = dict(list(grouped.items())[: int(max_packets)])
    return grouped


def symbol_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    """与 header_first_demod.demod_one_symbol 相同的 chip 中心采样口径。"""

    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    return int(start_sample) + int(os_value / 2) + os_value * np.arange(n_bins, dtype=np.int64)


def corrected_spectrum(samples: np.ndarray, row: dict[str, object]) -> np.ndarray:
    """重算一个 payload symbol 的 corrected chip-rate FFT 全谱。"""

    sf = int(row["sf"])
    indexes = symbol_indexes(
        start_sample=int(row["start_sample"]),
        sf=sf,
        os_factor=int(row["os_factor"]),
    )
    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(
            f"packet={row['packet_index']} payload_symbol={row['payload_symbol_index']} "
            "exceeds input sample range."
        )
    symbol = np.asarray(samples[indexes], dtype=np.complex64)
    downchirp = build_downchirp(sf, cfo_int=int(row["cfo_int"]), cfo_frac=float(row["cfo_frac"]))
    return dechirp_fft(symbol, downchirp)


def build_candidate_specs(n_bins: int, offset_bins: str, fixed_bins: str, fixed_fractions: str) -> list[CandidateSpec]:
    """生成 selected baseline + 多组 wrong-bin 候选。"""

    specs = [CandidateSpec(label="selected", kind="selected", value=0)]
    seen = {("selected", 0)}

    for offset in _parse_int_list(offset_bins):
        if positive_mod(offset, n_bins) == 0:
            continue
        key = ("offset", positive_mod(offset, n_bins))
        if key in seen:
            continue
        seen.add(key)
        sign = "+" if offset > 0 else ""
        specs.append(CandidateSpec(label=f"off{sign}{offset}", kind="offset", value=int(offset)))

    fixed_values = _parse_int_list(fixed_bins)
    if not fixed_values:
        fixed_values = [int(round(frac * n_bins)) for frac in _parse_float_list(fixed_fractions)]
    for fixed in fixed_values:
        fixed = positive_mod(fixed, n_bins)
        key = ("fixed", fixed)
        if key in seen:
            continue
        seen.add(key)
        specs.append(CandidateSpec(label=f"fix{fixed:04d}", kind="fixed", value=int(fixed)))
    return specs


def target_bin_for_spec(spec: CandidateSpec, selected_bin: int, n_bins: int) -> int:
    if spec.kind == "selected":
        return int(selected_bin)
    if spec.kind == "offset":
        return positive_mod(int(selected_bin) + int(spec.value), n_bins)
    if spec.kind == "fixed":
        return positive_mod(int(spec.value), n_bins)
    raise ValueError(f"unknown candidate kind: {spec.kind}")


def compute_feature_rows(
    samples: np.ndarray,
    grouped_rows: dict[int, list[dict[str, object]]],
    specs_by_bins: dict[int, list[CandidateSpec]],
) -> list[dict[str, object]]:
    """为每个 packet/symbol/candidate 计算 wrong-bin 特征。"""

    feature_rows: list[dict[str, object]] = []
    for packet_index, packet_rows in grouped_rows.items():
        for row in packet_rows:
            spectrum = corrected_spectrum(samples, row)
            n_bins = int(spectrum.size)
            power = np.abs(spectrum) ** 2
            total_power = float(np.sum(power, dtype=np.float64))
            selected_bin = int(np.argmax(power))
            selected_peak = complex(spectrum[selected_bin])
            selected_power = float(power[selected_bin])
            selected_amp = float(abs(selected_peak))
            specs = specs_by_bins[n_bins]

            for spec in specs:
                target_bin = target_bin_for_spec(spec, selected_bin=selected_bin, n_bins=n_bins)
                target_peak = complex(spectrum[target_bin])
                target_power = float(power[target_bin])
                target_amp = float(abs(target_peak))
                target_rank = int(1 + np.sum(power > target_power))
                is_selected = int(target_bin == selected_bin)
                feature_rows.append(
                    {
                        "packet_index": int(packet_index),
                        "event_index": int(row["event_index"]),
                        "frame_index": int(row["frame_index"]),
                        "payload_symbol_index": int(row["payload_symbol_index"]),
                        "candidate_label": spec.label,
                        "candidate_kind": spec.kind,
                        "candidate_value": int(spec.value),
                        "mode": "corrected_wrong_bin",
                        "sf": int(row["sf"]),
                        "os_factor": int(row["os_factor"]),
                        "cfo_int": int(row["cfo_int"]),
                        "cfo_frac": float(row["cfo_frac"]),
                        "sto_frac": float(row["sto_frac"]),
                        "sfo_hat": float(row["sfo_hat"]),
                        "csv_raw_fft_bin": int(row["csv_raw_fft_bin"]),
                        "selected_raw_fft_bin": int(selected_bin),
                        "selected_signed_fft_bin": signed_fft_bin(selected_bin, n_bins),
                        "selected_amp": selected_amp,
                        "selected_power": selected_power,
                        "selected_energy_ratio": float(selected_power / total_power) if total_power > 0.0 else float("nan"),
                        "target_raw_fft_bin": int(target_bin),
                        "target_signed_fft_bin": signed_fft_bin(target_bin, n_bins),
                        "target_is_selected_bin": is_selected,
                        "target_real": float(target_peak.real),
                        "target_imag": float(target_peak.imag),
                        "target_amp": target_amp,
                        "target_power": target_power,
                        "target_phase": float(math.atan2(target_peak.imag, target_peak.real)),
                        "target_energy_ratio": float(target_power / total_power) if total_power > 0.0 else float("nan"),
                        "target_bin_rank": target_rank,
                        "amp_vs_selected_db": float(20.0 * math.log10((target_amp + 1e-30) / (selected_amp + 1e-30))),
                        "power_vs_selected_db": float(10.0 * math.log10((target_power + 1e-30) / (selected_power + 1e-30))),
                        "total_fft_energy": total_power,
                        "selected_csv_bin_match": int(selected_bin == int(row["csv_raw_fft_bin"])),
                    }
                )
    add_unwrap_and_residual(feature_rows)
    return feature_rows


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    rmse = float(math.sqrt(np.mean((y - pred) ** 2)))
    return coef, pred, r2, rmse


def add_unwrap_and_residual(rows: list[dict[str, object]]) -> None:
    """给每个 packet/candidate 补充 unwrap phase、linear fit 和 residual。"""

    groups: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((int(row["packet_index"]), str(row["candidate_label"])), []).append(row)

    for items in groups.values():
        items.sort(key=lambda item: int(item["payload_symbol_index"]))
        k = np.asarray([int(item["payload_symbol_index"]) for item in items], dtype=np.float64)
        phase = np.asarray([float(item["target_phase"]) for item in items], dtype=np.float64)
        unwrap = np.unwrap(phase)
        coef, fit, r2, rmse = linear_fit(k, unwrap)
        residual = np.angle(np.exp(1j * (unwrap - fit)))
        for index, item in enumerate(items):
            item["target_phase_unwrap"] = float(unwrap[index])
            item["phase_linear_fit"] = float(fit[index])
            item["phase_residual"] = float(residual[index])
            item["group_phase_slope_pi_per_symbol"] = float(coef[0] / math.pi)
            item["group_linear_fit_r2"] = float(r2)
            item["group_linear_fit_rmse_pi"] = float(rmse / math.pi)


def summary_rows(feature_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """按 packet/candidate 汇总相位平滑性、幅度、能量占比和 rank。"""

    groups: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in feature_rows:
        groups.setdefault((int(row["packet_index"]), str(row["candidate_label"])), []).append(row)

    summaries: list[dict[str, object]] = []
    for (packet_index, label), items in sorted(groups.items()):
        items.sort(key=lambda item: int(item["payload_symbol_index"]))
        first = items[0]
        amps = np.asarray([float(item["target_amp"]) for item in items], dtype=np.float64)
        energy = np.asarray([float(item["target_energy_ratio"]) for item in items], dtype=np.float64)
        ranks = np.asarray([float(item["target_bin_rank"]) for item in items], dtype=np.float64)
        residual = np.asarray([float(item["phase_residual"]) for item in items], dtype=np.float64)
        amp_db = np.asarray([float(item["amp_vs_selected_db"]) for item in items], dtype=np.float64)
        power_db = np.asarray([float(item["power_vs_selected_db"]) for item in items], dtype=np.float64)
        target_hits = int(sum(int(item["target_is_selected_bin"]) for item in items))
        summaries.append(
            {
                "packet_index": int(packet_index),
                "event_index": int(first["event_index"]),
                "frame_index": int(first["frame_index"]),
                "candidate_label": label,
                "candidate_kind": first["candidate_kind"],
                "candidate_value": int(first["candidate_value"]),
                "is_selected_baseline": int(label == "selected"),
                "payload_symbol_count": len(items),
                "target_selected_hit_count": target_hits,
                "target_selected_hit_rate": float(target_hits / len(items)),
                "csv_selected_mismatch_count": int(sum(1 - int(item["selected_csv_bin_match"]) for item in items)),
                "phase_slope_pi_per_symbol": float(first["group_phase_slope_pi_per_symbol"]),
                "linear_fit_r2": float(first["group_linear_fit_r2"]),
                "linear_fit_rmse_pi": float(first["group_linear_fit_rmse_pi"]),
                "residual_std_pi": float(np.std(residual) / math.pi),
                "residual_peak_to_peak_pi": float((np.max(residual) - np.min(residual)) / math.pi),
                "target_amp_mean": _finite_mean(amps),
                "target_amp_std": float(np.std(amps)),
                "target_amp_cv": float(np.std(amps) / np.mean(amps)) if np.mean(amps) > 0.0 else float("nan"),
                "target_energy_ratio_mean": _finite_mean(energy),
                "target_energy_ratio_median": _finite_median(energy),
                "target_bin_rank_mean": _finite_mean(ranks),
                "target_bin_rank_median": _finite_median(ranks),
                "target_bin_rank_min": int(np.min(ranks)),
                "target_bin_rank_max": int(np.max(ranks)),
                "amp_vs_selected_db_mean": _finite_mean(amp_db),
                "power_vs_selected_db_mean": _finite_mean(power_db),
                "selected_energy_ratio_mean": _finite_mean([float(item["selected_energy_ratio"]) for item in items]),
                "selected_amp_mean": _finite_mean([float(item["selected_amp"]) for item in items]),
                "cfo_frac": float(first["cfo_frac"]),
                "sto_frac": float(first["sto_frac"]),
                "sfo_hat": float(first["sfo_hat"]),
            }
        )
    return summaries


def choose_plot_labels(summaries: list[dict[str, object]], packet_index: int, plot_wrong_count: int) -> list[str]:
    """每包绘制 selected + R2 最高的若干 wrong-bin，直接暴露最危险的反例。"""

    packet_items = [item for item in summaries if int(item["packet_index"]) == int(packet_index)]
    selected = [item for item in packet_items if int(item["is_selected_baseline"]) == 1]
    wrong = [item for item in packet_items if int(item["is_selected_baseline"]) == 0]
    wrong.sort(
        key=lambda item: (
            float(item["linear_fit_r2"]),
            float(item["target_energy_ratio_mean"]),
            -float(item["target_bin_rank_median"]),
        ),
        reverse=True,
    )
    labels = [str(item["candidate_label"]) for item in selected]
    labels.extend(str(item["candidate_label"]) for item in wrong[: int(plot_wrong_count)])
    return labels


def plot_packet_compare(
    packet_index: int,
    feature_rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    labels: list[str],
    out_path: Path,
    dpi: int,
) -> None:
    """绘制 selected vs wrong-bin 的四联图。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_by_label: dict[str, list[dict[str, object]]] = {}
    for row in feature_rows:
        if int(row["packet_index"]) != int(packet_index):
            continue
        label = str(row["candidate_label"])
        if label in labels:
            rows_by_label.setdefault(label, []).append(row)
    for rows in rows_by_label.values():
        rows.sort(key=lambda item: int(item["payload_symbol_index"]))

    summary_by_label = {
        str(item["candidate_label"]): item
        for item in summaries
        if int(item["packet_index"]) == int(packet_index)
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), dpi=int(dpi))
    axes = axes.ravel()
    cmap = plt.get_cmap("tab10")

    for idx, label in enumerate(labels):
        rows = rows_by_label.get(label, [])
        if not rows:
            continue
        summary = summary_by_label[label]
        k = np.asarray([int(item["payload_symbol_index"]) for item in rows], dtype=np.float64)
        phase = np.asarray([float(item["target_phase"]) for item in rows], dtype=np.float64)
        unwrap = np.asarray([float(item["target_phase_unwrap"]) for item in rows], dtype=np.float64)
        amp = np.asarray([float(item["target_amp"]) for item in rows], dtype=np.float64)
        energy = np.asarray([float(item["target_energy_ratio"]) for item in rows], dtype=np.float64)
        rank_median = float(summary["target_bin_rank_median"])
        r2 = float(summary["linear_fit_r2"])
        er_mean = float(summary["target_energy_ratio_mean"])
        is_selected = int(summary["is_selected_baseline"]) == 1
        color = "black" if is_selected else cmap((idx - 1) % 10)
        linewidth = 2.3 if is_selected else 1.15
        alpha = 0.95 if is_selected else 0.72
        legend = f"{label} R2={r2:.3f} ER={er_mean:.2e} rank~{rank_median:.0f}"

        axes[0].plot(k, phase / math.pi, "o-", markersize=2.8, linewidth=linewidth, color=color, alpha=alpha, label=legend)
        axes[1].plot(k, unwrap / math.pi, "o-", markersize=2.8, linewidth=linewidth, color=color, alpha=alpha, label=legend)
        axes[2].plot(k, np.maximum(amp, 1e-12), "o-", markersize=2.8, linewidth=linewidth, color=color, alpha=alpha, label=legend)
        axes[3].plot(k, np.maximum(energy, 1e-18), "o-", markersize=2.8, linewidth=linewidth, color=color, alpha=alpha, label=legend)

    axes[0].set_title("wrapped phase / pi")
    axes[0].set_ylabel("phase / pi")
    axes[0].set_ylim(-1.05, 1.05)

    axes[1].set_title("unwrap phase / pi")

    axes[2].set_title("amplitude, log scale")
    axes[2].set_xlabel("payload symbol index")
    axes[2].set_ylabel("amplitude")
    axes[2].set_yscale("log")

    axes[3].set_title("energy ratio, log scale")
    axes[3].set_xlabel("payload symbol index")
    axes[3].set_ylabel("target_power / total_fft_energy")
    axes[3].set_yscale("log")

    for axis in axes:
        axis.grid(True, color="#dddddd", linewidth=0.6)

    axes[1].legend(loc="best", fontsize=7)
    first_summary = next(item for item in summaries if int(item["packet_index"]) == int(packet_index))
    fig.suptitle(
        f"Packet {packet_index} corrected FFT wrong-bin control "
        f"| event={first_summary['event_index']} | plotted wrong bins=highest R2",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


FEATURE_FIELDS = [
    "packet_index",
    "event_index",
    "frame_index",
    "payload_symbol_index",
    "candidate_label",
    "candidate_kind",
    "candidate_value",
    "mode",
    "sf",
    "os_factor",
    "cfo_int",
    "cfo_frac",
    "sto_frac",
    "sfo_hat",
    "csv_raw_fft_bin",
    "selected_raw_fft_bin",
    "selected_signed_fft_bin",
    "selected_amp",
    "selected_power",
    "selected_energy_ratio",
    "target_raw_fft_bin",
    "target_signed_fft_bin",
    "target_is_selected_bin",
    "target_real",
    "target_imag",
    "target_amp",
    "target_power",
    "target_phase",
    "target_phase_unwrap",
    "phase_linear_fit",
    "phase_residual",
    "target_energy_ratio",
    "target_bin_rank",
    "amp_vs_selected_db",
    "power_vs_selected_db",
    "total_fft_energy",
    "selected_csv_bin_match",
    "group_phase_slope_pi_per_symbol",
    "group_linear_fit_r2",
    "group_linear_fit_rmse_pi",
]


SUMMARY_FIELDS = [
    "packet_index",
    "event_index",
    "frame_index",
    "candidate_label",
    "candidate_kind",
    "candidate_value",
    "is_selected_baseline",
    "payload_symbol_count",
    "target_selected_hit_count",
    "target_selected_hit_rate",
    "csv_selected_mismatch_count",
    "phase_slope_pi_per_symbol",
    "linear_fit_r2",
    "linear_fit_rmse_pi",
    "residual_std_pi",
    "residual_peak_to_peak_pi",
    "target_amp_mean",
    "target_amp_std",
    "target_amp_cv",
    "target_energy_ratio_mean",
    "target_energy_ratio_median",
    "target_bin_rank_mean",
    "target_bin_rank_median",
    "target_bin_rank_min",
    "target_bin_rank_max",
    "amp_vs_selected_db_mean",
    "power_vs_selected_db_mean",
    "selected_energy_ratio_mean",
    "selected_amp_mean",
    "cfo_frac",
    "sto_frac",
    "sfo_hat",
]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (WEAK_ROOT / "data" / "payload_wrong_bin_diagnostics")
    plot_dir = output_dir / "plots"
    basename = args.symbols_csv.stem.replace("_header_first_symbols", "")
    feature_output = output_dir / f"{basename}_wrong_bin_symbol_features.csv"
    summary_output = output_dir / f"{basename}_wrong_bin_summary.csv"

    packet_filter = set(args.packet) if args.packet is not None else None
    payload_rows = load_payload_rows(args.symbols_csv, packet_filter=packet_filter)
    grouped = group_payload_rows(payload_rows, max_packets=args.max_packets)
    if not grouped:
        raise ValueError("没有找到 header_valid=1 的 payload symbol 行。")

    samples = load_complex64_file(args.input)
    n_bins_values = sorted({1 << int(row["sf"]) for rows in grouped.values() for row in rows})
    specs_by_bins = {
        n_bins: build_candidate_specs(
            n_bins=n_bins,
            offset_bins=args.offset_bins,
            fixed_bins=args.fixed_bins,
            fixed_fractions=args.fixed_fractions,
        )
        for n_bins in n_bins_values
    }

    feature_rows = compute_feature_rows(samples, grouped, specs_by_bins=specs_by_bins)
    summaries = summary_rows(feature_rows)

    write_csv(feature_output, feature_rows, FEATURE_FIELDS)
    write_csv(summary_output, summaries, SUMMARY_FIELDS)

    png_count = 0
    for packet_index in grouped:
        labels = choose_plot_labels(summaries, packet_index=packet_index, plot_wrong_count=args.plot_wrong_count)
        out_path = plot_dir / f"packet_{packet_index:03d}_wrong_bin_phase_amplitude_compare.png"
        plot_packet_compare(
            packet_index=packet_index,
            feature_rows=feature_rows,
            summaries=summaries,
            labels=labels,
            out_path=out_path,
            dpi=args.dpi,
        )
        png_count += 1

    print(f"packets={len(grouped)}")
    print(f"payload_rows={len(payload_rows)}")
    print(f"candidate_specs={sum(len(specs) for specs in specs_by_bins.values())}")
    print(f"feature_rows={len(feature_rows)}")
    print(f"summary={summary_output}")
    print(f"features={feature_output}")
    print(f"plots={plot_dir}")
    print(f"wrote_pngs={png_count}")


if __name__ == "__main__":
    main()
