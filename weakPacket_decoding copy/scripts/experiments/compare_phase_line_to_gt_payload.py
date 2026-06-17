#!/usr/bin/env python3
"""对比 selector/header 拟合相位线与 GT-bin payload 真实相位线。

这是一个只用于诊断的实验脚本，不参与主解码链。
脚本复用现有低 SNR AWGN 构造和 symbol-level selector，再从已知 GT raw FFT bin
强行读取 payload 相位，拟合 payload-native phase line。CRC 不参与任何选择或搜索。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_low_snr_gt_bin_experiment import (  # noqa: E402
    estimate_payload_reference_power,
    load_gt_payload_symbols,
)
from run_symbol_phase_two_stage import (  # noqa: E402
    _argmax_bins,
    _extract_payload_spectra_with_coherence,
    _packet_phase_line,
    _ser,
    load_packets,
)
from weak_decoder.phase_guided_demod import PhaseLine, fit_phase_line  # noqa: E402
from weak_decoder.symbol_phase_two_stage import SymbolPhaseConfig, select_symbol_bins_two_stage  # noqa: E402


DEFAULT_DATASETS = ("0_0_0_10_14_8", "0_0_0_10_14_16", "0_0_0_10_14_32")
DEFAULT_SNRS = (-22.0, -23.0, -24.0, -25.0, -26.0, -27.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比 selector/header 拟合出的 phase line 与 GT payload phase line。"
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="要分析的数据集 stem。")
    parser.add_argument("--snr-db", type=float, nargs="+", default=list(DEFAULT_SNRS), help="要构造的目标 SNR 列表。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "phase_line_gt_compare",
        help="输出逐包比较 CSV 和按 SNR 汇总 CSV 的目录。",
    )
    parser.add_argument("--seed", type=int, default=20260531, help="AWGN 随机种子。")
    parser.add_argument("--independent-noise", action="store_true", help="每个 SNR 使用独立噪声 realization。")
    parser.add_argument(
        "--cfo-correction-mode",
        choices=("symbol", "continuous"),
        default="continuous",
        help="payload FFT 前的 CFO 相位补偿口径。",
    )
    parser.add_argument("--preamble-len", type=float, default=8.0, help="默认 preamble symbol 数。")
    parser.add_argument("--ldro-mode", type=int, default=2, help="payload codec 使用的 LDRO 模式。")
    parser.add_argument("--packet", type=int, action="append", default=None, help="只分析指定 packet_index，可重复传入。")
    return parser.parse_args()


def dataset_paths(dataset: str) -> dict[str, Path]:
    """根据数据集 stem 组织 IQ 和 header-first symbol CSV 路径。"""

    return {
        "iq": WEAK_ROOT.parent / "data" / "USRP_IQ" / f"{dataset}.bin",
        "symbols": WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / f"{dataset}_header_first_symbols.csv",
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的行: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def add_awgn_from_unit(samples: np.ndarray, unit_noise: np.ndarray, noise_power: float) -> np.ndarray:
    """用预生成的单位复高斯噪声构造指定噪声功率的 noisy IQ。"""

    sigma = math.sqrt(float(noise_power) / 2.0)
    return (samples + sigma * unit_noise).astype(np.complex64, copy=False)


def make_unit_noise(size: int, seed: int) -> np.ndarray:
    """生成单位方差复高斯噪声，便于多个 SNR 复用同一 realization。"""

    rng = np.random.default_rng(int(seed))
    noise_i = rng.normal(0.0, 1.0, size=int(size)).astype(np.float32)
    noise_q = rng.normal(0.0, 1.0, size=int(size)).astype(np.float32)
    return (noise_i + 1j * noise_q).astype(np.complex64)


def circular_residuals(phases: Sequence[float], line: PhaseLine, abs_indices: Sequence[float]) -> np.ndarray:
    """计算相位点到某条 phase line 的 circular residual。"""

    if int(line.anchor_count) < 2:
        return np.asarray([], dtype=np.float64)
    pred = np.asarray([line.predict(float(x)) for x in abs_indices], dtype=np.float64)
    phase = np.asarray(phases, dtype=np.float64)
    return np.angle(np.exp(1j * (phase - pred)))


def fit_gt_payload_line(
    center_spectra: Sequence[np.ndarray],
    gt_bins: Sequence[int],
    abs_indices: Sequence[float],
) -> tuple[PhaseLine, np.ndarray, np.ndarray]:
    """从已知 GT raw FFT bin 的相位拟合 payload-native phase line。"""

    phases: list[float] = []
    xs: list[float] = []
    for spec, gt_bin, abs_index in zip(center_spectra, gt_bins, abs_indices):
        b = int(gt_bin)
        if b < 0 or b >= len(spec):
            continue
        phases.append(float(np.angle(spec[b])))
        xs.append(float(abs_index))
    if len(phases) < 2:
        return PhaseLine(), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    unwrapped = np.unwrap(np.asarray(phases, dtype=np.float64))
    x = np.asarray(xs, dtype=np.float64)
    return fit_phase_line(x, unwrapped), x, unwrapped


def residual_metrics(prefix: str, residual: np.ndarray) -> dict[str, Any]:
    """把 residual 序列压成均值、标准差和峰峰值，单位统一归一化到 pi。"""

    if residual.size == 0:
        return {
            f"{prefix}_resid_mean_abs_pi": "",
            f"{prefix}_resid_std_pi": "",
            f"{prefix}_resid_ptp_pi": "",
        }
    return {
        f"{prefix}_resid_mean_abs_pi": float(np.mean(np.abs(residual)) / math.pi),
        f"{prefix}_resid_std_pi": float(np.std(residual) / math.pi),
        f"{prefix}_resid_ptp_pi": float((np.max(residual) - np.min(residual)) / math.pi),
    }


def line_fields(prefix: str, line: PhaseLine) -> dict[str, Any]:
    """导出一条 phase line 的基本拟合质量字段。"""

    return {
        f"{prefix}_anchors": int(line.anchor_count),
        f"{prefix}_slope_pi": float(line.slope_pi),
        f"{prefix}_r2": float(line.fit_r2) if math.isfinite(float(line.fit_r2)) else "",
        f"{prefix}_rmse_pi": float(line.fit_rmse_pi) if math.isfinite(float(line.fit_rmse_pi)) else "",
    }


def compare_packet(
    dataset: str,
    snr_db: float,
    packet: dict[str, Any],
    clean_samples: np.ndarray,
    noisy_samples: np.ndarray,
    args: argparse.Namespace,
    config: SymbolPhaseConfig,
) -> dict[str, Any] | None:
    """对单个 packet 做 selector line、header line 和 GT payload line 对比。"""

    center, multi, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
        noisy_samples, packet, args
    )
    clean_center, _clean_multi, clean_abs_indices, clean_gt_bins, _clean_coh = _extract_payload_spectra_with_coherence(
        clean_samples, packet, args
    )
    if not center or len(center) != len(gt_bins):
        return None

    evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi]
    header_line = _packet_phase_line(noisy_samples, packet, args)
    clean_header_line = _packet_phase_line(clean_samples, packet, args)
    result = select_symbol_bins_two_stage(
        center_spectra=center,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=config,
        fallback_line=header_line,
        offset_coherences=coherences,
    )

    gt_line, gt_x, gt_phase = fit_gt_payload_line(center, gt_bins, abs_indices)
    clean_gt_line, clean_gt_x, clean_gt_phase = fit_gt_payload_line(clean_center, clean_gt_bins, clean_abs_indices)
    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    center_bins = _argmax_bins(center)
    multi_bins = _argmax_bins(multi)
    selected_bins = tuple(int(v) for v in result.selected_raw_bins)
    _center_raw_ser, center_symbol_ser, _ = _ser(center_bins, gt_bins, sf=sf, ldro=ldro)
    _multi_raw_ser, multi_symbol_ser, _ = _ser(multi_bins, gt_bins, sf=sf, ldro=ldro)
    _selected_raw_ser, selected_symbol_ser, compared = _ser(selected_bins, gt_bins, sf=sf, ldro=ldro)

    selector_resid = circular_residuals(gt_phase, result.phase_line, gt_x)
    header_resid = circular_residuals(gt_phase, header_line, gt_x)
    clean_header_resid = circular_residuals(clean_gt_phase, clean_header_line, clean_gt_x)
    clean_selector_resid = circular_residuals(clean_gt_phase, result.phase_line, clean_gt_x)

    row: dict[str, Any] = {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "symbol_count": int(len(gt_bins)),
        "gt_compared_symbols": int(compared),
        "center_symbol_ser": float(center_symbol_ser),
        "multi_symbol_ser": float(multi_symbol_ser),
        "selected_symbol_ser": float(selected_symbol_ser),
        "locked_count": int(result.locked_count),
        "locked_ratio": float(result.locked_count / max(1, len(gt_bins))),
        "false_locked_gt_bins": int(
            sum(
                1
                for idx, locked in enumerate(result.locked_mask)
                if locked and idx < len(gt_bins) and idx < len(selected_bins) and int(selected_bins[idx]) != int(gt_bins[idx])
            )
        ),
    }
    row.update(line_fields("selector_line", result.phase_line))
    row.update(line_fields("header_line", header_line))
    row.update(line_fields("clean_header_line", clean_header_line))
    row.update(line_fields("gt_noisy_payload_line", gt_line))
    row.update(line_fields("gt_clean_payload_line", clean_gt_line))
    row.update(residual_metrics("selector_vs_gt_noisy", selector_resid))
    row.update(residual_metrics("header_vs_gt_noisy", header_resid))
    row.update(residual_metrics("selector_vs_gt_clean", clean_selector_resid))
    row.update(residual_metrics("clean_header_vs_gt_clean", clean_header_resid))
    row["selector_slope_minus_gt_noisy_pi"] = (
        float(result.phase_line.slope_pi - gt_line.slope_pi) if int(gt_line.anchor_count) >= 2 else ""
    )
    row["header_slope_minus_gt_noisy_pi"] = (
        float(header_line.slope_pi - gt_line.slope_pi)
        if int(header_line.anchor_count) >= 2 and int(gt_line.anchor_count) >= 2
        else ""
    )
    row["gt_noisy_slope_minus_clean_pi"] = (
        float(gt_line.slope_pi - clean_gt_line.slope_pi)
        if int(gt_line.anchor_count) >= 2 and int(clean_gt_line.anchor_count) >= 2
        else ""
    )
    return row


def avg(rows: Sequence[dict[str, Any]], key: str) -> float | str:
    """计算某个字段的有限数值平均；没有有效值时返回空字符串。"""

    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else ""


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 dataset/SNR 汇总逐包比较结果。"""

    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), float(row["target_snr_db"]))
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    metric_keys = (
        "center_symbol_ser",
        "multi_symbol_ser",
        "selected_symbol_ser",
        "locked_ratio",
        "false_locked_gt_bins",
        "selector_line_r2",
        "selector_line_rmse_pi",
        "header_line_r2",
        "header_line_rmse_pi",
        "gt_noisy_payload_line_r2",
        "gt_noisy_payload_line_rmse_pi",
        "gt_clean_payload_line_r2",
        "gt_clean_payload_line_rmse_pi",
        "selector_vs_gt_noisy_resid_mean_abs_pi",
        "selector_vs_gt_noisy_resid_std_pi",
        "selector_vs_gt_noisy_resid_ptp_pi",
        "header_vs_gt_noisy_resid_mean_abs_pi",
        "header_vs_gt_noisy_resid_std_pi",
        "selector_slope_minus_gt_noisy_pi",
        "header_slope_minus_gt_noisy_pi",
        "gt_noisy_slope_minus_clean_pi",
    )
    for (dataset, snr_db), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1]), reverse=False):
        summary: dict[str, Any] = {
            "dataset": dataset,
            "target_snr_db": float(snr_db),
            "packet_count": int(len(items)),
        }
        for key in metric_keys:
            summary[f"mean_{key}"] = avg(items, key)
        out.append(summary)
    return out


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = SymbolPhaseConfig()
    packet_filter = set(int(v) for v in args.packet) if args.packet else None

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        paths = dataset_paths(str(dataset))
        samples = np.fromfile(paths["iq"], dtype=np.complex64)
        if samples.size == 0:
            raise ValueError(f"空 IQ 文件: {paths['iq']}")
        gt_symbols = load_gt_payload_symbols(paths["symbols"], packet_filter=None)
        signal_power = estimate_payload_reference_power(samples, gt_symbols)
        packet_map = load_packets(paths["symbols"], None)
        packets = [
            packet
            for packet_index, packet in sorted(packet_map.items())
            if packet_filter is None or int(packet_index) in packet_filter
        ]
        if not packets:
            raise ValueError(f"没有读到可分析 packet: {dataset}: {paths['symbols']}")

        base_seed = int(args.seed)
        unit_noise = None
        if not args.independent_noise:
            unit_noise = make_unit_noise(samples.size, base_seed)

        for step_index, snr_db in enumerate(args.snr_db):
            noise_power = float(signal_power) * (10.0 ** (-float(snr_db) / 10.0))
            if unit_noise is None:
                unit_noise_snr = make_unit_noise(samples.size, base_seed + step_index)
            else:
                unit_noise_snr = unit_noise
            noisy_samples = add_awgn_from_unit(samples, unit_noise_snr, noise_power)
            count_before = len(rows)
            for packet in packets:
                row = compare_packet(str(dataset), float(snr_db), packet, samples, noisy_samples, args, config)
                if row is not None:
                    rows.append(row)
            print(
                f"{dataset} SNR {float(snr_db):.1f} dB: packets={len(rows) - count_before}, "
                f"noise_power={noise_power:.6e}",
                flush=True,
            )

    packet_csv = out_dir / "phase_line_gt_compare_packets.csv"
    summary_csv = out_dir / "phase_line_gt_compare_summary.csv"
    write_csv(packet_csv, rows)
    write_csv(summary_csv, summarize(rows))
    print(f"写入逐包结果={packet_csv}")
    print(f"写入汇总结果={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
