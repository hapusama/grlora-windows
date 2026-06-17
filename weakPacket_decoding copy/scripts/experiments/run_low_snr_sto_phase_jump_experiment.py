#!/usr/bin/env python3
"""低 SNR 下的 STO/SFO chirp phase-jump 补偿实验。

这个脚本复用 run_low_snr_gt_bin_experiment.py 生成的 noisy IQ，不重新做
weak detection / framesync / header decode。GT bin 仍然来自 clean
*_header_first_symbols.csv 中：

    stage == payload && header_valid == 1

的 raw_fft_bin。

实验差异只在 FFT 前多做一步：根据每个 payload symbol 的 residual STO
tau_s，找到 LoRa symbol 循环移位导致的 chirp wrap 位置，并对 wrap 之后
的 dechirped 片段补一个常相位。这样可以观察“去掉 STO 相位跳变亏损”后，
GT bin 的幅度、ER 和相位 residual 在极低 SNR 下是否更稳定。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_downchirp, positive_mod, signed_fft_bin  # noqa: E402


@dataclass(frozen=True)
class PayloadGt:
    frame_index: int
    packet_index: int
    event_index: int
    payload_symbol_index: int
    frame_symbol_index: int
    start_sample: int
    header_start_sample: int
    sf: int
    os_factor: int
    cfo_int: int
    cfo_frac: float
    sto_frac: float
    sfo_hat: float
    sfo_cum_before: float
    gt_raw_fft_bin: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 low-SNR noisy IQ，并在 corrected FFT 前额外补偿由 residual STO "
            "导致的 chirp wrap phase jump。"
        )
    )
    parser.add_argument(
        "-d",
        "--low-snr-dir",
        type=Path,
        required=True,
        help="包含 *_snr_mXXdB.bin 的目录，通常来自 run_low_snr_gt_bin_experiment.py。",
    )
    parser.add_argument(
        "-g",
        "--gt-symbol-csv",
        type=Path,
        required=True,
        help="clean header-first symbol CSV；payload 行的 raw_fft_bin 作为 GT bin。",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认 <low-snr-dir>/sto_phase_jump_corrected。",
    )
    parser.add_argument(
        "--input-stem",
        type=str,
        default=None,
        help="用于匹配 noisy bin 文件的 stem，默认从 GT CSV 文件名推断。",
    )
    parser.add_argument("--snr-db", type=float, nargs="*", default=None, help="可选 SNR 过滤。")
    parser.add_argument("--packet", type=int, action="append", default=None, help="可选 packet_index 过滤。")
    parser.add_argument(
        "--cfo-correction-mode",
        choices=("symbol", "continuous"),
        default="continuous",
        help="FFT 前的 CFO 补偿口径，默认 continuous。",
    )
    parser.add_argument(
        "--tau-source",
        choices=("sfo_cum_before", "sto_plus_sfo", "zero"),
        default="sfo_cum_before",
        help=(
            "residual STO tau_s 的来源。默认使用 header-first CSV 中的 sfo_cum_before；"
            "sto_plus_sfo 是备用近似；zero 用于 sanity check。"
        ),
    )
    parser.add_argument(
        "--phase-sign",
        choices=("plus", "minus"),
        default="plus",
        help=(
            "wrap 后片段相位补偿符号。plus 表示乘 exp(+j*2*pi*tau_s)，"
            "在当前 sfo_cum_before 符号定义下通常提升 GT-bin ER。"
        ),
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI。")
    parser.add_argument("--no-plots", action="store_true", default=False, help="Skip diagnostic PNG generation.")
    return parser.parse_args()


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


def _wrap_half(value: float) -> float:
    """把 fractional-chip offset 规约到 [-0.5, 0.5)。"""

    return float((float(value) + 0.5) % 1.0 - 0.5)


def _snr_label(snr_db: float) -> str:
    sign = "m" if float(snr_db) < 0 else "p"
    value = abs(float(snr_db))
    if abs(value - round(value)) < 1e-9:
        text = f"{int(round(value)):02d}"
    else:
        text = f"{value:.1f}".replace(".", "p")
    return f"snr_{sign}{text}dB"


def _infer_input_stem(gt_csv: Path) -> str:
    stem = gt_csv.stem
    for suffix in (
        "_header_first_symbols_continuous_cfo",
        "_header_first_symbols_framesync_valid_consistency",
        "_header_first_symbols",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _group_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (_int(row, "frame_index", -1), _int(row, "packet_index", -1), _int(row, "event_index", -1))


def load_payload_gt(path: Path, packet_filter: set[int] | None) -> list[PayloadGt]:
    """从 clean header-first CSV 中读取 payload GT bin 和同步估计量。"""

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    header_start_by_key: dict[tuple[int, int, int], int] = {}
    for row in rows:
        if row.get("stage") == "header" and _int(row, "stage_symbol_index", -1) == 0:
            header_start_by_key[_group_key(row)] = _int(row, "start_sample")

    symbols: list[PayloadGt] = []
    for row in rows:
        if row.get("stage") != "payload":
            continue
        if _int(row, "header_valid", 0) != 1:
            continue

        packet_index = _int(row, "packet_index", -1)
        if packet_filter is not None and packet_index not in packet_filter:
            continue

        sf = _int(row, "sf", 10)
        os_factor = _int(row, "os_factor", 4)
        frame_symbol_index = _int(row, "frame_symbol_index", -1)
        key = _group_key(row)
        header_start_sample = header_start_by_key.get(key)
        if header_start_sample is None:
            header_start_sample = _int(row, "start_sample") - frame_symbol_index * (1 << sf) * os_factor

        symbols.append(
            PayloadGt(
                frame_index=_int(row, "frame_index", -1),
                packet_index=packet_index,
                event_index=_int(row, "event_index", -1),
                payload_symbol_index=_int(row, "stage_symbol_index", -1),
                frame_symbol_index=frame_symbol_index,
                start_sample=_int(row, "start_sample"),
                header_start_sample=int(header_start_sample),
                sf=sf,
                os_factor=os_factor,
                cfo_int=_int(row, "cfo_int", 0),
                cfo_frac=_float(row, "cfo_frac", 0.0),
                sto_frac=_float(row, "sto_frac", 0.0),
                sfo_hat=_float(row, "sfo_hat", 0.0),
                sfo_cum_before=_float(row, "sfo_cum_before", _float(row, "sto_frac", 0.0)),
                gt_raw_fft_bin=_int(row, "raw_fft_bin", -1),
            )
        )

    symbols.sort(key=lambda item: (item.packet_index, item.payload_symbol_index))
    if not symbols:
        raise ValueError("No GT payload symbols found. Need stage=payload and header_valid=1.")
    return symbols


def find_noisy_bins(low_snr_dir: Path, input_stem: str, snr_filter: Iterable[float] | None) -> list[tuple[float, Path]]:
    pattern = re.compile(rf"^{re.escape(input_stem)}_snr_([mp])(\d+(?:p\d+)?)dB\.bin$")
    wanted = None if snr_filter is None else {round(float(value), 6) for value in snr_filter}
    found: list[tuple[float, Path]] = []
    for path in low_snr_dir.glob(f"{input_stem}_snr_*dB.bin"):
        match = pattern.match(path.name)
        if not match:
            continue
        sign, value_text = match.groups()
        value = float(value_text.replace("p", "."))
        snr_db = -value if sign == "m" else value
        if wanted is not None and round(float(snr_db), 6) not in wanted:
            continue
        found.append((float(snr_db), path))
    found.sort(key=lambda item: item[0], reverse=True)
    if not found:
        raise FileNotFoundError(f"No noisy IQ bins found under {low_snr_dir} for stem {input_stem}.")
    return found


def symbol_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    return int(start_sample) + int(os_value / 2) + os_value * np.arange(n_bins, dtype=np.int64)


def tau_for_symbol(symbol: PayloadGt, tau_source: str) -> float:
    """返回当前 symbol 的 residual STO，单位是 chip。"""

    if tau_source == "zero":
        return 0.0
    if tau_source == "sto_plus_sfo":
        return _wrap_half(float(symbol.sto_frac) + float(symbol.frame_symbol_index) * float(symbol.sfo_hat))
    return _wrap_half(float(symbol.sfo_cum_before))


def apply_phase_jump_compensation(
    dechirped: np.ndarray,
    gt_bin: int,
    tau_chips: float,
    phase_sign: str,
) -> tuple[np.ndarray, int, float]:
    """补偿 LoRa cyclic chirp 在 wrap 位置两侧由 residual STO 引入的常相位差。

    对 symbol_id=z 的 chirp，wrap 位置近似在 N - z + tau_s。当前工程中
    sfo_cum_before 的符号与用户推导中的 tau_s 相反，所以默认采用 plus：

        post-wrap *= exp(+j * 2*pi*tau_s)

    脚本保留 --phase-sign 方便做符号 sanity check。
    """

    n_bins = int(dechirped.size)
    z = positive_mod(int(gt_bin), n_bins)
    tau = _wrap_half(float(tau_chips))
    cut = int(math.ceil(float(n_bins - z) + tau))
    sign_value = 1.0 if str(phase_sign) == "plus" else -1.0
    correction_rad = float(sign_value * 2.0 * math.pi * tau)

    corrected = np.asarray(dechirped, dtype=np.complex64).copy()
    if 0 <= cut < n_bins:
        corrected[cut:] = (corrected[cut:] * np.exp(1j * correction_rad)).astype(np.complex64)
    return corrected, int(cut), correction_rad


def corrected_spectrum_with_sto_jump(
    samples: np.ndarray,
    symbol: PayloadGt,
    downchirps: dict[tuple[int, int, float], np.ndarray],
    cfo_correction_mode: str,
    tau_source: str,
    phase_sign: str,
) -> tuple[np.ndarray, float, float, int, float]:
    sf = int(symbol.sf)
    n_bins = 1 << sf
    indexes = symbol_indexes(symbol.start_sample, sf=symbol.sf, os_factor=symbol.os_factor)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(f"Symbol exceeds IQ range at start_sample={symbol.start_sample}.")

    chip_symbol = np.asarray(samples[indexes], dtype=np.complex64)
    cfo_total = float(symbol.cfo_int) + float(symbol.cfo_frac)
    if cfo_correction_mode == "continuous":
        relative_chip_start = float(symbol.start_sample - symbol.header_start_sample) / float(symbol.os_factor)
        cfo_common_phase_rad = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
        chip_symbol = (chip_symbol * np.exp(-1j * cfo_common_phase_rad)).astype(np.complex64)
    else:
        cfo_common_phase_rad = 0.0

    key = (sf, int(symbol.cfo_int), float(symbol.cfo_frac))
    downchirp = downchirps.get(key)
    if downchirp is None:
        downchirp = build_downchirp(symbol.sf, cfo_int=symbol.cfo_int, cfo_frac=symbol.cfo_frac)
        downchirps[key] = downchirp

    dechirped = np.asarray(chip_symbol * downchirp, dtype=np.complex64)
    tau_chips = tau_for_symbol(symbol, tau_source=tau_source)
    compensated, wrap_cut_chip, correction_rad = apply_phase_jump_compensation(
        dechirped,
        gt_bin=symbol.gt_raw_fft_bin,
        tau_chips=tau_chips,
        phase_sign=phase_sign,
    )
    return np.fft.fft(compensated).astype(np.complex64), cfo_common_phase_rad, tau_chips, wrap_cut_chip, correction_rad


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    rmse = float(math.sqrt(np.mean((y - pred) ** 2)))
    return coef, pred, r2, rmse


def quadratic_fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    coef = np.polyfit(x, y, deg=2)
    pred = np.polyval(coef, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")


def add_packet_phase_columns(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["target_snr_db"]), int(row["packet_index"]))].append(row)

    for packet_rows in grouped.values():
        packet_rows.sort(key=lambda item: int(item["payload_symbol_index"]))
        k = np.asarray([int(row["payload_symbol_index"]) for row in packet_rows], dtype=np.float64)
        phase = np.asarray([float(row["gt_bin_phase"]) for row in packet_rows], dtype=np.float64)
        phase_unwrap = np.unwrap(phase)
        fit_coef, fit_line, fit_r2, fit_rmse = linear_fit(k, phase_unwrap)
        residual = np.angle(np.exp(1j * (phase_unwrap - fit_line)))
        quad_r2 = quadratic_fit_r2(k, residual)
        for index, row in enumerate(packet_rows):
            row["gt_bin_phase_unwrap"] = float(phase_unwrap[index])
            row["phase_linear_fit"] = float(fit_line[index])
            row["phase_linear_residual"] = float(residual[index])
            row["packet_phase_slope_pi_per_symbol"] = float(fit_coef[0] / math.pi)
            row["packet_phase_linear_r2"] = float(fit_r2)
            row["packet_phase_linear_rmse_pi"] = float(fit_rmse / math.pi)
            row["packet_residual_quad_r2"] = float(quad_r2)


def compute_feature_rows(
    samples: np.ndarray,
    symbols: list[PayloadGt],
    target_snr_db: float,
    file_name: str,
    cfo_correction_mode: str,
    tau_source: str,
    phase_sign: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    downchirps: dict[tuple[int, int, float], np.ndarray] = {}
    for symbol in symbols:
        spectrum, cfo_common_phase_rad, tau_chips, wrap_cut_chip, correction_rad = corrected_spectrum_with_sto_jump(
            samples=samples,
            symbol=symbol,
            downchirps=downchirps,
            cfo_correction_mode=cfo_correction_mode,
            tau_source=tau_source,
            phase_sign=phase_sign,
        )
        n_bins = int(spectrum.size)
        gt_bin = positive_mod(symbol.gt_raw_fft_bin, n_bins)
        power = np.abs(spectrum) ** 2
        total_power = float(np.sum(power, dtype=np.float64))
        argmax_bin = int(np.argmax(power))
        gt_value = complex(spectrum[gt_bin])
        gt_power = float(power[gt_bin])
        gt_rank = int(np.where(np.argsort(power)[::-1] == gt_bin)[0][0] + 1)
        sorted_desc = np.sort(power)[::-1]
        second_power = float(sorted_desc[1]) if sorted_desc.size > 1 else 0.0

        rows.append(
            {
                "file_name": str(file_name),
                "target_snr_db": float(target_snr_db),
                "mode": "sto_phase_jump_corrected",
                "cfo_correction_mode": str(cfo_correction_mode),
                "tau_source": str(tau_source),
                "phase_sign": str(phase_sign),
                "frame_index": int(symbol.frame_index),
                "packet_index": int(symbol.packet_index),
                "event_index": int(symbol.event_index),
                "payload_symbol_index": int(symbol.payload_symbol_index),
                "frame_symbol_index": int(symbol.frame_symbol_index),
                "start_sample": int(symbol.start_sample),
                "header_start_sample": int(symbol.header_start_sample),
                "sf": int(symbol.sf),
                "os_factor": int(symbol.os_factor),
                "cfo_int": int(symbol.cfo_int),
                "cfo_frac": float(symbol.cfo_frac),
                "cfo_common_phase_rad": float(cfo_common_phase_rad),
                "sto_frac": float(symbol.sto_frac),
                "sfo_hat": float(symbol.sfo_hat),
                "sfo_cum_before": float(symbol.sfo_cum_before),
                "tau_chips": float(tau_chips),
                "wrap_cut_chip": int(wrap_cut_chip),
                "phase_jump_correction_rad": float(correction_rad),
                "phase_jump_correction_pi": float(correction_rad / math.pi),
                "gt_raw_fft_bin": int(gt_bin),
                "gt_signed_fft_bin": signed_fft_bin(gt_bin, n_bins),
                "argmax_bin": int(argmax_bin),
                "is_argmax_correct": int(argmax_bin == gt_bin),
                "gt_bin_rank": int(gt_rank),
                "gt_bin_real": float(gt_value.real),
                "gt_bin_imag": float(gt_value.imag),
                "gt_bin_amp": float(abs(gt_value)),
                "gt_bin_power": float(gt_power),
                "gt_bin_phase": float(math.atan2(gt_value.imag, gt_value.real)),
                "gt_peak_energy_ratio": float(gt_power / total_power) if total_power > 0.0 else float("nan"),
                "selected_peak_amp": float(abs(spectrum[argmax_bin])),
                "selected_peak_power": float(power[argmax_bin]),
                "selected_peak_margin_db": float(
                    10.0 * math.log10((float(power[argmax_bin]) + 1e-30) / (second_power + 1e-30))
                ),
                "total_fft_energy": float(total_power),
            }
        )

    add_packet_phase_columns(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["target_snr_db"]), int(row["packet_index"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (target_snr_db, packet_index), items in sorted(grouped.items()):
        items.sort(key=lambda row: int(row["payload_symbol_index"]))
        argmax_correct = np.asarray([int(row["is_argmax_correct"]) for row in items], dtype=np.float64)
        amp = np.asarray([float(row["gt_bin_amp"]) for row in items], dtype=np.float64)
        er = np.asarray([float(row["gt_peak_energy_ratio"]) for row in items], dtype=np.float64)
        rank = np.asarray([int(row["gt_bin_rank"]) for row in items], dtype=np.float64)
        residual = np.asarray([float(row["phase_linear_residual"]) for row in items], dtype=np.float64)
        tau = np.asarray([float(row["tau_chips"]) for row in items], dtype=np.float64)
        summary.append(
            {
                "target_snr_db": float(target_snr_db),
                "packet_index": int(packet_index),
                "event_index": int(items[0]["event_index"]),
                "frame_index": int(items[0]["frame_index"]),
                "payload_symbol_count": int(len(items)),
                "argmax_correct_rate": float(np.mean(argmax_correct)),
                "gt_bin_amp_mean": float(np.mean(amp)),
                "gt_bin_amp_std": float(np.std(amp)),
                "gt_peak_energy_ratio_mean": float(np.mean(er)),
                "gt_peak_energy_ratio_std": float(np.std(er)),
                "mean_gt_bin_rank": float(np.mean(rank)),
                "gt_top8_recall": float(np.mean(rank <= 8)),
                "gt_top16_recall": float(np.mean(rank <= 16)),
                "gt_top32_recall": float(np.mean(rank <= 32)),
                "tau_chips_mean": float(np.mean(tau)),
                "tau_chips_std": float(np.std(tau)),
                "phase_slope_pi_per_symbol": float(items[0]["packet_phase_slope_pi_per_symbol"]),
                "phase_linear_r2": float(items[0]["packet_phase_linear_r2"]),
                "phase_linear_rmse_pi": float(items[0]["packet_phase_linear_rmse_pi"]),
                "phase_residual_std_pi": float(np.std(residual) / math.pi),
                "phase_residual_peak_to_peak_pi": float((np.max(residual) - np.min(residual)) / math.pi),
                "phase_residual_quad_r2": float(items[0]["packet_residual_quad_r2"]),
            }
        )
    return summary


def plot_packet(packet_rows: list[dict[str, Any]], out_path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    packet_rows = sorted(packet_rows, key=lambda row: int(row["payload_symbol_index"]))
    k = np.asarray([int(row["payload_symbol_index"]) for row in packet_rows], dtype=np.float64)
    phase = np.asarray([float(row["gt_bin_phase"]) for row in packet_rows], dtype=np.float64)
    phase_unwrap = np.asarray([float(row["gt_bin_phase_unwrap"]) for row in packet_rows], dtype=np.float64)
    phase_fit = np.asarray([float(row["phase_linear_fit"]) for row in packet_rows], dtype=np.float64)
    residual = np.asarray([float(row["phase_linear_residual"]) for row in packet_rows], dtype=np.float64)
    amp = np.asarray([float(row["gt_bin_amp"]) for row in packet_rows], dtype=np.float64)
    energy_ratio = np.asarray([float(row["gt_peak_energy_ratio"]) for row in packet_rows], dtype=np.float64)
    argmax_correct = np.asarray([int(row["is_argmax_correct"]) for row in packet_rows], dtype=np.int32)
    tau = np.asarray([float(row["tau_chips"]) for row in packet_rows], dtype=np.float64)

    first = packet_rows[0]
    packet_index = int(first["packet_index"])
    event_index = int(first["event_index"])
    target_snr_db = float(first["target_snr_db"])
    slope = float(first["packet_phase_slope_pi_per_symbol"])
    linear_r2 = float(first["packet_phase_linear_r2"])
    quad_r2 = float(first["packet_residual_quad_r2"])
    argmax_rate = float(np.mean(argmax_correct))
    mean_er = float(np.mean(energy_ratio))
    phase_sign = str(first["phase_sign"])

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.2), dpi=int(dpi))
    axes = axes.ravel()
    marker_style = {"marker": "o", "markersize": 3.0, "linewidth": 1.1}

    axes[0].plot(k, phase / math.pi, **marker_style)
    axes[0].set_title("GT-bin wrapped phase after STO-jump comp")
    axes[0].set_ylabel("phase / pi")
    axes[0].set_ylim(-1.05, 1.05)

    axes[1].plot(k, phase_unwrap / math.pi, **marker_style, label="unwrap")
    axes[1].plot(k, phase_fit / math.pi, "-", linewidth=2.0, label=f"linear fit, slope={slope:.3f} pi/sym")
    axes[1].set_title(f"GT-bin unwrap phase, R2={linear_r2:.4f}")
    axes[1].legend()

    axes[2].plot(k, residual / math.pi, **marker_style, label="residual")
    ax_tau = axes[2].twinx()
    ax_tau.plot(k, tau, color="#2ca02c", linewidth=0.9, alpha=0.65, label="tau")
    axes[2].axhline(0.0, color="black", linewidth=0.75)
    axes[2].set_title(f"residual after linear detrend, quad R2={quad_r2:.4f}")
    axes[2].set_xlabel("payload symbol index")
    axes[2].set_ylabel("residual / pi")
    ax_tau.set_ylabel("tau chips")

    colors = np.where(argmax_correct == 1, "#1f77b4", "#d62728")
    axes[3].scatter(k, amp, c=colors, s=22)
    axes[3].plot(k, amp, color="#1f77b4", linewidth=0.85, alpha=0.55)
    ax_er = axes[3].twinx()
    ax_er.plot(k, energy_ratio, color="#ff7f0e", linewidth=1.0, alpha=0.75)
    axes[3].set_title(f"GT-bin amplitude / energy ratio, argmax acc={argmax_rate:.2f}")
    axes[3].set_xlabel("payload symbol index")
    axes[3].set_ylabel("amplitude")
    ax_er.set_ylabel("energy ratio")

    for axis in axes:
        axis.grid(True, color="#dddddd", linewidth=0.6)

    fig.suptitle(
        (
            f"Packet {packet_index} STO-jump compensated GT-bin diagnostics | event {event_index} | "
            f"SNR={target_snr_db:.1f} dB | mean ER={mean_er:.3f} | sign={phase_sign}"
        ),
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_all(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> list[Path]:
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["target_snr_db"]), int(row["packet_index"]))].append(row)

    paths: list[Path] = []
    for (snr_db, packet_index), items in sorted(grouped.items()):
        event_index = int(items[0]["event_index"])
        label = _snr_label(snr_db)
        path = (
            output_dir
            / "plots"
            / label
            / f"packet_{packet_index:03d}_event_{event_index:03d}_sto_jump_gt_bin_diagnostics.png"
        )
        plot_packet(items, out_path=path, dpi=dpi)
        paths.append(path)
    return paths


def main() -> int:
    args = parse_args()
    low_snr_dir = args.low_snr_dir.resolve()
    gt_csv = args.gt_symbol_csv.resolve()
    input_stem = args.input_stem or _infer_input_stem(gt_csv)
    output_dir = (args.output_dir or (low_snr_dir / "sto_phase_jump_corrected")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    packet_filter = set(args.packet) if args.packet else None
    symbols = load_payload_gt(gt_csv, packet_filter=packet_filter)
    noisy_bins = find_noisy_bins(low_snr_dir=low_snr_dir, input_stem=input_stem, snr_filter=args.snr_db)

    all_rows: list[dict[str, Any]] = []
    for snr_db, path in noisy_bins:
        samples = np.fromfile(path, dtype=np.complex64)
        if samples.size == 0:
            raise ValueError(f"Empty noisy IQ file: {path}")
        rows = compute_feature_rows(
            samples=samples,
            symbols=symbols,
            target_snr_db=snr_db,
            file_name=str(path),
            cfo_correction_mode=args.cfo_correction_mode,
            tau_source=args.tau_source,
            phase_sign=args.phase_sign,
        )
        label = _snr_label(snr_db)
        write_csv(output_dir / f"{input_stem}_{label}_sto_jump_gt_bin_features.csv", rows)
        all_rows.extend(rows)
        print(f"[{label}] symbols={len(symbols)}, rows={len(rows)}")

    all_feature_csv = output_dir / f"{input_stem}_sto_jump_gt_bin_features_all.csv"
    summary_csv = output_dir / f"{input_stem}_sto_jump_gt_bin_summary.csv"
    write_csv(all_feature_csv, all_rows)
    write_csv(summary_csv, summarize_rows(all_rows))
    plot_paths = [] if args.no_plots else plot_all(all_rows, output_dir=output_dir, dpi=args.dpi)

    print(f"summary={summary_csv}")
    print(f"features={all_feature_csv}")
    print(f"plots={output_dir / 'plots'}" if not args.no_plots else "plots=skipped")
    print(f"wrote_pngs={len(plot_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
