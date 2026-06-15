#!/usr/bin/env python3
"""preamble-only phase line 构建与可靠性验证实验。

这个脚本只做 phase-line 诊断，不做 payload FFT bin 重选。

实验流程：

1. 从 sync_chain 中读取 grlora_framesync_valid 的 packet 边界和 CFO/STO/SFO。
2. 默认以 grlora_fine_payload_start_sample 倒推 preamble 起点，并用 header/payload
   FFT demod 同款 CFO-aware downchirp 读取已知 upchirp bin0 相位。
3. 只用 preamble anchors 拟合 packet-level 线性相位线。
4. 如果提供 header-first symbol CSV，则把这根线外推到 payload GT bin 相位上，
   统计 residual，看 preamble-only phase line 是否可靠。

注意：payload 端会同时输出“直接外推 residual”和“允许 payload 端有一个常相位偏置
后的 residual”。后者用于排除 preamble -> sync/SFD/header 之间固定相位跳变的影响，
更关注 slope/trend 是否可迁移。
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[3]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_downchirp, build_upchirp, positive_mod, signed_fft_bin  # noqa: E402
from weak_decoder.grlora_frame_sync import _build_corrected_preamble_chirps, _extract_chip_rate_chirps  # noqa: E402
from weak_decoder.preamble_detector import PreambleDetectorConfig  # noqa: E402


@dataclass(frozen=True)
class SyncCandidate:
    packet_index: int
    event_index: int
    synced_preamble_start_sample: int
    fine_preamble_start_sample: int
    fine_payload_start_sample: int
    up_symbols_used: int
    cfo_int: int
    cfo_frac: float
    sto_sample_correction: int
    sfo_hat: float
    fs_p: float
    netid_valid: bool
    framesync_valid: bool


@dataclass(frozen=True)
class PayloadGtSymbol:
    frame_index: int
    packet_index: int
    event_index: int
    payload_symbol_index: int
    frame_symbol_index: int
    absolute_symbol_index: float
    start_sample: int
    header_start_sample: int
    sf: int
    os_factor: int
    cfo_int: int
    cfo_frac: float
    gt_raw_fft_bin: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate packet phase lines from preamble-only anchors.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="complex64 IQ .bin 文件。")
    parser.add_argument("-s", "--sync-chain-csv", type=Path, required=True, help="run_weak_sync_chain.py 输出的 sync_chain CSV。")
    parser.add_argument(
        "-g",
        "--gt-symbol-csv",
        type=Path,
        default=None,
        help="可选 header-first symbol CSV；用于 payload GT-bin 外推验证。",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认 data/phase_line/preamble_only/<input-stem>。",
    )
    parser.add_argument("--sf", type=int, default=None, help="LoRa SF；默认从文件名推断。")
    parser.add_argument("--bw", type=float, default=125000.0, help="LoRa BW Hz，默认 125000。")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ sample rate Hz，默认 500000。")
    parser.add_argument("--preamble-len", type=float, default=None, help="preamble upchirp 数；默认从文件名推断。")
    parser.add_argument(
        "--anchor-count",
        type=int,
        default=None,
        help="使用多少个 preamble upchirp 做 anchor；默认用 sync_chain 中 grlora_up_symbols_used。",
    )
    parser.add_argument(
        "--anchor-bin-mode",
        choices=("modal_argmax", "bin0"),
        default="bin0",
        help=(
            "preamble anchor 读取哪个 FFT bin。bin0 读取已知 upchirp 的语义 bin；"
            "modal_argmax 使用当前 packet 中最常见的主峰 bin。默认 bin0。"
        ),
    )
    parser.add_argument(
        "--anchor-grid",
        choices=("payload_backtrack", "framesync_corrected"),
        default="payload_backtrack",
        help=(
            "preamble anchor 的采样/补偿口径。payload_backtrack 从 fine_payload_start 倒推 preamble，"
            "并使用 header/payload 同款 CFO-aware downchirp；framesync_corrected 保留旧的 framesync preamble 校正口径。"
        ),
    )
    parser.add_argument(
        "--include-netid-invalid",
        action="store_true",
        default=False,
        help="默认只处理 grlora_netid_valid=1；打开后只要求 grlora_framesync_valid=1。",
    )
    parser.add_argument("--packet", type=int, action="append", default=None, help="只处理指定 packet_index。")
    parser.add_argument("--max-packets", type=int, default=None, help="最多处理前 N 个 packet。")
    parser.add_argument(
        "--payload-cfo-correction-mode",
        choices=("symbol", "continuous"),
        default="continuous",
        help="重新读取 payload GT bin 时的 CFO 相位口径，默认 continuous。",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI。")
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


def _flag(row: dict[str, str], key: str, default: bool = False) -> bool:
    value = str(row.get(key, "")).strip()
    if value == "":
        return bool(default)
    return bool(int(float(value)))


def infer_params_from_filename(path: Path) -> tuple[int | None, int | None]:
    parts = path.stem.split("_")
    sf = None
    preamble_len = None
    if len(parts) >= 6:
        try:
            sf = int(parts[3])
        except ValueError:
            sf = None
        try:
            preamble_len = int(parts[-1])
        except ValueError:
            preamble_len = None
    return sf, preamble_len


def circular_mean_angle(phases: np.ndarray) -> float:
    values = np.exp(1j * np.asarray(phases, dtype=np.float64))
    mean_value = complex(np.mean(values))
    return float(math.atan2(mean_value.imag, mean_value.real))


def circular_rmse(residual: np.ndarray) -> float:
    values = np.asarray(residual, dtype=np.float64)
    return float(math.sqrt(np.mean(values**2))) if values.size else float("nan")


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    rmse = float(math.sqrt(np.mean((y - pred) ** 2)))
    return coef, pred, r2, rmse


def read_sync_candidates(
    path: Path,
    packet_filter: set[int] | None,
    include_netid_invalid: bool,
    max_packets: int | None,
) -> list[SyncCandidate]:
    candidates: list[SyncCandidate] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            packet_index = _int(row, "packet_index", -1)
            if packet_filter is not None and packet_index not in packet_filter:
                continue
            if not _flag(row, "grlora_framesync_valid", False):
                continue
            netid_valid = _flag(row, "grlora_netid_valid", False)
            if not include_netid_invalid and not netid_valid:
                continue

            candidates.append(
                SyncCandidate(
                    packet_index=packet_index,
                    event_index=_int(row, "event_index", -1),
                    synced_preamble_start_sample=_int(row, "grlora_synced_preamble_start_sample"),
                    fine_preamble_start_sample=_int(row, "grlora_fine_preamble_start_sample"),
                    fine_payload_start_sample=_int(row, "grlora_fine_payload_start_sample"),
                    up_symbols_used=_int(row, "grlora_up_symbols_used", 0),
                    cfo_int=_int(row, "grlora_cfo_int_est", 0),
                    cfo_frac=_float(row, "grlora_cfo_frac_est", 0.0),
                    sto_sample_correction=_int(row, "grlora_sto_sample_correction", 0),
                    sfo_hat=_float(row, "grlora_sfo_hat", 0.0),
                    fs_p=_float(row, "grlora_fs_p", 125000.0),
                    netid_valid=netid_valid,
                    framesync_valid=True,
                )
            )

    candidates.sort(key=lambda item: (item.packet_index, item.event_index))
    if max_packets is not None:
        candidates = candidates[: int(max_packets)]
    if not candidates:
        raise ValueError("No framesync-valid candidates selected from sync_chain CSV.")
    return candidates


def _group_key(row: dict[str, str]) -> tuple[int, int]:
    return (_int(row, "packet_index", -1), _int(row, "event_index", -1))


def read_payload_gt(path: Path, preamble_len: float, packet_filter: set[int] | None) -> dict[tuple[int, int], list[PayloadGtSymbol]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    header_start_by_key: dict[tuple[int, int], int] = {}
    for row in rows:
        if row.get("stage") == "header" and _int(row, "stage_symbol_index", -1) == 0:
            header_start_by_key[_group_key(row)] = _int(row, "start_sample")

    grouped: dict[tuple[int, int], list[PayloadGtSymbol]] = defaultdict(list)
    for row in rows:
        if row.get("stage") != "payload":
            continue
        if _int(row, "header_valid", 0) != 1:
            continue
        packet_index = _int(row, "packet_index", -1)
        if packet_filter is not None and packet_index not in packet_filter:
            continue

        key = _group_key(row)
        sf = _int(row, "sf", 10)
        os_factor = _int(row, "os_factor", 4)
        frame_symbol_index = _int(row, "frame_symbol_index", -1)
        header_start = header_start_by_key.get(key)
        if header_start is None:
            header_start = _int(row, "start_sample") - frame_symbol_index * (1 << sf) * os_factor
        absolute_symbol_index = float(preamble_len) + 4.25 + float(frame_symbol_index)

        grouped[key].append(
            PayloadGtSymbol(
                frame_index=_int(row, "frame_index", -1),
                packet_index=packet_index,
                event_index=_int(row, "event_index", -1),
                payload_symbol_index=_int(row, "stage_symbol_index", -1),
                frame_symbol_index=frame_symbol_index,
                absolute_symbol_index=absolute_symbol_index,
                start_sample=_int(row, "start_sample"),
                header_start_sample=int(header_start),
                sf=sf,
                os_factor=os_factor,
                cfo_int=_int(row, "cfo_int", 0),
                cfo_frac=_float(row, "cfo_frac", 0.0),
                gt_raw_fft_bin=_int(row, "raw_fft_bin", -1),
            )
        )

    for items in grouped.values():
        items.sort(key=lambda item: item.payload_symbol_index)
    return dict(grouped)


def symbol_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    return int(start_sample) + int(os_factor / 2) + int(os_factor) * np.arange(n_bins, dtype=np.int64)


def extract_payload_gt_phase(
    samples: np.ndarray,
    symbol: PayloadGtSymbol,
    downchirps: dict[tuple[int, int, float], np.ndarray],
    cfo_correction_mode: str,
) -> dict[str, Any]:
    n_bins = 1 << int(symbol.sf)
    gt_bin = positive_mod(symbol.gt_raw_fft_bin, n_bins)
    indexes = symbol_indexes(symbol.start_sample, symbol.sf, symbol.os_factor)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(f"payload symbol exceeds IQ range at start={symbol.start_sample}")

    chip_symbol = np.asarray(samples[indexes], dtype=np.complex64)
    cfo_total = float(symbol.cfo_int) + float(symbol.cfo_frac)
    if cfo_correction_mode == "continuous":
        relative_chip_start = float(symbol.start_sample - symbol.header_start_sample) / float(symbol.os_factor)
        cfo_common_phase_rad = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
        chip_symbol = (chip_symbol * np.exp(-1j * cfo_common_phase_rad)).astype(np.complex64)
    else:
        cfo_common_phase_rad = 0.0

    key = (int(symbol.sf), int(symbol.cfo_int), float(symbol.cfo_frac))
    downchirp = downchirps.get(key)
    if downchirp is None:
        downchirp = build_downchirp(symbol.sf, cfo_int=symbol.cfo_int, cfo_frac=symbol.cfo_frac)
        downchirps[key] = downchirp

    spectrum = np.fft.fft(chip_symbol * downchirp).astype(np.complex64)
    power = np.abs(spectrum) ** 2
    peak_value = complex(spectrum[gt_bin])
    total_power = float(np.sum(power, dtype=np.float64))
    rank = int(1 + np.sum(power > float(power[gt_bin])))
    argmax_bin = int(np.argmax(power))
    return {
        "phase": float(math.atan2(peak_value.imag, peak_value.real)),
        "amp": float(abs(peak_value)),
        "power": float(power[gt_bin]),
        "energy_ratio": float(power[gt_bin] / total_power) if total_power > 0.0 else float("nan"),
        "rank": rank,
        "argmax_bin": argmax_bin,
        "is_argmax_correct": int(argmax_bin == gt_bin),
        "cfo_common_phase_rad": cfo_common_phase_rad,
    }


def extract_preamble_anchors(
    samples: np.ndarray,
    candidate: SyncCandidate,
    config: PreambleDetectorConfig,
    anchor_count: int,
    anchor_bin_mode: str,
    anchor_grid: str,
    preamble_len: float,
    cfo_correction_mode: str,
) -> list[dict[str, Any]]:
    """读取 preamble anchor phase。

    ``payload_backtrack`` 是主口径：从 fine payload/header 起点倒回已知数量的
    preamble/sync/SFD symbol，并使用 header_first_demod 同款 CFO-aware downchirp。
    这样 preamble anchor 和 payload GT phase 处于同一个 FFT 相位坐标系。
    """

    n_bins = int(config.n_bins)
    os_factor = int(config.os_factor)
    chirp_samples = int(config.chirp_samples)
    mode = str(anchor_grid)
    if mode == "payload_backtrack":
        back_symbols = float(preamble_len) + 4.25
        preamble_start = int(round(float(candidate.fine_payload_start_sample) - back_symbols * chirp_samples))
        chirps = _extract_chip_rate_chirps(
            samples,
            preamble_start,
            config,
            int(anchor_count),
            sample_correction=0,
        )
        down_ref = build_downchirp(config.sf, cfo_int=int(candidate.cfo_int), cfo_frac=float(candidate.cfo_frac))
        cfo_total = float(candidate.cfo_int) + float(candidate.cfo_frac)
        cfo_common_phases: list[float] = []
        if str(cfo_correction_mode) == "continuous":
            for idx in range(int(anchor_count)):
                symbol_start = preamble_start + idx * chirp_samples
                relative_chip_start = float(symbol_start - int(candidate.fine_payload_start_sample)) / float(os_factor)
                cfo_common_phase_rad = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
                chirps[idx] = (chirps[idx] * np.exp(-1j * cfo_common_phase_rad)).astype(np.complex64)
                cfo_common_phases.append(cfo_common_phase_rad)
        else:
            cfo_common_phases = [0.0 for _ in range(int(anchor_count))]
    elif mode == "framesync_corrected":
        preamble_start = int(candidate.synced_preamble_start_sample)
        chirps = _build_corrected_preamble_chirps(
            samples=samples,
            synced_preamble_start=preamble_start,
            detector_config=config,
            chirp_count=int(anchor_count),
            sto_sample_correction=int(candidate.sto_sample_correction),
            cfo_int=int(candidate.cfo_int),
            cfo_frac=float(candidate.cfo_frac),
            fs_p=float(candidate.fs_p),
        )
        down_ref = np.conjugate(build_upchirp(config.sf, symbol_id=0, os_factor=1)).astype(np.complex64)
        cfo_common_phases = [0.0 for _ in range(int(anchor_count))]
    else:
        raise ValueError(f"unknown anchor grid: {anchor_grid}")

    spectra = np.fft.fft(np.asarray(chirps, dtype=np.complex64) * down_ref[np.newaxis, :], axis=1).astype(np.complex64)
    powers = np.abs(spectra) ** 2
    peak_bins = np.argmax(powers, axis=1).astype(np.int64)
    if str(anchor_bin_mode) == "bin0":
        anchor_bin = 0
    else:
        anchor_bin = int(np.argmax(np.bincount(peak_bins, minlength=n_bins)))

    rows: list[dict[str, Any]] = []
    for idx, spectrum in enumerate(spectra):
        power = powers[idx]
        total_power = float(np.sum(power, dtype=np.float64))
        peak_bin = int(peak_bins[idx])
        anchor_value = complex(spectrum[anchor_bin])
        anchor_power = float(power[anchor_bin])
        second_power = float(np.partition(power, -2)[-2]) if power.size > 1 else 0.0
        rows.append(
            {
                "packet_index": int(candidate.packet_index),
                "event_index": int(candidate.event_index),
                "anchor_symbol_index": int(idx),
                "absolute_symbol_index": float(idx),
                "anchor_grid": str(anchor_grid),
                "anchor_start_sample": int(preamble_start + idx * chirp_samples),
                "anchor_cfo_common_phase_rad": float(cfo_common_phases[idx]),
                "anchor_bin_mode": str(anchor_bin_mode),
                "anchor_bin": int(anchor_bin),
                "anchor_signed_bin": signed_fft_bin(anchor_bin, n_bins),
                "argmax_bin": int(peak_bin),
                "argmax_signed_bin": signed_fft_bin(peak_bin, n_bins),
                "is_anchor_bin_argmax": int(peak_bin == anchor_bin),
                "anchor_real": float(anchor_value.real),
                "anchor_imag": float(anchor_value.imag),
                "anchor_amp": float(abs(anchor_value)),
                "anchor_power": float(anchor_power),
                "anchor_phase": float(math.atan2(anchor_value.imag, anchor_value.real)),
                "anchor_energy_ratio": float(anchor_power / total_power) if total_power > 0.0 else float("nan"),
                "peak_margin_db": float(10.0 * math.log10((float(power[peak_bin]) + 1e-30) / (second_power + 1e-30))),
                "total_fft_energy": total_power,
            }
        )
    return rows


def fit_anchor_line(anchor_rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray([float(row["absolute_symbol_index"]) for row in anchor_rows], dtype=np.float64)
    phase = np.asarray([float(row["anchor_phase"]) for row in anchor_rows], dtype=np.float64)
    phase_unwrap = np.unwrap(phase)
    coef, fit_line, r2, rmse = linear_fit(x, phase_unwrap)
    residual = np.angle(np.exp(1j * (phase_unwrap - fit_line)))
    for i, row in enumerate(anchor_rows):
        row["anchor_phase_unwrap"] = float(phase_unwrap[i])
        row["phase_line_fit"] = float(fit_line[i])
        row["phase_line_residual"] = float(residual[i])
    return {
        "slope_rad_per_symbol": float(coef[0]),
        "intercept_rad": float(coef[1]),
        "slope_pi_per_symbol": float(coef[0] / math.pi),
        "r2": float(r2),
        "rmse_pi": float(rmse / math.pi),
        "residual_std_pi": float(np.std(residual) / math.pi),
        "residual_peak_to_peak_pi": float((np.max(residual) - np.min(residual)) / math.pi),
    }


def validate_payload_against_line(
    samples: np.ndarray,
    payload_symbols: list[PayloadGtSymbol],
    line: dict[str, Any],
    cfo_correction_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    downchirps: dict[tuple[int, int, float], np.ndarray] = {}
    slope = float(line["slope_rad_per_symbol"])
    intercept = float(line["intercept_rad"])

    for symbol in payload_symbols:
        feature = extract_payload_gt_phase(
            samples=samples,
            symbol=symbol,
            downchirps=downchirps,
            cfo_correction_mode=cfo_correction_mode,
        )
        pred = float(slope * symbol.absolute_symbol_index + intercept)
        phase = float(feature["phase"])
        residual_direct = float(np.angle(np.exp(1j * (phase - pred))))
        rows.append(
            {
                "frame_index": int(symbol.frame_index),
                "packet_index": int(symbol.packet_index),
                "event_index": int(symbol.event_index),
                "payload_symbol_index": int(symbol.payload_symbol_index),
                "frame_symbol_index": int(symbol.frame_symbol_index),
                "absolute_symbol_index": float(symbol.absolute_symbol_index),
                "gt_raw_fft_bin": int(symbol.gt_raw_fft_bin),
                "gt_phase": phase,
                "gt_amp": float(feature["amp"]),
                "gt_energy_ratio": float(feature["energy_ratio"]),
                "gt_rank": int(feature["rank"]),
                "argmax_bin": int(feature["argmax_bin"]),
                "is_argmax_correct": int(feature["is_argmax_correct"]),
                "phase_line_pred": pred,
                "phase_residual_direct": residual_direct,
                "phase_offset_aligned": "",
                "phase_residual_offset_aligned": "",
                "payload_cfo_common_phase_rad": float(feature["cfo_common_phase_rad"]),
            }
        )

    if not rows:
        return rows, {}

    residual_direct = np.asarray([float(row["phase_residual_direct"]) for row in rows], dtype=np.float64)
    offset = circular_mean_angle(residual_direct)
    residual_aligned = np.angle(np.exp(1j * (residual_direct - offset)))
    payload_x = np.asarray([float(row["absolute_symbol_index"]) for row in rows], dtype=np.float64)
    payload_phase = np.unwrap(np.asarray([float(row["gt_phase"]) for row in rows], dtype=np.float64))
    payload_coef, payload_fit, payload_r2, payload_rmse = linear_fit(payload_x, payload_phase)
    for i, row in enumerate(rows):
        row["phase_offset_aligned"] = float(offset)
        row["phase_residual_offset_aligned"] = float(residual_aligned[i])
        row["payload_own_fit"] = float(payload_fit[i])

    stats = {
        "payload_phase_offset_rad": float(offset),
        "payload_phase_offset_pi": float(offset / math.pi),
        "payload_residual_direct_rmse_pi": float(circular_rmse(residual_direct) / math.pi),
        "payload_residual_direct_std_pi": float(np.std(residual_direct) / math.pi),
        "payload_residual_aligned_rmse_pi": float(circular_rmse(residual_aligned) / math.pi),
        "payload_residual_aligned_std_pi": float(np.std(residual_aligned) / math.pi),
        "payload_residual_aligned_peak_to_peak_pi": float((np.max(residual_aligned) - np.min(residual_aligned)) / math.pi),
        "payload_own_slope_pi_per_symbol": float(payload_coef[0] / math.pi),
        "payload_own_fit_r2": float(payload_r2),
        "payload_own_fit_rmse_pi": float(payload_rmse / math.pi),
        "preamble_payload_slope_delta_pi_per_symbol": float(payload_coef[0] / math.pi - float(line["slope_pi_per_symbol"])),
        "payload_argmax_correct_rate": float(np.mean([int(row["is_argmax_correct"]) for row in rows])),
        "payload_gt_top8_recall": float(np.mean([int(row["gt_rank"]) <= 8 for row in rows])),
        "payload_gt_top16_recall": float(np.mean([int(row["gt_rank"]) <= 16 for row in rows])),
        "payload_gt_top32_recall": float(np.mean([int(row["gt_rank"]) <= 32 for row in rows])),
        "payload_mean_gt_rank": float(np.mean([int(row["gt_rank"]) for row in rows])),
        "payload_mean_gt_er": float(np.mean([float(row["gt_energy_ratio"]) for row in rows])),
    }
    return rows, stats


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _match_phase_branch(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Move a predicted phase line onto the nearest 2*pi branch for plotting."""

    if pred.size == 0 or target.size == 0:
        return pred
    shift = 2.0 * math.pi * round(float(np.mean(target - pred)) / (2.0 * math.pi))
    return pred + shift


def plot_packet(
    anchor_rows: list[dict[str, Any]],
    payload_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anchor_x = np.asarray([float(row["absolute_symbol_index"]) for row in anchor_rows], dtype=np.float64)
    anchor_phase = np.asarray([float(row["anchor_phase_unwrap"]) for row in anchor_rows], dtype=np.float64)
    anchor_fit = np.asarray([float(row["phase_line_fit"]) for row in anchor_rows], dtype=np.float64)
    anchor_resid = np.asarray([float(row["phase_line_residual"]) for row in anchor_rows], dtype=np.float64)
    anchor_er = np.asarray([float(row["anchor_energy_ratio"]) for row in anchor_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.4), dpi=int(dpi))
    axes = axes.ravel()
    marker_style = {"marker": "o", "markersize": 3.0, "linewidth": 1.1}

    axes[0].plot(anchor_x, anchor_phase / math.pi, **marker_style, label="preamble anchor")
    axes[0].plot(anchor_x, anchor_fit / math.pi, linewidth=2.0, label="line fit")
    axes[0].set_title(f"preamble-only phase line, R2={float(summary['anchor_fit_r2']):.4f}")
    axes[0].set_ylabel("phase / pi")
    axes[0].legend()

    axes[1].plot(anchor_x, anchor_resid / math.pi, **marker_style, label="residual")
    ax_er = axes[1].twinx()
    ax_er.plot(anchor_x, anchor_er, color="#ff7f0e", linewidth=1.1, label="ER")
    axes[1].axhline(0.0, color="black", linewidth=0.75)
    axes[1].set_title("preamble fit residual / anchor ER")
    axes[1].set_ylabel("residual / pi")
    ax_er.set_ylabel("energy ratio")

    if payload_rows:
        payload_x = np.asarray([float(row["absolute_symbol_index"]) for row in payload_rows], dtype=np.float64)
        payload_phase = np.unwrap(np.asarray([float(row["gt_phase"]) for row in payload_rows], dtype=np.float64))
        payload_pred_direct = np.asarray([float(row["phase_line_pred"]) for row in payload_rows], dtype=np.float64)
        payload_pred_aligned = np.asarray(
            [float(row["phase_line_pred"]) + float(row["phase_offset_aligned"]) for row in payload_rows],
            dtype=np.float64,
        )
        payload_pred_direct = _match_phase_branch(payload_pred_direct, payload_phase)
        payload_pred_aligned = _match_phase_branch(payload_pred_aligned, payload_phase)
        payload_own_fit = np.asarray([float(row["payload_own_fit"]) for row in payload_rows], dtype=np.float64)
        payload_resid = np.asarray([float(row["phase_residual_offset_aligned"]) for row in payload_rows], dtype=np.float64)
        payload_rank = np.asarray([int(row["gt_rank"]) for row in payload_rows], dtype=np.float64)

        axes[2].plot(payload_x, payload_phase / math.pi, **marker_style, label="payload GT phase")
        axes[2].plot(payload_x, payload_pred_direct / math.pi, linewidth=1.2, alpha=0.55, label="preamble line direct")
        axes[2].plot(payload_x, payload_pred_aligned / math.pi, linewidth=2.0, label="preamble line + offset")
        axes[2].plot(payload_x, payload_own_fit / math.pi, "--", linewidth=1.5, label="payload own fit")
        axes[2].set_title(
            "payload GT phase validation, "
            f"aligned RMSE={float(summary['payload_residual_aligned_rmse_pi']):.3f} pi"
        )
        axes[2].set_xlabel("absolute symbol index")
        axes[2].set_ylabel("phase / pi")
        axes[2].legend()

        axes[3].plot(payload_x, payload_resid / math.pi, **marker_style, label="aligned residual")
        ax_rank = axes[3].twinx()
        ax_rank.plot(payload_x, payload_rank, color="#d62728", linewidth=0.9, alpha=0.55, label="GT rank")
        axes[3].axhline(0.0, color="black", linewidth=0.75)
        axes[3].set_title("payload residual to preamble-only line")
        axes[3].set_xlabel("absolute symbol index")
        axes[3].set_ylabel("residual / pi")
        ax_rank.set_ylabel("GT rank")
        ax_rank.set_yscale("log")
    else:
        axes[2].text(0.5, 0.5, "No payload GT CSV provided", ha="center", va="center", transform=axes[2].transAxes)
        axes[3].axis("off")

    for axis in axes:
        axis.grid(True, color="#dddddd", linewidth=0.6)

    fig.suptitle(
        (
            f"Packet {int(summary['packet_index'])} preamble-only phase line | "
            f"event {int(summary['event_index'])} | slope={float(summary['anchor_slope_pi_per_symbol']):.4f} pi/sym"
        ),
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    sync_csv = args.sync_chain_csv.resolve()
    gt_csv = args.gt_symbol_csv.resolve() if args.gt_symbol_csv is not None else None
    inferred_sf, inferred_preamble_len = infer_params_from_filename(input_path)
    sf = int(args.sf if args.sf is not None else inferred_sf)
    preamble_len = float(args.preamble_len if args.preamble_len is not None else inferred_preamble_len)
    if sf <= 0 or preamble_len <= 0:
        raise ValueError("sf and preamble_len are required; pass --sf/--preamble-len if filename inference fails.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = WEAK_ROOT / "data" / "phase_line" / "preamble_only" / input_path.stem
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = np.fromfile(input_path, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"Empty IQ file: {input_path}")

    config = PreambleDetectorConfig(
        sf=sf,
        bw=float(args.bw),
        samp_rate=float(args.samp_rate),
        win_chirps=2,
        hop_samples=None,
        min_periodic_peaks=2,
        bin_tol=0,
    )
    config.validate()

    packet_filter = set(args.packet) if args.packet else None
    candidates = read_sync_candidates(
        sync_csv,
        packet_filter=packet_filter,
        include_netid_invalid=bool(args.include_netid_invalid),
        max_packets=args.max_packets,
    )
    payload_by_key = read_payload_gt(gt_csv, preamble_len=preamble_len, packet_filter=packet_filter) if gt_csv else {}

    all_anchor_rows: list[dict[str, Any]] = []
    all_payload_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    plot_paths: list[Path] = []

    for candidate in candidates:
        anchor_count = int(args.anchor_count or candidate.up_symbols_used or max(2, int(round(preamble_len)) - 4))
        anchor_count = max(2, min(anchor_count, int(round(preamble_len))))
        anchor_rows = extract_preamble_anchors(
            samples,
            candidate,
            config=config,
            anchor_count=anchor_count,
            anchor_bin_mode=str(args.anchor_bin_mode),
            anchor_grid=str(args.anchor_grid),
            preamble_len=preamble_len,
            cfo_correction_mode=str(args.payload_cfo_correction_mode),
        )
        line = fit_anchor_line(anchor_rows)
        payload_symbols = payload_by_key.get((candidate.packet_index, candidate.event_index), [])
        payload_rows, payload_stats = validate_payload_against_line(
            samples=samples,
            payload_symbols=payload_symbols,
            line=line,
            cfo_correction_mode=str(args.payload_cfo_correction_mode),
        )

        summary: dict[str, Any] = {
            "file_name": str(input_path),
            "packet_index": int(candidate.packet_index),
            "event_index": int(candidate.event_index),
            "anchor_count": int(anchor_count),
            "preamble_len": float(preamble_len),
            "sf": int(sf),
            "bw": float(args.bw),
            "samp_rate": float(args.samp_rate),
            "cfo_int": int(candidate.cfo_int),
            "cfo_frac": float(candidate.cfo_frac),
            "sfo_hat": float(candidate.sfo_hat),
            "netid_valid": int(candidate.netid_valid),
            "anchor_grid": str(args.anchor_grid),
            "payload_cfo_correction_mode": str(args.payload_cfo_correction_mode),
            "anchor_slope_pi_per_symbol": float(line["slope_pi_per_symbol"]),
            "anchor_intercept_pi": float(line["intercept_rad"] / math.pi),
            "anchor_fit_r2": float(line["r2"]),
            "anchor_fit_rmse_pi": float(line["rmse_pi"]),
            "anchor_residual_std_pi": float(line["residual_std_pi"]),
            "anchor_residual_peak_to_peak_pi": float(line["residual_peak_to_peak_pi"]),
            "anchor_bin_mode": str(args.anchor_bin_mode),
            "anchor_bin": int(anchor_rows[0]["anchor_bin"]),
            "anchor_signed_bin": int(anchor_rows[0]["anchor_signed_bin"]),
            "anchor_bin_argmax_rate": float(np.mean([int(row["is_anchor_bin_argmax"]) for row in anchor_rows])),
            "anchor_mean_er": float(np.mean([float(row["anchor_energy_ratio"]) for row in anchor_rows])),
            "anchor_mean_margin_db": float(np.mean([float(row["peak_margin_db"]) for row in anchor_rows])),
            "payload_symbol_count": int(len(payload_rows)),
        }
        summary.update(payload_stats)
        summary_rows.append(summary)

        for row in anchor_rows:
            row.update(
                {
                    "file_name": str(input_path),
                    "sf": int(sf),
                    "bw": float(args.bw),
                    "samp_rate": float(args.samp_rate),
                    "cfo_int": int(candidate.cfo_int),
                    "cfo_frac": float(candidate.cfo_frac),
                    "sfo_hat": float(candidate.sfo_hat),
                    "anchor_grid": str(args.anchor_grid),
                    "anchor_line_slope_pi_per_symbol": float(line["slope_pi_per_symbol"]),
                    "anchor_line_r2": float(line["r2"]),
                }
            )
        all_anchor_rows.extend(anchor_rows)

        for row in payload_rows:
            row.update(
                {
                    "file_name": str(input_path),
                    "anchor_grid": str(args.anchor_grid),
                    "anchor_line_slope_pi_per_symbol": float(line["slope_pi_per_symbol"]),
                    "anchor_line_r2": float(line["r2"]),
                    "payload_residual_aligned_rmse_pi": payload_stats.get("payload_residual_aligned_rmse_pi", ""),
                }
            )
        all_payload_rows.extend(payload_rows)

        out_path = output_dir / "plots" / f"packet_{candidate.packet_index:03d}_event_{candidate.event_index:03d}_preamble_phase_line.png"
        plot_packet(anchor_rows, payload_rows, summary, out_path=out_path, dpi=int(args.dpi))
        plot_paths.append(out_path)

    write_csv(output_dir / f"{input_path.stem}_preamble_phase_anchor_features.csv", all_anchor_rows)
    write_csv(output_dir / f"{input_path.stem}_preamble_phase_payload_validation.csv", all_payload_rows)
    write_csv(output_dir / f"{input_path.stem}_preamble_phase_line_summary.csv", summary_rows)

    print(f"summary={output_dir / f'{input_path.stem}_preamble_phase_line_summary.csv'}")
    print(f"anchors={output_dir / f'{input_path.stem}_preamble_phase_anchor_features.csv'}")
    if all_payload_rows:
        print(f"payload_validation={output_dir / f'{input_path.stem}_preamble_phase_payload_validation.csv'}")
    print(f"plots={output_dir / 'plots'}")
    print(f"wrote_pngs={len(plot_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
