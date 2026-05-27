#!/usr/bin/env python3
"""导出 payload no-offset FFT peak 特征。

这个脚本是 offsets compensation 的消融对照实验入口：

1. 只复用已有 sync_chain 里的有效帧边界；
2. 不重新做 weak detection、sync word / netID 检查；
3. payload FFT 阶段不使用 CFO/STO/SFO 补偿；
4. 用 gr-lora_sdr 同款 chip-rate FFT：每 chip 取中心样点，再用无补偿 downchirp 做 FFT。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[1]
if str(WEAK_ROOT) not in sys.path:
    # 允许直接从 scripts/ 目录运行，同时还能导入旁边的 weak_decoder 包。
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_downchirp, dechirp_fft  # noqa: E402
from weak_decoder.preamble_detector import load_complex64_file  # noqa: E402


# LoRa explicit PHY header 固定为 8 个符号；payload 起点 = header 起点 + 8 symbols。
HEADER_SYMBOLS = 8


def parse_int_auto(text: str) -> int:
    """解析十进制或 0x 前缀十六进制整数。"""

    return int(str(text), 0)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    注意：sync_word / preamble_len 只保留为运行元数据，no-offset FFT 本身不会使用它们。
    """

    parser = argparse.ArgumentParser(
        description=(
            "Export a no-offset payload FFT feature table from existing sync_chain "
            "candidates. The script does not rerun weak detection, sync-word checks, "
            "CFO/STO/SFO correction, or gr-lora_sdr corrected demodulation."
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Raw complex64 IQ bin file.")
    parser.add_argument(
        "-s",
        "--sync-csv",
        type=Path,
        required=True,
        help="sync_chain CSV produced by run_weak_sync_chain.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Feature CSV output. Default: data/payload_feature_no_offset/<iq-stem>_payload_no_offset_features.csv.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="PNG output directory. Default: data/payload_feature_no_offset/plots.",
    )
    parser.add_argument("--sf", type=int, default=None, help="LoRa spreading factor. Default: infer from filename or CSV.")
    parser.add_argument("--bw", type=float, default=125000.0, help="LoRa bandwidth in Hz.")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="Raw IQ sample rate in Hz.")
    parser.add_argument("--sync-word", type=parse_int_auto, default=0x34, help="Accepted for run metadata; not used in FFT demod.")
    parser.add_argument("--preamble-len", type=float, default=None, help="Accepted for run metadata; not used in FFT demod.")
    parser.add_argument(
        "--os-factor",
        type=int,
        default=None,
        help="Raw samples per LoRa chip. Default: round(samp_rate / bw).",
    )
    parser.add_argument(
        "--anchor",
        choices=("located", "synced", "fine", "header"),
        default="located",
        help=(
            "Header-symbol slicing anchor. located uses frame_locator's located_payload_start_sample; "
            "synced uses grlora_synced_payload_start_sample; fine uses grlora_fine_payload_start_sample; "
            "header prefers an explicit header_start_sample and then falls back to fine. Default: located."
        ),
    )
    parser.add_argument(
        "--header-frames-csv",
        type=Path,
        default=None,
        help=(
            "Optional run_header_first_demod.py frame summary CSV. Used only to read payload_symbol_count. "
            "If omitted, the script tries to auto-discover the matching *_header_first_frames.csv."
        ),
    )
    parser.add_argument(
        "--payload-symbol-count",
        type=int,
        default=None,
        help="Fallback payload symbol count when neither sync CSV nor header frame CSV provides it.",
    )
    parser.add_argument(
        "--peak-gt-csv",
        type=Path,
        default=None,
        help="Optional peak groundtruth CSV. Adds gt_bin and no-offset gt-bin diagnostics.",
    )
    parser.add_argument(
        "--corrected-csv",
        type=Path,
        default=None,
        help=(
            "Optional offset-corrected header_first symbol CSV for comparison plots. "
            "If omitted, the script tries to auto-discover the matching *_header_first_symbols.csv."
        ),
    )
    parser.add_argument(
        "--no-auto-corrected",
        action="store_true",
        default=False,
        help="Disable auto-discovery of a corrected header_first symbol CSV.",
    )
    parser.add_argument("--packet", type=int, default=None, help="Only export one packet_index.")
    parser.add_argument("--max-packets", type=int, default=None, help="Only export the first N selected packets.")
    parser.add_argument("--dpi", type=int, default=220, help="Plot DPI.")
    return parser.parse_args()


def _to_int(row: dict[str, str], key: str, default: int | None = 0) -> int | None:
    """从 CSV 行里读取整数；空字段返回 default。"""

    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def _to_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    """从 CSV 行里读取浮点数；空字段返回 default。"""

    value = row.get(key, "")
    if value == "":
        return float(default)
    return float(value)


def _valid_flag(row: dict[str, str], key: str) -> bool:
    """兼容 CSV 中常见的 1/true/True 有效标记。"""

    return str(row.get(key, "0")).strip() in {"1", "true", "True"}


def _infer_sf_from_filename(path: Path) -> int | None:
    """按现有 IQ 文件命名约定从第 4 段推断 SF。"""

    parts = path.stem.split("_")
    if len(parts) >= 4:
        try:
            return int(parts[3])
        except ValueError:
            return None
    return None


def _resolve_os_factor(args: argparse.Namespace) -> int:
    """解析过采样倍数，默认要求 samp_rate / bw 为整数。"""

    if args.os_factor is not None:
        if int(args.os_factor) <= 0:
            raise ValueError("--os-factor must be positive.")
        return int(args.os_factor)
    ratio = float(args.samp_rate) / float(args.bw)
    os_factor = int(round(ratio))
    if os_factor <= 0 or not math.isclose(ratio, os_factor, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"--samp-rate / --bw must be an integer, got {ratio:.9g}.")
    return os_factor


def _capture_stem_from_sync_csv(path: Path) -> str:
    """从 sync_chain 文件名还原原始 capture stem。"""

    stem = path.stem
    marker = "_sync_chain"
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem


def _auto_header_frames_csv(sync_csv: Path, input_file: Path) -> Path | None:
    """自动寻找 header-first frame summary，用来补 payload_symbol_count。"""

    capture_stem = _capture_stem_from_sync_csv(sync_csv)
    candidates = [
        sync_csv.parent.parent / "header_first" / f"{capture_stem}_header_first_frames.csv",
        WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / f"{input_file.stem}_header_first_frames.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _auto_corrected_csv(sync_csv: Path, input_file: Path) -> Path | None:
    """自动寻找 offset-corrected symbol CSV，用来画 raw vs corrected 对比图。"""

    capture_stem = _capture_stem_from_sync_csv(sync_csv)
    candidates = [
        sync_csv.parent.parent / "header_first" / f"{capture_stem}_header_first_symbols.csv",
        WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / f"{input_file.stem}_header_first_symbols.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _row_key(row: dict[str, str]) -> tuple[str, str]:
    """用 packet_index + event_index 对齐 sync_chain 和 header_first 输出。"""

    return str(row.get("packet_index", "")), str(row.get("event_index", ""))


def load_header_frame_rows(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    """读取 header-first frame summary。

    这里只拿已经存在的 header 解码摘要，绝不在本脚本里重新解 header。
    """

    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {_row_key(row): row for row in rows}


def load_sync_rows(
    sync_csv: Path,
    header_frames: dict[tuple[str, str], dict[str, str]],
    sf_default: int | None,
    payload_symbol_count_default: int | None,
    packet_filter: int | None,
    max_packets: int | None,
) -> list[dict[str, object]]:
    """读取并筛选 sync_chain 行。

    本脚本只处理 grlora_framesync_valid == 1 的候选。payload_symbol_count
    优先来自 sync CSV；如果没有，则从 header_first frame summary 中查找。
    """

    rows: list[dict[str, object]] = []
    with sync_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            # 严格复用已有同步结果，不重新做检测或 netID 验证。
            if not _valid_flag(row, "grlora_framesync_valid"):
                continue
            packet_id = _to_int(row, "packet_index", len(rows))
            if packet_filter is not None and packet_id != int(packet_filter):
                continue
            header_frame = header_frames.get(_row_key(row), {})
            payload_count = _to_int(row, "payload_symbol_count", None)
            if payload_count is None:
                # 当前 sync_chain 不含 payload 长度，所以通常从 header_first_frames.csv 补齐。
                payload_count = _to_int(header_frame, "payload_symbol_count", None)
            if payload_count is None:
                payload_count = payload_symbol_count_default
            if payload_count is None:
                raise ValueError(
                    "payload_symbol_count is missing. Pass --header-frames-csv or --payload-symbol-count."
                )
            if int(payload_count) < 0:
                raise ValueError("payload_symbol_count must be non-negative.")

            frame_id = _to_int(header_frame, "frame_index", len(rows))
            frame_sf = _to_int(row, "sf", None)
            if frame_sf is None:
                frame_sf = _to_int(header_frame, "sf", sf_default)
            if frame_sf is None:
                raise ValueError("SF is missing. Pass --sf or use an input filename with an SF field.")

            merged = dict(row)
            for key, value in header_frame.items():
                merged.setdefault(key, value)
            rows.append(
                {
                    "raw": merged,
                    "packet_id": int(packet_id),
                    "event_id": int(_to_int(row, "event_index", -1)),
                    "frame_id": int(frame_id),
                    "sf": int(frame_sf),
                    "payload_symbol_count": int(payload_count),
                }
            )
            if max_packets is not None and len(rows) >= int(max_packets):
                break
    return rows


def header_start_from_row(row: dict[str, str], anchor: str) -> int:
    """按 anchor 策略选取 PHY header 第 0 个 symbol 的起点。

    located：使用 frame_locator 的物理边界，最接近“完全不做 offsets 补偿”；
    synced/fine/header：用于更细的对照实验，仍然只影响切片锚点，不启用 FFT 补偿。
    """

    if anchor == "located":
        keys = ("located_payload_start_sample", "header_start_sample", "grlora_fine_payload_start_sample")
    elif anchor == "synced":
        keys = ("grlora_synced_payload_start_sample", "located_payload_start_sample", "header_start_sample")
    elif anchor == "fine":
        keys = ("grlora_fine_payload_start_sample", "header_start_sample", "located_payload_start_sample")
    elif anchor == "header":
        keys = ("header_start_sample", "grlora_fine_payload_start_sample", "located_payload_start_sample")
    else:
        raise ValueError(f"unknown anchor: {anchor}")
    for key in keys:
        value = row.get(key, "")
        if value != "":
            return int(float(value))
    raise ValueError(f"row does not contain a usable {anchor} header anchor.")


def build_no_offset_downchirp(sf: int) -> np.ndarray:
    """构造 chip-rate no-offset downchirp。

    这和 corrected 路径使用同一个 downchirp 构造函数，但显式把 CFO_int/CFO_frac 置零。
    """

    return build_downchirp(sf, cfo_int=0, cfo_frac=0.0)


def chiprate_sample_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    """复刻 corrected demod 的 chip-rate 抽样位置。

    每个 LoRa chip 只取一个中心样点，因此 FFT 长度是 2^SF，而不是 2^SF * os_factor。
    """

    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    return int(start_sample) + int(os_value / 2) + os_value * np.arange(n_bins, dtype=np.int64)


def demod_no_offset_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    downchirp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float, float, float, float]:
    """对单个 payload symbol 做 no-offset chip-rate dechirp + FFT。

    禁用项：
    - 不做 CFO_int 采样偏移；
    - 不做 CFO_frac 相位旋转；
    - 不做 STO_frac 小数采样对齐；
    - 不做 SFO 累计插删点；
    - 不使用 gr-lora_sdr corrected downchirp。
    """

    indexes = chiprate_sample_indexes(start_sample, sf=sf, os_factor=os_factor)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(
            f"chip-rate symbol indexes [{int(indexes[0])}, {int(indexes[-1])}] exceed input samples."
        )
    # 只复刻 corrected 的 chip-rate 抽样和 FFT 口径；不引入任何 offsets 补偿。
    raw_symbol = np.asarray(samples[indexes], dtype=np.complex64)
    spectrum = dechirp_fft(raw_symbol, downchirp)
    power = np.abs(spectrum) ** 2
    raw_bin = int(np.argmax(power))
    peak = complex(spectrum[raw_bin])
    peak_amp = float(abs(peak))
    peak_power = float(power[raw_bin])
    total_power = float(np.sum(power, dtype=np.float64))
    if power.size > 1:
        top2 = np.partition(power, -2)[-2:]
        second_amp = math.sqrt(float(np.min(top2)))
    else:
        second_amp = 0.0
    peak_margin_db = float(20.0 * math.log10((peak_amp + 1e-30) / (second_amp + 1e-30)))
    peak_ratio = float(peak_power / total_power) if total_power > 0.0 else float("nan")
    phase = float(math.atan2(peak.imag, peak.real))
    return spectrum, power, raw_bin, peak_amp, peak_power, phase, peak_margin_db, total_power, peak_ratio


def load_peak_gt(path: Path | None) -> dict[tuple[str, int], dict[str, str]]:
    """读取可选 peak groundtruth。

    groundtruth 只用于标注和评估，不参与 no-offset argmax 选择。
    """

    if path is None:
        return {}
    mapping: dict[tuple[str, int], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("is_header", "0")) == "1":
                continue
            gt_packet = row.get("packet_index", "")
            gt_symbol = _to_int(row, "payload_chirp_index", None)
            if gt_packet != "" and gt_symbol is not None:
                mapping[(str(gt_packet), int(gt_symbol))] = row
            frame_count = row.get("frame_count", "")
            if frame_count != "" and gt_symbol is not None:
                mapping.setdefault((str(frame_count), int(gt_symbol)), row)
    return mapping


def find_gt_row(
    gt_rows: dict[tuple[str, int], dict[str, str]],
    packet_id: int,
    frame_id: int,
    payload_symbol_idx: int,
) -> dict[str, str] | None:
    """用 packet_id 或 frame_id 匹配某个 payload symbol 的 groundtruth 行。"""

    for key in (
        (str(packet_id), int(payload_symbol_idx)),
        (str(frame_id), int(payload_symbol_idx)),
    ):
        row = gt_rows.get(key)
        if row is not None:
            return row
    return None


def gt_features(
    gt_row: dict[str, str] | None,
    spectrum: np.ndarray,
    power: np.ndarray,
    raw_fft_bin: int,
) -> dict[str, object]:
    """计算 GT bin 在 no-offset 频谱中的幅度、相位和排名。"""

    empty = {
        "gt_bin": "",
        "is_argmax_correct": "",
        "gt_bin_amp": "",
        "gt_bin_phase": "",
        "gt_bin_rank": "",
    }
    if gt_row is None:
        return empty
    gt_value = gt_row.get("label_fft_bin", "") or gt_row.get("hard_bin", "")
    if gt_value == "":
        return empty
    gt_bin = int(float(gt_value))
    if gt_bin < 0 or gt_bin >= spectrum.size:
        return {
            "gt_bin": gt_bin,
            "is_argmax_correct": int(raw_fft_bin == gt_bin),
            "gt_bin_amp": "",
            "gt_bin_phase": "",
            "gt_bin_rank": "",
        }
    gt_peak = complex(spectrum[gt_bin])
    gt_power = float(power[gt_bin])
    return {
        "gt_bin": int(gt_bin),
        "is_argmax_correct": int(raw_fft_bin == gt_bin),
        "gt_bin_amp": float(abs(gt_peak)),
        "gt_bin_phase": float(math.atan2(gt_peak.imag, gt_peak.real)),
        "gt_bin_rank": int(1 + np.sum(power > gt_power)),
    }


def unwrap_feature_phases(rows: list[dict[str, object]]) -> None:
    """按 packet 对 selected peak phase 做 unwrap，便于观察包内相位走势。"""

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["packet_id"])].append(row)
    for packet_rows in grouped.values():
        packet_rows.sort(key=lambda item: int(item["payload_symbol_idx"]))
        phases = np.asarray([float(item["selected_peak_phase"]) for item in packet_rows], dtype=np.float64)
        unwrapped = np.unwrap(phases)
        for row, phase in zip(packet_rows, unwrapped):
            row["selected_peak_phase_unwrap"] = float(phase)


def export_features(
    samples: np.ndarray,
    selected_rows: list[dict[str, object]],
    gt_rows: dict[tuple[str, int], dict[str, str]],
    input_file: Path,
    bw: float,
    samp_rate: float,
    os_factor: int,
    anchor: str,
) -> list[dict[str, object]]:
    """主特征导出逻辑。

    输入是已有同步候选，输出是一行一个 payload symbol 的 no-offset FFT peak 特征。
    """

    rows: list[dict[str, object]] = []
    downchirps: dict[int, np.ndarray] = {}

    for selected in selected_rows:
        source = selected["raw"]
        assert isinstance(source, dict)
        packet_id = int(selected["packet_id"])
        event_id = int(selected["event_id"])
        frame_id = int(selected["frame_id"])
        sf = int(selected["sf"])
        payload_count = int(selected["payload_symbol_count"])
        samples_per_symbol = int((1 << sf) * os_factor)
        header_start = header_start_from_row(source, anchor)
        # 这里把 header 起点向后平移 8 个 explicit-header symbol，得到 payload 第 0 个符号起点。
        payload_start = int(header_start + HEADER_SYMBOLS * samples_per_symbol)
        key = sf
        if key not in downchirps:
            # 同一组 SF 的 chip-rate no-offset downchirp 可以复用。
            downchirps[key] = build_no_offset_downchirp(sf)
        downchirp = downchirps[key]

        for payload_idx in range(payload_count):
            start = int(payload_start + payload_idx * samples_per_symbol)
            spectrum, power, raw_bin, amp, peak_power, phase, margin_db, total_energy, ratio = (
                demod_no_offset_symbol(samples, start, sf=sf, os_factor=os_factor, downchirp=downchirp)
            )
            gt_row = find_gt_row(gt_rows, packet_id, frame_id, payload_idx)
            row: dict[str, object] = {
                "file_name": input_file.name,
                "packet_id": packet_id,
                "event_id": event_id,
                "frame_id": frame_id,
                "payload_symbol_idx": int(payload_idx),
                "mode": "no_offset",
                "anchor_mode": anchor,
                "header_start_sample": int(header_start),
                "payload_start_sample": int(payload_start),
                "symbol_start_sample": int(start),
                "sf": sf,
                "bw": float(bw),
                "sample_rate": float(samp_rate),
                "os_factor": int(os_factor),
                "samples_per_symbol": int(samples_per_symbol),
                "raw_fft_bin": int(raw_bin),
                "selected_peak_amp": float(amp),
                "selected_peak_power": float(peak_power),
                "selected_peak_phase": float(phase),
                "selected_peak_phase_unwrap": "",
                "peak_margin_db": float(margin_db),
                "total_fft_energy": float(total_energy),
                "peak_energy_ratio": float(ratio),
            }
            row.update(gt_features(gt_row, spectrum, power, raw_bin))
            rows.append(row)

    unwrap_feature_phases(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    """原子写 CSV，避免中途失败留下半截结果文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp_path.replace(path)


def group_by_packet(rows: Iterable[dict[str, object]], packet_key: str = "packet_id") -> dict[int, list[dict[str, object]]]:
    """按 packet_id 分组，并保证每组内部按 payload symbol 顺序排列。"""

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row[packet_key])].append(row)
    for packet_rows in grouped.values():
        packet_rows.sort(key=lambda item: int(item["payload_symbol_idx"]))
    return dict(sorted(grouped.items()))


def _finite_min_max(values: np.ndarray) -> tuple[float, float]:
    """给绘图坐标轴计算一个带留白的有限值范围。"""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if math.isclose(y_min, y_max, rel_tol=1e-12, abs_tol=1e-12):
        pad = max(1.0, abs(y_min) * 0.05)
        return y_min - pad, y_max + pad
    pad = 0.06 * (y_max - y_min)
    return y_min - pad, y_max + pad


def _plot_panels_pillow(out_path: Path, title: str, panels: list[dict[str, object]], dpi: int) -> None:
    """matplotlib 不可用时，用 Pillow 画一个简洁的多面板折线图。"""

    from PIL import Image, ImageDraw, ImageFont

    width = 1800
    panel_height = 330
    top_margin = 82
    bottom_margin = 70
    left_margin = 130
    right_margin = 42
    gap = 42
    height = top_margin + bottom_margin + len(panels) * panel_height + (len(panels) - 1) * gap

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        label_font = ImageFont.truetype("arial.ttf", 17)
        tick_font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        tick_font = ImageFont.load_default()

    draw.text((left_margin, 28), title, fill="#111111", font=title_font)
    plot_left = left_margin
    plot_right = width - right_margin
    plot_width = plot_right - plot_left

    for panel_index, panel in enumerate(panels):
        y_top = top_margin + panel_index * (panel_height + gap)
        y_bottom = y_top + panel_height
        series = panel["series"]
        assert isinstance(series, list)
        x_values = np.concatenate([item["x"] for item in series if len(item["x"]) > 0])
        y_values = np.concatenate([item["y"] for item in series if len(item["y"]) > 0])
        x_min, x_max = _finite_min_max(x_values.astype(np.float64))
        y_min, y_max = _finite_min_max(y_values.astype(np.float64))
        if math.isclose(x_min, x_max, rel_tol=1e-12, abs_tol=1e-12):
            x_min -= 1.0
            x_max += 1.0

        # 背景网格保持轻量，只用于快速观察趋势，不追求论文图美化。
        draw.rectangle((plot_left, y_top, plot_right, y_bottom), outline="#333333", width=1)
        for grid_idx in range(1, 5):
            gy = y_top + grid_idx * panel_height / 5.0
            gx = plot_left + grid_idx * plot_width / 5.0
            draw.line((plot_left, gy, plot_right, gy), fill="#e5e5e5", width=1)
            draw.line((gx, y_top, gx, y_bottom), fill="#eeeeee", width=1)

        ylabel = str(panel.get("ylabel", ""))
        draw.text((14, y_top + panel_height / 2 - 8), ylabel, fill="#111111", font=label_font)
        draw.text((plot_left, y_top - 24), str(panel.get("title", "")), fill="#111111", font=label_font)

        def point_xy(x_value: float, y_value: float) -> tuple[int, int]:
            """把数据坐标映射到图像像素坐标。"""

            px = plot_left + (float(x_value) - x_min) / (x_max - x_min) * plot_width
            py = y_bottom - (float(y_value) - y_min) / (y_max - y_min) * panel_height
            return int(round(px)), int(round(py))

        legend_x = plot_right - 210
        legend_y = y_top + 10
        for series_index, item in enumerate(series):
            color = str(item.get("color", "#1f77b4"))
            label = str(item.get("label", ""))
            x_arr = np.asarray(item["x"], dtype=np.float64)
            y_arr = np.asarray(item["y"], dtype=np.float64)
            points = [
                point_xy(x_value, y_value)
                for x_value, y_value in zip(x_arr, y_arr)
                if np.isfinite(x_value) and np.isfinite(y_value)
            ]
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            for px, py in points:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline=color, fill="white", width=2)
            if label:
                ly = legend_y + 20 * series_index
                draw.line((legend_x, ly + 7, legend_x + 28, ly + 7), fill=color, width=3)
                draw.text((legend_x + 36, ly), label, fill="#222222", font=tick_font)

        draw.text((plot_left, y_bottom + 8), f"{x_min:.0f}", fill="#333333", font=tick_font)
        x_max_text = f"{x_max:.0f}"
        text_width = draw.textbbox((0, 0), x_max_text, font=tick_font)[2]
        draw.text((plot_right - text_width, y_bottom + 8), x_max_text, fill="#333333", font=tick_font)
        draw.text((plot_left - 82, y_top - 3), f"{y_max:.3g}", fill="#333333", font=tick_font)
        draw.text((plot_left - 82, y_bottom - 12), f"{y_min:.3g}", fill="#333333", font=tick_font)

    draw.text((plot_left, height - 42), "Payload symbol index", fill="#111111", font=label_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, dpi=(int(dpi), int(dpi)))


def _plot_no_offset_packet_pillow(
    packet_id: int,
    packet_rows: list[dict[str, object]],
    out_path: Path,
    dpi: int,
) -> None:
    """Pillow fallback：绘制单包 no-offset selected peak 趋势。"""

    x = np.asarray([int(row["payload_symbol_idx"]) for row in packet_rows], dtype=np.float64)
    phase_pi = np.asarray([float(row["selected_peak_phase"]) / math.pi for row in packet_rows], dtype=np.float64)
    amp = np.asarray([float(row["selected_peak_amp"]) for row in packet_rows], dtype=np.float64)
    frame_id = int(packet_rows[0]["frame_id"]) if packet_rows else -1
    event_id = int(packet_rows[0]["event_id"]) if packet_rows else -1
    panels = [
        {"title": "Payload selected FFT peak phase", "ylabel": "Phase / pi", "series": [{"x": x, "y": phase_pi, "color": "#1f77b4"}]},
        {"title": "Payload selected FFT peak amplitude", "ylabel": "Amplitude", "series": [{"x": x, "y": amp, "color": "#1f77b4"}]},
    ]
    _plot_panels_pillow(
        out_path,
        f"packet_id={packet_id} | frame_id={frame_id} | event_id={event_id} | mode=no_offset_chiprate",
        panels,
        dpi,
    )


def plot_no_offset_packet(packet_id: int, packet_rows: list[dict[str, object]], out_path: Path, dpi: int) -> None:
    """绘制单包 no-offset 图。

    优先使用 matplotlib，版式对齐现有 plot_payload_peak_trends.py 的两联图风格。
    """

    try:
        import matplotlib
    except ModuleNotFoundError:
        _plot_no_offset_packet_pillow(packet_id, packet_rows, out_path, dpi)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([int(row["payload_symbol_idx"]) for row in packet_rows], dtype=np.float64)
    phase_pi = np.asarray([float(row["selected_peak_phase"]) / math.pi for row in packet_rows], dtype=np.float64)
    amp = np.asarray([float(row["selected_peak_amp"]) for row in packet_rows], dtype=np.float64)
    frame_id = int(packet_rows[0]["frame_id"]) if packet_rows else -1
    event_id = int(packet_rows[0]["event_id"]) if packet_rows else -1

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.4), dpi=int(dpi), sharex=True)
    marker_style = {
        "marker": "o",
        "markersize": 3.2,
        "markerfacecolor": "white",
        "linewidth": 1.05,
        "alpha": 0.92,
        "color": "#1f77b4",
    }
    axes[0].plot(x, phase_pi, **marker_style)
    axes[1].plot(x, amp, **marker_style)

    axes[0].set_title("Payload selected FFT peak phase")
    axes[0].set_ylabel("Phase / pi")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(True, color="#dddddd", linewidth=0.55)

    axes[1].set_title("Payload selected FFT peak amplitude")
    axes[1].set_xlabel("Payload symbol index")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, color="#dddddd", linewidth=0.55)

    for axis in axes:
        axis.tick_params(axis="both", labelsize=9)
    for axis in axes:
        axis.title.set_fontsize(12)
    fig.suptitle(
        f"Packet {packet_id} no-offset chip-rate FFT selected peak trends | frame {frame_id} | event {event_id}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def load_corrected_rows(path: Path | None) -> dict[tuple[int, int, int], dict[str, object]]:
    """读取 offset-corrected payload symbol CSV，供 raw vs corrected 图使用。"""

    if path is None:
        return {}
    mapping: dict[tuple[int, int, int], dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("stage") != "payload":
                continue
            if str(row.get("header_valid", "0")).strip() != "1":
                continue
            packet_id = int(float(row.get("packet_index", "0")))
            event_id = int(float(row.get("event_index", "-1")))
            symbol_idx = int(float(row.get("stage_symbol_index", "0")))
            peak_power = _to_float(row, "peak_power")
            total_power = _to_float(row, "total_power")
            mapping[(packet_id, event_id, symbol_idx)] = {
                "packet_id": packet_id,
                "event_id": event_id,
                "payload_symbol_idx": symbol_idx,
                "corrected_amp": _to_float(row, "peak_amp"),
                "corrected_phase": _to_float(row, "peak_phase"),
                "corrected_margin_db": _to_float(row, "peak_margin_db"),
                "corrected_energy_ratio": float(peak_power / total_power)
                if np.isfinite(peak_power) and np.isfinite(total_power) and total_power > 0.0
                else float("nan"),
            }
    corrected_by_packet: dict[int, list[dict[str, object]]] = defaultdict(list)
    for item in mapping.values():
        corrected_by_packet[int(item["packet_id"])].append(item)
    for packet_rows in corrected_by_packet.values():
        packet_rows.sort(key=lambda item: int(item["payload_symbol_idx"]))
        phases = np.asarray([float(item["corrected_phase"]) for item in packet_rows], dtype=np.float64)
        for item, unwrapped in zip(packet_rows, np.unwrap(phases)):
            item["corrected_phase_unwrap"] = float(unwrapped)
    return mapping


def _plot_comparison_packet_pillow(
    packet_id: int,
    matched: list[tuple[dict[str, object], dict[str, object]]],
    packet_rows: list[dict[str, object]],
    out_path: Path,
    dpi: int,
) -> None:
    """Pillow fallback：绘制 no-offset 和 corrected 的相位/幅度对比。"""

    x = np.asarray([int(raw["payload_symbol_idx"]) for raw, _ in matched], dtype=np.float64)
    raw_amp = np.asarray([float(raw["selected_peak_amp"]) for raw, _ in matched], dtype=np.float64)
    corr_amp = np.asarray([float(corr["corrected_amp"]) for _, corr in matched], dtype=np.float64)
    raw_phase = np.asarray([float(raw["selected_peak_phase"]) / math.pi for raw, _ in matched], dtype=np.float64)
    corr_phase = np.asarray([float(corr["corrected_phase"]) / math.pi for _, corr in matched], dtype=np.float64)
    event_id = int(packet_rows[0]["event_id"]) if packet_rows else -1
    frame_id = int(packet_rows[0]["frame_id"]) if packet_rows else -1
    panels = [
        {
            "title": "Payload selected FFT peak phase",
            "ylabel": "Phase / pi",
            "series": [
                {"x": x, "y": raw_phase, "label": "no_offset", "color": "#d62728"},
                {"x": x, "y": corr_phase, "label": "corrected", "color": "#1f77b4"},
            ],
        },
        {
            "title": "Payload selected FFT peak amplitude",
            "ylabel": "Amplitude",
            "series": [
                {"x": x, "y": raw_amp, "label": "no_offset", "color": "#d62728"},
                {"x": x, "y": corr_amp, "label": "corrected", "color": "#1f77b4"},
            ],
        },
    ]
    _plot_panels_pillow(
        out_path,
        f"packet_id={packet_id} | frame_id={frame_id} | event_id={event_id} | no_offset_chiprate vs corrected",
        panels,
        dpi,
    )


def plot_comparison_packet(
    packet_id: int,
    packet_rows: list[dict[str, object]],
    corrected_rows: dict[tuple[int, int, int], dict[str, object]],
    out_path: Path,
    dpi: int,
) -> bool:
    """绘制单包 raw/no-offset 与 offset-corrected 的对比图。

    只画 phase/pi 和 amplitude，返回 False 表示该 packet 没有找到可对齐的 corrected 行。
    """

    matched: list[tuple[dict[str, object], dict[str, object]]] = []
    for row in packet_rows:
        key = (int(row["packet_id"]), int(row["event_id"]), int(row["payload_symbol_idx"]))
        corrected = corrected_rows.get(key)
        if corrected is not None:
            matched.append((row, corrected))
    if not matched:
        return False

    try:
        import matplotlib
    except ModuleNotFoundError:
        _plot_comparison_packet_pillow(packet_id, matched, packet_rows, out_path, dpi)
        return True

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([int(raw["payload_symbol_idx"]) for raw, _ in matched], dtype=np.float64)
    raw_amp = np.asarray([float(raw["selected_peak_amp"]) for raw, _ in matched], dtype=np.float64)
    corr_amp = np.asarray([float(corr["corrected_amp"]) for _, corr in matched], dtype=np.float64)
    raw_phase = np.asarray([float(raw["selected_peak_phase"]) / math.pi for raw, _ in matched], dtype=np.float64)
    corr_phase = np.asarray([float(corr["corrected_phase"]) / math.pi for _, corr in matched], dtype=np.float64)
    event_id = int(packet_rows[0]["event_id"]) if packet_rows else -1
    frame_id = int(packet_rows[0]["frame_id"]) if packet_rows else -1

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.4), dpi=int(dpi), sharex=True)
    raw_style = {"marker": "o", "markersize": 3.0, "linewidth": 1.0, "alpha": 0.9, "color": "#d62728"}
    corr_style = {"marker": "s", "markersize": 2.8, "linewidth": 1.0, "alpha": 0.86, "color": "#1f77b4"}
    series = (
        (raw_phase, corr_phase, "Phase / pi", "Payload selected FFT peak phase"),
        (raw_amp, corr_amp, "Amplitude", "Payload selected FFT peak amplitude"),
    )
    for axis, (raw_y, corr_y, label, title) in zip(axes, series):
        axis.plot(x, raw_y, label="no_offset", **raw_style)
        axis.plot(x, corr_y, label="corrected", **corr_style)
        axis.set_ylabel(label)
        axis.set_title(title)
        axis.grid(True, color="#dddddd", linewidth=0.55)
        axis.legend(loc="best", fontsize=8)
    axes[0].set_ylim(-1.05, 1.05)
    axes[-1].set_xlabel("Payload symbol index")
    fig.suptitle(
        f"packet_id={packet_id} | frame_id={frame_id} | event_id={event_id} | no_offset_chiprate vs corrected",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main() -> None:
    """脚本主入口：读取输入、导出 CSV、生成趋势图和可选对比图。"""

    args = parse_args()
    input_file = args.input.resolve()
    sync_csv = args.sync_csv.resolve()
    output = args.output or (
        WEAK_ROOT
        / "data"
        / "payload_feature_no_offset"
        / f"{input_file.stem}_payload_no_offset_features.csv"
    )
    plots_dir = args.plots_dir or (WEAK_ROOT / "data" / "payload_feature_no_offset" / "plots")
    sf_default = args.sf if args.sf is not None else _infer_sf_from_filename(input_file)
    os_factor = _resolve_os_factor(args)

    # sync_chain 通常没有 payload_symbol_count，因此默认自动找 header_first frame summary。
    header_frames_csv = args.header_frames_csv or _auto_header_frames_csv(sync_csv, input_file)
    header_frames = load_header_frame_rows(header_frames_csv)
    selected_rows = load_sync_rows(
        sync_csv,
        header_frames,
        sf_default=sf_default,
        payload_symbol_count_default=args.payload_symbol_count,
        packet_filter=args.packet,
        max_packets=args.max_packets,
    )
    if not selected_rows:
        raise ValueError("No grlora_framesync_valid candidates were selected.")

    # GT 只用于附加诊断字段，不影响 no-offset FFT 的 selected peak。
    gt_rows = load_peak_gt(args.peak_gt_csv.resolve() if args.peak_gt_csv else None)
    samples = load_complex64_file(input_file)
    try:
        feature_rows = export_features(
            samples=samples,
            selected_rows=selected_rows,
            gt_rows=gt_rows,
            input_file=input_file,
            bw=args.bw,
            samp_rate=args.samp_rate,
            os_factor=os_factor,
            anchor=args.anchor,
        )
    finally:
        mmap_handle = getattr(samples, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    fields = [
        "file_name",
        "packet_id",
        "event_id",
        "frame_id",
        "payload_symbol_idx",
        "mode",
        "anchor_mode",
        "header_start_sample",
        "payload_start_sample",
        "symbol_start_sample",
        "sf",
        "bw",
        "sample_rate",
        "os_factor",
        "samples_per_symbol",
        "raw_fft_bin",
        "selected_peak_amp",
        "selected_peak_power",
        "selected_peak_phase",
        "selected_peak_phase_unwrap",
        "peak_margin_db",
        "total_fft_energy",
        "peak_energy_ratio",
        "gt_bin",
        "is_argmax_correct",
        "gt_bin_amp",
        "gt_bin_phase",
        "gt_bin_rank",
    ]
    write_csv(output, feature_rows, fields)

    grouped = group_by_packet(feature_rows)
    no_offset_pngs = 0
    for packet_id, packet_rows in grouped.items():
        out_path = plots_dir / f"packet_{packet_id:03d}_no_offset_peak_trends.png"
        plot_no_offset_packet(packet_id, packet_rows, out_path, dpi=args.dpi)
        no_offset_pngs += 1

    corrected_csv = args.corrected_csv
    if corrected_csv is None and not args.no_auto_corrected:
        # 找得到 corrected CSV 就自动生成对照图；找不到也不影响 no-offset CSV 导出。
        corrected_csv = _auto_corrected_csv(sync_csv, input_file)
    corrected_pngs = 0
    if corrected_csv is not None and corrected_csv.exists():
        corrected_rows = load_corrected_rows(corrected_csv)
        for packet_id, packet_rows in grouped.items():
            out_path = plots_dir / f"packet_{packet_id:03d}_raw_vs_corrected_peak_trends.png"
            if plot_comparison_packet(packet_id, packet_rows, corrected_rows, out_path, dpi=args.dpi):
                corrected_pngs += 1

    gt_matches = sum(1 for row in feature_rows if row.get("gt_bin", "") != "")
    gt_correct = sum(1 for row in feature_rows if row.get("is_argmax_correct", "") == 1)
    print(f"selected_packets={len(grouped)}")
    print(f"payload_rows={len(feature_rows)}")
    print(f"anchor={args.anchor}")
    print(f"os_factor={os_factor}")
    if header_frames_csv is not None:
        print(f"header_frames_csv={header_frames_csv}")
    if args.peak_gt_csv is not None:
        print(f"gt_matches={gt_matches}, gt_argmax_correct={gt_correct}/{gt_matches}")
    print(f"wrote={output}")
    print(f"wrote_no_offset_plots={no_offset_pngs}")
    if corrected_csv is not None and corrected_csv.exists():
        print(f"corrected_csv={corrected_csv}")
        print(f"wrote_comparison_plots={corrected_pngs}")


if __name__ == "__main__":
    main()
