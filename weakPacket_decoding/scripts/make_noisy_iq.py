#!/usr/bin/env python3
"""Generate lower-SNR LoRa IQ captures by adding complex AWGN.

The input and output format is raw GNU Radio ``gr_complex``:
interleaved float32 I/Q samples, readable as ``numpy.complex64``.

中文说明：
这个脚本用于把真实 USRP 采集的 LoRa IQ 文件人工降信噪比。输入/输出
都是 GNU Radio 常用的 raw complex64 文件，也就是连续的 float32 I/Q
复数采样点。脚本只做离线加噪，不修改 gr-lora_sdr 的 C++ 解码链路。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Iterator

import numpy as np


# 默认路径都按当前脚本位置反推，避免运行时强依赖当前工作目录。
SCRIPT_DIR = Path(__file__).resolve().parent
GRLORA_ROOT = SCRIPT_DIR.parents[1]
WEAKPACKET_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = GRLORA_ROOT / "data" / "USRP_IQ" / "0_0_0_10_14_8.bin"
DEFAULT_OUTPUT_ROOT = WEAKPACKET_ROOT / "data" / "noisy_iq"
FILE_SOURCE_STAGING_DIR = WEAKPACKET_ROOT / "_file_source_staging"
COMPLEX64_BYTES = np.dtype(np.complex64).itemsize
DEFAULT_NOISE_START_DB = -30.0
DEFAULT_NOISE_STOP_DB = 0.0
DEFAULT_NOISE_STEP_DB = 5.0
# 这一块没办法硬编码一个内容，因为payload有一部分是关于发多少个包的，这个玩意是递增的
DEFAULT_EXPECTED_PAYLOAD_HEXES = [
    "404433221100130058303132333435363738393a3b3c3d3e3f4041424378563412",
    "404433221100150058303132333435363738393a3b3c3d3e3f4041424378563412",
    "404433221100170058303132333435363738393a3b3c3d3e3f4041424378563412",
    "404433221100190058303132333435363738393a3b3c3d3e3f4041424378563412",
    "4044332211001a0058303132333435363738393a3b3c3d3e3f4041424378563412",
    "4044332211001b0058303132333435363738393a3b3c3d3e3f4041424378563412",
    "4044332211001c0058303132333435363738393a3b3c3d3e3f4041424378563412",
]


def parse_payload_hex(value: str) -> bytes:
    """Parse a payload hex string; separators such as spaces, commas and 0x are allowed."""
    text = str(value).strip().lower().replace("0x", "")
    compact = re.sub(r"[^0-9a-f]", "", text)
    if len(compact) % 2:
        raise ValueError(f"Payload hex string has an odd number of hex digits: {value!r}")
    return bytes.fromhex(compact)


def payload_bytes_to_text(payload: bytes) -> str:
    """Return a compact printable representation for JSON/CSV diagnostics."""
    return payload.decode("ascii", errors="backslashreplace")


def parse_int_auto(value: str) -> int:
    """支持十进制和 0x 前缀十六进制参数，例如 --sync-word 0x34。"""
    return int(str(value), 0)


def db10(value: float) -> float:
    """把线性功率比转换成 dB。非正功率用 -inf 表示，后面写 JSON 时会转成 null。"""
    if value <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(value)


def db_to_label(value_db: float) -> str:
    """把 dB 数值变成文件名友好的标签，例如 -5.0 -> m5p0。"""
    text = f"{value_db:.3f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")


def format_db_value(value_db: float) -> str:
    """Format a dB value without hiding fine sweep steps."""
    return f"{float(value_db):.3f}".rstrip("0").rstrip(".")


def parse_capture_metadata_value(value: str) -> int | str:
    """从文件名字段中提取整数；没有数字时保留原字符串。"""
    match = re.search(r"-?\d+", str(value))
    if match is None:
        return str(value)
    return int(match.group(0))


def parse_capture_metadata(path: Path) -> dict[str, Any]:
    """解析 USRP_IQ 文件名中的实验参数。

    文件名描述约定为：
        实验编号_走廊编号_位置编号_SF_TP_Preamble数量.bin

    这里按前 6 个字段解析；这样输入已经由本脚本生成、后面带有
    ``_noise_rel_...`` 后缀时，仍然能读回原始 SF 和 preamble 数量。
    """
    parts = [part for part in re.split(r"[_-]", Path(path).stem) if part != ""]
    keys = ["experiment_id", "corridor_id", "position_id", "sf", "tx_power_dbm", "preamble_len"]
    metadata: dict[str, Any] = {
        "filename": Path(path).name,
        "filename_parts": parts,
        "parsed": False,
        "experiment_id": "",
        "corridor_id": "",
        "position_id": "",
        "tx_power_dbm": "",
        "filename_sf": "",
        "filename_tx_power_dbm": "",
        "filename_preamble_len": "",
    }
    if len(parts) < 6:
        return metadata

    for key, value in zip(keys, parts[:6]):
        parsed = parse_capture_metadata_value(value)
        if key == "sf":
            metadata["filename_sf"] = parsed
        elif key == "tx_power_dbm":
            metadata["tx_power_dbm"] = parsed
            metadata["filename_tx_power_dbm"] = parsed
        elif key == "preamble_len":
            metadata["filename_preamble_len"] = parsed
        else:
            metadata[key] = parsed
    metadata["parsed"] = isinstance(metadata["filename_sf"], int) and isinstance(
        metadata["filename_preamble_len"], int
    )
    return metadata


def resolve_capture_parameters(args: argparse.Namespace, input_path: Path) -> dict[str, Any]:
    """把命令行参数和文件名参数合并成最终给 gr-lora_sdr 使用的配置。"""
    metadata = parse_capture_metadata(input_path)

    if args.sf is None:
        filename_sf = metadata.get("filename_sf", "")
        if not isinstance(filename_sf, int):
            raise ValueError(
                f"Cannot infer SF from filename {input_path.name}; pass --sf explicitly."
            )
        args.sf = int(filename_sf)
        sf_source = "filename"
    else:
        args.sf = int(args.sf)
        sf_source = "cli"

    if args.preamble_len is None:
        filename_preamble_len = metadata.get("filename_preamble_len", "")
        if not isinstance(filename_preamble_len, int):
            raise ValueError(
                f"Cannot infer preamble length from filename {input_path.name}; "
                "pass --preamble-len explicitly."
            )
        args.preamble_len = int(filename_preamble_len)
        preamble_source = "filename"
    else:
        args.preamble_len = int(args.preamble_len)
        preamble_source = "cli"

    metadata.update(
        {
            "resolved_sf": int(args.sf),
            "resolved_sf_source": sf_source,
            "resolved_preamble_len": int(args.preamble_len),
            "resolved_preamble_len_source": preamble_source,
        }
    )
    args.capture_metadata = metadata
    return metadata


def build_noise_power_db_values(args: argparse.Namespace) -> list[float]:
    """生成要测试的加噪功率步进，单位是相对参考功率的 dB。"""
    if args.noise_power_db is not None:
        values = [float(value) for value in args.noise_power_db]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("--noise-power-db values must be finite numbers.")
        return values

    start = float(args.noise_start_db)
    stop = float(args.noise_stop_db)
    step = float(args.noise_step_db)
    if not all(math.isfinite(value) for value in (start, stop, step)):
        raise ValueError("--noise-start-db/--noise-stop-db/--noise-step-db must be finite numbers.")
    if step == 0.0:
        raise ValueError("--noise-step-db must not be 0.")
    if (stop - start) * step < 0.0:
        raise ValueError("--noise-step-db sign must move from --noise-start-db toward --noise-stop-db.")

    values = []
    current = start
    epsilon = abs(step) * 1e-9
    if step > 0.0:
        while current <= stop + epsilon:
            values.append(round(current, 10))
            current += step
    else:
        while current >= stop - epsilon:
            values.append(round(current, 10))
            current += step
    if not values:
        raise ValueError("No noise power steps were generated.")
    return values


def validate_capture_args(args: argparse.Namespace) -> None:
    """检查会直接影响 GNU Radio 流图和文件写出的基础参数。"""
    if not 5 <= int(args.sf) <= 12:
        raise ValueError(f"LoRa SF must be in [5, 12], got {args.sf}.")
    if int(args.preamble_len) <= 0:
        raise ValueError(f"--preamble-len must be positive, got {args.preamble_len}.")
    if float(args.bw) <= 0.0 or float(args.samp_rate) <= 0.0:
        raise ValueError("--bw and --samp-rate must be positive.")
    os_factor = float(args.samp_rate) / float(args.bw)
    rounded_os_factor = round(os_factor)
    if rounded_os_factor <= 0 or not math.isclose(os_factor, rounded_os_factor, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "--samp-rate must be an integer multiple of --bw for gr-lora_sdr "
            f"(got samp_rate/bw={os_factor:.9g})."
        )
    if args.block_samples <= 0 or args.chunk_samples <= 0:
        raise ValueError("--block-samples and --chunk-samples must be positive.")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise ValueError("--sample-limit must be positive when provided.")


def load_complex64_memmap(path: Path) -> np.memmap:
    """用 memmap 打开大 IQ 文件，避免一次性把几百 MB 数据全部读进内存。"""
    size_bytes = path.stat().st_size
    if size_bytes % COMPLEX64_BYTES != 0:
        raise ValueError(
            f"{path} size {size_bytes} is not divisible by {COMPLEX64_BYTES}; "
            "expected raw complex64 IQ samples."
        )
    return np.memmap(path, dtype=np.complex64, mode="r")


def iter_chunks(
    samples: np.ndarray,
    chunk_samples: int,
    sample_limit: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """按固定采样点数切块，后续估计功率和写文件都走这个迭代器。"""
    n_samples = samples.size if sample_limit is None else min(samples.size, sample_limit)
    for start in range(0, n_samples, chunk_samples):
        stop = min(start + chunk_samples, n_samples)
        yield start, samples[start:stop]


def mean_power(samples: np.ndarray) -> float:
    """计算复 IQ 平均功率 E[|I+jQ|^2] = E[I^2 + Q^2]。"""
    if samples.size == 0:
        raise ValueError("Cannot estimate power from an empty sample range.")
    real = samples.real.astype(np.float64, copy=False)
    imag = samples.imag.astype(np.float64, copy=False)
    return float(np.mean(real * real + imag * imag, dtype=np.float64))


def sum_power(samples: np.ndarray) -> tuple[float, int]:
    """返回一段复 IQ 的功率和以及采样点数，方便后面做加权平均。"""
    if samples.size == 0:
        return 0.0, 0
    real = samples.real.astype(np.float64, copy=False)
    imag = samples.imag.astype(np.float64, copy=False)
    return float(np.sum(real * real + imag * imag, dtype=np.float64)), int(samples.size)


def mean_power_chunked(
    samples: np.ndarray,
    chunk_samples: int,
    sample_limit: int | None = None,
) -> float:
    """分块计算整段平均功率，避免大文件上产生额外的大数组副本。"""
    total_power = 0.0
    total_count = 0
    for _, chunk in iter_chunks(samples, chunk_samples, sample_limit):
        real = chunk.real.astype(np.float64, copy=False)
        imag = chunk.imag.astype(np.float64, copy=False)
        total_power += float(np.sum(real * real + imag * imag, dtype=np.float64))
        total_count += chunk.size
    if total_count == 0:
        raise ValueError("No samples available for power estimation.")
    return total_power / total_count


def estimate_block_powers(
    samples: np.ndarray,
    block_samples: int,
    sample_limit: int | None = None,
) -> np.ndarray:
    """把整段 IQ 切成多个块，并计算每个块的平均功率。"""
    n_samples = samples.size if sample_limit is None else min(samples.size, sample_limit)
    powers: list[float] = []
    for start in range(0, n_samples, block_samples):
        stop = min(start + block_samples, n_samples)
        if stop > start:
            powers.append(mean_power(samples[start:stop]))
    if not powers:
        raise ValueError("No blocks available for power estimation.")
    return np.asarray(powers, dtype=np.float64)


def prepare_file_source_path(input_file: Path) -> Path:
    """为 GNU Radio file_source 创建 ASCII-only 硬链接。

    Windows 下 GNU Radio C++ file_source 对中文路径不稳定；Python 能直接读中文路径，
    但 GNU Radio 侧最好给它一个纯 ASCII 的临时 hardlink。hardlink 必须在同一盘符，
    所以 staging 目录放在 weakPacket_decoding 下面。
    """
    source_path = Path(input_file).resolve()
    FILE_SOURCE_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()
    staged_path = FILE_SOURCE_STAGING_DIR / f"{digest}{source_path.suffix.lower()}"

    if staged_path.exists():
        try:
            if staged_path.samefile(source_path):
                return staged_path
        except OSError:
            pass
        staged_path.unlink()

    try:
        os.link(source_path, staged_path)
    except OSError as exc:
        raise RuntimeError(
            f"failed to create ASCII hardlink for GNU Radio file_source: {source_path}"
        ) from exc
    return staged_path


def cleanup_file_source_path(path: Path) -> None:
    """清理 GNU Radio file_source 的临时 hardlink。"""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def resolve_ldro(sf: int, bw: float, ldro_mode: int) -> int:
    """解析低数据率优化开关：0 关闭，1 开启，2 按 symbol time 自动判断。"""
    if int(ldro_mode) == 0:
        return 0
    if int(ldro_mode) == 1:
        return 1
    return 1 if ((1 << int(sf)) / float(bw)) >= 0.016 else 0


def lora_payload_symbol_count(
    sf: int,
    bw: float,
    cr: int,
    payload_len: int,
    has_crc: bool,
    impl_head: bool,
    ldro_mode: int,
) -> int:
    """按 LoRa airtime 公式估计 PHY header 之后的 payload 部分 symbol 数。

    这里的返回值包含 LoRa 公式中的固定 8 个 payload symbol 以及编码块展开后的
    symbol 数。整包长度后面再加 preamble_len + 4.25 个 preamble/sync/SFD symbol。
    """
    sf = int(sf)
    cr = int(cr)
    payload_len = max(0, int(payload_len))
    crc = 1 if has_crc else 0
    ih = 1 if impl_head else 0
    de = resolve_ldro(sf, bw, ldro_mode)
    denominator = 4 * max(1, sf - 2 * de)
    numerator = 8 * payload_len - 4 * sf + 28 + 16 * crc - 20 * ih
    coded_blocks = max(math.ceil(numerator / denominator), 0)
    return 8 + coded_blocks * (cr + 4)


def estimate_packet_range(iq_size: int, frame: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """从 frame_sync 的 header 对齐范围推导整包样本范围。

    frame_sync 发布的 start/end 只覆盖 preamble + sync word + SFD，也就是
    payload 之前的非 payload 片段。完整 packet_end 需要结合 header_decoder
    解出来的 pay_len、CR、CRC、LDRO，再用 LoRa airtime 公式估算。
    """
    sf = int(frame.get("sf", args.sf))
    bw = float(frame.get("bw", args.bw))
    samples_per_symbol = int(frame.get("samples_per_symbol", (1 << sf) * int(round(args.samp_rate / args.bw))))

    pay_len = int(frame.get("pay_len", args.pay_len))
    if pay_len < 0:
        pay_len = int(args.pay_len)
    cr = int(frame.get("cr", args.cr))
    if cr < 1:
        cr = int(args.cr)
    has_crc = bool(int(frame.get("crc", int(args.has_crc))))
    ldro_mode = int(frame.get("ldro_mode", args.ldro_mode))

    payload_symbols = lora_payload_symbol_count(
        sf,
        bw,
        cr,
        pay_len,
        has_crc,
        args.impl_head,
        ldro_mode,
    )
    packet_symbols = float(frame.get("preamble_len", args.preamble_len)) + 4.25 + float(payload_symbols)
    packet_start = max(0, int(frame["start_sample"]))
    preamble_end = max(packet_start, int(frame["end_sample"]))
    packet_end = packet_start + int(math.ceil(packet_symbols * samples_per_symbol))
    packet_end = max(preamble_end, min(int(iq_size), packet_end))

    result = dict(frame)
    result.update(
        {
            "payload_symbols": int(payload_symbols),
            "packet_symbols": float(packet_symbols),
            "packet_start_sample": int(packet_start),
            "packet_end_sample": int(packet_end),
            "packet_samples": int(max(0, packet_end - packet_start)),
        }
    )
    return result


def normalize_ranges(ranges: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    """裁剪并合并重叠区间，用于包外噪声估计。"""
    clipped = []
    for start, end in ranges:
        start = max(0, min(int(start), int(limit)))
        end = max(start, min(int(end), int(limit)))
        if end > start:
            clipped.append((start, end))
    if not clipped:
        return []

    clipped.sort()
    merged = [clipped[0]]
    for start, end in clipped[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def mean_power_outside_ranges(
    samples: np.ndarray,
    ranges: list[tuple[int, int]],
    sample_limit: int | None,
) -> tuple[float, int]:
    """计算所有 packet 区间之外的平均功率，作为已有底噪估计。"""
    limit = samples.size if sample_limit is None else min(samples.size, int(sample_limit))
    merged = normalize_ranges(ranges, limit)
    total_sum = 0.0
    total_count = 0
    cursor = 0
    for start, end in merged:
        if start > cursor:
            part_sum, part_count = sum_power(samples[cursor:start])
            total_sum += part_sum
            total_count += part_count
        cursor = max(cursor, end)
    if cursor < limit:
        part_sum, part_count = sum_power(samples[cursor:limit])
        total_sum += part_sum
        total_count += part_count
    if total_count == 0:
        return float("nan"), 0
    return total_sum / total_count, total_count


def summarize_values(values: list[float]) -> dict[str, float | int]:
    """输出 mean/median/std/min/max，metadata 和终端打印都会用到。"""
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def expected_payloads_from_args(args: argparse.Namespace) -> list[bytes]:
    """Resolve the groundtruth payload list used to judge decode correctness."""
    if getattr(args, "no_expected_payload_check", False):
        return []
    values = getattr(args, "expected_payload_hex", None) or DEFAULT_EXPECTED_PAYLOAD_HEXES
    return [parse_payload_hex(value) for value in values]


def compare_payloads(
    decoded_packets: list[dict[str, Any]],
    expected_payloads: list[bytes],
) -> dict[str, Any]:
    """Compare decoded payloads against the expected clean-packet groundtruth."""
    decoded: list[bytes | None] = []
    for packet in decoded_packets:
        if packet.get("payload_decode_error"):
            decoded.append(None)
            continue
        decoded.append(parse_payload_hex(str(packet.get("decoded_payload_hex", ""))))
    unmatched_expected = list(range(len(expected_payloads)))
    matches = []
    wrong_decoded_indexes = []

    for decoded_index, payload in enumerate(decoded):
        match_pos = None
        if payload is not None:
            for expected_index in unmatched_expected:
                if payload == expected_payloads[expected_index]:
                    match_pos = expected_index
                    break
        if match_pos is None:
            wrong_decoded_indexes.append(decoded_index)
            continue
        unmatched_expected.remove(match_pos)
        matches.append(
            {
                "decoded_index": int(decoded_index),
                "expected_index": int(match_pos),
                "payload_hex": payload.hex(),
            }
        )

    crc_valid_count = sum(1 for packet in decoded_packets if bool(packet.get("crc_valid", False)))
    correct_count = len(matches)
    wrong_count = len(wrong_decoded_indexes)
    missed_correct = max(0, len(expected_payloads) - correct_count)

    return {
        "expected_packet_count": int(len(expected_payloads)),
        "decoded_payload_count": int(len(decoded)),
        "crc_valid_packets": int(crc_valid_count),
        "crc_invalid_packets": int(max(0, len(decoded_packets) - crc_valid_count)),
        "correct_payload_packets": int(correct_count),
        "wrong_payload_packets": int(wrong_count),
        "missed_correct_payload_packets": int(missed_correct),
        "all_expected_payloads_correct": bool(
            len(expected_payloads) > 0 and correct_count == len(expected_payloads) and wrong_count == 0
        ),
        "matches": matches,
        "unmatched_expected_indexes": [int(index) for index in unmatched_expected],
        "wrong_decoded_indexes": [int(index) for index in wrong_decoded_indexes],
    }


def merge_detector_frames(
    frames: list[dict[str, Any]],
    headers: list[dict[str, Any]],
    payloads: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """按 frame_count 合并 preamble message 和 header message。

    当前 frame_sync 发布 preamble message 时通常已经带上 pay_len/cr/crc 等 header 字段；
    这里再合并一次 header_sink，是为了兼容后续本地代码变动或旧版消息字段缺失。
    """
    headers_by_id = {
        int(header["frame_count"]): header
        for header in headers
        if int(header.get("frame_count", -1)) >= 0 and int(header.get("header_err", 1)) == 0
    }
    merged = []
    for frame in frames:
        item = dict(frame)
        header = headers_by_id.get(int(frame.get("frame_count", -1)))
        if header:
            item.update(header)
        merged.append(item)

    if payloads:
        payloads_by_id = {
            int(payload["frame_count"]): payload
            for payload in payloads
            if int(payload.get("frame_count", -1)) >= 0
        }
        used_payload_indexes: set[int] = set()
        for index, item in enumerate(merged):
            payload = payloads_by_id.get(int(item.get("frame_count", -1)))
            if payload is None and index < len(payloads):
                payload = payloads[index]
                used_payload_indexes.add(index)
            if payload is not None:
                item.update(payload)
        if len(payloads) > len(merged):
            for index, payload in enumerate(payloads):
                if index in used_payload_indexes:
                    continue
                if int(payload.get("frame_count", -1)) in {
                    int(item.get("frame_count", -2)) for item in merged
                }:
                    continue
                merged.append(dict(payload))
    return merged


def run_grlora_packet_detector(input_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """运行 gr-lora_sdr 接收链，返回每个 header-valid 包的整包样本范围。

    这个函数只在 --power-mode packet 或加噪后实测 SNR 时调用，因此 GNU Radio
    依赖是惰性导入的。用普通 Python 只做 dry-run 的 total/window 时，不需要安装 gnuradio。
    """
    try:
        from gnuradio import blocks, gr
        import gnuradio.lora_sdr as lora_sdr
        import pmt
    except ImportError as exc:
        raise RuntimeError(
            "gr-lora_sdr SNR measurement requires GNU Radio and gr-lora_sdr. "
            "Run this script in the gr-lora conda environment, or use --dry-run if you only need planned outputs."
        ) from exc

    class PreambleMetadataSink(gr.basic_block):
        """收集 frame_sync 发布的 preamble/sync/SFD 对齐样本范围和 gr-lora_sdr SNR。"""

        def __init__(self):
            gr.basic_block.__init__(self, name="noisy_iq_preamble_sink", in_sig=None, out_sig=None)
            self.frames: list[dict[str, Any]] = []
            self._lock = threading.Lock()
            self.message_port_register_in(pmt.intern("preamble"))
            self.set_msg_handler(pmt.intern("preamble"), self.handle_preamble)

        def _dict_value(self, msg, key, default=None):
            value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
            if pmt.is_null(value):
                return default
            try:
                return pmt.to_python(value)
            except Exception:
                return default

        def handle_preamble(self, msg):
            if not pmt.is_dict(msg):
                return
            start_sample = self._dict_value(msg, "start_sample", None)
            end_sample = self._dict_value(msg, "end_sample", None)
            if start_sample is None or end_sample is None:
                return
            frame = {
                "frame_count": int(self._dict_value(msg, "frame_count", 0)),
                "sf": int(self._dict_value(msg, "sf", args.sf)),
                "bw": float(self._dict_value(msg, "bw", args.bw)),
                "sample_rate": float(self._dict_value(msg, "sample_rate", args.samp_rate)),
                "samples_per_symbol": int(
                    self._dict_value(msg, "samples_per_symbol", (1 << int(args.sf)) * int(round(args.samp_rate / args.bw)))
                ),
                "preamble_len": int(self._dict_value(msg, "preamble_len", args.preamble_len)),
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "n_samples": int(self._dict_value(msg, "n_samples", int(end_sample) - int(start_sample))),
                "n_symbols": float(self._dict_value(msg, "n_symbols", 0.0)),
                "grlora_snr_db": float(self._dict_value(msg, "snr_db", float("nan"))),
                "cfo": float(self._dict_value(msg, "cfo", float("nan"))),
                "sto": float(self._dict_value(msg, "sto", float("nan"))),
                "sfo": float(self._dict_value(msg, "sfo", float("nan"))),
                "netid1": int(self._dict_value(msg, "netid1", -1)),
                "netid2": int(self._dict_value(msg, "netid2", -1)),
            }
            for source_key, output_key in (
                ("cr", "cr"),
                ("pay_len", "pay_len"),
                ("crc", "crc"),
                ("ldro_mode", "ldro_mode"),
                ("err", "header_err"),
            ):
                value = self._dict_value(msg, source_key, None)
                if value is not None:
                    frame[output_key] = int(value)
            with self._lock:
                self.frames.append(frame)

    class HeaderMetadataSink(gr.basic_block):
        """收集 header_decoder 输出的 PHY header 元数据。"""

        def __init__(self):
            gr.basic_block.__init__(self, name="noisy_iq_header_sink", in_sig=None, out_sig=None)
            self.headers: list[dict[str, Any]] = []
            self._lock = threading.Lock()
            self.message_port_register_in(pmt.intern("frame_info"))
            self.set_msg_handler(pmt.intern("frame_info"), self.handle_frame_info)

        def _dict_value(self, msg, key, default=None):
            value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
            if pmt.is_null(value):
                return default
            try:
                return pmt.to_python(value)
            except Exception:
                return default

        def handle_frame_info(self, msg):
            if not pmt.is_dict(msg):
                return
            header = {
                "frame_count": int(self._dict_value(msg, "frame_count", -1)),
                "cr": int(self._dict_value(msg, "cr", -1)),
                "pay_len": int(self._dict_value(msg, "pay_len", -1)),
                "crc": int(self._dict_value(msg, "crc", int(args.has_crc))),
                "ldro_mode": int(self._dict_value(msg, "ldro_mode", args.ldro_mode)),
                "header_err": int(self._dict_value(msg, "err", 1)),
            }
            for key in ("start_sample", "end_sample"):
                value = self._dict_value(msg, key, None)
                if value is not None:
                    header[key] = int(value)
            with self._lock:
                self.headers.append(header)

    class PayloadMetadataSink(gr.basic_block):
        """Collect crc_verif payload messages for correctness checks."""

        def __init__(self):
            gr.basic_block.__init__(self, name="noisy_iq_payload_sink", in_sig=None, out_sig=None)
            self.payloads: list[dict[str, Any]] = []
            self._lock = threading.Lock()
            self.message_port_register_in(pmt.intern("payload_metadata"))
            self.set_msg_handler(pmt.intern("payload_metadata"), self.handle_payload_metadata)

        def _dict_value(self, msg, key, default=None):
            value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
            if pmt.is_null(value):
                return default
            try:
                return pmt.to_python(value)
            except Exception:
                return default

        def _payload_to_bytes(self, payload: Any) -> bytes:
            if payload is None:
                return b""
            if isinstance(payload, bytes):
                return payload
            if isinstance(payload, bytearray):
                return bytes(payload)
            if isinstance(payload, str):
                return payload.encode("latin-1", errors="replace")
            if isinstance(payload, (list, tuple)):
                return bytes(int(item) & 0xFF for item in payload)
            return str(payload).encode("utf-8", errors="replace")

        def handle_payload_metadata(self, msg):
            if not pmt.is_dict(msg):
                return
            payload_pmt = pmt.dict_ref(msg, pmt.intern("payload"), pmt.PMT_NIL)
            payload_decode_error = ""
            try:
                payload_value = None if pmt.is_null(payload_pmt) else pmt.to_python(payload_pmt)
            except Exception as exc:
                payload_value = None
                payload_decode_error = f"{type(exc).__name__}: {exc}"
            payload_bytes = self._payload_to_bytes(payload_value)
            payload = {
                "frame_count": int(self._dict_value(msg, "frame_count", -1)),
                "decoded_payload_len": int(self._dict_value(msg, "decoded_payload_len", len(payload_bytes))),
                "decoded_payload_available": True,
                "decoded_payload_hex": payload_bytes.hex(),
                "decoded_payload_text": payload_bytes_to_text(payload_bytes),
                "crc_valid": bool(self._dict_value(msg, "crc_valid", False)),
            }
            if payload_decode_error:
                payload["payload_decode_error"] = payload_decode_error
            for key in (
                "cr",
                "pay_len",
                "crc",
                "ldro_mode",
                "err",
                "start_sample",
                "end_sample",
                "sf",
                "samples_per_symbol",
            ):
                value = self._dict_value(msg, key, None)
                if value is not None:
                    output_key = "header_err" if key == "err" else key
                    try:
                        payload[output_key] = int(value)
                    except (TypeError, ValueError):
                        payload[output_key] = value
            with self._lock:
                self.payloads.append(payload)

    class PacketDetectorTopBlock(gr.top_block):
        """最短 gr-lora_sdr 接收链：file_source -> frame_sync -> header_decoder。"""

        def __init__(self, file_source_path: Path):
            gr.top_block.__init__(self, "LoRa Packet Range Detector", catch_exceptions=True)
            os_factor = int(round(float(args.samp_rate) / float(args.bw)))
            min_buf = int(np.ceil(os_factor * ((1 << int(args.sf)) + 2)))

            self.file_source = blocks.file_source(gr.sizeof_gr_complex, str(file_source_path), False, 0, 0)
            self.file_source.set_min_output_buffer(min_buf)
            self.frame_sync = lora_sdr.frame_sync(
                int(args.center_freq),
                int(args.bw),
                int(args.sf),
                bool(args.impl_head),
                [int(args.sync_word)],
                os_factor,
                int(args.preamble_len),
            )
            self.fft_demod = lora_sdr.fft_demod(bool(args.soft_decoding), bool(args.print_grlora))
            self.gray_mapping = lora_sdr.gray_mapping(bool(args.soft_decoding))
            self.deinterleaver = lora_sdr.deinterleaver(bool(args.soft_decoding))
            self.hamming_dec = lora_sdr.hamming_dec(bool(args.soft_decoding))
            self.header_decoder = lora_sdr.header_decoder(
                bool(args.impl_head),
                int(args.cr),
                int(args.pay_len),
                bool(args.has_crc),
                int(args.ldro_mode),
                bool(args.print_header),
            )
            self.dewhitening = lora_sdr.dewhitening()
            crc_mode = lora_sdr.Crc_mode.SX1276 if int(args.crc_mode) == 1 else lora_sdr.Crc_mode.GRLORA
            self.crc_verif = lora_sdr.crc_verif(
                0,
                False,
                crc_mode,
            )
            self.preamble_sink = PreambleMetadataSink()
            self.header_sink = HeaderMetadataSink()
            self.payload_sink = PayloadMetadataSink()

            self.connect((self.file_source, 0), (self.frame_sync, 0))
            self.connect((self.frame_sync, 0), (self.fft_demod, 0))
            self.connect((self.fft_demod, 0), (self.gray_mapping, 0))
            self.connect((self.gray_mapping, 0), (self.deinterleaver, 0))
            self.connect((self.deinterleaver, 0), (self.hamming_dec, 0))
            self.connect((self.hamming_dec, 0), (self.header_decoder, 0))
            self.connect((self.header_decoder, 0), (self.dewhitening, 0))
            self.connect((self.dewhitening, 0), (self.crc_verif, 0))
            self.msg_connect((self.header_decoder, "frame_info"), (self.frame_sync, "frame_info"))
            self.msg_connect((self.header_decoder, "frame_info"), (self.header_sink, "frame_info"))
            self.msg_connect((self.frame_sync, "preamble"), (self.preamble_sink, "preamble"))
            self.msg_connect((self.crc_verif, "payload_metadata"), (self.payload_sink, "payload_metadata"))

    file_source_path = prepare_file_source_path(input_path)
    tb = None
    try:
        tb = PacketDetectorTopBlock(file_source_path)
        tb.start()
        tb.wait()
        frames = list(tb.preamble_sink.frames)
        headers = list(tb.header_sink.headers)
        payloads = list(tb.payload_sink.payloads)
    finally:
        tb = None
        cleanup_file_source_path(file_source_path)

    merged = merge_detector_frames(frames, headers, payloads)
    iq_size = load_complex64_memmap(input_path).size
    return [estimate_packet_range(iq_size, frame, args) for frame in merged]


def estimate_packet_power(samples: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    """用 gr-lora_sdr 对齐结果进行 packet-level 功率估计。"""
    packets = run_grlora_packet_detector(Path(args.input).resolve(), args)
    limit = samples.size if args.sample_limit is None else min(samples.size, int(args.sample_limit))
    packet_ranges = normalize_ranges(
        [(packet["packet_start_sample"], packet["packet_end_sample"]) for packet in packets],
        limit,
    )
    if not packet_ranges:
        raise ValueError(
            "gr-lora_sdr did not publish any valid packet ranges. "
            "Check --sf/--samp-rate/--bw/--sync-word/--preamble-len, or use --power-mode active/window."
        )

    packet_power_sum = 0.0
    packet_sample_count = 0
    packet_records = []
    for packet in packets:
        start = max(0, min(int(packet["packet_start_sample"]), limit))
        end = max(start, min(int(packet["packet_end_sample"]), limit))
        if end <= start:
            continue
        current_sum, current_count = sum_power(samples[start:end])
        current_power = current_sum / current_count if current_count else float("nan")
        packet_power_sum += current_sum
        packet_sample_count += current_count
        record = dict(packet)
        record["packet_start_sample"] = int(start)
        record["packet_end_sample"] = int(end)
        record["packet_samples"] = int(current_count)
        record["packet_mean_power"] = float(current_power)
        record["packet_mean_power_db"] = db10(float(current_power))
        packet_records.append(record)

    if packet_sample_count == 0:
        raise ValueError("Detected packet ranges are empty after applying --sample-limit.")

    packet_mean_power = packet_power_sum / packet_sample_count
    outside_power, outside_count = mean_power_outside_ranges(samples, packet_ranges, args.sample_limit)
    noise_power = 0.0 if args.ignore_existing_noise or not np.isfinite(outside_power) else outside_power
    signal_power = packet_mean_power if args.ignore_existing_noise else packet_mean_power - noise_power
    if signal_power <= 0.0:
        raise ValueError(
            "Packet-level signal power is non-positive after subtracting packet-outside noise. "
            "Try --ignore-existing-noise or inspect detected packet ranges."
        )

    grlora_snr_values = [float(packet.get("grlora_snr_db", float("nan"))) for packet in packet_records]
    packet_power_values = [float(packet.get("packet_mean_power", float("nan"))) for packet in packet_records]
    decoded_packets = [
        packet
        for packet in packet_records
        if packet.get("decoded_payload_available") or packet.get("decoded_payload_hex")
    ]
    expected_payloads = expected_payloads_from_args(args)
    payload_check = compare_payloads(decoded_packets, expected_payloads) if expected_payloads else {}
    if payload_check:
        payload_check["missed_detection_packets"] = int(
            max(0, payload_check["expected_packet_count"] - len(packet_records))
        )
    current_snr_db = db10(signal_power / noise_power) if noise_power > 0.0 else float("nan")

    return {
        "power_mode": "packet",
        "signal_power": float(signal_power),
        "signal_power_db": db10(float(signal_power)),
        "existing_noise_power": float(noise_power),
        "existing_noise_power_db": db10(float(noise_power)) if np.isfinite(noise_power) else float("nan"),
        "current_snr_db": float(current_snr_db),
        "packet_mean_power": float(packet_mean_power),
        "packet_mean_power_db": db10(float(packet_mean_power)),
        "packet_count": int(len(packet_records)),
        "packet_total_samples": int(packet_sample_count),
        "outside_noise_samples": int(outside_count),
        "packet_power_summary": summarize_values(packet_power_values),
        "grlora_snr_db_summary": summarize_values(grlora_snr_values),
        "payload_check": payload_check,
        "packet_ranges": packet_records,
    }


def measure_grlora_snr(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """对 IQ 文件跑一次 gr-lora_sdr，记录 frame_sync 给出的 preamble SNR。"""
    verify_args = argparse.Namespace(**vars(args))
    verify_args.input = path
    packets = run_grlora_packet_detector(path.resolve(), verify_args)
    snr_values = [float(packet.get("grlora_snr_db", float("nan"))) for packet in packets]
    decoded_packets = [
        packet
        for packet in packets
        if packet.get("decoded_payload_available") or packet.get("decoded_payload_hex")
    ]
    expected_payloads = expected_payloads_from_args(args)
    payload_check = compare_payloads(decoded_packets, expected_payloads) if expected_payloads else {}
    if payload_check:
        payload_check["missed_detection_packets"] = int(
            max(0, payload_check["expected_packet_count"] - len(packets))
        )
    summary = summarize_values(snr_values)
    return {
        "file": str(path.resolve()),
        "detected_packets": int(len(packets)),
        "decoded_payload_packets": int(len(decoded_packets)),
        "grlora_snr_db_summary": summary,
        "payload_check": payload_check,
        "packet_measurements": packets,
    }


def clean_measurement_from_packet_power(input_path: Path, power_info: dict[str, Any]) -> dict[str, Any]:
    """复用 --power-mode packet 已经跑过的原始文件检测结果。"""
    packets = power_info.get("packet_ranges", [])
    return {
        "file": str(input_path.resolve()),
        "detected_packets": int(power_info.get("packet_count", 0)),
        "decoded_payload_packets": int(
            sum(1 for packet in packets if packet.get("decoded_payload_available") or packet.get("decoded_payload_hex"))
        ),
        "grlora_snr_db_summary": power_info.get("grlora_snr_db_summary", summarize_values([])),
        "payload_check": power_info.get("payload_check", {}),
        "packet_measurements": power_info.get("packet_ranges", []),
    }


def estimate_reference_power(
    samples: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """估计用于加噪的“参考信号功率”。

    支持三种模式：
    - packet：先用 gr-lora_sdr 对齐每个包，再按完整 packet 窗口估计功率。最适合严谨 SNR sweep。
    - total：直接把全文件均方功率当作信号功率，适合文件几乎全是连续发射信号的情况。
    - window：用户手动指定一个已知包所在的采样窗口，用该窗口估计信号功率。
    - active：默认模式。先估计低分位块功率作为底噪，再把明显高于底噪的块当作发包区间。

    packet 模式依赖 gr-lora_sdr 能在原始高 SNR 文件中正常检测并解析 PHY header。
    如果只想快速粗略加噪，active 仍然可用；但它不是整包级功率估计。
    """
    total_power = mean_power_chunked(samples, args.chunk_samples, args.sample_limit)

    if args.power_mode == "packet":
        packet_power = estimate_packet_power(samples, args)
        packet_power.update(
            {
                "total_power": float(total_power),
                "total_power_db": db10(float(total_power)),
                # packet 模式不使用 block active 统计，保留这些字段方便 metadata schema 稳定。
                "noise_floor_power": float(packet_power["existing_noise_power"]),
                "noise_floor_power_db": packet_power["existing_noise_power_db"],
                "active_mean_power": float(packet_power["packet_mean_power"]),
                "active_mean_power_db": packet_power["packet_mean_power_db"],
                "active_blocks": 0,
                "total_blocks": 0,
            }
        )
        return packet_power

    if args.power_mode == "total":
        # total 模式不区分包和空闲间隔，最简单，也最不依赖检测假设。
        signal_power = total_power
        noise_power = 0.0 if args.ignore_existing_noise else float("nan")
        current_snr_db = float("nan")
        active_blocks = 0
        total_blocks = 0
        noise_floor = float("nan")
        active_mean_power = total_power
    elif args.power_mode == "window":
        if args.signal_start is None or args.signal_samples is None:
            raise ValueError("--power-mode window requires --signal-start and --signal-samples.")
        start = max(0, int(args.signal_start))
        stop = min(samples.size, start + int(args.signal_samples))
        if stop <= start:
            raise ValueError("The requested signal window is empty.")
        # window 模式仍然用全局低分位块功率估计已有底噪，再从窗口功率中扣除。
        block_powers = estimate_block_powers(samples, args.block_samples, args.sample_limit)
        noise_floor = float(np.percentile(block_powers, args.noise_percentile))
        active_mean_power = mean_power(samples[start:stop])
        noise_power = 0.0 if args.ignore_existing_noise else noise_floor
        signal_power = active_mean_power if args.ignore_existing_noise else active_mean_power - noise_power
        current_snr_db = db10(signal_power / noise_power) if noise_power > 0.0 else float("nan")
        active_blocks = 1
        total_blocks = int(block_powers.size)
    else:
        # active 模式的核心假设：真实包所在块的均方功率会明显高于静默/底噪块。
        block_powers = estimate_block_powers(samples, args.block_samples, args.sample_limit)
        # 低分位数比最小值稳健一些，避免单个全零块或异常块把底噪估计拉歪。
        noise_floor = float(np.percentile(block_powers, args.noise_percentile))
        threshold = noise_floor * (10.0 ** (args.active_threshold_db / 10.0))
        active = block_powers > threshold
        active_blocks = int(np.count_nonzero(active))
        total_blocks = int(block_powers.size)
        if active_blocks == 0:
            # 如果没找到活跃块，就退回到全文件功率，避免脚本直接不可用。
            active_mean_power = total_power
            signal_power = total_power if args.ignore_existing_noise else max(total_power - noise_floor, 0.0)
        else:
            active_mean_power = float(np.mean(block_powers[active], dtype=np.float64))
            signal_power = active_mean_power if args.ignore_existing_noise else active_mean_power - noise_floor
        noise_power = 0.0 if args.ignore_existing_noise else noise_floor
        current_snr_db = db10(signal_power / noise_power) if noise_power > 0.0 else float("nan")

    if not np.isfinite(signal_power) or signal_power <= 0.0:
        raise ValueError(
            "Estimated non-positive signal power. Try --power-mode total, "
            "--ignore-existing-noise, or a manual --power-mode window."
        )

    return {
        "power_mode": args.power_mode,
        "total_power": float(total_power),
        "total_power_db": db10(float(total_power)),
        "signal_power": float(signal_power),
        "signal_power_db": db10(float(signal_power)),
        "existing_noise_power": float(noise_power),
        "existing_noise_power_db": db10(float(noise_power)) if np.isfinite(noise_power) else float("nan"),
        "current_snr_db": float(current_snr_db),
        "noise_floor_power": float(noise_floor),
        "noise_floor_power_db": db10(float(noise_floor)) if np.isfinite(noise_floor) else float("nan"),
        "active_mean_power": float(active_mean_power),
        "active_mean_power_db": db10(float(active_mean_power)),
        "active_blocks": int(active_blocks),
        "total_blocks": int(total_blocks),
    }


def generate_noisy_file(
    samples: np.ndarray,
    out_path: Path,
    add_noise_power: float,
    seed: int,
    chunk_samples: int,
    sample_limit: int | None,
    overwrite: bool,
) -> None:
    """把加噪后的 IQ 流式写入文件。

    这里按块读取、按块生成噪声、按块写出，主要是为了处理几百 MB 甚至更大的 IQ 文件。
    """
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 先写 .tmp，全部成功后再原子替换，避免中途失败留下半个输出文件。
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    rng = np.random.default_rng(seed)
    # 复高斯噪声 n = n_i + j*n_q，总功率为 E[|n|^2]。
    # 因为 I/Q 两路独立同方差，所以每路方差 = 总噪声功率 / 2。
    sigma = math.sqrt(add_noise_power / 2.0) if add_noise_power > 0.0 else 0.0

    with tmp_path.open("wb") as handle:
        for _, chunk in iter_chunks(samples, chunk_samples, sample_limit):
            iq = np.asarray(chunk, dtype=np.complex64)
            if sigma > 0.0:
                noise_i = rng.normal(0.0, sigma, size=iq.size).astype(np.float32)
                noise_q = rng.normal(0.0, sigma, size=iq.size).astype(np.float32)
                iq = (iq + (noise_i + 1j * noise_q)).astype(np.complex64, copy=False)
            iq.tofile(handle)

    os.replace(tmp_path, out_path)


def json_safe(value):
    """把 NaN/Inf 等非标准 JSON 数值转成 null，方便后续用任意 JSON 工具读取。"""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_metadata(path: Path, metadata: dict) -> None:
    """写出同名 metadata，记录加噪功率、实测 SNR、随机种子和生成参数。"""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metadata), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def csv_value(value: Any) -> Any:
    """把 summary CSV 中不适合直接写出的值转成空字符串或短文本。"""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return f"{value:.9g}" if math.isfinite(value) else ""
    if isinstance(value, Path):
        return str(value)
    return value


def append_measurement_record(
    records: list[dict[str, Any]],
    *,
    kind: str,
    step_index: int,
    source_file: Path,
    output_file: Path,
    noise_power_db_relative: float | None,
    added_noise_power: float,
    seed: int | None,
    measurement: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """把一次 gr-lora_sdr SNR 实测结果压平成 summary 行。"""
    summary = measurement.get("grlora_snr_db_summary", {})
    payload_check = measurement.get("payload_check", {}) or {}
    records.append(
        {
            "kind": kind,
            "step_index": step_index,
            "source_file": str(source_file),
            "output_file": str(output_file),
            "noise_power_db_relative": noise_power_db_relative,
            "added_noise_power": float(added_noise_power),
            "seed": "" if seed is None else int(seed),
            "detected_packets": int(measurement.get("detected_packets", 0)),
            "decoded_payload_packets": int(measurement.get("decoded_payload_packets", 0)),
            "expected_packet_count": int(payload_check.get("expected_packet_count", 0)),
            "crc_valid_packets": int(payload_check.get("crc_valid_packets", 0)),
            "crc_invalid_packets": int(payload_check.get("crc_invalid_packets", 0)),
            "correct_payload_packets": int(payload_check.get("correct_payload_packets", 0)),
            "wrong_payload_packets": int(payload_check.get("wrong_payload_packets", 0)),
            "missed_detection_packets": int(payload_check.get("missed_detection_packets", 0)),
            "missed_correct_payload_packets": int(payload_check.get("missed_correct_payload_packets", 0)),
            "all_expected_payloads_correct": bool(payload_check.get("all_expected_payloads_correct", False)),
            "grlora_snr_count": summary.get("count", 0),
            "grlora_snr_mean": summary.get("mean", float("nan")),
            "grlora_snr_median": summary.get("median", float("nan")),
            "grlora_snr_std": summary.get("std", float("nan")),
            "grlora_snr_min": summary.get("min", float("nan")),
            "grlora_snr_max": summary.get("max", float("nan")),
            "sf": int(args.sf),
            "preamble_len": int(args.preamble_len),
            "samp_rate": float(args.samp_rate),
            "bw": float(args.bw),
            "sync_word": f"0x{int(args.sync_word):02x}",
        }
    )


def write_sweep_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """写出便于 Excel / pandas 查看的一行一步 summary。"""
    fieldnames = [
        "kind",
        "step_index",
        "source_file",
        "output_file",
        "noise_power_db_relative",
        "added_noise_power",
        "seed",
        "detected_packets",
        "decoded_payload_packets",
        "expected_packet_count",
        "crc_valid_packets",
        "crc_invalid_packets",
        "correct_payload_packets",
        "wrong_payload_packets",
        "missed_detection_packets",
        "missed_correct_payload_packets",
        "all_expected_payloads_correct",
        "grlora_snr_count",
        "grlora_snr_mean",
        "grlora_snr_median",
        "grlora_snr_std",
        "grlora_snr_min",
        "grlora_snr_max",
        "sf",
        "preamble_len",
        "samp_rate",
        "bw",
        "sync_word",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: csv_value(record.get(key, "")) for key in fieldnames})


def planned_output_paths(input_path: Path, output_dir: Path, noise_power_db_values: list[float]) -> list[tuple[float, Path, Path]]:
    """提前计算每个加噪步对应的 bin/json 路径。"""
    outputs = []
    for noise_power_db in noise_power_db_values:
        label = db_to_label(float(noise_power_db))
        outputs.append(
            (
                float(noise_power_db),
                output_dir / f"{input_path.stem}_noise_rel_{label}dB.bin",
                output_dir / f"{input_path.stem}_noise_rel_{label}dB.json",
            )
        )
    return outputs


def check_output_collisions(outputs: list[tuple[float, Path, Path]], overwrite: bool, input_path: Path) -> None:
    """在运行 GNU Radio 测量前先发现输出冲突，避免长时间运行后才失败。"""
    for _, bin_path, meta_path in outputs:
        if bin_path.resolve() == input_path.resolve():
            raise ValueError(f"Refusing to overwrite the input file: {bin_path}")
        if not overwrite and (bin_path.exists() or meta_path.exists()):
            raise FileExistsError(
                f"{bin_path} or {meta_path} already exists; pass --overwrite to replace existing outputs."
            )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。默认从文件名读取 SF / preamble，并做粗步进加噪。"""
    parser = argparse.ArgumentParser(
        description=(
            "Add complex AWGN to raw complex64 LoRa IQ captures step by step, "
            "then measure each output with gr-lora_sdr."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input raw complex64 IQ file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: weakPacket_decoding/data/noisy_iq/<input-stem>",
    )
    parser.add_argument(
        "--noise-power-db",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Explicit added-noise powers in dB relative to the chosen reference power. "
            "This is NOT target SNR. If omitted, use --noise-start-db/--noise-stop-db/--noise-step-db."
        ),
    )
    parser.add_argument(
        "--noise-start-db",
        type=float,
        default=DEFAULT_NOISE_START_DB,
        help=f"First added-noise power in dB relative to the reference. Default: {DEFAULT_NOISE_START_DB}.",
    )
    parser.add_argument(
        "--noise-stop-db",
        type=float,
        default=DEFAULT_NOISE_STOP_DB,
        help=f"Last added-noise power in dB relative to the reference. Default: {DEFAULT_NOISE_STOP_DB}.",
    )
    parser.add_argument(
        "--noise-step-db",
        type=float,
        default=DEFAULT_NOISE_STEP_DB,
        help=f"Step size for added-noise power in dB. Default: {DEFAULT_NOISE_STEP_DB}.",
    )
    parser.add_argument(
        "--power-mode",
        choices=("packet", "active", "total", "window"),
        default="total",
        help=(
            "Reference used only to scale the added-noise steps. packet uses gr-lora_sdr aligned "
            "whole-packet ranges; active estimates packet blocks above the noise floor; total uses "
            "the whole-file mean power; window uses --signal-start/--signal-samples. Default: total."
        ),
    )
    parser.add_argument("--sf", type=int, default=None, help="LoRa spreading factor. Default: infer from filename.")
    parser.add_argument("--bw", type=float, default=125000.0, help="LoRa bandwidth in Hz. Default: 125000.")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ sample rate in Hz. Default: 500000.")
    parser.add_argument("--cr", type=int, default=1, help="LoRa coding-rate index used by gr-lora_sdr. Default: 1.")
    parser.add_argument(
        "--pay-len",
        type=int,
        default=255,
        help="Fallback payload length for implicit header or missing header metadata. Default: 255.",
    )
    parser.add_argument(
        "--has-crc",
        action="store_true",
        default=True,
        help="Packet has PHY CRC. Default: enabled.",
    )
    parser.add_argument(
        "--no-crc",
        action="store_false",
        dest="has_crc",
        help="Packet has no PHY CRC.",
    )
    parser.add_argument("--impl-head", action="store_true", default=False, help="Use implicit header mode.")
    parser.add_argument("--soft-decoding", action="store_true", default=False, help="Enable gr-lora_sdr soft decoding.")
    parser.add_argument(
        "--center-freq",
        type=float,
        default=487.7e6,
        help="RF center frequency for gr-lora_sdr SFO estimation. Default: 487.7e6.",
    )
    parser.add_argument(
        "--sync-word",
        type=parse_int_auto,
        default=0x34,
        help="LoRa sync word, decimal or hex. Default: 0x34.",
    )
    parser.add_argument(
        "--preamble-len",
        type=int,
        default=None,
        help="Expected preamble upchirp count / frame_sync trigger parameter. Default: infer from filename.",
    )
    parser.add_argument("--ldro-mode", type=int, default=2, help="LDRO mode: 0 off, 1 on, 2 auto. Default: 2.")
    parser.add_argument(
        "--crc-mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="CRC algorithm mode for payload verification: 0=GRLORA, 1=SX1276. Default: 0.",
    )
    parser.add_argument(
        "--expected-payload-hex",
        type=str,
        nargs="+",
        default=None,
        help="Expected clean decoded payloads as hex strings. Default: hard-coded groundtruth for 0_0_0_10_14_8.bin.",
    )
    parser.add_argument(
        "--no-expected-payload-check",
        action="store_true",
        help="Disable groundtruth payload comparison and only measure detection/SNR.",
    )
    parser.add_argument(
        "--print-header",
        action="store_true",
        default=False,
        help="Let gr-lora_sdr header_decoder print decoded PHY headers while detecting packet ranges.",
    )
    parser.add_argument(
        "--print-grlora",
        action="store_true",
        default=False,
        help="Let gr-lora_sdr fft_demod print demodulator info while detecting packet ranges.",
    )
    parser.add_argument(
        "--noise-percentile",
        type=float,
        default=10.0,
        help="Percentile of block powers used as noise floor in active/window modes.",
    )
    parser.add_argument(
        "--active-threshold-db",
        type=float,
        default=6.0,
        help="A block is active when its power is this many dB above the estimated noise floor.",
    )
    parser.add_argument(
        "--block-samples",
        type=int,
        default=32768,
        help="Block size for active/noise-floor power estimation.",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=1_000_000,
        help="Samples per streaming write chunk.",
    )
    parser.add_argument("--signal-start", type=int, default=None, help="Start sample for --power-mode window.")
    parser.add_argument("--signal-samples", type=int, default=None, help="Number of samples for --power-mode window.")
    parser.add_argument(
        "--ignore-existing-noise",
        action="store_true",
        help="In packet/active/window reference modes, do not subtract estimated existing noise from the reference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260503,
        help="Random seed. By default the same unit-noise realization is scaled for every noise step.",
    )
    parser.add_argument(
        "--independent-noise",
        action="store_true",
        help="Use a different random realization for each noise step.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Only process the first N samples. Mainly useful for smoke tests.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate powers and print planned outputs only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    args.input = input_path
    capture_metadata = resolve_capture_parameters(args, input_path)
    noise_power_db_values = build_noise_power_db_values(args)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_ROOT / input_path.stem).resolve()
    )
    validate_capture_args(args)
    outputs = planned_output_paths(input_path, output_dir, noise_power_db_values)
    if not args.dry_run:
        check_output_collisions(outputs, args.overwrite, input_path)

    samples = load_complex64_memmap(input_path)
    process_samples = samples.size if args.sample_limit is None else min(samples.size, args.sample_limit)
    # 先估计一次参考功率。这里只把它当作“加噪强度标尺”，不再推导目标 SNR。
    power_info = estimate_reference_power(samples, args)
    noise_reference_power = float(power_info["signal_power"])
    if not np.isfinite(noise_reference_power) or noise_reference_power <= 0.0:
        raise ValueError("Estimated non-positive reference power; cannot scale added noise.")

    print(f"Input: {input_path}")
    print(f"Samples: {samples.size} total, {process_samples} to process")
    print(f"Output directory: {output_dir}")
    print(
        "Resolved capture parameters: "
        f"sf={args.sf} ({capture_metadata['resolved_sf_source']}), "
        f"preamble_len={args.preamble_len} ({capture_metadata['resolved_preamble_len_source']}), "
        f"samp_rate={args.samp_rate:.0f}, bw={args.bw:.0f}, sync_word=0x{int(args.sync_word):02x}"
    )
    print(
        "Noise-step reference: "
        f"mode={args.power_mode}, power={noise_reference_power:.6e} ({db10(noise_reference_power):.2f} dB)"
    )
    if args.power_mode == "packet":
        grlora_snr = power_info.get("grlora_snr_db_summary", {})
        print(
            "Packet power estimate: "
            f"{power_info.get('packet_count', 0)} packet(s), "
            f"packet_mean={power_info.get('packet_mean_power', float('nan')):.6e} "
            f"({power_info.get('packet_mean_power_db', float('nan')):.2f} dB), "
            f"grlora_snr_median={grlora_snr.get('median', float('nan')):.2f} dB"
        )
    if power_info["total_blocks"]:
        print(f"Active blocks: {power_info['active_blocks']}/{power_info['total_blocks']}")

    print(
        "Noise steps (relative dB, not target SNR): "
        + ", ".join(format_db_value(value) for value in noise_power_db_values)
    )

    summary_records: list[dict[str, Any]] = []
    if not args.dry_run:
        clean_measurement = (
            clean_measurement_from_packet_power(input_path, power_info)
            if args.power_mode == "packet"
            else measure_grlora_snr(input_path, args)
        )
        clean_summary = clean_measurement["grlora_snr_db_summary"]
        append_measurement_record(
            summary_records,
            kind="clean",
            step_index=0,
            source_file=input_path,
            output_file=input_path,
            noise_power_db_relative=None,
            added_noise_power=0.0,
            seed=None,
            measurement=clean_measurement,
            args=args,
        )
        clean_payload = clean_measurement.get("payload_check", {}) or {}
        print(
            "[MEASURE] clean input: "
            f"detected={clean_measurement['detected_packets']}, "
            f"decoded={clean_measurement.get('decoded_payload_packets', 0)}, "
            f"correct={clean_payload.get('correct_payload_packets', 0)}/"
            f"{clean_payload.get('expected_packet_count', 0)}, "
            f"wrong={clean_payload.get('wrong_payload_packets', 0)}, "
            f"miss_detect={clean_payload.get('missed_detection_packets', 0)}, "
            f"grlora_snr_median={clean_summary['median']:.2f} dB"
        )

    for index, (noise_power_db, out_path, meta_path) in enumerate(outputs, start=1):
        # 默认情况下每个加噪步都复用同一个 seed。generate_noisy_file() 会从同一个随机数
        # 状态生成单位高斯噪声，只改变 sigma，因此各步噪声形态一致、幅度逐级增大。
        # 如果要每一步使用独立噪声实现，运行时加 --independent-noise。
        added_noise_power = noise_reference_power * (10.0 ** (float(noise_power_db) / 10.0))
        seed = args.seed + (index - 1) if args.independent_noise else args.seed
        # metadata 是后续复现实验的关键：保存加噪步进、实测 SNR 和当时的解码参数。
        metadata = {
            "input_file": str(input_path),
            "output_file": str(out_path),
            "format": "raw numpy.complex64 / GNU Radio gr_complex",
            "noise_power_db_relative": float(noise_power_db),
            "noise_reference_power": float(noise_reference_power),
            "noise_reference_power_db": db10(float(noise_reference_power)),
            "noise_reference_mode": args.power_mode,
            "added_noise_power": float(added_noise_power),
            "added_noise_sigma_per_iq_component": math.sqrt(added_noise_power / 2.0)
            if added_noise_power > 0.0
            else 0.0,
            "seed": int(seed),
            "sample_limit": args.sample_limit,
            "processed_samples": int(process_samples),
            "capture_metadata": capture_metadata,
            "args": {
                "power_mode": args.power_mode,
                "sf": args.sf,
                "bw": args.bw,
                "samp_rate": args.samp_rate,
                "cr": args.cr,
                "pay_len": args.pay_len,
                "has_crc": args.has_crc,
                "impl_head": args.impl_head,
                "center_freq": args.center_freq,
                "sync_word": args.sync_word,
                "preamble_len": args.preamble_len,
                "ldro_mode": args.ldro_mode,
                "crc_mode": args.crc_mode,
                "expected_payload_hex": args.expected_payload_hex,
                "no_expected_payload_check": args.no_expected_payload_check,
                "noise_percentile": args.noise_percentile,
                "active_threshold_db": args.active_threshold_db,
                "block_samples": args.block_samples,
                "chunk_samples": args.chunk_samples,
                "ignore_existing_noise": args.ignore_existing_noise,
                "independent_noise": args.independent_noise,
                "noise_power_db": args.noise_power_db,
                "noise_start_db": args.noise_start_db,
                "noise_stop_db": args.noise_stop_db,
                "noise_step_db": args.noise_step_db,
            },
            "power_estimate": power_info,
        }

        status = "DRY-RUN" if args.dry_run else "WRITE"
        print(
            f"[{status}] step {index:02d}, noise_rel={format_db_value(noise_power_db):>8} dB -> {out_path.name}; "
            f"add_noise_power={added_noise_power:.6e}"
        )
        if not args.dry_run:
            generate_noisy_file(
                samples,
                out_path,
                added_noise_power,
                seed,
                args.chunk_samples,
                args.sample_limit,
                args.overwrite,
            )
            measurement = measure_grlora_snr(out_path, args)
            metadata["grlora_snr_measurement"] = measurement
            summary = measurement["grlora_snr_db_summary"]
            payload_check = measurement.get("payload_check", {}) or {}
            append_measurement_record(
                summary_records,
                kind="noisy",
                step_index=index,
                source_file=input_path,
                output_file=out_path,
                noise_power_db_relative=float(noise_power_db),
                added_noise_power=float(added_noise_power),
                seed=seed,
                measurement=measurement,
                args=args,
            )
            print(
                "[MEASURE] "
                f"{out_path.name}: detected={measurement['detected_packets']}, "
                f"decoded={measurement.get('decoded_payload_packets', 0)}, "
                f"correct={payload_check.get('correct_payload_packets', 0)}/"
                f"{payload_check.get('expected_packet_count', 0)}, "
                f"wrong={payload_check.get('wrong_payload_packets', 0)}, "
                f"miss_detect={payload_check.get('missed_detection_packets', 0)}, "
                f"grlora_snr_median={summary['median']:.2f} dB"
            )
            write_metadata(meta_path, metadata)

    if not args.dry_run:
        summary_json = output_dir / f"{input_path.stem}_noise_sweep_summary.json"
        summary_csv = output_dir / f"{input_path.stem}_noise_sweep_summary.csv"
        write_metadata(
            summary_json,
            {
                "input_file": str(input_path),
                "output_dir": str(output_dir),
                "capture_metadata": capture_metadata,
                "noise_power_db_values": noise_power_db_values,
                "noise_reference_power": float(noise_reference_power),
                "noise_reference_power_db": db10(float(noise_reference_power)),
                "noise_reference_mode": args.power_mode,
                "records": summary_records,
            },
        )
        write_sweep_summary_csv(summary_csv, summary_records)
        print(f"Summary JSON: {summary_json}")
        print(f"Summary CSV:  {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
