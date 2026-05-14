# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0
"""CSV and NPZ feature export.

These outputs preserve the historical packet_features.csv and
preamble_features.npz schema while the plotting code lives separately.
"""

from pathlib import Path

import numpy as np

from .signal_analysis import (
    analyze_frame,
    compute_packet_average_metrics,
    estimate_packet_ranges,
    peak_spectrum_offsets,
)
from .utils import fmt_float, int_or_default, write_dict_csv


def build_packet_row(frame, metrics, args):
    """生成唯一 CSV 的一行：一个数据包一行，只保留核心平均特征和必要索引。"""
    row = {
        "file_name": frame["file_name"],
        "lab_name": frame.get("lab_name", ""),
        "experiment_id": frame.get("experiment_id", ""),
        "corridor_id": frame.get("corridor_id", ""),
        "position_id": frame.get("position_id", ""),
        "tx_power_dbm": frame.get("tx_power_dbm", ""),
        "filename_sf": frame.get("filename_sf", ""),
        "filename_tx_power_dbm": frame.get("filename_tx_power_dbm", ""),
        "filename_preamble_len": frame.get("filename_preamble_len", ""),
        "header_packet_counter": frame.get("header_packet_counter", frame.get("payload_packet_number", "")),
        "packet_avg_power_db": fmt_float(metrics["packet_avg_power_db"]),
        "preamble_peak_to_residual_db": fmt_float(metrics["preamble_peak_to_residual_db"]),
        "preamble_peak_width_3db_bins_avg": fmt_float(metrics["preamble_peak_width_bins_avg"]),
    }
    for offset, value in zip(peak_spectrum_offsets(args), metrics["preamble_peak_spectrum"]):
        row[f"preamble_peak_mag_bin_{int(offset):+d}"] = fmt_float(value)
    return row


def save_results(args, capture_results):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 旧版本会生成这些明细 CSV；现在输出收敛后主动清掉，避免 Excel 里误看旧文件。
    for stale_name in ("rssi_samples_5ms.csv", "preamble_symbol_features.csv", "position_summary.csv"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    packet_rows = []
    npz_frames = []
    packet_avg_power_db = []
    preamble_peak_to_residual_db = []
    preamble_peak_spectrum = []
    preamble_peak_width_bins_avg = []

    for capture_args, frames in capture_results:
        iq_path = Path(capture_args.input_file)
        iq = np.memmap(iq_path, dtype=np.complex64, mode="r")
        for frame in frames:
            analysis = analyze_frame(iq, frame, capture_args)
            ranges = estimate_packet_ranges(iq.size, frame, capture_args)
            metrics = compute_packet_average_metrics(analysis, ranges, iq, capture_args)
            npz_frames.append(frame)
            packet_rows.append(build_packet_row(frame, metrics, capture_args))
            packet_avg_power_db.append(metrics["packet_avg_power_db"])
            preamble_peak_to_residual_db.append(metrics["preamble_peak_to_residual_db"])
            preamble_peak_spectrum.append(metrics["preamble_peak_spectrum"])
            preamble_peak_width_bins_avg.append(metrics["preamble_peak_width_bins_avg"])

    packet_path = output_dir / "packet_features.csv"

    # packet_features.csv：Excel 友好版，一行代表一个数据包，只保留平均 IQ 功率、
    # 前导码主峰集中度、FHDR 包计数、文件名元数据和主峰附近平均幅度谱。
    packet_fields = [
        "file_name", "lab_name",
        "experiment_id", "corridor_id", "position_id", "tx_power_dbm",
        "filename_sf", "filename_tx_power_dbm", "filename_preamble_len", "header_packet_counter",
        "packet_avg_power_db", "preamble_peak_to_residual_db",
        "preamble_peak_width_3db_bins_avg",
    ]
    packet_fields.extend(f"preamble_peak_mag_bin_{int(offset):+d}" for offset in peak_spectrum_offsets(args))

    write_dict_csv(packet_path, packet_rows, packet_fields)

    npz_path = output_dir / "preamble_features.npz"
    # preamble_features.npz：Python 分析版。每个下标对应一个数据包，字段和 CSV 保持同一口径。
    # preamble_peak_spectrum[i] 是第 i 个包的主峰对齐平均局部幅度谱。
    np.savez_compressed(
        npz_path,
        file_names=np.asarray([f["file_name"] for f in npz_frames]),
        lab_names=np.asarray([f.get("lab_name", "") for f in npz_frames]),
        position_labels=np.asarray([str(f.get("position_id", "")) for f in npz_frames]),
        experiment_id=np.asarray(
            [int_or_default(f.get("experiment_id", ""), -1) for f in npz_frames],
            dtype=np.int32,
        ),
        corridor_id=np.asarray(
            [int_or_default(f.get("corridor_id", ""), -1) for f in npz_frames],
            dtype=np.int32,
        ),
        position_id=np.asarray(
            [int_or_default(f.get("position_id", ""), -1) for f in npz_frames],
            dtype=np.int32,
        ),
        tx_power_dbm=np.asarray(
            [int_or_default(f.get("tx_power_dbm", ""), -9999) for f in npz_frames],
            dtype=np.int32,
        ),
        filename_tx_power_dbm=np.asarray(
            [int_or_default(f.get("filename_tx_power_dbm", ""), -9999) for f in npz_frames],
            dtype=np.int32,
        ),
        filename_sf=np.asarray(
            [int_or_default(f.get("filename_sf", ""), -1) for f in npz_frames],
            dtype=np.int32,
        ),
        filename_preamble_len=np.asarray(
            [int_or_default(f.get("filename_preamble_len", ""), -1) for f in npz_frames],
            dtype=np.int32,
        ),
        header_packet_counter=np.asarray(
            [
                int_or_default(f.get("header_packet_counter", f.get("payload_packet_number", "")), -1)
                for f in npz_frames
            ],
            dtype=np.int32,
        ),
        packet_avg_power_db=np.asarray(packet_avg_power_db, dtype=np.float32),
        preamble_peak_to_residual_db=np.asarray(preamble_peak_to_residual_db, dtype=np.float32),
        preamble_peak_width_3db_bins_avg=np.asarray(preamble_peak_width_bins_avg, dtype=np.float32),
        preamble_peak_spectrum_offsets=peak_spectrum_offsets(args),
        preamble_peak_spectrum=np.asarray(preamble_peak_spectrum, dtype=np.float32),
        normalization=args.normalize,
        peak_width_db=args.peak_width_db,
        peak_spectrum_half_width=int(args.peak_spectrum_half_width),
    )
    return {
        "npz": npz_path,
        "packet": packet_path,
        "packet_count": len(packet_rows),
    }
