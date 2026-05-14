# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0
"""Input discovery and packet metadata merging.

This module turns file names/lab notes/GNU Radio metadata into one stable packet
record per detected packet.
"""

import copy
import re
from pathlib import Path

from .utils import int_or_default, prepare_file_source_path, read_text_file


def parse_capture_metadata_value(value):
    match = re.search(r"-?\d+", str(value))
    if match is None:
        return value
    return int(match.group(0))


def parse_capture_metadata(path):
    parts = re.split(r"[_-]", Path(path).stem)
    keys = ["experiment_id", "corridor_id", "position_id", "sf", "tx_power_dbm", "preamble_len"]
    metadata = {
        "lab_name": "",
        "lab_path": "",
        "lab_note": "",
        "experiment_id": "",
        "corridor_id": "",
        "position_id": "",
        "tx_power_dbm": "",
        "filename_sf": "",
        "filename_tx_power_dbm": "",
        "filename_preamble_len": "",
    }
    if len(parts) >= 6:
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
    return metadata


def parse_lab_note_overrides(note_text):
    """从 lab 文件夹的补充说明中解析会影响解码参数的修正项。"""
    overrides = {}
    if not note_text:
        return overrides

    sf_match = re.search(r"sf\s*其实是\s*(\d+)", note_text, flags=re.IGNORECASE)
    if sf_match is None:
        sf_match = re.search(r"(?:sf|spreading\s*factor)\s*(?:=|是|为|:)\s*(\d+)", note_text, flags=re.IGNORECASE)
    if sf_match is not None:
        overrides["sf"] = int(sf_match.group(1))

    return overrides


def load_lab_metadata(lab_dir):
    """读取一个 lab 文件夹的补充说明，并生成会写入输出结果的 lab 元数据。"""
    lab_dir = Path(lab_dir)
    note_paths = sorted(lab_dir.rglob("补充.txt"))
    note_text = "\n".join(read_text_file(path) for path in note_paths)
    overrides = parse_lab_note_overrides(note_text)
    return {
        "lab_name": lab_dir.name,
        "lab_path": str(lab_dir),
        "lab_note": note_text,
        "lab_note_path": ";".join(str(path) for path in note_paths),
        "override_sf": overrides.get("sf", ""),
    }


def default_output_dir():
    return Path(__file__).resolve().parents[1] / "preamble_fft"


def all_bin_root_output_dir(args, input_dir):
    """Use the lab/input folder itself for root-level --all-bin unless the user chose an output dir."""
    output_dir = Path(args.output_dir)
    if output_dir == default_output_dir():
        return Path(input_dir)
    return output_dir


def discover_lab_jobs(args):
    """按 USRP_IQ 下的 lab 子文件夹分组；每组单独输出到自己的 lab 文件夹。"""
    input_dir = Path(args.input_dir)
    jobs = []
    for lab_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        input_files = sorted(lab_dir.rglob("*.bin"))
        if not input_files:
            continue
        jobs.append(
            {
                "label": lab_dir.name,
                "input_files": input_files,
                "output_dir": lab_dir,
                "lab_metadata": load_lab_metadata(lab_dir),
            }
        )

    root_input_files = sorted(input_dir.glob("*.bin"))
    if jobs:
        if root_input_files:
            print(
                f"[preamble_fft] found {len(root_input_files)} root-level .bin file(s) in {input_dir}; "
                "skip them because lab-folder mode is active"
            )
        return jobs

    if root_input_files:
        # 兼容旧目录结构：如果没有任何 lab 子文件夹，就退回到根目录 .bin 批处理。
        return [
            {
                "label": input_dir.name,
                "input_files": root_input_files,
                "output_dir": all_bin_root_output_dir(args, input_dir),
                "lab_metadata": {
                    "lab_name": input_dir.name,
                    "lab_path": str(input_dir),
                    "lab_note": "",
                    "lab_note_path": "",
                    "override_sf": "",
                },
            }
        ]

    return []


def filter_jobs_by_position_ids(jobs, position_ids):
    if not position_ids:
        return jobs

    filtered_jobs = []
    for job in jobs:
        filtered_files = []
        for input_file in job["input_files"]:
            metadata = parse_capture_metadata(input_file)
            position_id = int_or_default(metadata.get("position_id", ""), None)
            if position_id in position_ids:
                filtered_files.append(input_file)
        if filtered_files:
            filtered_job = dict(job)
            filtered_job["input_files"] = filtered_files
            filtered_jobs.append(filtered_job)
    return filtered_jobs


def resolve_capture_args(base_args, input_file, lab_metadata=None, prepare_source=True):
    args = copy.copy(base_args)
    args.input_file = str(input_file)
    args.file_source_path = prepare_file_source_path(input_file) if prepare_source else ""
    metadata = parse_capture_metadata(input_file)
    if lab_metadata:
        metadata.update(
            {
                "lab_name": lab_metadata.get("lab_name", ""),
                "lab_path": lab_metadata.get("lab_path", ""),
                "lab_note": lab_metadata.get("lab_note", ""),
                "lab_note_path": lab_metadata.get("lab_note_path", ""),
                "lab_note_override_sf": lab_metadata.get("override_sf", ""),
            }
        )
    args.capture_metadata = metadata
    filename_sf = metadata.get("filename_sf", "")
    filename_preamble_len = metadata.get("filename_preamble_len", "")
    override_sf = lab_metadata.get("override_sf", "") if lab_metadata else ""
    args.sf = int(
        base_args.sf
        if base_args.sf is not None
        else (override_sf if override_sf != "" else (filename_sf if filename_sf != "" else 7))
    )
    args.preamble_len = int(
        base_args.preamble_len
        if base_args.preamble_len is not None
        else (filename_preamble_len if filename_preamble_len != "" else 16)
    )
    args.capture_metadata["resolved_sf"] = args.sf
    args.capture_metadata["resolved_preamble_len"] = args.preamble_len
    return args


def frame_metadata_key(item):
    """Return the stable packet key carried by frame/header/payload metadata."""
    for key in ("frame_count", "start_sample"):
        value = item.get(key, None)
        if value in (None, ""):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if key == "frame_count" and parsed < 0:
            continue
        return key, parsed
    return None


def default_header_metadata(capture_args):
    return {
        "cr": int(capture_args.cr),
        "pay_len": int(capture_args.pay_len),
        "crc": int(capture_args.has_crc),
        "ldro_mode": int(capture_args.ldro_mode),
        "header_err": -1,
    }


def default_payload_metadata():
    return {
        "header_packet_counter": "",
        "payload_packet_number": "",
        "decoded_payload_len": 0,
    }


def finalize_packet_metadata(item, capture_args, packet_index):
    metadata = capture_args.capture_metadata
    item.update(metadata)
    item["input_file"] = str(capture_args.input_file)
    item["file_name"] = Path(capture_args.input_file).name
    item["file_stem"] = Path(capture_args.input_file).stem
    item["packet_index_in_file"] = packet_index
    item["global_packet_id"] = ""
    return item


def merge_frame_and_header_metadata_by_index(frames, headers, payloads, capture_args):
    require_valid_payload = bool(getattr(capture_args, "require_valid_payload", False))
    valid_headers = [header for header in headers if int(header.get("header_err", 1)) == 0]
    merged = []
    for index, frame in enumerate(frames):
        payload = payloads[index] if index < len(payloads) else None
        if require_valid_payload and not (payload and payload.get("crc_valid", False)):
            continue

        item = dict(frame)
        if index < len(valid_headers):
            item.update(valid_headers[index])
        else:
            item.update(default_header_metadata(capture_args))
        if payload:
            item.update(payload)
        else:
            item.update(default_payload_metadata())
        merged.append(finalize_packet_metadata(item, capture_args, index))
    return merged


def merge_frame_and_header_metadata(frames, headers, payloads, capture_args):
    require_valid_payload = bool(getattr(capture_args, "require_valid_payload", False))
    frame_keys = [frame_metadata_key(frame) for frame in frames]
    header_keys = [frame_metadata_key(header) for header in headers]
    payload_keys = [frame_metadata_key(payload) for payload in payloads]
    has_frame_ids = all(key is not None for key in frame_keys)
    has_header_ids = any(key is not None for key in header_keys)
    has_payload_ids = (not payloads) or any(key is not None for key in payload_keys)

    if not (has_frame_ids and has_header_ids and has_payload_ids):
        print(
            f"[preamble_fft] metadata IDs incomplete for {Path(capture_args.input_file).name}; "
            "falling back to detection-order merge"
        )
        return merge_frame_and_header_metadata_by_index(frames, headers, payloads, capture_args)

    headers_by_key = {}
    for header, key in zip(headers, header_keys):
        if key is not None:
            headers_by_key[key] = header

    payloads_by_key = {}
    for payload, key in zip(payloads, payload_keys):
        if key is not None:
            payloads_by_key[key] = payload

    merged = []
    frame_key_set = set(frame_keys)
    for index, (frame, key) in enumerate(zip(frames, frame_keys)):
        header = headers_by_key.get(key)
        payload = payloads_by_key.get(key)
        if require_valid_payload and not (payload and payload.get("crc_valid", False)):
            continue

        item = dict(frame)
        if header is not None:
            item.update(header)
        else:
            item.update(default_header_metadata(capture_args))
        if payload is not None:
            item.update(payload)
        else:
            item.update(default_payload_metadata())
        merged.append(finalize_packet_metadata(item, capture_args, index))

    unmatched_valid_header_keys = {
        key
        for header, key in zip(headers, header_keys)
        if key is not None and int(header.get("header_err", 1)) == 0 and key not in frame_key_set
    }
    unmatched_payload_keys = {
        key for key in payload_keys if key is not None and key not in frame_key_set
    }
    if unmatched_valid_header_keys or unmatched_payload_keys:
        print(
            f"[preamble_fft] unmatched metadata for {Path(capture_args.input_file).name}: "
            f"{len(unmatched_valid_header_keys)} header(s), {len(unmatched_payload_keys)} payload(s)"
        )
    return merged
