# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0
"""Detector execution helpers.

Batch mode can run each capture in a child process so a native GNU Radio crash
only skips that file instead of aborting the whole lab job.
"""

import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .constants import WINDOWS_ACCESS_VIOLATION
from .file_graph_path import SCRIPT_PATH
from .flowgraph import lora_file_preamble_fft_rx
from .utils import cleanup_file_source_path, json_safe, suppress_native_output, tail_lines


def run_detector_once(capture_args, current_tb=None):
    """Run GNU Radio detector for one IQ file and return copied metadata lists."""
    tb = None
    try:
        with suppress_native_output(getattr(capture_args, "quiet_gnuradio", True)):
            tb = lora_file_preamble_fft_rx(capture_args)
            if current_tb is not None:
                current_tb[0] = tb
            tb.start()
            tb.wait()
            if current_tb is not None:
                current_tb[0] = None

            frames = list(tb.metadata_sink.frames)
            headers = list(tb.header_sink.headers)
            payloads = []
            if getattr(capture_args, "require_valid_payload", False):
                crc_valid_flags = [bool(item) for item in tb.crc_valid_sink.data()]
                for index, payload in enumerate(tb.payload_sink.payloads):
                    item = dict(payload)
                    if "crc_valid" not in item:
                        item["crc_valid"] = crc_valid_flags[index] if index < len(crc_valid_flags) else False
                    payloads.append(item)
        return frames, headers, payloads
    finally:
        if current_tb is not None:
            current_tb[0] = None
        cleanup_file_source_path(capture_args)
        tb = None
        gc.collect()


def write_detector_json(capture_args, output_path):
    """Hidden worker mode: detect one file and write metadata as JSON for the parent process."""
    frames, headers, payloads = run_detector_once(capture_args)
    data = {
        "frames": frames,
        "headers": headers,
        "payloads": payloads,
    }
    Path(output_path).write_text(json.dumps(json_safe(data), ensure_ascii=False), encoding="utf-8")


def child_detector_command(capture_args, json_path):
    """Build a one-file worker command using the same Python executable."""
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "-f",
        str(capture_args.input_file),
        "--detect-only-json",
        str(json_path),
        "--sf",
        str(int(capture_args.sf)),
        "--preamble-len",
        str(int(capture_args.preamble_len)),
        "--bw",
        str(float(capture_args.bw)),
        "--samp-rate",
        str(float(capture_args.samp_rate)),
        "--cr",
        str(int(capture_args.cr)),
        "--pay-len",
        str(int(capture_args.pay_len)),
        "--center-freq",
        str(float(capture_args.center_freq)),
        "--sync-word",
        hex(int(capture_args.sync_word)),
        "--ldro-mode",
        str(int(capture_args.ldro_mode)),
        "--crc-mode",
        str(int(capture_args.crc_mode)),
        "--no-print-header",
        "--print-payload",
        str(capture_args.print_payload),
    ]
    cmd.append("--has-crc" if capture_args.has_crc else "--no-crc")
    cmd.append("--soft-decoding" if capture_args.soft_decoding else "--hard-decoding")
    if capture_args.impl_head:
        cmd.append("--impl-head")
    cmd.append("--throttle" if capture_args.throttle else "--no-throttle")
    if getattr(capture_args, "require_valid_payload", False):
        cmd.append("--require-valid-payload")
    if capture_args.downsample_phase is not None:
        cmd.extend(["--downsample-phase", str(int(capture_args.downsample_phase))])
    if capture_args.nfft:
        cmd.extend(["--nfft", str(int(capture_args.nfft))])
    if not getattr(capture_args, "quiet_gnuradio", True):
        cmd.append("--show-gnuradio-log")
    return cmd


def format_returncode(returncode):
    if returncode == WINDOWS_ACCESS_VIOLATION:
        return f"{returncode} (0x{returncode:08X}, Windows access violation)"
    if returncode < 0:
        return f"{returncode} (signal {-returncode})"
    return str(returncode)


def run_detector_isolated(capture_args):
    """Run one IQ file in a fresh child process to avoid GNU Radio resource buildup in batch mode."""
    attempts = max(1, int(capture_args.worker_retries) + 1)
    last_result = None
    for attempt in range(1, attempts + 1):
        with tempfile.TemporaryDirectory(prefix="lora_preamble_detect_") as temp_dir:
            json_path = Path(temp_dir) / "detected.json"
            result = subprocess.run(
                child_detector_command(capture_args, json_path),
                cwd=str(Path.cwd()),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            last_result = result
            if result.returncode == 0 and json_path.exists():
                data = json.loads(json_path.read_text(encoding="utf-8"))
                return data.get("frames", []), data.get("headers", []), data.get("payloads", [])

            if attempt < attempts:
                print(
                    f"[preamble_fft] worker failed for {capture_args.input_file} "
                    f"(exit={format_returncode(result.returncode)}), retry {attempt}/{attempts - 1}"
                )

    if last_result is None:
        return [], [], []

    if last_result.returncode == 0:
        print(f"[preamble_fft] worker produced no JSON for {capture_args.input_file}")
    else:
        print(
            f"[preamble_fft] worker failed for {capture_args.input_file} "
            f"(exit={format_returncode(last_result.returncode)})"
        )
    combined = "\n".join(part for part in (last_result.stdout, last_result.stderr) if part)
    if combined and int(capture_args.worker_log_lines) > 0:
        print(tail_lines(combined, int(capture_args.worker_log_lines)))
    elif combined:
        print("[preamble_fft] worker log suppressed; use --worker-log-lines 40 to show GNU Radio details")
    return [], [], []
