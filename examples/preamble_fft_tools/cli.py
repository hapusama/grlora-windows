# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0
"""Command-line interface for the offline LoRa preamble FFT exporter."""

import copy
import signal
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

from gnuradio.eng_arg import eng_float

from .detector import run_detector_isolated, run_detector_once, write_detector_json
from .jobs import (
    default_output_dir,
    discover_lab_jobs,
    filter_jobs_by_position_ids,
    merge_frame_and_header_metadata,
    resolve_capture_args,
)
from .outputs import save_results
from .plots import save_dechirp_fft_plots
from .utils import parse_position_ids


def build_arg_parser():
    parser = ArgumentParser(
        description="Export LoRa packet IQ power and preamble dechirp FFT features from fc32/cfile IQ captures."
    )
    parser.add_argument("-f", "--input-file", type=str, default=None, help="Input fc32/cfile IQ file.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "USRP_IQ"),
        help="Directory containing lab subdirectories with USRP IQ .bin files for --all-bin.",
    )
    parser.add_argument(
        "--all-bin",
        action="store_true",
        default=False,
        help="Process each lab subdirectory under --input-dir separately and save outputs into that lab directory.",
    )
    parser.add_argument(
        "--sf",
        "--spreading-factor",
        type=int,
        default=None,
        help="LoRa spreading factor. In --all-bin mode, omit this to use each file name's SF field.",
    )
    parser.add_argument("--bw", "--bandwidth", type=eng_float, default=125e3, help="LoRa bandwidth in Hz.")
    parser.add_argument("--samp-rate", type=eng_float, default=500e3, help="IQ sample rate in Hz.")
    parser.add_argument("--cr", "--coding-rate", type=int, default=1, help="Coding rate index, 1..4.")
    parser.add_argument("--pay-len", type=int, default=255, help="Payload length for implicit header mode.")
    parser.add_argument("--has-crc", action="store_true", default=True, help="Packet has payload CRC.")
    parser.add_argument("--no-crc", action="store_false", dest="has_crc", help="Packet has no payload CRC.")
    parser.add_argument("--impl-head", action="store_true", default=False, help="Use implicit header mode.")
    parser.add_argument("--soft-decoding", action="store_true", default=True, help="Use soft decoding.")
    parser.add_argument("--hard-decoding", action="store_false", dest="soft_decoding", help="Use hard decoding.")
    parser.add_argument("--center-freq", type=eng_float, default=487.7e6, help="RF center frequency used by frame_sync SFO estimation.")
    parser.add_argument("--sync-word", type=lambda x: int(x, 0), default=0x34, help="LoRa sync word, decimal or 0x hex.")
    parser.add_argument("--ldro-mode", type=int, default=2, help="LDRO mode: 0 disabled, 1 enabled, 2 auto.")
    parser.add_argument(
        "--preamble-len",
        type=int,
        default=None,
        help="Expected preamble upchirp count. In --all-bin mode, omit this to use each file name's preamble field.",
    )
    parser.add_argument("--crc-mode", type=int, choices=[0, 1], default=0, help="CRC mode used only with --require-valid-payload: 0 GRLORA, 1 SX1276.")
    parser.add_argument(
        "--print-header",
        action="store_true",
        dest="print_header",
        help="Print decoded PHY header information. This is enabled by default.",
    )
    parser.add_argument(
        "--no-print-header",
        action="store_false",
        dest="print_header",
        help="Do not print decoded PHY header information.",
    )
    parser.add_argument(
        "--print-payload",
        choices=["none", "ascii", "hex"],
        default="ascii",
        help="Payload print format used only with --require-valid-payload.",
    )
    parser.add_argument(
        "--require-valid-payload",
        "--with-fcnt",
        action="store_true",
        default=False,
        help="Decode payloads, keep only packets with CRC-valid payloads, and fill FCnt when parseable. Disabled by default.",
    )
    parser.add_argument(
        "--throttle",
        action="store_true",
        dest="throttle",
        help="Throttle file playback to sample rate. This is enabled by default.",
    )
    parser.add_argument(
        "--no-throttle",
        action="store_false",
        dest="throttle",
        help="Process the file as fast as GNU Radio can schedule it.",
    )
    parser.set_defaults(throttle=True)
    parser.set_defaults(print_header=True)

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir()),
        help="Output directory for single-file mode, or fallback output directory when --all-bin has no lab subdirectories.",
    )
    parser.add_argument(
        "--position-ids",
        type=str,
        nargs="+",
        default=None,
        help="Location/position IDs to process in --all-bin mode, for example 8,9,10 or 8 9 10.",
    )
    parser.add_argument(
        "--plot-dechirp-fft",
        action="store_true",
        default=False,
        help="Save complete raw preamble dechirp FFT spectrum plots.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=False,
        help="Only save plots; do not overwrite packet_features.csv or preamble_features.npz.",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default=None,
        help="Directory for --plot-dechirp-fft PNG files. Default is <output-dir>/preamble_fft_plots.",
    )
    parser.add_argument(
        "--plot-db-floor",
        type=float,
        default=-60.0,
        help="Legacy option kept for older commands; peak-normalized plots ignore this value.",
    )
    parser.add_argument(
        "--plot-zoom-bins",
        type=int,
        default=8,
        help="Show an additional +/-N bin zoom panel around the raw main peak.",
    )
    parser.add_argument(
        "--normalize",
        choices=["max", "sum", "l2", "none"],
        default="max",
        help="FFT amplitude normalization mode.",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=0,
        help="FFT length. Default is 2**SF, matching frame_sync.",
    )
    parser.add_argument(
        "--downsample-phase",
        type=int,
        default=None,
        help="Oversampled symbol decimation phase. Default is os_factor//2, matching frame_sync.",
    )
    parser.add_argument(
        "--peak-width-db",
        type=float,
        default=-3.0,
        help="Peak-width threshold relative to peak amplitude in dB. Default -3 dB.",
    )
    parser.add_argument(
        "--peak-spectrum-half-width",
        type=int,
        default=8,
        help="Number of FFT bins to keep on each side of the aligned preamble main peak.",
    )
    parser.add_argument(
        "--isolated-workers",
        action="store_true",
        default=False,
        help="In --all-bin mode, run each .bin in a child process so native GNU Radio crashes only skip that file.",
    )
    parser.add_argument(
        "--show-gnuradio-log",
        action="store_false",
        dest="quiet_gnuradio",
        help="Show GNU Radio/C++ stdout and stderr during direct detector runs. Hidden by default.",
    )
    parser.set_defaults(quiet_gnuradio=True)
    parser.add_argument(
        "--worker-retries",
        type=int,
        default=1,
        help="Extra retries for each --isolated-workers child process after a native crash or missing JSON.",
    )
    parser.add_argument(
        "--worker-log-lines",
        type=int,
        default=0,
        help="Print this many tail lines from a failed --isolated-workers child process. Default suppresses noisy C++ logs.",
    )
    # 已收敛输出：不再支持 per-frame 明细 CSV 和未使用的 RSSI 采样间隔参数。
    parser.add_argument(
        "--detect-only-json",
        type=str,
        default=None,
        help=SUPPRESS,
    )
    return parser


def main():
    """命令行入口。

    主流程分为六步：
    1. 解析参数，判断是单文件、批处理，还是隐藏 worker 模式。
    2. 把输入文件整理成一个或多个 job；一个 lab 文件夹对应一个 job。
    3. 为 Ctrl+C/终止信号安装清理函数，避免 GNU Radio top_block 悬挂。
    4. 对每个 IQ 文件运行 detector，拿到 preamble 范围、PHY header 和可选 payload 信息。
    5. 合并每包元数据，然后重新读取 IQ 计算功率/前导码 FFT 特征。
    6. 每个 job 写出 packet_features.csv 和 preamble_features.npz。
    """

    # Step 1: 解析命令行参数。
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        args.position_id_filter = parse_position_ids(args.position_ids)
    except ValueError as exc:
        parser.error(str(exc))
    if args.plot_only and not args.plot_dechirp_fft:
        parser.error("--plot-only requires --plot-dechirp-fft")

    # 隐藏 worker 模式：父进程在 --isolated-workers 下会启动子进程。
    # 子进程只检测一个文件，把 frame/header/payload 元数据写成 JSON 后退出。
    if args.detect_only_json:
        if not args.input_file:
            parser.error("--detect-only-json requires -f/--input-file")
        capture_args = resolve_capture_args(args, Path(args.input_file), None)
        write_detector_json(capture_args, args.detect_only_json)
        return 0

    # Step 2: 组织待处理任务。
    # --all-bin: 扫描 input_dir 下的 lab 子目录，每个 lab 单独输出。
    # 单文件模式: 只创建一个 job，输出到 --output-dir。
    if args.all_bin:
        jobs = discover_lab_jobs(args)
        if not jobs:
            parser.error(f"no .bin files found in {Path(args.input_dir)} or its lab subdirectories")
    else:
        if not args.input_file:
            parser.error("either -f/--input-file or --all-bin is required")
        jobs = [
            {
                "label": Path(args.input_file).stem,
                "input_files": [Path(args.input_file)],
                "output_dir": Path(args.output_dir),
                "lab_metadata": None,
            }
        ]
    jobs = filter_jobs_by_position_ids(jobs, args.position_id_filter)
    if not jobs:
        parser.error(f"no .bin files match --position-ids {args.position_ids!r}")

    # Step 3: 保存当前正在运行的 GNU Radio top_block。
    # 这里用单元素列表，是为了让内部 sig_handler 可以修改外层引用。
    current_tb = [None]

    def sig_handler(sig=None, frame=None):
        # 收到 Ctrl+C / SIGTERM 时，先停止当前流图，再用 130 退出。
        if current_tb[0] is not None:
            current_tb[0].stop()
            current_tb[0].wait()
        sys.exit(130)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    total_packets = 0
    successful_jobs = 0
    for job in jobs:
        # Step 4a: 初始化当前 job。
        # 每个 lab 文件夹作为一个独立任务：单独读取其中所有 bin，输出也写回该 lab 文件夹。
        job_args = copy.copy(args)
        job_args.output_dir = str(job["output_dir"])
        lab_metadata = job.get("lab_metadata")
        print(
            f"[preamble_fft] ===== job {job['label']}: "
            f"{len(job['input_files'])} file(s), output -> {job_args.output_dir} ====="
        )
        if lab_metadata and lab_metadata.get("lab_note"):
            print(f"[preamble_fft] lab note: {lab_metadata['lab_note']}")
        if lab_metadata and lab_metadata.get("override_sf") != "":
            print(f"[preamble_fft] lab note overrides SF to {lab_metadata['override_sf']}")

        capture_results = []
        for input_file in job["input_files"]:
            # Step 4b: 为单个 IQ 文件准备实际参数。
            # 根据文件名补全实验编号、位置编号、SF、发射功率、前导码长度等信息。
            # 如果 lab 的补充.txt 写明了修正参数，会在 resolve_capture_args 里覆盖文件名参数。
            # 子进程隔离模式下，父进程只调度，不提前创建 GNU Radio file_source 的 hardlink。
            use_isolated_worker = bool(args.all_bin and args.isolated_workers)
            capture_args = resolve_capture_args(
                job_args,
                input_file,
                lab_metadata,
                prepare_source=not use_isolated_worker,
            )
            print(
                f"[preamble_fft] processing {input_file} "
                f"(sf={capture_args.sf}, preamble_len={capture_args.preamble_len})"
            )

            # Step 4c: 运行 detector。
            # 普通模式在当前进程跑；隔离模式为每个文件启动一个子进程，
            # 防止单个 native 崩溃影响整个批处理。
            if use_isolated_worker:
                frames, headers, payloads = run_detector_isolated(capture_args)
            else:
                frames, headers, payloads = run_detector_once(capture_args, current_tb)

            # Step 4d: 合并 frame_sync/header_decoder/payload 三路元数据。
            # 默认不要求 payload CRC valid；如果开启 --require-valid-payload，只保留 CRC 通过的包。
            if not frames:
                print(f"[preamble_fft] no valid preamble ranges for {input_file}")
                continue
            merged_frames = merge_frame_and_header_metadata(frames, headers, payloads, capture_args)
            if capture_args.require_valid_payload:
                dropped = len(frames) - len(merged_frames)
                print(
                    f"[preamble_fft] kept {len(merged_frames)}/{len(frames)} "
                    f"CRC-valid payload packet(s) from {input_file.name}"
                    + (f"; dropped {dropped}" if dropped else "")
                )
                if not merged_frames:
                    continue
            capture_results.append((capture_args, merged_frames))
            print(f"[preamble_fft] collected {len(merged_frames)} packet(s) from {input_file.name}")

        # 当前 job 没有任何有效包时跳过输出，继续处理下一个 job。
        if not capture_results:
            print(f"[preamble_fft] no valid packets for job {job['label']}")
            continue

        # Step 5/6: 重新读取 IQ，计算每包特征，并按需写出 CSV/NPZ 和 PNG 图。
        job_packet_count = sum(len(frames) for _, frames in capture_results)
        wrote_anything = False
        if args.plot_dechirp_fft:
            plot_outputs = save_dechirp_fft_plots(job_args, capture_results)
            wrote_anything = plot_outputs["plot_count"] > 0
            if plot_outputs["plots"]:
                plot_list = ", ".join(str(path) for path in plot_outputs["plots"])
                print(
                    f"[preamble_fft] wrote {plot_outputs['plot_count']} plot(s) "
                    f"for {plot_outputs['packet_count']} packet(s) -> {plot_list}"
                )

        if not args.plot_only:
            outputs = save_results(job_args, capture_results)
            job_packet_count = outputs["packet_count"]
            wrote_anything = True
            print(f"[preamble_fft] wrote {outputs['packet_count']} packet(s) -> {outputs['packet']}, {outputs['npz']}")

        if wrote_anything:
            total_packets += job_packet_count
            successful_jobs += 1

    # 所有 job 都处理完后，根据是否成功写出任何包返回退出码。
    if successful_jobs == 0:
        print("[preamble_fft] no valid packets were published by frame_sync/header_decoder")
        return 1
    print(f"[preamble_fft] finished {successful_jobs} job(s), {total_packets} packet(s) total")
    return 0
