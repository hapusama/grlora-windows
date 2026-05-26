#!/usr/bin/env python3
"""基于弱包前导码检测事件估计初始同步状态。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


WEAK_ROOT = Path(__file__).resolve().parents[1]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.initial_state import (  # noqa: E402
    InitialStateEstimate,
    InitialStateSearchConfig,
    InitialStateSeed,
    estimate_initial_state,
    load_complex64_file,
)
from weak_decoder.preamble_detector import PreambleDetectorConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 run_weak_sync_chain.py 的帧定界 CSV，"
            "在 located_preamble_start_sample 上做多 upchirp 相干叠加，估计 tau0 / beta / zeta。"
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="raw complex64 IQ 文件。")
    parser.add_argument("-d", "--detections", type=Path, required=True, help="帧定界 CSV；兼容旧的前导码检测 events.csv。")
    parser.add_argument("-o", "--output", type=Path, required=True, help="初始状态估计 CSV。")
    parser.add_argument("--sf", type=int, required=True, help="LoRa spreading factor。")
    parser.add_argument("--bw", type=float, default=125000.0, help="LoRa 带宽 Hz，默认 125000。")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ 采样率 Hz，默认 500000。")
    parser.add_argument("--preamble-len", type=float, default=8.0, help="前导码 upchirp 数，默认 8。")
    parser.add_argument("--estimate-chirps", type=int, default=6, help="用于相干估计的连续 upchirp 数，默认 6。")
    parser.add_argument("--max-events", type=int, default=None, help="最多估计多少个检测事件，默认全部。")
    parser.add_argument("--tau-min", type=float, default=None, help="tau0 搜索下界，单位 chip。默认 -2^(SF-1)。")
    parser.add_argument("--tau-max", type=float, default=None, help="tau0 搜索上界，单位 chip。默认 +2^(SF-1)。")
    parser.add_argument("--tau-step", type=float, default=4.0, help="tau0 粗搜索步长，单位 chip，默认 4。")
    parser.add_argument("--beta-min", type=float, default=-64.0, help="beta/CFO 搜索下界，单位 bin，默认 -64。")
    parser.add_argument("--beta-max", type=float, default=64.0, help="beta/CFO 搜索上界，单位 bin，默认 64。")
    parser.add_argument("--beta-step", type=float, default=1.0, help="beta/CFO 粗搜索步长，默认 1。")
    parser.add_argument("--fine-tau-radius", type=float, default=4.0, help="细搜索 tau0 半径，默认 4。")
    parser.add_argument("--fine-tau-step", type=float, default=1.0, help="细搜索 tau0 步长，默认 1。")
    parser.add_argument("--fine-beta-radius", type=float, default=2.0, help="细搜索 beta 半径，默认 2。")
    parser.add_argument("--fine-beta-step", type=float, default=0.25, help="细搜索 beta 步长，默认 0.25。")
    parser.add_argument("--zeta-span", type=float, default=0.0, help="SFO zeta 搜索半径，默认 0，即不估 SFO。")
    parser.add_argument("--zeta-step", type=float, default=1e-6, help="SFO zeta 搜索步长，默认 1e-6。")
    parser.add_argument("--frequency-chunk", type=int, default=128, help="粗搜索频偏预计算分块大小。")
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "")))
    except ValueError:
        return default


def load_detection_seeds(path: Path, max_events: int | None = None) -> list[InitialStateSeed]:
    """从帧定界 CSV 读取初始状态估计种子；兼容旧检测 events.csv。"""

    seeds: list[InitialStateSeed] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seeds.append(
                InitialStateSeed(
                    event_index=_int(row, "packet_index", _int(row, "event_index", len(seeds))),
                    start_sample=_int(
                        row,
                        "located_preamble_start_sample",
                        _int(row, "start_sample", 0),
                    ),
                    end_sample=_int(
                        row,
                        "located_payload_start_sample",
                        _int(row, "end_sample", 0),
                    ),
                    reference_bin=_int(
                        row,
                        "preamble_ref_bin",
                        _int(row, "reference_bin", 0),
                    ),
                    window_count=_int(
                        row,
                        "preamble_stable_count",
                        _int(row, "window_count", 0),
                    ),
                )
            )
            if max_events is not None and len(seeds) >= int(max_events):
                break
    return seeds


def write_estimates_csv(path: Path, estimates: list[InitialStateEstimate]) -> None:
    """写出每个检测事件的初始同步状态估计。"""

    fields = [
        "event_index",
        "coarse_start_sample",
        "reference_bin",
        "signed_reference_bin",
        "estimate_chirps",
        "tau0_chip",
        "tau0_sample",
        "beta_bin",
        "cfo_hz",
        "zeta",
        "payload_sto_chip",
        "payload_sto_sample",
        "payload_start_sample",
        "objective",
        "noncoherent_power",
        "coherent_gain_db",
        "mean_abs_z0",
        "coarse_tau0_chip",
        "coarse_beta_bin",
        "coarse_objective",
        "hit_tau_boundary",
        "hit_beta_boundary",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in estimates:
            writer.writerow({field: getattr(item, field) for field in fields})


def main() -> None:
    args = parse_args()
    detector_config = PreambleDetectorConfig(
        sf=args.sf,
        bw=args.bw,
        samp_rate=args.samp_rate,
        win_chirps=1,
        hop_samples=None,
        min_periodic_peaks=2,
        bin_tol=0,
    )
    detector_config.validate()
    search_config = InitialStateSearchConfig(
        estimate_chirps=args.estimate_chirps,
        preamble_len=args.preamble_len,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_step=args.tau_step,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_step=args.beta_step,
        fine_tau_radius=args.fine_tau_radius,
        fine_tau_step=args.fine_tau_step,
        fine_beta_radius=args.fine_beta_radius,
        fine_beta_step=args.fine_beta_step,
        zeta_span=args.zeta_span,
        zeta_step=args.zeta_step,
        frequency_chunk=args.frequency_chunk,
    )
    search_config.validate()

    seeds = load_detection_seeds(args.detections, args.max_events)
    samples = load_complex64_file(args.input)
    estimates: list[InitialStateEstimate] = []
    try:
        for seed in seeds:
            estimates.append(
                estimate_initial_state(
                    samples,
                    seed,
                    detector_config,
                    search_config,
                )
            )
    finally:
        mmap_handle = getattr(samples, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    write_estimates_csv(args.output, estimates)
    print(f"events={len(seeds)}")
    print(f"estimated={len(estimates)}")
    print(f"chirp_samples={detector_config.chirp_samples}")
    print(f"estimate_chirps={search_config.estimate_chirps}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
