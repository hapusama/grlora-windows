#!/usr/bin/env python3
"""弱包同步链路入口：前导码检测、chirp 对齐、初始状态估计。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[1]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_upchirp  # noqa: E402
from weak_decoder.initial_state import (  # noqa: E402
    InitialStateEstimate,
    InitialStateSearchConfig,
    InitialStateSeed,
    estimate_initial_state,
    load_complex64_file,
)
from weak_decoder.frame_locator import (  # noqa: E402
    FrameLocation,
    FrameLocatorConfig,
    locate_frame_from_event,
)
from weak_decoder.preamble_detector import (  # noqa: E402
    DetectionEvent,
    PreambleDetectorConfig,
    WindowPeak,
    detect_preamble_runs,
)


def parse_int_auto(text: str) -> int:
    """解析十进制或 0x 前缀十六进制整数。"""

    return int(str(text), 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "一体化弱包同步链：先做多 chirp 滑窗前导码检测，"
            "再对每个检测事件独立做 chirp 起点对齐和初始状态估计。"
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="raw complex64 IQ 文件。")
    parser.add_argument("-o", "--output", type=Path, required=True, help="链路结果 CSV。")
    parser.add_argument("--events-csv", type=Path, default=None, help="可选：另存检测事件 CSV。")
    parser.add_argument("--windows-csv", type=Path, default=None, help="可选：另存逐滑窗 peak CSV。")
    parser.add_argument("--sf", type=int, default=None, help="LoRa SF。默认从文件名第 4 段推断。")
    parser.add_argument("--bw", type=float, default=125000.0, help="LoRa 带宽 Hz，默认 125000。")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ 采样率 Hz，默认 500000。")
    parser.add_argument("--sync-word", type=parse_int_auto, default=0x34, help="LoRa sync word，默认 0x34。")
    parser.add_argument("--preamble-len", type=float, default=None, help="前导码 upchirp 数。默认从文件名最后一段推断。")
    parser.add_argument("--win-chirps", type=int, default=2, help="检测窗口内 chirp 数，默认 2。")
    parser.add_argument("--hop-chirps", type=float, default=1.0, help="滑窗步长，单位 chirp，默认 1。")
    parser.add_argument("--hop-samples", type=int, default=None, help="滑窗步长，单位 sample；给出后覆盖 --hop-chirps。")
    parser.add_argument(
        "--min-periodic-peaks",
        type=int,
        default=None,
        help="连续稳定窗口数。默认 preamble_len - win_chirps + 1。",
    )
    parser.add_argument("--bin-tol", type=int, default=2, help="peak bin 循环距离容差，默认 2。")
    parser.add_argument("--sample-limit", type=int, default=None, help="只扫描前 N 个 sample。")
    parser.add_argument("--max-windows", type=int, default=None, help="最多扫描多少个滑窗。")
    parser.add_argument("--max-events", type=int, default=None, help="最多处理多少个检测事件。")
    parser.add_argument(
        "--min-event-gap-chirps",
        type=float,
        default=None,
        help="检测事件去重间隔，单位 chirp；默认 preamble_len。",
    )
    parser.add_argument("--align-search-chirps", type=float, default=1.0, help="chirp 起点对齐搜索半径，单位 chirp，默认 1。")
    parser.add_argument("--align-step-samples", type=int, default=1, help="chirp 起点对齐步长，单位 sample，默认 1。")
    parser.add_argument("--align-chirps", type=int, default=None, help="对齐评分使用的 upchirp 数，默认 min(4, preamble_len)。")
    parser.add_argument("--frame-search-samples", type=int, default=None, help="SFD 定位搜索半径，单位 sample；默认 Ns/8。")
    parser.add_argument("--frame-step-samples", type=int, default=1, help="SFD 定位搜索步长，单位 sample，默认 1。")
    parser.add_argument("--frame-preamble-bin-tol", type=int, default=2, help="SFD 定位阶段前导码稳定 bin 容差，默认 2。")
    parser.add_argument("--frame-sync-bin-tol", type=int, default=4, help="sync word 相对 bin 容差，默认 4。")
    parser.add_argument("--frame-sfd-bin-tol", type=int, default=4, help="两个 SFD downchirp bin 稳定容差，默认 4。")
    parser.add_argument("--frame-min-preamble-peaks", type=int, default=None, help="SFD 定位阶段最少稳定前导码符号数，默认 preamble_len-2。")
    parser.add_argument("--frame-symbol-search-span", type=int, default=2, help="SFD 定位额外搜索前后多少个整 chirp，默认 2。")
    parser.add_argument("--stft-dir", type=Path, default=None, help="可选：保存定位出的 preamble+sync+SFD STFT 验证图。")
    parser.add_argument("--estimate-chirps", type=int, default=None, help="初始状态估计使用的 upchirp 数，默认 min(6, preamble_len)。")
    parser.add_argument("--tau-min", type=float, default=None, help="tau0 搜索下界，单位 chip。")
    parser.add_argument("--tau-max", type=float, default=None, help="tau0 搜索上界，单位 chip。")
    parser.add_argument("--tau-step", type=float, default=8.0, help="tau0 粗搜索步长，默认 8。")
    parser.add_argument("--beta-min", type=float, default=-64.0, help="beta 搜索下界，默认 -64。")
    parser.add_argument("--beta-max", type=float, default=64.0, help="beta 搜索上界，默认 64。")
    parser.add_argument("--beta-step", type=float, default=2.0, help="beta 粗搜索步长，默认 2。")
    parser.add_argument("--fine-tau-radius", type=float, default=4.0, help="细搜索 tau0 半径，默认 4。")
    parser.add_argument("--fine-tau-step", type=float, default=1.0, help="细搜索 tau0 步长，默认 1。")
    parser.add_argument("--fine-beta-radius", type=float, default=2.0, help="细搜索 beta 半径，默认 2。")
    parser.add_argument("--fine-beta-step", type=float, default=0.25, help="细搜索 beta 步长，默认 0.25。")
    parser.add_argument("--zeta-span", type=float, default=0.0, help="SFO zeta 搜索半径，默认 0。")
    parser.add_argument("--zeta-step", type=float, default=1e-6, help="SFO zeta 搜索步长，默认 1e-6。")
    parser.add_argument("--frequency-chunk", type=int, default=128, help="粗搜索频偏分块大小。")
    return parser.parse_args()


def infer_params_from_filename(path: Path) -> tuple[int | None, int | None]:
    """按 experiment_corridor_position_sf_txpower_preamble 的命名约定推断参数。"""

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


def resolve_positive_int(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required and could not be inferred from the filename.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def resolve_chirp_samples(sf: int, bw: float, samp_rate: float) -> tuple[int, int]:
    ratio = float(samp_rate) / float(bw)
    os_factor = int(round(ratio))
    if os_factor <= 0 or not math.isclose(ratio, os_factor, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"samp_rate / bw must be an integer, got {ratio:.9g}.")
    return int((1 << int(sf)) * os_factor), os_factor


def detection_to_seed(event: DetectionEvent, event_index: int, start_sample: int) -> InitialStateSeed:
    """把检测事件转换成初始状态估计种子。"""

    return InitialStateSeed(
        event_index=int(event_index),
        start_sample=int(start_sample),
        end_sample=int(event.end_sample),
        reference_bin=int(event.reference_bin),
        window_count=int(event.window_count),
    )


def select_spaced_events(
    events: list[DetectionEvent],
    chirp_samples: int,
    min_gap_chirps: float,
    max_events: int | None,
) -> list[DetectionEvent]:
    """按起点间隔做轻量去重，避免同一段前导码被重复送进估计器。"""

    selected: list[DetectionEvent] = []
    min_gap_samples = int(round(float(min_gap_chirps) * int(chirp_samples)))
    last_start: int | None = None
    for event in sorted(events, key=lambda item: item.start_sample):
        if last_start is not None and event.start_sample - last_start < min_gap_samples:
            continue
        selected.append(event)
        last_start = int(event.start_sample)
        if max_events is not None and len(selected) >= int(max_events):
            break
    return selected


def align_event_start(
    samples: np.ndarray,
    event: DetectionEvent,
    config: PreambleDetectorConfig,
    search_radius_samples: int,
    step_samples: int,
    align_chirps: int,
) -> dict[str, float | int]:
    """在检测粗起点附近搜索更好的 chirp 边界。"""

    chirp_samples = config.chirp_samples
    downchirp = np.conjugate(
        build_upchirp(config.sf, symbol_id=0, os_factor=config.os_factor)
    ).astype(np.complex64)
    n_required = int(align_chirps * chirp_samples)
    start_min = max(0, int(event.start_sample) - int(search_radius_samples))
    start_max = min(samples.size - n_required, int(event.start_sample) + int(search_radius_samples))
    if start_max < start_min:
        raise ValueError(f"event {event.event_index} does not have enough samples for alignment.")

    best: dict[str, float | int] | None = None
    for candidate_start in range(start_min, start_max + 1, int(step_samples)):
        block = np.asarray(samples[candidate_start : candidate_start + n_required], dtype=np.complex64)
        chirps = block.reshape(int(align_chirps), chirp_samples)
        spectrum = np.fft.fft(chirps * downchirp[np.newaxis, :], axis=1)
        energy = np.sum(np.abs(spectrum) ** 2, axis=0, dtype=np.float64)
        total_power = float(np.sum(energy, dtype=np.float64))
        if total_power <= 0.0:
            continue
        peak_bin = int(np.argmax(energy))
        peak_power = float(energy[peak_bin])
        second_power = float(np.partition(energy, -2)[-2]) if energy.size > 1 else 0.0
        confidence_db = 10.0 * math.log10((peak_power + 1e-30) / (second_power + 1e-30))
        peak_share = peak_power / total_power
        score = peak_power
        if best is None or score > float(best["align_score"]):
            best = {
                "aligned_start_sample": int(candidate_start),
                "align_offset_samples": int(candidate_start - int(event.start_sample)),
                "align_peak_bin": peak_bin,
                "align_peak_power": peak_power,
                "align_second_power": second_power,
                "align_total_power": total_power,
                "align_confidence_db": float(confidence_db),
                "align_peak_share": float(peak_share),
                "align_score": float(score),
            }
    if best is None:
        raise ValueError(f"event {event.event_index} has no valid alignment candidate.")
    return best


def write_windows_csv(path: Path, windows: list[WindowPeak], config: PreambleDetectorConfig) -> None:
    """写出检测阶段逐滑窗结果。"""

    fields = [
        "window_index",
        "start_sample",
        "end_sample",
        "peak_bin",
        "peak_bin_div_os",
        "peak_power",
        "second_power",
        "total_power",
        "confidence_db",
        "peak_share",
        "valid",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in windows:
            writer.writerow(
                {
                    "window_index": item.window_index,
                    "start_sample": item.start_sample,
                    "end_sample": item.end_sample,
                    "peak_bin": item.peak_bin,
                    "peak_bin_div_os": item.peak_bin / float(config.os_factor),
                    "peak_power": item.peak_power,
                    "second_power": item.second_power,
                    "total_power": item.total_power,
                    "confidence_db": item.confidence_db,
                    "peak_share": item.peak_share,
                    "valid": int(item.valid),
                }
            )


def write_events_csv(path: Path, events: list[DetectionEvent], config: PreambleDetectorConfig) -> None:
    """写出检测事件。"""

    fields = [
        "event_index",
        "start_sample",
        "end_sample",
        "first_window_index",
        "last_window_index",
        "window_count",
        "reference_bin",
        "reference_bin_div_os",
        "bin_min",
        "bin_max",
        "mean_peak_power",
        "mean_confidence_db",
        "max_peak_share",
        "sf",
        "bw",
        "samp_rate",
        "os_factor",
        "chirp_samples",
        "win_chirps",
        "hop_samples",
        "min_periodic_peaks",
        "bin_tol",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in events:
            writer.writerow(
                {
                    "event_index": item.event_index,
                    "start_sample": item.start_sample,
                    "end_sample": item.end_sample,
                    "first_window_index": item.first_window_index,
                    "last_window_index": item.last_window_index,
                    "window_count": item.window_count,
                    "reference_bin": item.reference_bin,
                    "reference_bin_div_os": item.reference_bin / float(config.os_factor),
                    "bin_min": item.bin_min,
                    "bin_max": item.bin_max,
                    "mean_peak_power": item.mean_peak_power,
                    "mean_confidence_db": item.mean_confidence_db,
                    "max_peak_share": item.max_peak_share,
                    "sf": config.sf,
                    "bw": config.bw,
                    "samp_rate": config.samp_rate,
                    "os_factor": config.os_factor,
                    "chirp_samples": config.chirp_samples,
                    "win_chirps": config.win_chirps,
                    "hop_samples": config.resolved_hop_samples,
                    "min_periodic_peaks": config.min_periodic_peaks,
                    "bin_tol": config.bin_tol,
                }
            )


def write_chain_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """写出一体化链路结果。"""

    fields = [
        "packet_index",
        "event_index",
        "detected_start_sample",
        "detected_end_sample",
        "detected_window_count",
        "detected_reference_bin",
        "detected_mean_confidence_db",
        "aligned_start_sample",
        "align_offset_samples",
        "align_peak_bin",
        "align_confidence_db",
        "align_peak_share",
        "frame_valid",
        "located_preamble_start_sample",
        "located_sfd_start_sample",
        "located_payload_start_sample",
        "locator_score",
        "preamble_ref_bin",
        "preamble_stable_count",
        "sync1_bin",
        "sync2_bin",
        "sync1_expected_bin",
        "sync2_expected_bin",
        "sync1_distance",
        "sync2_distance",
        "sfd1_bin",
        "sfd2_bin",
        "sfd_bin_distance",
        "mean_preamble_confidence_db",
        "mean_sfd_confidence_db",
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
        "hit_tau_boundary",
        "hit_beta_boundary",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def estimate_to_row(
    packet_index: int,
    event: DetectionEvent,
    alignment: dict[str, float | int],
    frame_location: FrameLocation,
    estimate: InitialStateEstimate,
) -> dict[str, object]:
    """合并检测、对齐、SFD 定位、初始状态估计四层结果。"""

    return {
        "packet_index": int(packet_index),
        "event_index": int(event.event_index),
        "detected_start_sample": int(event.start_sample),
        "detected_end_sample": int(event.end_sample),
        "detected_window_count": int(event.window_count),
        "detected_reference_bin": int(event.reference_bin),
        "detected_mean_confidence_db": float(event.mean_confidence_db),
        "aligned_start_sample": alignment["aligned_start_sample"],
        "align_offset_samples": alignment["align_offset_samples"],
        "align_peak_bin": alignment["align_peak_bin"],
        "align_confidence_db": alignment["align_confidence_db"],
        "align_peak_share": alignment["align_peak_share"],
        "frame_valid": int(frame_location.valid),
        "located_preamble_start_sample": int(frame_location.preamble_start_sample),
        "located_sfd_start_sample": int(frame_location.sfd_start_sample),
        "located_payload_start_sample": int(frame_location.payload_start_sample),
        "locator_score": float(frame_location.score),
        "preamble_ref_bin": int(frame_location.preamble_ref_bin),
        "preamble_stable_count": int(frame_location.preamble_stable_count),
        "sync1_bin": int(frame_location.sync1_bin),
        "sync2_bin": int(frame_location.sync2_bin),
        "sync1_expected_bin": int(frame_location.sync1_expected_bin),
        "sync2_expected_bin": int(frame_location.sync2_expected_bin),
        "sync1_distance": int(frame_location.sync1_distance),
        "sync2_distance": int(frame_location.sync2_distance),
        "sfd1_bin": int(frame_location.sfd1_bin),
        "sfd2_bin": int(frame_location.sfd2_bin),
        "sfd_bin_distance": int(frame_location.sfd_bin_distance),
        "mean_preamble_confidence_db": float(frame_location.mean_preamble_confidence_db),
        "mean_sfd_confidence_db": float(frame_location.mean_sfd_confidence_db),
        "tau0_chip": estimate.tau0_chip,
        "tau0_sample": estimate.tau0_sample,
        "beta_bin": estimate.beta_bin,
        "cfo_hz": estimate.cfo_hz,
        "zeta": estimate.zeta,
        "payload_sto_chip": estimate.payload_sto_chip,
        "payload_sto_sample": estimate.payload_sto_sample,
        "payload_start_sample": estimate.payload_start_sample,
        "objective": estimate.objective,
        "noncoherent_power": estimate.noncoherent_power,
        "coherent_gain_db": estimate.coherent_gain_db,
        "mean_abs_z0": estimate.mean_abs_z0,
        "coarse_tau0_chip": estimate.coarse_tau0_chip,
        "coarse_beta_bin": estimate.coarse_beta_bin,
        "hit_tau_boundary": int(estimate.hit_tau_boundary),
        "hit_beta_boundary": int(estimate.hit_beta_boundary),
    }


def write_frame_stft_plot(
    samples: np.ndarray,
    frame_location: FrameLocation,
    detector_config: PreambleDetectorConfig,
    preamble_len: float,
    output_path: Path,
) -> None:
    """把定位出的 preamble+sync+SFD 区间画成 STFT 验证图。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = int(frame_location.preamble_start_sample)
    stop = int(frame_location.payload_start_sample)
    if start < 0 or stop <= start or stop > samples.size:
        raise ValueError("invalid STFT sample range.")

    segment = np.asarray(samples[start:stop], dtype=np.complex64)
    chirp_samples = detector_config.chirp_samples
    nperseg = min(max(128, chirp_samples // 16), max(128, segment.size // 4))
    nperseg = min(nperseg, segment.size)
    noverlap = int(round(nperseg * 0.75))
    hop = max(1, nperseg - noverlap)
    nfft = max(1024, int(2 ** math.ceil(math.log2(max(nperseg * 4, 2)))))
    starts = np.arange(0, segment.size - nperseg + 1, hop, dtype=np.int64)
    if starts.size == 0:
        starts = np.asarray([0], dtype=np.int64)

    window = np.hanning(nperseg).astype(np.float32)
    frames = np.empty((starts.size, nperseg), dtype=np.complex64)
    for row, offset in enumerate(starts):
        frames[row, :] = segment[offset : offset + nperseg] * window

    spec = np.fft.fftshift(np.fft.fft(frames, n=nfft, axis=1), axes=1)
    spec_db = 20.0 * np.log10(np.maximum(np.abs(spec).T, 1e-12))
    spec_db -= float(np.max(spec_db))
    times_ms = (starts + nperseg / 2.0) / detector_config.samp_rate * 1e3
    freqs_khz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / detector_config.samp_rate)) / 1e3

    fig, ax = plt.subplots(figsize=(12, 4), dpi=160)
    image = ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        extent=[times_ms[0], times_ms[-1], freqs_khz[0], freqs_khz[-1]],
        cmap="viridis",
        vmin=-75,
        vmax=0,
    )
    symbol_ms = chirp_samples / detector_config.samp_rate * 1e3
    boundaries = [
        (float(preamble_len) * symbol_ms, "sync"),
        ((float(preamble_len) + 2.0) * symbol_ms, "SFD"),
        ((float(preamble_len) + 4.0) * symbol_ms, "quarter"),
        ((float(preamble_len) + 4.25) * symbol_ms, "payload"),
    ]
    for x_ms, label in boundaries:
        ax.axvline(x_ms, color="white", linewidth=0.9, alpha=0.85)
        ax.text(x_ms, freqs_khz[-1] * 0.92, label, color="white", fontsize=8, rotation=90, va="top")

    for idx in range(1, int(math.floor(float(preamble_len) + 4.25)) + 1):
        ax.axvline(idx * symbol_ms, color="white", linewidth=0.35, alpha=0.35)

    ax.set_title(
        "Located LoRa preamble + sync + SFD "
        f"(packet event {frame_location.event_index}, valid={int(frame_location.valid)})"
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (kHz)")
    cbar = fig.colorbar(image, ax=ax, pad=0.01)
    cbar.set_label("Relative power (dB)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inferred_sf, inferred_preamble_len = infer_params_from_filename(args.input)
    sf = resolve_positive_int(args.sf if args.sf is not None else inferred_sf, "sf")
    resolved_preamble_len = args.preamble_len if args.preamble_len is not None else inferred_preamble_len
    if resolved_preamble_len is None:
        raise ValueError("preamble_len is required and could not be inferred from the filename.")
    preamble_len = float(resolved_preamble_len)
    if preamble_len <= 0.0:
        raise ValueError("preamble_len must be positive and could not be inferred from the filename.")

    chirp_samples, _ = resolve_chirp_samples(sf, args.bw, args.samp_rate)
    hop_samples = int(args.hop_samples) if args.hop_samples is not None else int(round(args.hop_chirps * chirp_samples))
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive.")
    min_periodic_peaks = (
        int(args.min_periodic_peaks)
        if args.min_periodic_peaks is not None
        else max(2, int(round(preamble_len)) - int(args.win_chirps) + 1)
    )
    align_chirps = int(args.align_chirps) if args.align_chirps is not None else max(1, min(4, int(round(preamble_len))))
    frame_search_samples = (
        int(args.frame_search_samples)
        if args.frame_search_samples is not None
        else max(1, int(round(chirp_samples / 8)))
    )
    estimate_chirps = (
        int(args.estimate_chirps)
        if args.estimate_chirps is not None
        else max(1, min(6, int(round(preamble_len))))
    )
    min_event_gap_chirps = float(args.min_event_gap_chirps) if args.min_event_gap_chirps is not None else float(preamble_len)

    detector_config = PreambleDetectorConfig(
        sf=sf,
        bw=args.bw,
        samp_rate=args.samp_rate,
        win_chirps=args.win_chirps,
        hop_samples=hop_samples,
        min_periodic_peaks=min_periodic_peaks,
        bin_tol=args.bin_tol,
    )
    detector_config.validate()
    locator_config = FrameLocatorConfig(
        preamble_len=preamble_len,
        sync_word=args.sync_word,
        search_radius_samples=frame_search_samples,
        step_samples=args.frame_step_samples,
        preamble_bin_tol=args.frame_preamble_bin_tol,
        sync_bin_tol=args.frame_sync_bin_tol,
        sfd_bin_tol=args.frame_sfd_bin_tol,
        min_preamble_peaks=args.frame_min_preamble_peaks,
        symbol_search_span=args.frame_symbol_search_span,
    )
    locator_config.validate()
    search_config = InitialStateSearchConfig(
        estimate_chirps=estimate_chirps,
        preamble_len=preamble_len,
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

    samples = load_complex64_file(args.input)
    rows: list[dict[str, object]] = []
    try:
        windows, events = detect_preamble_runs(
            samples,
            detector_config,
            sample_limit=args.sample_limit,
            max_windows=args.max_windows,
        )
        selected_events = select_spaced_events(
            events,
            detector_config.chirp_samples,
            min_event_gap_chirps,
            args.max_events,
        )
        search_radius = int(round(float(args.align_search_chirps) * detector_config.chirp_samples))
        for packet_index, event in enumerate(selected_events):
            alignment = align_event_start(
                samples,
                event,
                detector_config,
                search_radius_samples=search_radius,
                step_samples=args.align_step_samples,
                align_chirps=align_chirps,
            )
            frame_location = locate_frame_from_event(
                samples,
                event,
                detector_config,
                locator_config,
                coarse_start_sample=int(alignment["aligned_start_sample"]),
            )
            seed = detection_to_seed(
                event,
                event_index=packet_index,
                start_sample=int(frame_location.preamble_start_sample),
            )
            estimate = estimate_initial_state(samples, seed, detector_config, search_config)
            rows.append(estimate_to_row(packet_index, event, alignment, frame_location, estimate))
            if args.stft_dir is not None:
                write_frame_stft_plot(
                    samples,
                    frame_location,
                    detector_config,
                    preamble_len,
                    args.stft_dir / f"packet_{packet_index:03d}_event_{event.event_index:03d}_stft.png",
                )

        write_chain_csv(args.output, rows)
        if args.events_csv is not None:
            write_events_csv(args.events_csv, events, detector_config)
        if args.windows_csv is not None:
            write_windows_csv(args.windows_csv, windows, detector_config)
    finally:
        mmap_handle = getattr(samples, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    print(f"windows={len(windows)}")
    print(f"detections={len(events)}")
    print(f"selected_packets={len(rows)}")
    print(f"sf={sf}")
    print(f"preamble_len={preamble_len:g}")
    print(f"chirp_samples={detector_config.chirp_samples}")
    print(f"win_chirps={detector_config.win_chirps}")
    print(f"hop_samples={detector_config.resolved_hop_samples}")
    print(f"min_periodic_peaks={detector_config.min_periodic_peaks}")
    print(f"align_chirps={align_chirps}")
    print(f"frame_search_samples={frame_search_samples}")
    print(f"frame_symbol_search_span={args.frame_symbol_search_span}")
    print(f"sync_word=0x{int(args.sync_word):02x}")
    print(f"estimate_chirps={estimate_chirps}")
    print(f"wrote={args.output}")
    if args.stft_dir is not None:
        print(f"wrote_stft_dir={args.stft_dir}")
    if args.events_csv is not None:
        print(f"wrote_events={args.events_csv}")
    if args.windows_csv is not None:
        print(f"wrote_windows={args.windows_csv}")


if __name__ == "__main__":
    main()
