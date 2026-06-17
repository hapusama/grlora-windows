#!/usr/bin/env python3
"""Plot payload-index phase traces for GT, selector, and center FFT argmax."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from compare_phase_line_to_gt_payload import (  # noqa: E402
    DEFAULT_DATASETS,
    DEFAULT_SNRS,
    add_awgn_from_unit,
    circular_residuals,
    dataset_paths,
    fit_gt_payload_line,
    make_unit_noise,
)
from run_low_snr_gt_bin_experiment import (  # noqa: E402
    estimate_payload_reference_power,
    load_gt_payload_symbols,
)
from run_symbol_phase_two_stage import (  # noqa: E402
    _argmax_bins,
    _extract_payload_spectra_with_coherence,
    _packet_phase_line,
    _ser,
    load_packets,
)
from weak_decoder.symbol_phase_two_stage import SymbolPhaseConfig, select_symbol_bins_two_stage  # noqa: E402


@dataclass(frozen=True)
class PhaseTrace:
    dataset: str
    snr_db: float
    packet_index: int
    payload_len: int
    selected_ser: float
    center_argmax_ser: float
    locked_ratio: float
    selector_line_r2: float
    gt_line_r2: float
    center_argmax_line_r2: float
    selector_line_resid_mean_abs_pi: float
    x_payload: np.ndarray
    gt_phase_pi: np.ndarray
    selector_phase_pi: np.ndarray
    center_argmax_phase_pi: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw per-SNR payload phase traces.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snr-db", type=float, nargs="+", default=list(DEFAULT_SNRS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "phase_line_gt_compare" / "m22_m27_payload_phase_traces_single_packet",
    )
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--cfo-correction-mode", choices=("symbol", "continuous"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--packet", type=int, action="append", default=None)
    parser.add_argument(
        "--grid-per-snr",
        action="store_true",
        help="Draw the old three-panel per-SNR layout instead of one PNG per selected packet.",
    )
    return parser.parse_args()


def phases_at_bins(spectra: Sequence[np.ndarray], raw_bins: Sequence[int]) -> np.ndarray:
    phases: list[float] = []
    count = min(len(spectra), len(raw_bins))
    for idx in range(count):
        spec = spectra[idx]
        raw_bin = int(raw_bins[idx])
        if 0 <= raw_bin < len(spec):
            phases.append(float(np.angle(spec[raw_bin])))
    return np.asarray(phases, dtype=np.float64)


def unwrap_and_align(reference_wrapped: np.ndarray, wrapped: np.ndarray) -> np.ndarray:
    reference = np.unwrap(np.asarray(reference_wrapped, dtype=np.float64))
    values = np.unwrap(np.asarray(wrapped, dtype=np.float64))
    count = min(reference.size, values.size)
    if count == 0:
        return values
    shift = 2.0 * math.pi * round(float(reference[0] - values[0]) / (2.0 * math.pi))
    return values + shift


def normalized_phase_traces(
    gt_wrapped: np.ndarray,
    selector_wrapped: np.ndarray,
    center_argmax_wrapped: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = np.unwrap(gt_wrapped)
    selector = unwrap_and_align(gt_wrapped, selector_wrapped)
    center_argmax = unwrap_and_align(gt_wrapped, center_argmax_wrapped)
    origin = float(gt[0]) if gt.size else 0.0
    return (gt - origin) / math.pi, (selector - origin) / math.pi, (center_argmax - origin) / math.pi


def collect_trace(
    dataset: str,
    snr_db: float,
    packet: dict[str, Any],
    noisy_samples: np.ndarray,
    args: argparse.Namespace,
    config: SymbolPhaseConfig,
) -> PhaseTrace | None:
    center, multi, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
        noisy_samples, packet, args
    )
    if not center or len(center) != len(gt_bins):
        return None

    evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi]
    header_line = _packet_phase_line(noisy_samples, packet, args)
    result = select_symbol_bins_two_stage(
        center_spectra=center,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=config,
        fallback_line=header_line,
        offset_coherences=coherences,
    )
    selected_bins = tuple(int(v) for v in result.selected_raw_bins)
    center_bins = _argmax_bins(center)
    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    _selected_raw_ser, selected_ser, compared = _ser(selected_bins, gt_bins, sf=sf, ldro=ldro)
    _center_raw_ser, center_ser, _ = _ser(center_bins, gt_bins, sf=sf, ldro=ldro)
    if compared <= 0:
        return None

    gt_line, gt_x, gt_phase_unwrapped = fit_gt_payload_line(center, gt_bins, abs_indices)
    center_line, _center_x, _center_phase = fit_gt_payload_line(center, center_bins, abs_indices)
    selector_residual = circular_residuals(gt_phase_unwrapped, result.phase_line, gt_x)
    selector_resid_mean = (
        float(np.mean(np.abs(selector_residual)) / math.pi) if selector_residual.size else float("nan")
    )

    gt_phase = phases_at_bins(center, gt_bins)
    selector_phase = phases_at_bins(center, selected_bins)
    center_argmax_phase = phases_at_bins(center, center_bins)
    count = min(gt_phase.size, selector_phase.size, center_argmax_phase.size)
    if count < 2:
        return None
    gt_pi, selector_pi, center_pi = normalized_phase_traces(
        gt_phase[:count],
        selector_phase[:count],
        center_argmax_phase[:count],
    )

    return PhaseTrace(
        dataset=str(dataset),
        snr_db=float(snr_db),
        packet_index=int(packet["packet_index"]),
        payload_len=int(packet["payload_len"]),
        selected_ser=float(selected_ser),
        center_argmax_ser=float(center_ser),
        locked_ratio=float(result.locked_count / max(1, len(selected_bins))),
        selector_line_r2=float(result.phase_line.fit_r2) if math.isfinite(float(result.phase_line.fit_r2)) else float("nan"),
        gt_line_r2=float(gt_line.fit_r2) if math.isfinite(float(gt_line.fit_r2)) else float("nan"),
        center_argmax_line_r2=float(center_line.fit_r2) if math.isfinite(float(center_line.fit_r2)) else float("nan"),
        selector_line_resid_mean_abs_pi=float(selector_resid_mean),
        x_payload=np.arange(count, dtype=np.float64),
        gt_phase_pi=gt_pi[:count],
        selector_phase_pi=selector_pi[:count],
        center_argmax_phase_pi=center_pi[:count],
    )


def choose_median_trace(traces: Sequence[PhaseTrace]) -> PhaseTrace | None:
    finite = [trace for trace in traces if math.isfinite(float(trace.selector_line_resid_mean_abs_pi))]
    if not finite:
        return traces[0] if traces else None
    values = np.asarray([trace.selector_line_resid_mean_abs_pi for trace in finite], dtype=np.float64)
    target = float(np.median(values))
    return min(finite, key=lambda trace: abs(float(trace.selector_line_resid_mean_abs_pi) - target))


def fit_line(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.size < 2 or y.size < 2:
        return y
    coef = np.polyfit(x, y, deg=1)
    return np.polyval(coef, x)


def snr_label(snr_db: float) -> str:
    value = int(round(abs(float(snr_db))))
    prefix = "m" if float(snr_db) < 0 else "p"
    return f"{prefix}{value}"


def plot_snr(traces: Sequence[PhaseTrace], snr_db: float, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces_sorted = sorted(traces, key=lambda trace: trace.payload_len)
    fig, axes = plt.subplots(1, len(traces_sorted), figsize=(5.6 * len(traces_sorted), 4.6), dpi=160)
    if len(traces_sorted) == 1:
        axes = [axes]

    colors = {
        "gt": "#2f8a46",
        "selector": "#2764b3",
        "argmax": "#b35424",
    }
    for ax, trace in zip(axes, traces_sorted):
        series = [
            ("GT bin", trace.gt_phase_pi, colors["gt"], "o"),
            ("selector bin", trace.selector_phase_pi, colors["selector"], "s"),
            ("center FFT argmax", trace.center_argmax_phase_pi, colors["argmax"], "^"),
        ]
        for label, values, color, marker in series:
            ax.plot(trace.x_payload, values, color=color, alpha=0.35, linewidth=1.0)
            ax.scatter(trace.x_payload, values, color=color, marker=marker, s=16, alpha=0.85, label=label)
            ax.plot(trace.x_payload, fit_line(trace.x_payload, values), color=color, linewidth=2.1, linestyle="--")

        ax.set_title(
            "{}  pkt {}  len {}\nselSER={:.3f} argSER={:.3f} lock={:.2f}".format(
                trace.dataset,
                trace.packet_index,
                trace.payload_len,
                trace.selected_ser,
                trace.center_argmax_ser,
                trace.locked_ratio,
            ),
            fontsize=10,
        )
        ax.set_xlabel("payload symbol index")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.02,
            0.03,
            "R2: selector {:.3f}\nGT {:.3f}  argmax {:.3f}".format(
                trace.selector_line_r2,
                trace.gt_line_r2,
                trace.center_argmax_line_r2,
            ),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 3},
        )
    axes[0].set_ylabel("(unwrapped phase - GT[0]) / pi")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(f"Payload phase traces at SNR {float(snr_db):.0f} dB", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"payload_phase_traces_snr_{snr_label(snr_db)}dB.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_single_trace(trace: PhaseTrace, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.2), dpi=170)
    colors = {
        "gt": "#2f8a46",
        "selector": "#2764b3",
        "argmax": "#b35424",
    }
    series = [
        ("selector bin", trace.selector_phase_pi, colors["selector"], "s"),
        ("center FFT argmax", trace.center_argmax_phase_pi, colors["argmax"], "^"),
        ("GT bin", trace.gt_phase_pi, colors["gt"], "o"),
    ]
    for label, values, color, marker in series:
        ax.plot(trace.x_payload, values, color=color, alpha=0.28, linewidth=1.15)
        if label == "GT bin":
            ax.scatter(
                trace.x_payload,
                values,
                facecolors="white",
                edgecolors=color,
                linewidths=1.8,
                marker=marker,
                s=44,
                alpha=0.98,
                label=label,
                zorder=5,
            )
        else:
            ax.scatter(
                trace.x_payload,
                values,
                color=color,
                marker=marker,
                s=25,
                alpha=0.88,
                label=label,
                zorder=4,
            )

    ax.set_title(
        "{}  SNR {:.0f} dB  packet {}  len {}\nselSER={:.3f}  argSER={:.3f}  lock={:.2f}".format(
            trace.dataset,
            trace.snr_db,
            trace.packet_index,
            trace.payload_len,
            trace.selected_ser,
            trace.center_argmax_ser,
            trace.locked_ratio,
        ),
        fontsize=12,
    )
    ax.set_xlabel("payload symbol index")
    ax.set_ylabel("(unwrapped phase - GT[0]) / pi")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=10)
    ax.text(
        0.02,
        0.03,
        "phase-line R2: selector {:.3f}  GT {:.3f}  argmax {:.3f}".format(
            trace.selector_line_r2,
            trace.gt_line_r2,
            trace.center_argmax_line_r2,
        ),
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 4},
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (
        f"payload_phase_trace_{trace.dataset}_snr_{snr_label(trace.snr_db)}dB"
        f"_packet_{trace.packet_index}.png"
    )
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    config = SymbolPhaseConfig()
    packet_filter = set(int(v) for v in args.packet) if args.packet else None
    traces_by_snr: dict[float, list[PhaseTrace]] = {float(snr): [] for snr in args.snr_db}

    for dataset in args.datasets:
        paths = dataset_paths(str(dataset))
        samples = np.fromfile(paths["iq"], dtype=np.complex64)
        if samples.size == 0:
            raise ValueError(f"empty IQ file: {paths['iq']}")
        gt_symbols = load_gt_payload_symbols(paths["symbols"], packet_filter=None)
        signal_power = estimate_payload_reference_power(samples, gt_symbols)
        packet_map = load_packets(paths["symbols"], None)
        packets = [
            packet
            for packet_index, packet in sorted(packet_map.items())
            if packet_filter is None or int(packet_index) in packet_filter
        ]
        unit_noise = make_unit_noise(samples.size, int(args.seed))

        for snr_db in args.snr_db:
            noise_power = float(signal_power) * (10.0 ** (-float(snr_db) / 10.0))
            noisy_samples = add_awgn_from_unit(samples, unit_noise, noise_power)
            candidates: list[PhaseTrace] = []
            for packet in packets:
                trace = collect_trace(str(dataset), float(snr_db), packet, noisy_samples, args, config)
                if trace is not None:
                    candidates.append(trace)
            chosen = choose_median_trace(candidates)
            if chosen is not None:
                traces_by_snr[float(snr_db)].append(chosen)
                print(
                    f"{dataset} SNR {float(snr_db):.1f}: "
                    f"packet={chosen.packet_index} resid={chosen.selector_line_resid_mean_abs_pi:.3f}pi",
                    flush=True,
                )

    for snr_db in args.snr_db:
        traces = traces_by_snr.get(float(snr_db), [])
        if not traces:
            print(f"skip SNR {float(snr_db):.1f}: no traces")
            continue
        if bool(args.grid_per_snr):
            out_path = plot_snr(traces, float(snr_db), args.output_dir)
            print(f"wrote={out_path}", flush=True)
            continue
        for trace in traces:
            out_path = plot_single_trace(trace, args.output_dir)
            print(f"wrote={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
