#!/usr/bin/env python3
"""Evaluate gated Huber Top-K demodulation on frozen real-IQ payload windows.

The synchronizer and symbol ground truth remain frozen.  Each Savaux/robust
pair sees the same original IQ window, added complex AWGN, and optional sparse
impulsive interference.  The optional interference is useful for checking the
failure mode the robust detector is designed for; pure AWGN is the safety
control and should normally leave the Savaux decision unchanged.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[4]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.common import (  # noqa: E402
    DEFAULT_DATASETS,
    dataset_paths,
    load_packets,
    noise_samples,
    signal_reference_power,
    write_csv,
)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)
from weak_decoder.decoding.robust_sparse_demod import (  # noqa: E402
    RobustSparseConfig,
    robust_sparse_rerank,
)
from weak_decoder.os_lora.system.nonuniform_sampling import (  # noqa: E402
    prepare_dechirped_symbol,
)


DEFAULT_OUTPUT = WEAK_ROOT / "data" / "experiments" / "robust_sparse_demod"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=[-18.0, -20.0, -22.0],
        help="added-AWGN SNR values in dB",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--max-symbols-per-dataset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-power-mode",
        choices=("payload", "header_payload", "whole"),
        default="payload",
    )
    parser.add_argument("--signal-power", type=float, default=None)
    parser.add_argument(
        "--impulse-fraction",
        type=float,
        default=0.0,
        help="fraction of symbol samples receiving extra sparse complex Gaussian noise",
    )
    parser.add_argument(
        "--impulse-isr-db",
        type=float,
        default=20.0,
        help="active-sample impulse power relative to the signal reference power",
    )
    parser.add_argument(
        "--impulse-layout",
        choices=("random", "contiguous"),
        default="random",
    )
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--huber-delta", type=float, default=2.5)
    parser.add_argument("--irls-iterations", type=int, default=3)
    parser.add_argument("--min-outlier-fraction", type=float, default=0.02)
    parser.add_argument("--min-robust-gain-db", type=float, default=3.0)
    return parser.parse_args()


def _trial_seed(
    base_seed: int,
    dataset_index: int,
    packet_index: int,
    symbol_index: int,
    stream: int,
) -> int:
    """Return a stable seed without depending on Python's salted hash."""

    value = (
        int(base_seed) * 1_000_003
        + int(dataset_index) * 100_003
        + int(packet_index) * 1_009
        + int(symbol_index) * 17
        + int(stream)
    )
    return int(value % (2**32 - 1))


def _add_sparse_impulses(
    samples: np.ndarray,
    signal_power: float,
    fraction: float,
    isr_db: float,
    layout: str,
    seed: int,
) -> tuple[np.ndarray, int]:
    values = np.asarray(samples, dtype=np.complex64)
    if float(fraction) <= 0.0:
        return values, 0
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("impulse_fraction must be in (0, 1]")
    count = max(1, min(values.size, int(round(float(fraction) * values.size))))
    rng = np.random.default_rng(int(seed))
    if str(layout) == "contiguous":
        start = int(rng.integers(0, values.size - count + 1))
        indexes = np.arange(start, start + count, dtype=np.int64)
    elif str(layout) == "random":
        indexes = np.asarray(rng.choice(values.size, size=count, replace=False))
    else:
        raise ValueError(f"unknown impulse layout: {layout}")
    impulse_power = float(signal_power) * 10.0 ** (float(isr_db) / 10.0)
    impulses = (
        rng.normal(0.0, math.sqrt(0.5 * impulse_power), count)
        + 1j * rng.normal(0.0, math.sqrt(0.5 * impulse_power), count)
    ).astype(np.complex64)
    output = values.copy()
    output[indexes] += impulses
    return output, int(count)


def _selected_packets(
    packets: Sequence[dict[str, Any]],
    packet_index: int | None,
    max_packets: int,
) -> list[dict[str, Any]]:
    selected = list(packets)
    if packet_index is not None:
        selected = [
            packet
            for packet in selected
            if int(packet["packet_index"]) == int(packet_index)
        ]
    if int(max_packets) > 0:
        selected = selected[: int(max_packets)]
    return selected


def _evaluate_dataset(
    dataset: str,
    dataset_index: int,
    args: argparse.Namespace,
    config: RobustSparseConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    iq_path, symbol_path = dataset_paths(dataset)
    if not iq_path.exists():
        raise FileNotFoundError(iq_path)
    if not symbol_path.exists():
        raise FileNotFoundError(symbol_path)
    packets = _selected_packets(
        load_packets(symbol_path),
        packet_index=args.packet,
        max_packets=int(args.max_packets),
    )
    if not packets:
        raise ValueError(f"no selected packets for {dataset}")
    samples = np.memmap(iq_path, dtype=np.complex64, mode="r")
    reference_power, reference_samples, reference_packets = signal_reference_power(
        samples=samples,
        packets=packets,
        mode=str(args.reference_power_mode),
        explicit_power=args.signal_power,
    )
    rows: list[dict[str, Any]] = []
    max_symbols = int(args.max_symbols_per_dataset)
    for snr_db in args.snrs:
        for base_seed in args.seeds:
            trial_symbols = 0
            for packet in packets:
                sf = int(packet["sf"])
                os_factor = int(packet["os_factor"])
                origin_shift = os_factor // 2
                symbol_samples = (1 << sf) * os_factor
                for item in packet["payload_symbols"]:
                    if max_symbols > 0 and trial_symbols >= max_symbols:
                        break
                    symbol_index = int(item["payload_symbol_index"])
                    raw_start = int(item["start_sample"])
                    start = raw_start + origin_shift
                    stop = start + symbol_samples
                    if start < 0 or stop > samples.size:
                        continue
                    clean = np.asarray(samples[start:stop], dtype=np.complex64)
                    noisy = noise_samples(
                        clean=clean,
                        snr_db=float(snr_db),
                        seed=_trial_seed(
                            int(base_seed),
                            dataset_index,
                            int(packet["packet_index"]),
                            symbol_index,
                            stream=1,
                        ),
                        signal_reference_power=reference_power,
                        noise_shape="white",
                        os_factor=os_factor,
                    )
                    observed, impulse_count = _add_sparse_impulses(
                        samples=noisy,
                        signal_power=reference_power,
                        fraction=float(args.impulse_fraction),
                        isr_db=float(args.impulse_isr_db),
                        layout=str(args.impulse_layout),
                        seed=_trial_seed(
                            int(base_seed),
                            dataset_index,
                            int(packet["packet_index"]),
                            symbol_index,
                            stream=2,
                        ),
                    )

                    baseline_start = time.perf_counter()
                    spectrum, _branches, _phase = paper_oversampled_spectrum(
                        samples=observed,
                        start_sample=0,
                        sf=sf,
                        os_factor=os_factor,
                        cfo_int=int(packet["cfo_int"]),
                        cfo_frac=float(packet["cfo_frac"]),
                        header_start_sample=0,
                        cfo_correction_mode="symbol",
                    )
                    dechirped = prepare_dechirped_symbol(
                        samples=observed,
                        start_sample=0,
                        sf=sf,
                        os_factor=os_factor,
                        cfo_int=int(packet["cfo_int"]),
                        cfo_frac=float(packet["cfo_frac"]),
                        header_start_sample=0,
                        cfo_correction_mode="symbol",
                    )
                    baseline_seconds = time.perf_counter() - baseline_start
                    robust_start = time.perf_counter()
                    result = robust_sparse_rerank(
                        dechirped=dechirped,
                        savaux_spectrum=spectrum,
                        sf=sf,
                        os_factor=os_factor,
                        config=config,
                    )
                    robust_seconds = time.perf_counter() - robust_start
                    gt_bin = int(item["gt_bin"])
                    savaux_error = int(result.savaux_bin != gt_bin)
                    robust_error = int(result.selected_bin != gt_bin)
                    rows.append(
                        {
                            "dataset": dataset,
                            "scenario": (
                                "awgn"
                                if float(args.impulse_fraction) <= 0.0
                                else f"awgn_sparse_{args.impulse_layout}"
                            ),
                            "snr_db": float(snr_db),
                            "seed": int(base_seed),
                            "packet_index": int(packet["packet_index"]),
                            "payload_symbol_index": symbol_index,
                            "raw_start_sample": raw_start,
                            "demod_start_sample": start,
                            "origin_shift": origin_shift,
                            "sf": sf,
                            "os_factor": os_factor,
                            "signal_reference_power": reference_power,
                            "impulse_fraction": float(args.impulse_fraction),
                            "impulse_isr_db": (
                                float(args.impulse_isr_db)
                                if float(args.impulse_fraction) > 0.0
                                else ""
                            ),
                            "impulse_sample_count": impulse_count,
                            "gt_bin": gt_bin,
                            "savaux_bin": int(result.savaux_bin),
                            "robust_bin": int(result.selected_bin),
                            "savaux_error": savaux_error,
                            "robust_error": robust_error,
                            "fix": int(savaux_error == 1 and robust_error == 0),
                            "break": int(savaux_error == 0 and robust_error == 1),
                            "gate_triggered": int(result.gate_triggered),
                            "changed": int(result.changed_from_savaux),
                            "outlier_fraction": float(result.outlier_fraction),
                            "estimated_noise_scale": float(result.noise_scale),
                            "robust_gain_db": float(result.robust_gain_db),
                            "gt_in_top_k": int(gt_bin in result.candidate_bins),
                            "candidate_bins": " ".join(
                                str(value) for value in result.candidate_bins
                            ),
                            "savaux_seconds": float(baseline_seconds),
                            "robust_overhead_seconds": float(robust_seconds),
                        }
                    )
                    trial_symbols += 1
                if max_symbols > 0 and trial_symbols >= max_symbols:
                    break
            print(
                f"{dataset} snr={float(snr_db):g} seed={int(base_seed)} "
                f"symbols={trial_symbols}",
                flush=True,
            )
    metadata = {
        "dataset": dataset,
        "iq_path": str(iq_path),
        "symbol_path": str(symbol_path),
        "selected_packet_count": len(packets),
        "signal_reference_power": reference_power,
        "signal_reference_sample_count": reference_samples,
        "signal_reference_packet_count": reference_packets,
    }
    return rows, metadata


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _aggregate(rows: Sequence[dict[str, Any]], seed: int | str) -> dict[str, Any]:
    count = len(rows)
    savaux_errors = sum(int(row["savaux_error"]) for row in rows)
    robust_errors = sum(int(row["robust_error"]) for row in rows)
    fixes = sum(int(row["fix"]) for row in rows)
    breaks = sum(int(row["break"]) for row in rows)
    return {
        "dataset": str(rows[0]["dataset"]),
        "scenario": str(rows[0]["scenario"]),
        "snr_db": float(rows[0]["snr_db"]),
        "seed": seed,
        "symbol_count": count,
        "savaux_errors": savaux_errors,
        "robust_errors": robust_errors,
        "savaux_ser": float(savaux_errors / count) if count else 0.0,
        "robust_ser": float(robust_errors / count) if count else 0.0,
        "net_fixed_errors": int(savaux_errors - robust_errors),
        "fixes": fixes,
        "breaks": breaks,
        "gate_count": sum(int(row["gate_triggered"]) for row in rows),
        "change_count": sum(int(row["changed"]) for row in rows),
        "gate_rate": _mean(rows, "gate_triggered"),
        "change_rate": _mean(rows, "changed"),
        "gt_top_k_rate": _mean(rows, "gt_in_top_k"),
        "mean_outlier_fraction": _mean(rows, "outlier_fraction"),
        "mean_savaux_seconds": _mean(rows, "savaux_seconds"),
        "mean_robust_overhead_seconds": _mean(rows, "robust_overhead_seconds"),
    }


def _summary_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    seed_groups: dict[tuple[str, str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset"]), str(row["scenario"]), float(row["snr_db"]))
        groups[key].append(row)
        seed_groups[key + (int(row["seed"]),)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        output.append(_aggregate(groups[key], seed="all"))
        matching = [item for item in seed_groups if item[:3] == key]
        for seed_key in sorted(matching):
            output.append(_aggregate(seed_groups[seed_key], seed=seed_key[3]))
    return output


def _print_summary(rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        if row["seed"] != "all":
            continue
        print(
            "{dataset} {scenario} snr={snr_db:g}: n={symbol_count} "
            "Savaux={savaux_errors} ({savaux_ser:.4f}) robust={robust_errors} "
            "({robust_ser:.4f}) fixes={fixes} breaks={breaks} gates={gate_count}".format(
                **row
            ),
            flush=True,
        )


def main() -> int:
    args = _parse_args()
    config = RobustSparseConfig(
        candidate_count=int(args.candidate_count),
        huber_delta=float(args.huber_delta),
        irls_iterations=int(args.irls_iterations),
        min_outlier_fraction=float(args.min_outlier_fraction),
        min_robust_gain_db=float(args.min_robust_gain_db),
    )
    config.validate()
    if not args.snrs:
        raise ValueError("at least one SNR is required")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if not 0.0 <= float(args.impulse_fraction) <= 1.0:
        raise ValueError("impulse_fraction must be in [0, 1]")

    symbol_rows: list[dict[str, Any]] = []
    datasets_metadata: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(args.datasets):
        dataset_rows, metadata = _evaluate_dataset(
            dataset=str(dataset),
            dataset_index=dataset_index,
            args=args,
            config=config,
        )
        symbol_rows.extend(dataset_rows)
        datasets_metadata.append(metadata)
    summaries = _summary_rows(symbol_rows)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "symbols.csv", symbol_rows)
    write_csv(output_dir / "summary.csv", summaries)
    configuration = {
        "datasets": [str(value) for value in args.datasets],
        "snrs": [float(value) for value in args.snrs],
        "seeds": [int(value) for value in args.seeds],
        "packet": args.packet,
        "max_packets": int(args.max_packets),
        "max_symbols_per_dataset": int(args.max_symbols_per_dataset),
        "reference_power_mode": str(args.reference_power_mode),
        "explicit_signal_power": args.signal_power,
        "impulse_fraction": float(args.impulse_fraction),
        "impulse_isr_db": float(args.impulse_isr_db),
        "impulse_layout": str(args.impulse_layout),
        "robust_config": {
            "candidate_count": config.candidate_count,
            "huber_delta": config.huber_delta,
            "irls_iterations": config.irls_iterations,
            "min_outlier_fraction": config.min_outlier_fraction,
            "min_robust_gain_db": config.min_robust_gain_db,
        },
        "synchronization": (
            "frozen header-first CSV; payload symbols only; "
            "demodulation origin shifted by os_factor // 2"
        ),
        "datasets_metadata": datasets_metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _print_summary(summaries)
    print(f"wrote={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
