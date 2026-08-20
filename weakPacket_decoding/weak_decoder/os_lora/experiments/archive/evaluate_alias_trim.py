#!/usr/bin/env python3
"""Evaluate AliasTrim on real LoRa IQ with reproducible sampled alias tones.

An out-of-band continuous-wave blocker and its aliased discrete-time tone are
indistinguishable after sampling.  The evaluator therefore injects the exact
post-sampling equivalent at the existing sample rate; it does not synthesize
or reconstruct a higher-rate waveform.  Frozen synchronization and payload
ground truth are retained, and Savaux/AliasTrim see the same noisy symbol.
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
from weak_decoder.decoding.alias_trim import (  # noqa: E402
    AliasTrimConfig,
    _oversampled_downchirp,
    alias_trim_rerank,
)
from weak_decoder.os_lora.system.nonuniform_sampling import (  # noqa: E402
    prepare_dechirped_symbol,
)


DEFAULT_OUTPUT = WEAK_ROOT / "data" / "experiments" / "alias_trim"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snrs", nargs="+", type=float, default=[-22.0, -24.0, -26.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1401, 1502, 1603])
    parser.add_argument(
        "--blocker-isrs",
        nargs="*",
        type=float,
        default=[25.0, 30.0, 35.0],
        help="active continuous-wave blocker power relative to payload reference power",
    )
    parser.add_argument(
        "--include-no-blocker",
        action="store_true",
        help="also run an AWGN-only safety control",
    )
    parser.add_argument(
        "--blocker-frequency",
        type=float,
        default=0.05,
        help="post-alias normalized frequency in cycles/sample, in [-0.5, 0.5)",
    )
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
    parser.add_argument("--max-blockers", type=int, default=2)
    parser.add_argument("--fft-oversampling", type=int, default=4)
    parser.add_argument("--min-tone-prominence-db", type=float, default=18.0)
    parser.add_argument("--min-tone-power-fraction", type=float, default=0.02)
    parser.add_argument("--min-clean-gain-db", type=float, default=3.0)
    return parser.parse_args()


def _trial_seed(
    base_seed: int,
    dataset_index: int,
    packet_index: int,
    symbol_index: int,
    stream: int,
) -> int:
    value = (
        int(base_seed) * 1_000_003
        + int(dataset_index) * 100_003
        + int(packet_index) * 1_009
        + int(symbol_index) * 17
        + int(stream)
    )
    return int(value % (2**32 - 1))


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


def _blocker_phase(seed: int, dataset_index: int) -> float:
    rng = np.random.default_rng(
        _trial_seed(seed, dataset_index, packet_index=0, symbol_index=0, stream=9)
    )
    return float(rng.uniform(-np.pi, np.pi))


def _add_alias_tone(
    samples: np.ndarray,
    absolute_start: int,
    signal_power: float,
    blocker_isr_db: float | None,
    normalized_frequency: float,
    initial_phase: float,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.complex64)
    if blocker_isr_db is None:
        return values
    amplitude = math.sqrt(
        float(signal_power) * 10.0 ** (float(blocker_isr_db) / 10.0)
    )
    indexes = int(absolute_start) + np.arange(values.size, dtype=np.float64)
    blocker = amplitude * np.exp(
        1j * (2.0 * np.pi * float(normalized_frequency) * indexes + initial_phase)
    )
    return (values + blocker).astype(np.complex64)


def _frequency_error(estimate: float, truth: float) -> float:
    return float((float(estimate) - float(truth) + 0.5) % 1.0 - 0.5)


def _evaluate_dataset(
    dataset: str,
    dataset_index: int,
    args: argparse.Namespace,
    config: AliasTrimConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    iq_path, symbol_path = dataset_paths(dataset)
    if not iq_path.exists():
        raise FileNotFoundError(iq_path)
    if not symbol_path.exists():
        raise FileNotFoundError(symbol_path)
    packets = _selected_packets(
        load_packets(symbol_path), args.packet, int(args.max_packets)
    )
    if not packets:
        raise ValueError(f"no selected packets for {dataset}")
    samples = np.memmap(iq_path, dtype=np.complex64, mode="r")
    reference_power, reference_samples, reference_packets = signal_reference_power(
        samples,
        packets,
        str(args.reference_power_mode),
        args.signal_power,
    )
    blocker_conditions: list[float | None] = [
        float(value) for value in args.blocker_isrs
    ]
    if bool(args.include_no_blocker):
        blocker_conditions.insert(0, None)
    rows: list[dict[str, Any]] = []
    max_symbols = int(args.max_symbols_per_dataset)
    for snr_db in args.snrs:
        for blocker_isr_db in blocker_conditions:
            for base_seed in args.seeds:
                phase = _blocker_phase(int(base_seed), dataset_index)
                trial_symbols = 0
                for packet in packets:
                    sf = int(packet["sf"])
                    os_factor = int(packet["os_factor"])
                    origin_shift = os_factor // 2
                    symbol_samples = (1 << sf) * os_factor
                    downchirp = _oversampled_downchirp(
                        sf,
                        os_factor,
                        int(packet["cfo_int"]),
                        float(packet["cfo_frac"]),
                    )
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
                            clean,
                            float(snr_db),
                            _trial_seed(
                                int(base_seed),
                                dataset_index,
                                int(packet["packet_index"]),
                                symbol_index,
                                stream=1,
                            ),
                            reference_power,
                            noise_shape="white",
                            os_factor=os_factor,
                        )
                        observed = _add_alias_tone(
                            noisy,
                            absolute_start=start,
                            signal_power=reference_power,
                            blocker_isr_db=blocker_isr_db,
                            normalized_frequency=float(args.blocker_frequency),
                            initial_phase=phase,
                        )

                        baseline_start = time.perf_counter()
                        spectrum, _branches, _common_phase = paper_oversampled_spectrum(
                            observed,
                            0,
                            sf,
                            os_factor,
                            int(packet["cfo_int"]),
                            float(packet["cfo_frac"]),
                            0,
                            "symbol",
                        )
                        baseline_seconds = time.perf_counter() - baseline_start
                        alias_start = time.perf_counter()
                        dechirped = prepare_dechirped_symbol(
                            observed,
                            0,
                            sf,
                            os_factor,
                            int(packet["cfo_int"]),
                            float(packet["cfo_frac"]),
                            0,
                            "symbol",
                        )
                        result = alias_trim_rerank(
                            dechirped,
                            spectrum,
                            downchirp,
                            sf,
                            os_factor,
                            config,
                        )
                        alias_seconds = time.perf_counter() - alias_start
                        gt_bin = int(item["gt_bin"])
                        savaux_error = int(result.savaux_bin != gt_bin)
                        alias_error = int(result.selected_bin != gt_bin)
                        first_blocker = result.blockers[0] if result.blockers else None
                        rows.append(
                            {
                                "dataset": dataset,
                                "scenario": (
                                    "awgn"
                                    if blocker_isr_db is None
                                    else "awgn_alias_tone"
                                ),
                                "snr_db": float(snr_db),
                                "blocker_isr_db": (
                                    "" if blocker_isr_db is None else float(blocker_isr_db)
                                ),
                                "blocker_frequency": (
                                    "" if blocker_isr_db is None else float(args.blocker_frequency)
                                ),
                                "seed": int(base_seed),
                                "packet_index": int(packet["packet_index"]),
                                "payload_symbol_index": symbol_index,
                                "raw_start_sample": raw_start,
                                "demod_start_sample": start,
                                "origin_shift": origin_shift,
                                "sf": sf,
                                "os_factor": os_factor,
                                "signal_reference_power": reference_power,
                                "gt_bin": gt_bin,
                                "savaux_bin": int(result.savaux_bin),
                                "cleaned_bin": int(result.cleaned_bin),
                                "alias_trim_bin": int(result.selected_bin),
                                "savaux_error": savaux_error,
                                "alias_trim_error": alias_error,
                                "fix": int(savaux_error == 1 and alias_error == 0),
                                "break": int(savaux_error == 0 and alias_error == 1),
                                "gate_triggered": int(result.gate_triggered),
                                "changed": int(result.changed_from_savaux),
                                "blocker_count": len(result.blockers),
                                "clean_gain_db": float(result.clean_gain_db),
                                "strongest_tone_prominence_db": float(
                                    result.strongest_tone_prominence_db
                                ),
                                "strongest_tone_power_fraction": float(
                                    result.strongest_tone_power_fraction
                                ),
                                "estimated_blocker_frequency": (
                                    ""
                                    if first_blocker is None
                                    else float(first_blocker.normalized_frequency)
                                ),
                                "blocker_frequency_error": (
                                    ""
                                    if first_blocker is None or blocker_isr_db is None
                                    else _frequency_error(
                                        first_blocker.normalized_frequency,
                                        float(args.blocker_frequency),
                                    )
                                ),
                                "savaux_seconds": float(baseline_seconds),
                                "alias_trim_overhead_seconds": float(alias_seconds),
                            }
                        )
                        trial_symbols += 1
                    if max_symbols > 0 and trial_symbols >= max_symbols:
                        break
                blocker_label = "none" if blocker_isr_db is None else f"{blocker_isr_db:g}"
                print(
                    f"{dataset} snr={float(snr_db):g} blocker_isr={blocker_label} "
                    f"seed={int(base_seed)} symbols={trial_symbols}",
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


def _condition_key(row: dict[str, Any]) -> tuple[str, str, float, str]:
    blocker = str(row["blocker_isr_db"])
    return (
        str(row["dataset"]),
        str(row["scenario"]),
        float(row["snr_db"]),
        blocker,
    )


def _aggregate(rows: Sequence[dict[str, Any]], seed: int | str) -> dict[str, Any]:
    count = len(rows)
    savaux_errors = sum(int(row["savaux_error"]) for row in rows)
    alias_errors = sum(int(row["alias_trim_error"]) for row in rows)
    return {
        "dataset": str(rows[0]["dataset"]),
        "scenario": str(rows[0]["scenario"]),
        "snr_db": float(rows[0]["snr_db"]),
        "blocker_isr_db": rows[0]["blocker_isr_db"],
        "seed": seed,
        "symbol_count": count,
        "savaux_errors": savaux_errors,
        "alias_trim_errors": alias_errors,
        "savaux_ser": float(savaux_errors / count) if count else 0.0,
        "alias_trim_ser": float(alias_errors / count) if count else 0.0,
        "net_fixed_errors": int(savaux_errors - alias_errors),
        "fixes": sum(int(row["fix"]) for row in rows),
        "breaks": sum(int(row["break"]) for row in rows),
        "gate_count": sum(int(row["gate_triggered"]) for row in rows),
        "change_count": sum(int(row["changed"]) for row in rows),
        "gate_rate": _mean(rows, "gate_triggered"),
        "change_rate": _mean(rows, "changed"),
        "mean_tone_prominence_db": _mean(rows, "strongest_tone_prominence_db"),
        "mean_tone_power_fraction": _mean(rows, "strongest_tone_power_fraction"),
        "mean_savaux_seconds": _mean(rows, "savaux_seconds"),
        "mean_alias_trim_overhead_seconds": _mean(
            rows, "alias_trim_overhead_seconds"
        ),
    }


def _summary_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    seed_groups: dict[tuple[str, str, float, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _condition_key(row)
        groups[key].append(row)
        seed_groups[key + (int(row["seed"]),)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        output.append(_aggregate(groups[key], seed="all"))
        matching = [item for item in seed_groups if item[:4] == key]
        for seed_key in sorted(matching):
            output.append(_aggregate(seed_groups[seed_key], seed=seed_key[4]))
    return output


def _print_summary(rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        if row["seed"] != "all":
            continue
        blocker = row["blocker_isr_db"] if row["blocker_isr_db"] != "" else "none"
        print(
            "{dataset} snr={snr_db:g} blocker={blocker}: n={symbol_count} "
            "Savaux={savaux_errors} ({savaux_ser:.4f}) AliasTrim={alias_trim_errors} "
            "({alias_trim_ser:.4f}) fixes={fixes} breaks={breaks} gates={gate_count}".format(
                blocker=blocker, **row
            ),
            flush=True,
        )


def main() -> int:
    args = _parse_args()
    config = AliasTrimConfig(
        max_blockers=int(args.max_blockers),
        fft_oversampling=int(args.fft_oversampling),
        min_tone_prominence_db=float(args.min_tone_prominence_db),
        min_tone_power_fraction=float(args.min_tone_power_fraction),
        min_clean_gain_db=float(args.min_clean_gain_db),
    )
    config.validate()
    frequency = float(args.blocker_frequency)
    if not -0.5 <= frequency < 0.5:
        raise ValueError("blocker_frequency must be in [-0.5, 0.5)")
    if not args.snrs or not args.seeds:
        raise ValueError("at least one SNR and seed are required")
    if not args.blocker_isrs and not bool(args.include_no_blocker):
        raise ValueError("select a blocker ISR or --include-no-blocker")

    symbol_rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(args.datasets):
        current, current_metadata = _evaluate_dataset(
            str(dataset), dataset_index, args, config
        )
        symbol_rows.extend(current)
        metadata.append(current_metadata)
    summaries = _summary_rows(symbol_rows)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "symbols.csv", symbol_rows)
    write_csv(output_dir / "summary.csv", summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "datasets": [str(value) for value in args.datasets],
        "snrs": [float(value) for value in args.snrs],
        "seeds": [int(value) for value in args.seeds],
        "blocker_isrs": [float(value) for value in args.blocker_isrs],
        "include_no_blocker": bool(args.include_no_blocker),
        "post_alias_normalized_frequency": frequency,
        "alias_model": (
            "sampled tone equivalent of an out-of-band continuous-wave blocker; "
            "no higher-rate waveform is generated"
        ),
        "origin_shift_samples": "os_factor // 2",
        "reference_power_mode": str(args.reference_power_mode),
        "explicit_signal_power": args.signal_power,
        "max_packets": int(args.max_packets),
        "max_symbols_per_dataset": int(args.max_symbols_per_dataset),
        "alias_trim_config": {
            "max_blockers": config.max_blockers,
            "fft_oversampling": config.fft_oversampling,
            "min_tone_prominence_db": config.min_tone_prominence_db,
            "min_tone_power_fraction": config.min_tone_power_fraction,
            "min_clean_gain_db": config.min_clean_gain_db,
        },
        "datasets_metadata": metadata,
    }
    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _print_summary(summaries)
    print(f"wrote={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
