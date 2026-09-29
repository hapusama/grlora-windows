#!/usr/bin/env python3
"""Evaluate single-symbol multi-rate LoRa structure under pure complex AWGN.

Synchronization, CFO, SFO, and channel gain are perfect.  Every lower-rate
view is a deterministic phase-zero decimation of the same noisy high-rate
symbol.  The experiment therefore measures detector structure, not additional
information created by resampling.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[4]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.common import write_csv  # noqa: E402
from weak_decoder.chirp import build_upchirp  # noqa: E402
from weak_decoder.os_lora.system.multirate_structure import (  # noqa: E402
    awgn_multirate_glrt_scores,
    build_multirate_spectra,
    coherent_ml_scores,
    fold_pair_components,
    fold_pair_energy_scores,
    fold_profile_scores,
    fold_ratio_consistency,
    mapped_fft_argmax_scores,
    multirate_structure_scores,
)


DEFAULT_OUTPUT = WEAK_ROOT / "data" / "experiments" / "multirate_structure_awgn"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--bandwidth", type=float, default=125_000.0)
    parser.add_argument("--source-os-factor", type=int, default=8)
    parser.add_argument("--rates", nargs="+", type=int, default=[8, 4, 2, 1])
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=[-34.0, -32.0, -30.0, -28.0, -26.0, -24.0, -22.0],
        help="per-high-rate-sample SNR values in dB",
    )
    parser.add_argument("--trials-per-snr", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--timing-offsets",
        nargs="+",
        type=float,
        default=[0.0],
        help="candidate timing grid in chips; zero is the fixed-perfect-sync control",
    )
    parser.add_argument(
        "--rate-weights",
        nargs="+",
        type=float,
        default=None,
        help="optional weights aligned with --rates; default is an unweighted sum",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _method_names(rates: Sequence[int], source_os_factor: int) -> tuple[str, ...]:
    names = [f"fft_argmax_q{int(rate)}" for rate in rates]
    names.extend(
        (
            f"pair_energy_q{int(source_os_factor)}",
            f"fold_profile_q{int(source_os_factor)}",
            "multirate_equal_sum",
            "multirate_awgn_glrt",
            f"coherent_ml_q{int(source_os_factor)}",
        )
    )
    return tuple(names)


def _best_false_score(scores: np.ndarray, groundtruth: int) -> float:
    values = np.asarray(scores, dtype=np.float64).copy()
    values[int(groundtruth)] = -np.inf
    return float(np.max(values))


def _margin_db(scores: np.ndarray, groundtruth: int) -> float:
    gt_score = float(scores[int(groundtruth)])
    false_score = _best_false_score(scores, groundtruth)
    return float(10.0 * math.log10((gt_score + 1e-30) / (false_score + 1e-30)))


def _invariant_rows(
    sf: int,
    source_os_factor: int,
    rates: Sequence[int],
    bandwidth_hz: float,
) -> list[dict[str, Any]]:
    n_bins = 1 << int(sf)
    symbols = tuple(dict.fromkeys((0, 1, n_bins // 8, n_bins // 2, n_bins - 1)))
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        clean = build_upchirp(sf, symbol, source_os_factor)
        observation = build_multirate_spectra(
            clean, sf, source_os_factor=source_os_factor, rates=rates
        )
        for rate in rates:
            spectrum = observation.spectrum(rate)
            primary, secondary = fold_pair_components(spectrum, sf, rate)
            primary_amplitude = float(abs(primary[symbol]))
            length = n_bins * int(rate)
            expected_primary = float(
                int(rate) * (n_bins - symbol) / math.sqrt(float(length))
            )
            if secondary is None:
                expected_primary = math.sqrt(float(n_bins))
                secondary_amplitude = primary_amplitude
                expected_secondary = expected_primary
                separation = 0
                ratio_error = 0.0
            else:
                secondary_amplitude = float(abs(secondary[symbol]))
                expected_secondary = float(
                    int(rate) * symbol / math.sqrt(float(length))
                )
                primary_index = symbol
                secondary_index = symbol + length - n_bins
                separation = int((primary_index - secondary_index) % length)
                observed_total = primary_amplitude + secondary_amplitude
                expected_total = expected_primary + expected_secondary
                observed_ratio = primary_amplitude / max(observed_total, 1e-30)
                expected_ratio = expected_primary / max(expected_total, 1e-30)
                ratio_error = float(abs(observed_ratio - expected_ratio))
            rows.append(
                {
                    "sf": int(sf),
                    "n_bins": n_bins,
                    "symbol": int(symbol),
                    "os_factor": int(rate),
                    "sample_rate_hz": float(rate) * float(bandwidth_hz),
                    "pair_separation_bins": separation,
                    "expected_separation_bins": n_bins if int(rate) > 1 else 0,
                    "primary_amplitude": primary_amplitude,
                    "expected_primary_amplitude": expected_primary,
                    "secondary_amplitude": secondary_amplitude,
                    "expected_secondary_amplitude": expected_secondary,
                    "primary_amplitude_error": float(
                        abs(primary_amplitude - expected_primary)
                    ),
                    "secondary_amplitude_error": float(
                        abs(secondary_amplitude - expected_secondary)
                    ),
                    "wrap_ratio_error": ratio_error,
                }
            )
    return rows


def _run_trials(args: argparse.Namespace) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    sf = int(args.sf)
    n_bins = 1 << sf
    source = int(args.source_os_factor)
    rates = tuple(int(value) for value in args.rates)
    if source not in rates:
        raise ValueError("--rates must include --source-os-factor")
    if int(args.trials_per_snr) <= 0:
        raise ValueError("--trials-per-snr must be positive")
    if args.rate_weights is not None and len(args.rate_weights) != len(rates):
        raise ValueError("--rate-weights length must match --rates")
    methods = _method_names(rates, source)
    reference = build_upchirp(sf, symbol_id=0, os_factor=source)
    length = int(reference.size)
    rows: list[dict[str, Any]] = []
    root_seed = np.random.SeedSequence(int(args.seed))
    snr_sequences = root_seed.spawn(len(args.snrs))
    for snr_index, (snr_db, sequence) in enumerate(
        zip(args.snrs, snr_sequences, strict=True)
    ):
        rng = np.random.default_rng(sequence)
        noise_sigma = math.sqrt(0.5 * 10.0 ** (-float(snr_db) / 10.0))
        errors = {method: 0 for method in methods}
        for trial in range(int(args.trials_per_snr)):
            groundtruth = int(rng.integers(0, n_bins))
            common_phase = float(rng.uniform(-math.pi, math.pi))
            clean = np.roll(reference, -groundtruth * source) * np.exp(
                1j * common_phase
            )
            noise = noise_sigma * (
                rng.normal(size=length) + 1j * rng.normal(size=length)
            )
            samples = (clean + noise).astype(np.complex64)
            observation = build_multirate_spectra(
                samples,
                sf,
                source_os_factor=source,
                rates=rates,
            )
            decisions: dict[str, int] = {}
            for rate in rates:
                method = f"fft_argmax_q{rate}"
                decisions[method] = int(
                    np.argmax(
                        mapped_fft_argmax_scores(
                            observation.spectrum(rate), sf, rate
                        )
                    )
                )
            source_spectrum = observation.spectrum(source)
            pair_scores = fold_pair_energy_scores(source_spectrum, sf, source)
            profile_scores = fold_profile_scores(
                source_spectrum,
                sf,
                source,
                timing_offsets_chips=args.timing_offsets,
            )
            multirate_scores, _rate_scores = multirate_structure_scores(
                observation,
                timing_offsets_chips=args.timing_offsets,
                rate_weights=args.rate_weights,
            )
            ml_scores = coherent_ml_scores(samples, sf, source)
            multirate_glrt_scores = awgn_multirate_glrt_scores(
                observation,
                timing_offsets_chips=args.timing_offsets,
            )
            decisions[f"pair_energy_q{source}"] = int(np.argmax(pair_scores))
            decisions[f"fold_profile_q{source}"] = int(np.argmax(profile_scores))
            decisions["multirate_equal_sum"] = int(np.argmax(multirate_scores))
            decisions["multirate_awgn_glrt"] = int(
                np.argmax(multirate_glrt_scores)
            )
            decisions[f"coherent_ml_q{source}"] = int(np.argmax(ml_scores))
            for method, decision in decisions.items():
                errors[method] += int(decision != groundtruth)

            ratio_fields: dict[str, float] = {}
            for rate in rates:
                consistency = fold_ratio_consistency(
                    observation.spectrum(rate), sf, rate
                )
                ratio_fields[f"gt_ratio_consistency_q{rate}"] = float(
                    consistency[groundtruth]
                )
            row: dict[str, Any] = {
                "snr_db": float(snr_db),
                "snr_index": snr_index,
                "trial": trial,
                "groundtruth": groundtruth,
                "common_phase_rad": common_phase,
                "multirate_equal_sum_gt_margin_db": _margin_db(
                    multirate_scores, groundtruth
                ),
                "multirate_awgn_glrt_gt_margin_db": _margin_db(
                    multirate_glrt_scores, groundtruth
                ),
                "coherent_ml_gt_margin_db": _margin_db(ml_scores, groundtruth),
                **ratio_fields,
            }
            for method in methods:
                row[f"{method}_bin"] = decisions[method]
                row[f"{method}_error"] = int(decisions[method] != groundtruth)
            rows.append(row)
        compact = " ".join(
            f"{method}={errors[method] / int(args.trials_per_snr):.4f}"
            for method in methods
        )
        print(f"snr={float(snr_db):g} n={int(args.trials_per_snr)} {compact}", flush=True)
    return rows, methods


def _summary_rows(
    rows: Sequence[dict[str, Any]],
    methods: Sequence[str],
    source_os_factor: int,
) -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[float(row["snr_db"])].append(row)
    baseline = f"fft_argmax_q{int(source_os_factor)}"
    output: list[dict[str, Any]] = []
    for snr_db in sorted(groups):
        current = groups[snr_db]
        for method in methods:
            error_key = f"{method}_error"
            errors = int(sum(int(row[error_key]) for row in current))
            fixes = int(
                sum(
                    int(row[f"{baseline}_error"] == 1 and row[error_key] == 0)
                    for row in current
                )
            )
            breaks = int(
                sum(
                    int(row[f"{baseline}_error"] == 0 and row[error_key] == 1)
                    for row in current
                )
            )
            output.append(
                {
                    "snr_db": snr_db,
                    "method": method,
                    "trials": len(current),
                    "errors": errors,
                    "ser": float(errors / len(current)),
                    "fixes_vs_source_fft": fixes,
                    "breaks_vs_source_fft": breaks,
                    "net_fixes_vs_source_fft": fixes - breaks,
                }
            )
    return output


def _wrap_region_rows(
    rows: Sequence[dict[str, Any]],
    methods: Sequence[str],
    sf: int,
    source_os_factor: int,
) -> list[dict[str, Any]]:
    """Aggregate errors by distance of the symbol fold from an edge."""

    n_bins = 1 << int(sf)
    edges = (0.0, 0.125, 0.25, 0.375, 0.5000001)
    labels = ("edge", "outer", "inner", "center")
    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        symbol = int(row["groundtruth"])
        balance = min(symbol, n_bins - symbol) / float(n_bins)
        index = int(np.searchsorted(edges, balance, side="right") - 1)
        index = min(max(index, 0), len(labels) - 1)
        groups[(float(row["snr_db"]), labels[index])].append(row)
    baseline = f"fft_argmax_q{int(source_os_factor)}"
    output: list[dict[str, Any]] = []
    for (snr_db, label), current in sorted(groups.items()):
        balances = [
            min(int(row["groundtruth"]), n_bins - int(row["groundtruth"]))
            / float(n_bins)
            for row in current
        ]
        for method in methods:
            error_key = f"{method}_error"
            errors = int(sum(int(row[error_key]) for row in current))
            fixes = int(
                sum(
                    int(row[f"{baseline}_error"] == 1 and row[error_key] == 0)
                    for row in current
                )
            )
            breaks = int(
                sum(
                    int(row[f"{baseline}_error"] == 0 and row[error_key] == 1)
                    for row in current
                )
            )
            output.append(
                {
                    "snr_db": snr_db,
                    "wrap_region": label,
                    "mean_min_wrap_fraction": float(np.mean(balances)),
                    "method": method,
                    "trials": len(current),
                    "errors": errors,
                    "ser": float(errors / len(current)),
                    "fixes_vs_source_fft": fixes,
                    "breaks_vs_source_fft": breaks,
                    "net_fixes_vs_source_fft": fixes - breaks,
                }
            )
    return output


def _write_results(
    path: Path,
    summary: Sequence[dict[str, Any]],
    methods: Sequence[str],
    args: argparse.Namespace,
) -> None:
    by_key = {
        (float(row["snr_db"]), str(row["method"])): row for row in summary
    }
    snrs = sorted({float(row["snr_db"]) for row in summary})
    lines = [
        "# Multi-rate single-symbol AWGN experiment",
        "",
        "All lower-rate views are nested phase-zero decimations of the same noisy input. ",
        "No anti-alias filter, resynchronization, learned noise model, or independent noise draw is used.",
        "",
        f"- SF: {int(args.sf)}",
        f"- Bandwidth: {float(args.bandwidth):g} Hz",
        f"- Source rate: q={int(args.source_os_factor)} ({float(args.bandwidth) * int(args.source_os_factor):g} sample/s)",
        f"- Views: {', '.join(f'q={int(value)}' for value in args.rates)}",
        f"- Trials per SNR: {int(args.trials_per_snr)}",
        "- SNR convention: per-source-sample signal power divided by complex AWGN power.",
        "- Synchronization/CFO/SFO: perfect; unknown common symbol phase is randomized.",
        "",
        "## Symbol error rate",
        "",
        "| SNR (dB) | " + " | ".join(methods) + " |",
        "|---:|" + "---:|" * len(methods),
    ]
    for snr_db in snrs:
        values = [float(by_key[(snr_db, method)]["ser"]) for method in methods]
        lines.append(
            f"| {snr_db:g} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "## Method boundary",
            "",
            f"- `fft_argmax_q{int(args.source_os_factor)}` maps the stronger legal full-rate peak back modulo N; it does not combine the two segments.",
            f"- `pair_energy_q{int(args.source_os_factor)}` uses the N-bin spacing but not the expected wrap ratio.",
            f"- `fold_profile_q{int(args.source_os_factor)}` additionally projects onto the expected `(N-m):m` fold profile.",
            "- `multirate_equal_sum` directly sums the fold-profile scores from all requested rates. It is the simple proposed heuristic, not an independent-view likelihood.",
            "- `multirate_awgn_glrt` whitens the nested-view correlation. Under source-rate white AWGN it algebraically collapses to the source-rate fold profile because the conditional lower-rate residual contains no candidate signal.",
            f"- `coherent_ml_q{int(args.source_os_factor)}` correlates against the complete source-rate LoRa waveform and is the AWGN upper control.",
            "",
            "The invariant audit is in `invariants.csv`, and paired trial decisions are in `trials.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    rows, methods = _run_trials(args)
    summary = _summary_rows(rows, methods, int(args.source_os_factor))
    wrap_regions = _wrap_region_rows(
        rows,
        methods,
        int(args.sf),
        int(args.source_os_factor),
    )
    invariants = _invariant_rows(
        int(args.sf),
        int(args.source_os_factor),
        tuple(int(value) for value in args.rates),
        float(args.bandwidth),
    )
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "trials.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "wrap_regions.csv", wrap_regions)
    write_csv(output_dir / "invariants.csv", invariants)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "sf": int(args.sf),
        "n_bins": 1 << int(args.sf),
        "bandwidth_hz": float(args.bandwidth),
        "source_os_factor": int(args.source_os_factor),
        "source_sample_rate_hz": float(args.bandwidth) * int(args.source_os_factor),
        "rates": [int(value) for value in args.rates],
        "sample_rates_hz": [float(args.bandwidth) * int(value) for value in args.rates],
        "snrs_db": [float(value) for value in args.snrs],
        "trials_per_snr": int(args.trials_per_snr),
        "seed": int(args.seed),
        "timing_offsets_chips": [float(value) for value in args.timing_offsets],
        "rate_weights": (
            None
            if args.rate_weights is None
            else [float(value) for value in args.rate_weights]
        ),
        "noise": "independent circular complex AWGN at the source rate",
        "lower_rate_views": "deterministic phase-zero decimation; no filtering",
        "information_claim": "no independent information is created by rate conversion",
        "methods": list(methods),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_results(output_dir / "RESULTS.md", summary, methods, args)
    print(f"wrote={output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
