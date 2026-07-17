#!/usr/bin/env python3
"""Run aligned LoRa SER experiments over Sionna RT indoor channels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)
from weak_decoder.chirp import build_upchirp  # noqa: E402
from weak_decoder.os_lora.evaluate_pattern_fft_coherence import _background_bins  # noqa: E402
from weak_decoder.os_lora.nonuniform_sampling import (  # noqa: E402
    NonuniformPatternBank,
    build_pattern_bank,
    conditional_lora_gls_detect,
    crossfit_gls_spectrum_power,
    crossfit_weighted_spectrum,
    lora_branch_color_mismatch,
    lora_wrap_consistency_power,
    matrix_free_crossfit_gls_spectrum_power,
    pattern_bank_split_spectra,
    prepare_dechirped_symbol,
)


BOLTZMANN = 1.380649e-23
CURRENT_PATTERN_NAMES = (
    "multiscale_w32_s1_b1",
    "multiscale_w32_s1_b0",
    "multiscale_w32_s1_b2",
    "multiscale_w32_s1_b3",
    "block_w2_s3_b3",
    "block_w8_s1_b0",
    "block_w32_s1_b3",
    "multiscale_w2_s3_b3",
)
METHODS = (
    "branch0",
    "savaux",
    "equal8",
    "equal8_wrap",
    "gls8_direct",
    "gls8_cg",
    "gls8_cg_wrap",
    "conditional_gls",
)

DEFAULT_CIRS = (
    WEAK_ROOT / "Buildings" / "B" / "sionna_scene" / "hospital_cirs.npz"
)
DEFAULT_OUTPUT = WEAK_ROOT / "data" / "sionna_rt" / "final" / "aligned_ser"


def _float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _string_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cirs", type=Path, default=DEFAULT_CIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--os-factor", type=int, default=4)
    parser.add_argument("--bandwidth", type=float, default=125e3)
    parser.add_argument("--temperature", type=float, default=290.0)
    parser.add_argument("--noise-figure-db", type=float, default=6.0)
    parser.add_argument("--noise-filter-taps", type=int, default=129)
    parser.add_argument("--symbols", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--target-tx-dbm",
        type=_float_list,
        default=(-58.0, -60.0, -62.0, -64.0),
        help="Comma-separated target transmit powers.",
    )
    parser.add_argument("--interferer-tx-dbm", type=float, default=-74.0)
    parser.add_argument(
        "--scenarios",
        type=_string_list,
        default=("thermal_only", "one_interferer", "two_interferers"),
    )
    parser.add_argument("--exclude-top", type=int, default=8)
    parser.add_argument("--exclude-guard-bins", type=int, default=1)
    parser.add_argument("--diagonal-loading", type=float, default=0.05)
    parser.add_argument("--crossfit-folds", type=int, default=2)
    parser.add_argument("--cg-iterations", type=int, default=4)
    parser.add_argument("--savaux-gate-margin-db", type=float, default=0.5)
    parser.add_argument("--branch-color-threshold", type=float, default=0.107121)
    parser.add_argument("--wrap-exponent", type=float, default=0.5)
    return parser.parse_args()


def _dbm_to_watts(value: float) -> float:
    return 1e-3 * 10.0 ** (float(value) / 10.0)


def _power_db(value: float) -> float:
    return 10.0 * math.log10(max(float(value), 1e-300))


def _current_bank(sf: int, os_factor: int) -> NonuniformPatternBank:
    full = build_pattern_bank(sf, os_factor, kind="multiscale_only")
    index = {name: idx for idx, name in enumerate(full.names)}
    missing = [name for name in CURRENT_PATTERN_NAMES if name not in index]
    if missing:
        raise RuntimeError(f"current pattern bank is missing {missing}")
    selected = [index[name] for name in CURRENT_PATTERN_NAMES]
    return NonuniformPatternBank(
        names=tuple(full.names[idx] for idx in selected),
        offsets=tuple(full.offsets[idx] for idx in selected),
        os_factor=int(os_factor),
        sf=int(sf),
        kind="current_information8",
    )


def _target_response_diagnostics(
    links: dict[str, tuple[np.ndarray, np.ndarray]],
    bank: NonuniformPatternBank,
    sf: int,
    os_factor: int,
    bandwidth: float,
    receiver_filter_taps: int,
) -> dict[str, object]:
    """Measure the noiseless correct-bin response used by the GLS steering vector."""
    symbol_length = (1 << int(sf)) * int(os_factor)
    waveform = _symbol_stream(
        sf,
        os_factor,
        np.zeros(3, dtype=np.int64),
    )
    received = _apply_cir(
        waveform,
        links["target"][0],
        links["target"][1],
        float(bandwidth) * int(os_factor),
    )
    received = _receiver_filter(
        received,
        _lowpass_taps(
            int(receiver_filter_taps),
            float(bandwidth),
            float(bandwidth) * int(os_factor),
        ),
    )
    dechirped = prepare_dechirped_symbol(
        received,
        start_sample=symbol_length,
        sf=sf,
        os_factor=os_factor,
        cfo_correction_mode="none",
    )
    spectra, _, _ = pattern_bank_split_spectra(dechirped, bank)
    response = np.asarray(spectra[:, 0], dtype=np.complex128)
    mean_response = complex(np.mean(response))
    relative = response / mean_response
    amplitude_db = 20.0 * np.log10(np.maximum(np.abs(relative), 1e-300))
    phase_deg = np.rad2deg(np.angle(relative))
    rms_relative_error = float(
        np.linalg.norm(response - mean_response)
        / max(float(np.linalg.norm(response)), 1e-300)
    )
    return {
        "assumption": "correct-bin steering vector is proportional to the all-ones vector",
        "pattern_count": int(response.size),
        "max_amplitude_spread_db": float(np.ptp(amplitude_db)),
        "max_phase_spread_deg": float(np.ptp(phase_deg)),
        "rms_relative_error_from_equal_response": rms_relative_error,
    }


def _load_cirs(path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        links: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in metadata["links"]:
            links[name] = (
                np.asarray(archive[f"{name}_coefficients"], dtype=np.complex128),
                np.asarray(archive[f"{name}_delays_s"], dtype=np.float64),
            )
    return links, metadata


def _symbol_stream(sf: int, os_factor: int, values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [build_upchirp(sf=sf, symbol_id=int(value), os_factor=os_factor) for value in values]
    ).astype(np.complex128)


def _apply_cir(
    samples: np.ndarray,
    coefficients: np.ndarray,
    delays_s: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    signal = np.asarray(samples, dtype=np.complex128)
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    delays = np.asarray(delays_s, dtype=np.float64)
    if coefficients.size == 0 or delays.size == 0:
        return np.zeros_like(signal)
    relative_delays = delays - float(np.min(delays))
    extra = max(16, int(math.ceil(float(np.max(relative_delays)) * sample_rate)) + 16)
    fft_size = 1 << int(math.ceil(math.log2(max(2, signal.size + extra))))
    frequencies = np.fft.fftfreq(fft_size, d=1.0 / float(sample_rate))
    response = np.sum(
        coefficients[:, None]
        * np.exp(-2j * np.pi * relative_delays[:, None] * frequencies[None, :]),
        axis=0,
    )
    output = np.fft.ifft(np.fft.fft(signal, fft_size) * response)
    return np.asarray(output[: signal.size], dtype=np.complex128)


def _lowpass_taps(count: int, bandwidth: float, sample_rate: float) -> np.ndarray:
    tap_count = max(3, int(count))
    if tap_count % 2 == 0:
        tap_count += 1
    cutoff = 0.5 * float(bandwidth)
    if not (0.0 < cutoff < 0.5 * float(sample_rate)):
        raise ValueError("noise filter cutoff must lie inside Nyquist")
    centered = np.arange(tap_count, dtype=np.float64) - 0.5 * (tap_count - 1)
    normalized_cutoff = cutoff / float(sample_rate)
    taps = 2.0 * normalized_cutoff * np.sinc(2.0 * normalized_cutoff * centered)
    taps *= np.hamming(tap_count)
    taps /= np.sum(taps)
    return taps


def _receiver_filter(samples: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply the shared linear-phase receiver filter with delay compensation."""
    signal = np.asarray(samples, dtype=np.complex128)
    coefficients = np.asarray(taps, dtype=np.float64)
    group_delay = (coefficients.size - 1) // 2
    filtered = np.convolve(signal, coefficients, mode="full")
    return np.asarray(
        filtered[group_delay : group_delay + signal.size],
        dtype=np.complex128,
    )


def _thermal_noise_input(
    rng: np.random.Generator,
    length: int,
    bandwidth: float,
    temperature: float,
    noise_figure_db: float,
    taps: np.ndarray,
) -> tuple[np.ndarray, float]:
    output_noise_power = (
        BOLTZMANN
        * float(temperature)
        * float(bandwidth)
        * 10.0 ** (float(noise_figure_db) / 10.0)
    )
    filter_noise_gain = max(float(np.sum(np.abs(taps) ** 2)), 1e-300)
    input_noise_power = output_noise_power / filter_noise_gain
    noise = (
        rng.standard_normal(int(length)) + 1j * rng.standard_normal(int(length))
    ) / math.sqrt(2.0)
    noise *= math.sqrt(input_noise_power)
    return np.asarray(noise, dtype=np.complex128), float(output_noise_power)


def _interferer_component(
    rng: np.random.Generator,
    link: tuple[np.ndarray, np.ndarray],
    sf: int,
    os_factor: int,
    sample_rate: float,
    output_length: int,
    tx_power_dbm: float,
    timing_fraction: float,
    cfo_hz: float,
) -> np.ndarray:
    symbol_length = (1 << int(sf)) * int(os_factor)
    timing_samples = int(round(float(timing_fraction) * symbol_length))
    symbol_count = int(math.ceil((output_length + timing_samples) / symbol_length)) + 3
    values = rng.integers(0, 1 << int(sf), size=symbol_count, dtype=np.int64)
    waveform = _symbol_stream(sf, os_factor, values)
    time = np.arange(waveform.size, dtype=np.float64) / float(sample_rate)
    phase = rng.uniform(-np.pi, np.pi)
    waveform *= np.exp(1j * (2.0 * np.pi * float(cfo_hz) * time + phase))
    coefficients, delays = link
    propagated = _apply_cir(waveform, coefficients, delays, sample_rate)
    propagated *= math.sqrt(_dbm_to_watts(tx_power_dbm))
    return np.asarray(propagated[timing_samples : timing_samples + output_length], dtype=np.complex128)


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(materialized[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _evaluate_condition(
    args: argparse.Namespace,
    links: dict[str, tuple[np.ndarray, np.ndarray]],
    bank: NonuniformPatternBank,
    scenario: str,
    target_tx_dbm: float,
    trial: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    sf = int(args.sf)
    os_factor = int(args.os_factor)
    n_bins = 1 << sf
    symbol_length = n_bins * os_factor
    sample_rate = float(args.bandwidth) * os_factor
    guard_symbols = 2
    rng = np.random.default_rng(int(args.seed) + 100_003 * trial)
    target_values = rng.integers(
        0,
        n_bins,
        size=int(args.symbols) + 2 * guard_symbols,
        dtype=np.int64,
    )
    target_waveform = _symbol_stream(sf, os_factor, target_values)
    target_component = _apply_cir(
        target_waveform,
        links["target"][0],
        links["target"][1],
        sample_rate,
    )
    target_component *= math.sqrt(_dbm_to_watts(target_tx_dbm))

    interference = np.zeros_like(target_component)
    if scenario in {"one_interferer", "two_interferers"}:
        interference += _interferer_component(
            rng,
            links["interferer_a"],
            sf,
            os_factor,
            sample_rate,
            target_component.size,
            float(args.interferer_tx_dbm),
            timing_fraction=0.37,
            cfo_hz=1900.0,
        )
    if scenario == "two_interferers":
        interference += _interferer_component(
            rng,
            links["interferer_b"],
            sf,
            os_factor,
            sample_rate,
            target_component.size,
            float(args.interferer_tx_dbm) - 3.0,
            timing_fraction=0.61,
            cfo_hz=-3100.0,
        )
    if scenario not in {"thermal_only", "one_interferer", "two_interferers"}:
        raise ValueError(f"unknown scenario: {scenario}")

    receiver_taps = _lowpass_taps(
        int(args.noise_filter_taps),
        float(args.bandwidth),
        sample_rate,
    )
    noise_input, theoretical_noise_power = _thermal_noise_input(
        rng,
        target_component.size,
        float(args.bandwidth),
        float(args.temperature),
        float(args.noise_figure_db),
        receiver_taps,
    )
    target_component = _receiver_filter(target_component, receiver_taps)
    interference = _receiver_filter(interference, receiver_taps)
    noise = _receiver_filter(noise_input, receiver_taps)
    received = np.asarray(target_component + interference + noise, dtype=np.complex64)

    eval_start = guard_symbols * symbol_length
    eval_stop = eval_start + int(args.symbols) * symbol_length
    target_power = float(np.mean(np.abs(target_component[eval_start:eval_stop]) ** 2))
    interference_power = float(np.mean(np.abs(interference[eval_start:eval_stop]) ** 2))
    noise_power = float(np.mean(np.abs(noise[eval_start:eval_stop]) ** 2))

    errors = defaultdict(int)
    fixes = defaultdict(int)
    breaks = defaultdict(int)
    conditional_runs = 0
    conditional_screened = 0
    margins: list[float] = []
    color_values: list[float] = []
    symbol_rows: list[dict[str, object]] = []

    for symbol_index in range(int(args.symbols)):
        start = eval_start + symbol_index * symbol_length
        gt_bin = int(target_values[guard_symbols + symbol_index])
        savaux_spectrum, branch_spectra, _ = paper_oversampled_spectrum(
            received,
            start_sample=start,
            sf=sf,
            os_factor=os_factor,
            cfo_correction_mode="none",
        )
        savaux_power = np.abs(savaux_spectrum).astype(np.float64) ** 2
        savaux_bin = int(np.argmax(savaux_power))
        top_two = np.partition(savaux_power, -2)[-2:]
        margin_db = float(
            10.0
            * np.log10(
                (float(np.max(top_two)) + 1e-300)
                / (float(np.min(top_two)) + 1e-300)
            )
        )
        background = _background_bins(
            savaux_power,
            exclude_top=int(args.exclude_top),
            guard_bins=int(args.exclude_guard_bins),
        )
        color_mismatch = lora_branch_color_mismatch(branch_spectra, background)
        margins.append(margin_db)
        color_values.append(color_mismatch)

        dechirped = prepare_dechirped_symbol(
            received,
            start_sample=start,
            sf=sf,
            os_factor=os_factor,
            cfo_correction_mode="none",
        )
        spectra, head_spectra, tail_spectra = pattern_bank_split_spectra(dechirped, bank)
        equal_power = np.abs(np.sum(spectra, axis=0)) ** 2
        equal_wrap_power = lora_wrap_consistency_power(
            np.sum(head_spectra, axis=0),
            np.sum(tail_spectra, axis=0),
            base_power=equal_power,
            exponent=float(args.wrap_exponent),
            minimum_segment=16,
        )
        direct_power = crossfit_gls_spectrum_power(
            spectra,
            covariance_bins=background,
            diagonal_loading=float(args.diagonal_loading),
            folds=int(args.crossfit_folds),
        )
        cg_result = matrix_free_crossfit_gls_spectrum_power(
            spectra,
            covariance_bins=background,
            diagonal_loading=float(args.diagonal_loading),
            folds=int(args.crossfit_folds),
            max_iterations=int(args.cg_iterations),
            tolerance=0.0,
        )
        weighted_head = crossfit_weighted_spectrum(head_spectra, cg_result.inverse_targets)
        weighted_tail = crossfit_weighted_spectrum(tail_spectra, cg_result.inverse_targets)
        wrap_power = lora_wrap_consistency_power(
            weighted_head,
            weighted_tail,
            base_power=cg_result.power,
            exponent=float(args.wrap_exponent),
            minimum_segment=16,
        )
        conditional = conditional_lora_gls_detect(
            dechirped=dechirped,
            savaux_spectrum=savaux_spectrum,
            savaux_branch_spectra=branch_spectra,
            bank=bank,
            covariance_bins=background,
            savaux_margin_db=float(args.savaux_gate_margin_db),
            branch_color_threshold=float(args.branch_color_threshold),
            diagonal_loading=float(args.diagonal_loading),
            folds=int(args.crossfit_folds),
            max_iterations=int(args.cg_iterations),
            tolerance=0.0,
            wrap_consistency_exponent=float(args.wrap_exponent),
        )
        conditional_runs += int(conditional.backend_ran)
        conditional_screened += int(conditional.screened)

        selected = {
            "branch0": int(np.argmax(np.abs(branch_spectra[0]) ** 2)),
            "savaux": savaux_bin,
            "equal8": int(np.argmax(equal_power)),
            "equal8_wrap": int(np.argmax(equal_wrap_power)),
            "gls8_direct": int(np.argmax(direct_power)),
            "gls8_cg": int(np.argmax(cg_result.power)),
            "gls8_cg_wrap": int(np.argmax(wrap_power)),
            "conditional_gls": int(conditional.raw_fft_bin),
        }
        for method, raw_bin in selected.items():
            errors[method] += int(raw_bin != gt_bin)
            if method != "savaux":
                fixes[method] += int(savaux_bin != gt_bin and raw_bin == gt_bin)
                breaks[method] += int(savaux_bin == gt_bin and raw_bin != gt_bin)
        symbol_rows.append(
            {
                "scenario": scenario,
                "target_tx_dbm": float(target_tx_dbm),
                "trial": int(trial),
                "symbol_index": int(symbol_index),
                "gt_bin": gt_bin,
                **{f"{method}_bin": selected[method] for method in METHODS},
                "savaux_margin_db": margin_db,
                "branch_color_mismatch": color_mismatch,
                "conditional_backend_ran": int(conditional.backend_ran),
            }
        )

    result = {
        "scenario": scenario,
        "target_tx_dbm": float(target_tx_dbm),
        "interferer_tx_dbm": float(args.interferer_tx_dbm),
        "trial": int(trial),
        "symbol_count": int(args.symbols),
        "target_power_w": target_power,
        "interference_power_w": interference_power,
        "noise_power_w": noise_power,
        "theoretical_noise_power_w": theoretical_noise_power,
        "target_snr_db": _power_db(target_power / max(noise_power, 1e-300)),
        "target_sinr_db": _power_db(
            target_power / max(noise_power + interference_power, 1e-300)
        ),
        "mean_savaux_margin_db": float(np.mean(margins)),
        "mean_branch_color_mismatch": float(np.mean(color_values)),
        "conditional_screen_rate": float(conditional_screened / max(1, int(args.symbols))),
        "conditional_backend_rate": float(conditional_runs / max(1, int(args.symbols))),
    }
    for method in METHODS:
        result[f"{method}_errors"] = int(errors[method])
        result[f"{method}_ser"] = float(errors[method] / max(1, int(args.symbols)))
        result[f"{method}_fixes_vs_savaux"] = int(fixes[method])
        result[f"{method}_breaks_vs_savaux"] = int(breaks[method])
    return symbol_rows, result


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario"]), float(row["target_tx_dbm"]))].append(row)
    output: list[dict[str, object]] = []
    for (scenario, target_tx_dbm), condition in sorted(grouped.items()):
        symbol_count = sum(int(row["symbol_count"]) for row in condition)
        item: dict[str, object] = {
            "scenario": scenario,
            "target_tx_dbm": target_tx_dbm,
            "trials": len(condition),
            "symbol_count": symbol_count,
            "mean_target_snr_db": float(np.mean([float(row["target_snr_db"]) for row in condition])),
            "mean_target_sinr_db": float(np.mean([float(row["target_sinr_db"]) for row in condition])),
            "mean_branch_color_mismatch": float(
                np.mean([float(row["mean_branch_color_mismatch"]) for row in condition])
            ),
            "conditional_backend_rate": float(
                np.mean([float(row["conditional_backend_rate"]) for row in condition])
            ),
        }
        for method in METHODS:
            errors = sum(int(row[f"{method}_errors"]) for row in condition)
            item[f"{method}_errors"] = errors
            item[f"{method}_ser"] = float(errors / max(1, symbol_count))
            item[f"{method}_fixes_vs_savaux"] = sum(
                int(row[f"{method}_fixes_vs_savaux"]) for row in condition
            )
            item[f"{method}_breaks_vs_savaux"] = sum(
                int(row[f"{method}_breaks_vs_savaux"]) for row in condition
            )
        output.append(item)
    return output


def _aggregate_scopes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scopes = (
        ("thermal_only", lambda row: str(row["scenario"]) == "thermal_only"),
        ("interference", lambda row: str(row["scenario"]) != "thermal_only"),
        ("all", lambda _row: True),
    )
    output: list[dict[str, object]] = []
    for scope, predicate in scopes:
        condition = [row for row in rows if predicate(row)]
        if not condition:
            continue
        symbol_count = sum(int(row["symbol_count"]) for row in condition)
        for method in METHODS:
            errors = sum(int(row[f"{method}_errors"]) for row in condition)
            output.append(
                {
                    "scope": scope,
                    "method": method,
                    "symbol_count": symbol_count,
                    "errors": errors,
                    "ser": float(errors / max(1, symbol_count)),
                    "fixes_vs_savaux": sum(
                        int(row[f"{method}_fixes_vs_savaux"]) for row in condition
                    ),
                    "breaks_vs_savaux": sum(
                        int(row[f"{method}_breaks_vs_savaux"]) for row in condition
                    ),
                }
            )
    return output


def _plot(rows: list[dict[str, object]], output_path: Path) -> None:
    scenarios = sorted({str(row["scenario"]) for row in rows})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.2 * len(scenarios), 4.8), squeeze=False)
    colors = {
        "branch0": "#777777",
        "savaux": "#1f77b4",
        "equal8": "#ff7f0e",
        "equal8_wrap": "#8c564b",
        "gls8_direct": "#2ca02c",
        "gls8_cg": "#17becf",
        "gls8_cg_wrap": "#9467bd",
        "conditional_gls": "#d62728",
    }
    for axis, scenario in zip(axes[0], scenarios, strict=True):
        condition = sorted(
            (row for row in rows if str(row["scenario"]) == scenario),
            key=lambda row: float(row["mean_target_sinr_db"]),
        )
        x = [float(row["mean_target_sinr_db"]) for row in condition]
        for method in METHODS:
            y = [max(float(row[f"{method}_ser"]), 0.5 / max(1, int(row["symbol_count"]))) for row in condition]
            axis.semilogy(x, y, marker="o", linewidth=1.7, label=method, color=colors[method])
        axis.set_title(scenario.replace("_", " "))
        axis.set_xlabel("Measured target SINR (dB)")
        axis.set_ylabel("SER")
        axis.grid(True, which="both", alpha=0.3)
        axis.set_ylim(1e-3, 1.05)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Aligned SF10 LoRa over Sionna RT hospital channels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    links, cir_metadata = _load_cirs(args.cirs.resolve())
    bank = _current_bank(int(args.sf), int(args.os_factor))

    trial_rows: list[dict[str, object]] = []
    symbol_rows: list[dict[str, object]] = []
    for scenario in args.scenarios:
        for target_tx_dbm in args.target_tx_dbm:
            for trial in range(int(args.trials)):
                symbols, result = _evaluate_condition(
                    args,
                    links,
                    bank,
                    str(scenario),
                    float(target_tx_dbm),
                    trial,
                )
                symbol_rows.extend(symbols)
                trial_rows.append(result)
                print(json.dumps(result, ensure_ascii=True))

    summary_rows = _aggregate(trial_rows)
    aggregate_rows = _aggregate_scopes(summary_rows)
    _write_csv(args.output / "trial_metrics.csv", trial_rows)
    _write_csv(args.output / "summary.csv", summary_rows)
    _write_csv(args.output / "aggregate_summary.csv", aggregate_rows)
    _write_csv(args.output / "symbol_metrics.csv", symbol_rows)
    _plot(summary_rows, args.output / "ser.png")
    metadata = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cirs": cir_metadata,
        "patterns": list(bank.names),
        "target_pattern_response": _target_response_diagnostics(
            links=links,
            bank=bank,
            sf=int(args.sf),
            os_factor=int(args.os_factor),
            bandwidth=float(args.bandwidth),
            receiver_filter_taps=int(args.noise_filter_taps),
        ),
        "receiver_model": (
            "target, interference, and white receiver-input noise share one delay-compensated "
            "BW/2 low-pass FIR; input noise is scaled for kTB*NF output power"
        ),
        "interference_model": "independent asynchronous LoRa transmitters propagated through their own Sionna RT CIRs",
        "alignment": "perfect target symbol boundary and zero target CFO",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
