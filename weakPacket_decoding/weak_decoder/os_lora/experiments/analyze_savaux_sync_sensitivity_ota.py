"""Measure Savaux SER sensitivity to controlled STO/CFO mismatch on OTA IQ.

Clean 1 MS/s packets are synchronized once.  The resulting payload boundaries,
CFO, and SFO schedule are frozen.  LoRa-band AWGN is then added to guarded
payload symbols, and only the synchronization parameters presented to Savaux
are perturbed.  No synchronization search or multi-hypothesis decoder is used
in this first-stage experiment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterable, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
WEAK_PACKET_ROOT = SCRIPT_PATH.parents[3]
GR_LORA_ROOT = SCRIPT_PATH.parents[4]
WORKSPACE_ROOT = GR_LORA_ROOT.parent
if str(WEAK_PACKET_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_PACKET_ROOT))

from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)


SF = 12
N_BINS = 1 << SF
BW_HZ = 125_000.0
SOURCE_RATE_HZ = 1_000_000.0
OS_FACTOR = 8
SYMBOL_SAMPLES = N_BINS * OS_FACTOR
BIN_HZ = BW_HZ / N_BINS
FFT_COORDINATE_OFFSET = -1
INTERPOLATION_RADIUS = 16
GUARD_SAMPLES = 32
# FrameSync's payload cursor is consumed at the chip-center convention used by
# fft_demod. Savaux's full-OSR branch model uses the symbol-origin convention,
# which is exactly OSR/2 ADC samples earlier for this implementation.
SAVAUX_ORIGIN_OFFSET_SAMPLES = -OS_FACTOR / 2.0


@dataclass(frozen=True)
class GuardedCleanSymbol:
    packet_id: str
    reference_id: int
    payload_index: int
    gt_symbol: int
    start_sample: int
    cfo_total_bins: float
    signal_power: float
    offpacket_power: float
    guarded_samples: np.ndarray


def _parse_list(text: str, cast: type) -> tuple[Any, ...]:
    values = tuple(cast(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return values


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_single_packet_sync_module(dataset_repo: Path) -> ModuleType:
    source = dataset_repo / "weak_decoder" / "synchronization" / "single_packet.py"
    if not source.is_file():
        raise FileNotFoundError(f"missing clean synchronization wrapper: {source}")
    name = "weak_decoder.synchronization._savaux_sensitivity_single_packet"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load synchronization wrapper: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _packet_metadata_paths(ota_root: Path) -> Iterable[Path]:
    for path in sorted((ota_root / "metadata").glob("*_fulltrim.json")):
        metadata = _load_json(path)
        if int(metadata.get("view", {}).get("adc_phase_2m", -1)) != 0:
            continue
        if not bool(metadata.get("packet", {}).get("crc_valid", False)):
            continue
        yield path


def _advance_symbol_cursor(cursor: int, sfo_cum: float, sfo_hat: float) -> tuple[int, float]:
    step = SYMBOL_SAMPLES
    cumulative = float(sfo_cum)
    threshold = 1.0 / (2.0 * OS_FACTOR)
    if abs(cumulative) > threshold:
        sign = -1 if math.copysign(1.0, cumulative) < 0.0 else 1
        step -= sign
        cumulative -= sign * (1.0 / OS_FACTOR)
    cumulative += float(sfo_hat)
    return int(cursor + step), cumulative


def _split_cfo_bins(total_bins: float) -> tuple[int, float]:
    integer = int(math.floor(float(total_bins) + 0.5))
    return integer, float(total_bins - integer)


def _savaux_metrics(
    symbol: np.ndarray,
    cfo_total_bins: float,
    gt_symbol: int,
) -> tuple[int, float, float]:
    cfo_int, cfo_frac = _split_cfo_bins(cfo_total_bins)
    combined, _branches, _phase = paper_oversampled_spectrum(
        samples=np.asarray(symbol, dtype=np.complex64),
        start_sample=0,
        sf=SF,
        os_factor=OS_FACTOR,
        cfo_int=cfo_int,
        cfo_frac=cfo_frac,
        cfo_correction_mode="symbol",
    )
    power = np.abs(combined).astype(np.float64) ** 2
    raw_bin = int(np.argmax(power))
    decision = int((raw_bin - FFT_COORDINATE_OFFSET) % N_BINS)
    true_coordinate = int((int(gt_symbol) + FFT_COORDINATE_OFFSET) % N_BINS)
    true_power = float(power[true_coordinate])
    false = power.copy()
    false[true_coordinate] = -np.inf
    strongest_false = float(np.max(false))
    margin_db = float(
        10.0 * math.log10((true_power + 1e-30) / (strongest_false + 1e-30))
    )
    return decision, margin_db, true_power


def _unit_lora_band_awgn(rng: np.random.Generator, count: int) -> np.ndarray:
    """Unit-power complex AWGN flat inside B and zero outside B."""

    length = int(count)
    white = (
        rng.standard_normal(length) + 1j * rng.standard_normal(length)
    ) / math.sqrt(2.0)
    frequency = np.fft.fft(white)
    passband_bins = int(round(length / OS_FACTOR))
    if passband_bins % 2:
        passband_bins -= 1
    half = passband_bins // 2
    mask = np.zeros(length, dtype=bool)
    mask[:half] = True
    mask[-half:] = True
    frequency[~mask] = 0.0
    return (
        np.fft.ifft(frequency) * math.sqrt(float(length) / passband_bins)
    ).astype(np.complex64)


def _fractional_timing_window(
    guarded: np.ndarray,
    timing_error_samples: float,
    output_count: int = SYMBOL_SAMPLES,
    guard_samples: int = GUARD_SAMPLES,
    radius: int = INTERPOLATION_RADIUS,
) -> np.ndarray:
    """Sample a symbol window at ``oracle_start + timing_error_samples``.

    A positive error means the receiver hypothesis starts later. Integer shifts
    are exact; fractional shifts use a Kaiser-windowed sinc over real adjacent
    packet samples rather than circularly rotating an isolated symbol.
    """

    values = np.asarray(guarded, dtype=np.complex64)
    delta = float(timing_error_samples)
    integer = int(round(delta))
    fraction = delta - integer
    offsets = np.arange(-int(radius), int(radius) + 1, dtype=np.int64)
    weights = np.sinc(offsets.astype(np.float64) - fraction)
    weights *= np.kaiser(offsets.size, beta=8.6)
    weights /= np.sum(weights)
    base = int(guard_samples) + integer
    first = base + int(offsets[0])
    last = base + int(offsets[-1]) + int(output_count)
    if first < 0 or last > values.size:
        raise ValueError("guarded symbol is too short for the requested timing error")
    output = np.zeros(int(output_count), dtype=np.complex128)
    for offset, weight in zip(offsets, weights):
        start = base + int(offset)
        output += float(weight) * values[start : start + int(output_count)]
    return output.astype(np.complex64)


def _collect_clean_symbols(
    dataset_repo: Path,
    ota_root: Path,
    max_packets: int,
    symbols_per_packet: int,
) -> tuple[list[GuardedCleanSymbol], list[dict[str, Any]], list[dict[str, Any]]]:
    sync_module = _load_single_packet_sync_module(dataset_repo)
    config_type = sync_module.SinglePacketSyncConfig
    run_sync = sync_module.run_single_packet_sync
    clean: list[GuardedCleanSymbol] = []
    audits: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    packet_count = 0

    for metadata_path in _packet_metadata_paths(ota_root):
        if packet_count >= int(max_packets):
            break
        metadata = _load_json(metadata_path)
        reference_id = int(metadata["reference"]["reference_id"])
        reference = _load_json(
            ota_root.parent / "metadata" / f"{reference_id:06d}.json"
        )
        phy = reference["phy"]
        iq_path = ota_root / str(metadata["ota"]["relative_path"])
        samples = np.fromfile(iq_path, dtype=np.dtype("<c8"))
        sync_config = config_type(
            sf=int(phy["sf"]),
            bw_hz=float(phy["bandwidth_hz"]),
            sample_rate_hz=float(phy["sample_rate_hz"]),
            center_frequency_hz=float(metadata["capture"]["center_frequency_hz"]),
            preamble_symbols=int(phy["preamble_symbols"]),
            sync_word=int(phy["sync_word"]),
        )
        sync_result = run_sync(samples, sync_config)
        packet_id = str(metadata["ota_id"])
        audit: dict[str, Any] = {
            "packet_id": packet_id,
            "reference_id": reference_id,
            "sync_status": str(sync_result.status),
            "sync_valid": int(bool(sync_result.synchronized)),
            "tested_payload_symbols": 0,
            "admitted_clean_symbols": 0,
        }
        if not sync_result.synchronized or sync_result.frame_sync is None:
            audit["sync_error"] = str(sync_result.error or "")
            audits.append(audit)
            packet_count += 1
            continue

        frame_sync = sync_result.frame_sync
        audit.update(
            {
                "fine_payload_start_sample": int(frame_sync.fine_payload_start_sample),
                "cfo_total_bins": float(frame_sync.cfo_total_est),
                "cfo_hz": float(frame_sync.cfo_hz_est),
                "sfo_hat": float(frame_sync.sfo_hat),
                "sfo_cum_initial": float(frame_sync.sfo_cum_initial),
            }
        )
        header_ids = [int(value) for value in reference["symbols"]["header_ids"]]
        payload_ids = [int(value) for value in reference["symbols"]["payload_ids"]]
        all_ids = header_ids + payload_ids[: int(symbols_per_packet)]
        cursor = int(frame_sync.fine_payload_start_sample)
        sfo_cum = float(frame_sync.sfo_cum_initial)
        off_count = int(metadata["ota"]["leading_real_off_packet_samples"])
        offpacket_power = float(
            np.mean(np.abs(samples[:off_count]).astype(np.float64) ** 2)
        )

        for frame_index, gt_symbol in enumerate(all_ids):
            if frame_index >= len(header_ids):
                payload_index = frame_index - len(header_ids)
                left = cursor - GUARD_SAMPLES
                right = cursor + SYMBOL_SAMPLES + GUARD_SAMPLES
                if left < 0 or right > samples.size:
                    break
                guarded = np.asarray(samples[left:right], dtype=np.complex64).copy()
                center = _fractional_timing_window(
                    guarded, SAVAUX_ORIGIN_OFFSET_SAMPLES
                )
                decision, margin_db, _true_power = _savaux_metrics(
                    center, float(frame_sync.cfo_total_est), gt_symbol
                )
                accepted = decision == gt_symbol
                signal_power = max(
                    float(np.mean(np.abs(center).astype(np.float64) ** 2))
                    - offpacket_power,
                    np.finfo(np.float64).tiny,
                )
                row = {
                    "packet_id": packet_id,
                    "reference_id": reference_id,
                    "payload_index": payload_index,
                    "gt_symbol": gt_symbol,
                    "start_sample": cursor,
                    "savaux_origin_start_sample": cursor
                    + SAVAUX_ORIGIN_OFFSET_SAMPLES,
                    "clean_decision": decision,
                    "accepted": int(accepted),
                    "clean_true_margin_db": margin_db,
                    "signal_power": signal_power,
                    "offpacket_power": offpacket_power,
                }
                symbol_rows.append(row)
                audit["tested_payload_symbols"] += 1
                if accepted:
                    audit["admitted_clean_symbols"] += 1
                    clean.append(
                        GuardedCleanSymbol(
                            packet_id=packet_id,
                            reference_id=reference_id,
                            payload_index=payload_index,
                            gt_symbol=gt_symbol,
                            start_sample=cursor,
                            cfo_total_bins=float(frame_sync.cfo_total_est),
                            signal_power=signal_power,
                            offpacket_power=offpacket_power,
                            guarded_samples=guarded,
                        )
                    )
            cursor, sfo_cum = _advance_symbol_cursor(
                cursor, sfo_cum, float(frame_sync.sfo_hat)
            )
        audits.append(audit)
        packet_count += 1
    return clean, audits, symbol_rows


def _run_sensitivity(
    clean_symbols: Sequence[GuardedCleanSymbol],
    esn0_db: float,
    seeds: Sequence[int],
    timing_errors_samples: Sequence[float],
    cfo_errors_bins: Sequence[float],
    wide_cfo_errors_hz: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    linear_esn0 = 10.0 ** (float(esn0_db) / 10.0)
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        for symbol_index, clean in enumerate(clean_symbols):
            noise_power = clean.signal_power * N_BINS / linear_esn0
            noise = _unit_lora_band_awgn(rng, clean.guarded_samples.size)
            noisy_guarded = clean.guarded_samples + noise * math.sqrt(noise_power)
            timing_views = {
                float(error): _fractional_timing_window(
                    noisy_guarded,
                    SAVAUX_ORIGIN_OFFSET_SAMPLES + float(error),
                )
                for error in timing_errors_samples
            }
            trial_id = f"{seed}:{symbol_index}"
            common = {
                "trial_id": trial_id,
                "esn0_db": float(esn0_db),
                "seed": int(seed),
                "clean_symbol_index": symbol_index,
                "packet_id": clean.packet_id,
                "reference_id": clean.reference_id,
                "payload_index": clean.payload_index,
                "gt_symbol": clean.gt_symbol,
                "signal_power": clean.signal_power,
                "added_noise_power": noise_power,
            }
            for timing_error, view in timing_views.items():
                for cfo_error_bins in cfo_errors_bins:
                    hypothesis = clean.cfo_total_bins + float(cfo_error_bins)
                    decision, margin_db, true_power = _savaux_metrics(
                        view, hypothesis, clean.gt_symbol
                    )
                    rows.append(
                        {
                            **common,
                            "timing_error_samples_1m": float(timing_error),
                            "timing_error_chips": float(timing_error) / OS_FACTOR,
                            "cfo_error_bins": float(cfo_error_bins),
                            "cfo_error_hz": float(cfo_error_bins) * BIN_HZ,
                            "decision": decision,
                            "correct": int(decision == clean.gt_symbol),
                            "true_margin_db": margin_db,
                            "true_power": true_power,
                        }
                    )
            zero_view = _fractional_timing_window(
                noisy_guarded, SAVAUX_ORIGIN_OFFSET_SAMPLES
            )
            for cfo_error_hz in wide_cfo_errors_hz:
                cfo_error_bins = float(cfo_error_hz) / BIN_HZ
                decision, margin_db, true_power = _savaux_metrics(
                    zero_view,
                    clean.cfo_total_bins + cfo_error_bins,
                    clean.gt_symbol,
                )
                wide_rows.append(
                    {
                        **common,
                        "timing_error_samples_1m": 0.0,
                        "cfo_error_bins": cfo_error_bins,
                        "cfo_error_hz": float(cfo_error_hz),
                        "decision": decision,
                        "correct": int(decision == clean.gt_symbol),
                        "true_margin_db": margin_db,
                        "true_power": true_power,
                    }
                )
    return rows, wide_rows


def _summarize_grid(
    rows: Sequence[dict[str, Any]],
    timing_errors_samples: Sequence[float],
    cfo_errors_bins: Sequence[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for timing_error in timing_errors_samples:
        for cfo_error_bins in cfo_errors_bins:
            selected = [
                row
                for row in rows
                if math.isclose(
                    float(row["timing_error_samples_1m"]), float(timing_error), abs_tol=1e-12
                )
                and math.isclose(
                    float(row["cfo_error_bins"]), float(cfo_error_bins), abs_tol=1e-12
                )
            ]
            count = len(selected)
            errors = sum(1 - int(row["correct"]) for row in selected)
            output.append(
                {
                    "timing_error_samples_1m": float(timing_error),
                    "timing_error_chips": float(timing_error) / OS_FACTOR,
                    "cfo_error_bins": float(cfo_error_bins),
                    "cfo_error_hz": float(cfo_error_bins) * BIN_HZ,
                    "trial_count": count,
                    "error_count": errors,
                    "ser": errors / count if count else float("nan"),
                    "mean_true_margin_db": float(
                        np.mean([float(row["true_margin_db"]) for row in selected])
                    )
                    if count
                    else float("nan"),
                }
            )
    return output


def _summarize_wide(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    values = sorted({float(row["cfo_error_hz"]) for row in rows})
    output: list[dict[str, Any]] = []
    for value in values:
        selected = [
            row
            for row in rows
            if math.isclose(float(row["cfo_error_hz"]), value, abs_tol=1e-12)
        ]
        count = len(selected)
        errors = sum(1 - int(row["correct"]) for row in selected)
        output.append(
            {
                "cfo_error_hz": value,
                "cfo_error_bins": value / BIN_HZ,
                "trial_count": count,
                "error_count": errors,
                "ser": errors / count if count else float("nan"),
                "mean_true_margin_db": float(
                    np.mean([float(row["true_margin_db"]) for row in selected])
                )
                if count
                else float("nan"),
            }
        )
    return output


def _grid_cell(
    summary: Sequence[dict[str, Any]], timing_error: float, cfo_error_bins: float
) -> dict[str, Any]:
    return next(
        row
        for row in summary
        if math.isclose(float(row["timing_error_samples_1m"]), timing_error, abs_tol=1e-12)
        and math.isclose(float(row["cfo_error_bins"]), cfo_error_bins, abs_tol=1e-12)
    )


def _make_plots(
    output_dir: Path,
    summary: Sequence[dict[str, Any]],
    wide_summary: Sequence[dict[str, Any]],
    timing_errors_samples: Sequence[float],
    cfo_errors_bins: Sequence[float],
    esn0_db: float,
) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(
        [
            [float(_grid_cell(summary, tau, cfo)["ser"]) for cfo in cfo_errors_bins]
            for tau in timing_errors_samples
        ]
    )
    fig, axis = plt.subplots(figsize=(13.5, 7.0))
    image = axis.imshow(matrix, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(cfo_errors_bins)))
    axis.set_xticklabels(
        [f"{value * BIN_HZ:.1f}\n({value:g} bin)" for value in cfo_errors_bins],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(timing_errors_samples)))
    axis.set_yticklabels([f"{value:g}" for value in timing_errors_samples])
    axis.set_xlabel("CFO correction error: Hz (LoRa FFT bins)")
    axis.set_ylabel("Residual STO error around Savaux origin (1 MS/s samples)")
    axis.set_title(f"Savaux SER sensitivity, Es/N0={esn0_db:g} dB")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value > 0.55 else "black"
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=axis, label="SER")
    fig.tight_layout()
    fig.savefig(output_dir / "savaux_sto_cfo_ser_heatmap.png", dpi=180)
    plt.close(fig)

    zero_timing = min(timing_errors_samples, key=abs)
    zero_cfo = min(cfo_errors_bins, key=abs)
    cfo_slice = [_grid_cell(summary, zero_timing, value) for value in cfo_errors_bins]
    sto_slice = [_grid_cell(summary, value, zero_cfo) for value in timing_errors_samples]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(
        [float(row["cfo_error_hz"]) for row in cfo_slice],
        [float(row["ser"]) for row in cfo_slice],
        marker="o",
    )
    axes[0].axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("CFO correction error (Hz), STO error = 0")
    axes[0].set_ylabel("SER")
    axes[1].plot(
        [float(row["timing_error_samples_1m"]) for row in sto_slice],
        [float(row["ser"]) for row in sto_slice],
        marker="o",
    )
    axes[1].axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Residual STO error (1 MS/s samples), CFO error = 0")
    axes[1].set_ylabel("SER")
    for axis in axes:
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, alpha=0.25)
    fig.suptitle(f"Savaux synchronization sensitivity slices, Es/N0={esn0_db:g} dB")
    fig.tight_layout()
    fig.savefig(output_dir / "savaux_sync_sensitivity_slices.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(
        [float(row["cfo_error_hz"]) for row in wide_summary],
        [float(row["ser"]) for row in wide_summary],
        marker="o",
    )
    axis.set_xscale("symlog", linthresh=50.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("CFO correction error (Hz), STO error = 0")
    axis.set_ylabel("SER")
    axis.grid(True, alpha=0.25)
    axis.set_title(f"Wide CFO sensitivity, Es/N0={esn0_db:g} dB")
    fig.tight_layout()
    fig.savefig(output_dir / "savaux_wide_cfo_sensitivity.png", dpi=180)
    plt.close(fig)


def _build_report(
    output_dir: Path,
    audits: Sequence[dict[str, Any]],
    clean_count: int,
    summary: Sequence[dict[str, Any]],
    wide_summary: Sequence[dict[str, Any]],
    timing_errors_samples: Sequence[float],
    cfo_errors_bins: Sequence[float],
    esn0_db: float,
) -> None:
    tested = sum(int(row["tested_payload_symbols"]) for row in audits)
    zero_timing = min(timing_errors_samples, key=abs)
    zero_cfo = min(cfo_errors_bins, key=abs)
    oracle = _grid_cell(summary, zero_timing, zero_cfo)
    cfo_slice = [_grid_cell(summary, zero_timing, value) for value in cfo_errors_bins]
    sto_slice = [_grid_cell(summary, value, zero_cfo) for value in timing_errors_samples]
    lines = [
        "# Savaux STO/CFO sensitivity on OTA symbols",
        "",
        "This experiment perturbs only the synchronization hypothesis presented to",
        "Savaux. Clean FrameSync is frozen before the same B-wide AWGN realization is",
        "reused across every STO/CFO grid point.",
        "",
        f"- Es/N0: {esn0_db:g} dB",
        f"- Clean payload symbols tested/admitted: {tested}/{clean_count}",
        f"- Trials per grid cell: {int(oracle['trial_count'])}",
        f"- Oracle-hypothesis SER at (0,0): {float(oracle['ser']):.4f}",
        f"- LoRa bin spacing: {BIN_HZ:.6f} Hz",
        f"- FrameSync-to-Savaux origin conversion: {SAVAUX_ORIGIN_OFFSET_SAMPLES:g} ADC samples",
        "",
        "## CFO slice at zero STO error",
        "",
        "| CFO error (bin) | CFO error (Hz) | SER | mean true margin (dB) |",
        "|---:|---:|---:|---:|",
    ]
    for row in cfo_slice:
        lines.append(
            f"| {float(row['cfo_error_bins']):g} | {float(row['cfo_error_hz']):.3f} | "
            f"{float(row['ser']):.4f} | {float(row['mean_true_margin_db']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## STO slice at zero CFO error",
            "",
            "| STO error (1M sample) | STO error (chip) | SER | mean true margin (dB) |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in sto_slice:
        lines.append(
            f"| {float(row['timing_error_samples_1m']):g} | {float(row['timing_error_chips']):g} | "
            f"{float(row['ser']):.4f} | {float(row['mean_true_margin_db']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Wide CFO slice",
            "",
            "| CFO error (Hz) | CFO error (bin) | SER |",
            "|---:|---:|---:|",
        ]
    )
    for row in wide_summary:
        lines.append(
            f"| {float(row['cfo_error_hz']):g} | {float(row['cfo_error_bins']):.3f} | "
            f"{float(row['ser']):.4f} |"
        )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-repo", type=Path, default=WORKSPACE_ROOT / "lora-rfsr-savaux"
    )
    parser.add_argument("--ota-root", type=Path, default=None)
    parser.add_argument("--max-packets", type=int, default=4)
    parser.add_argument("--symbols-per-packet", type=int, default=16)
    parser.add_argument("--esn0-db", type=float, default=14.0)
    parser.add_argument("--seeds", default="20260819,20260820")
    parser.add_argument(
        "--timing-errors-samples",
        default="-4,-2,-1,-0.5,0,0.5,1,2,4",
        help="STO hypothesis error in 1 MS/s ADC samples",
    )
    parser.add_argument(
        "--cfo-errors-bins",
        default="-2,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,2",
        help="CFO hypothesis error in LoRa FFT bins (one bin is B/2^SF Hz)",
    )
    parser.add_argument(
        "--wide-cfo-errors-hz",
        default="-2000,-1000,-500,-250,-125,0,125,250,500,1000,2000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "savaux_sync_sensitivity_ota_awgn_20260820",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    dataset_repo = args.dataset_repo.resolve()
    ota_root = (
        args.ota_root.resolve()
        if args.ota_root is not None
        else dataset_repo / "data" / "reference_phy" / "rfsr_db"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_list(args.seeds, int)
    timing_errors = _parse_list(args.timing_errors_samples, float)
    cfo_errors = _parse_list(args.cfo_errors_bins, float)
    wide_cfo_errors = _parse_list(args.wide_cfo_errors_hz, float)
    if not any(math.isclose(value, 0.0, abs_tol=1e-12) for value in timing_errors):
        raise ValueError("timing error grid must include zero")
    if not any(math.isclose(value, 0.0, abs_tol=1e-12) for value in cfo_errors):
        raise ValueError("CFO error grid must include zero")

    clean, audits, clean_rows = _collect_clean_symbols(
        dataset_repo,
        ota_root,
        max_packets=int(args.max_packets),
        symbols_per_packet=int(args.symbols_per_packet),
    )
    if not clean:
        raise RuntimeError("no payload symbols passed the clean Savaux audit")
    _write_csv(output_dir / "sync_audit.csv", audits)
    _write_csv(output_dir / "clean_symbols.csv", clean_rows)

    trials, wide_trials = _run_sensitivity(
        clean,
        esn0_db=float(args.esn0_db),
        seeds=seeds,
        timing_errors_samples=timing_errors,
        cfo_errors_bins=cfo_errors,
        wide_cfo_errors_hz=wide_cfo_errors,
    )
    summary = _summarize_grid(trials, timing_errors, cfo_errors)
    wide_summary = _summarize_wide(wide_trials)
    _write_csv(output_dir / "trials.csv", trials)
    _write_csv(output_dir / "wide_cfo_trials.csv", wide_trials)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "wide_cfo_summary.csv", wide_summary)
    _make_plots(
        output_dir,
        summary,
        wide_summary,
        timing_errors,
        cfo_errors,
        float(args.esn0_db),
    )
    _build_report(
        output_dir,
        audits,
        len(clean),
        summary,
        wide_summary,
        timing_errors,
        cfo_errors,
        float(args.esn0_db),
    )
    config = {
        "dataset_repo": str(dataset_repo),
        "ota_root": str(ota_root),
        "max_packets": int(args.max_packets),
        "symbols_per_packet": int(args.symbols_per_packet),
        "esn0_db": float(args.esn0_db),
        "seeds": list(seeds),
        "timing_errors_samples_1m": list(timing_errors),
        "cfo_errors_bins": list(cfo_errors),
        "cfo_errors_hz": [value * BIN_HZ for value in cfo_errors],
        "wide_cfo_errors_hz": list(wide_cfo_errors),
        "clean_symbol_count": len(clean),
        "noise_model": "complex AWGN flat in B, injected once at 1 MS/s",
        "frame_sync_to_savaux_origin_offset_samples_1m": SAVAUX_ORIGIN_OFFSET_SAMPLES,
        "fractional_sto_interpolator": {
            "kind": "Kaiser-windowed sinc",
            "radius": INTERPOLATION_RADIUS,
            "guard_samples": GUARD_SAMPLES,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    oracle = _grid_cell(summary, 0.0, 0.0)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "clean_symbol_count": len(clean),
                "trials_per_cell": int(oracle["trial_count"]),
                "oracle_ser": float(oracle["ser"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
