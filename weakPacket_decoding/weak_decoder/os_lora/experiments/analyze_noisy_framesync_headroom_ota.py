"""Measure real noisy-FrameSync error and the resulting Savaux oracle gap.

AWGN is added once to an entire 1 MS/s OTA packet, including its preamble.
FrameSync then runs on that noisy packet.  The same noisy IQ is demodulated by
Savaux with either the noisy synchronization estimate or the clean estimate.
Strict FrameSync validation failure is reported separately from estimator
availability so that a gating loss is not mislabelled as CFO headroom.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import importlib.util
import json
import math
import os
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
SAVAUX_ORIGIN_OFFSET_SAMPLES = -OS_FACTOR // 2
LOCAL_CFO_LIMIT_BINS = 2.0
LOCAL_STO_LIMIT_SAMPLES = 32
_SYNC_MODULE: ModuleType | None = None


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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cfo_sto_coupled_residual_bins(row: dict[str, Any]) -> float:
    """First-order dechirp residual after LoRa CFO/STO ambiguity coupling."""

    return float(row["cfo_error_bins"]) - float(
        row["payload_sto_error_samples_1m"]
    ) / float(OS_FACTOR)


def _sync_gate_failures(sync_result: Any, config: Any) -> tuple[str, ...]:
    failures: list[str] = []
    location = sync_result.frame_location
    frame_sync = sync_result.frame_sync
    if location is None:
        failures.append("frame_location_missing")
    else:
        min_preamble = max(
            3, int(config.preamble_symbols) - int(config.bin_tolerance)
        )
        if int(location.preamble_stable_count) < min_preamble:
            failures.append("locator_preamble_stability")
        if int(location.sync1_distance) > int(config.sync_bin_tolerance):
            failures.append("locator_sync1")
        if int(location.sync2_distance) > int(config.sync_bin_tolerance):
            failures.append("locator_sync2")
        if int(location.sfd_bin_distance) > int(config.sfd_bin_tolerance):
            failures.append("locator_sfd")
    if frame_sync is None:
        failures.append("frame_sync_missing")
    else:
        if int(frame_sync.preamble_bin0_count) != int(
            frame_sync.preamble_peak_count
        ):
            failures.append("framesync_preamble_all_bin0")
        if not bool(frame_sync.netid_valid):
            failures.append("framesync_netid")
    return tuple(failures)


def _load_single_packet_sync_module(dataset_repo: Path) -> ModuleType:
    global _SYNC_MODULE
    if _SYNC_MODULE is not None:
        return _SYNC_MODULE
    source = dataset_repo / "weak_decoder" / "synchronization" / "single_packet.py"
    if not source.is_file():
        raise FileNotFoundError(f"missing clean synchronization wrapper: {source}")
    name = "weak_decoder.synchronization._noisy_headroom_single_packet"
    if name in sys.modules:
        _SYNC_MODULE = sys.modules[name]
        return _SYNC_MODULE
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load synchronization wrapper: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _SYNC_MODULE = module
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


def _payload_starts(
    data_start: int,
    sfo_cum_initial: float,
    sfo_hat: float,
    header_count: int,
    payload_count: int,
) -> list[int]:
    cursor = int(data_start)
    cumulative = float(sfo_cum_initial)
    output: list[int] = []
    for frame_index in range(int(header_count) + int(payload_count)):
        if frame_index >= int(header_count):
            output.append(cursor)
        cursor, cumulative = _advance_symbol_cursor(cursor, cumulative, sfo_hat)
    return output


def _split_cfo_bins(total_bins: float) -> tuple[int, float]:
    integer = int(math.floor(float(total_bins) + 0.5))
    return integer, float(total_bins - integer)


def _savaux_metrics(
    samples: np.ndarray,
    start_sample: int,
    cfo_total_bins: float,
    gt_symbol: int,
) -> dict[str, Any] | None:
    start = int(start_sample) + SAVAUX_ORIGIN_OFFSET_SAMPLES
    stop = start + SYMBOL_SAMPLES
    if start < 0 or stop > np.asarray(samples).size:
        return None
    cfo_int, cfo_frac = _split_cfo_bins(cfo_total_bins)
    combined, _branches, _phase = paper_oversampled_spectrum(
        samples=samples,
        start_sample=start,
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
    return {
        "decision": decision,
        "correct": int(decision == int(gt_symbol)),
        "true_margin_db": float(
            10.0
            * math.log10((true_power + 1e-30) / (strongest_false + 1e-30))
        ),
    }


def _unit_lora_band_awgn(rng: np.random.Generator, count: int) -> np.ndarray:
    """Generate unit-power complex AWGN flat in the B-wide LoRa channel."""

    length = int(count)
    white = (
        rng.standard_normal(length).astype(np.float32)
        + 1j * rng.standard_normal(length).astype(np.float32)
    ) / np.float32(math.sqrt(2.0))
    frequency = np.fft.fft(white)
    passband_bins = int(round(length / OS_FACTOR))
    if passband_bins % 2:
        passband_bins -= 1
    half = passband_bins // 2
    frequency[half : length - half] = 0.0
    noise = np.fft.ifft(frequency)
    noise *= math.sqrt(float(length) / passband_bins)
    return noise.astype(np.complex64)


def _sync_config(module: ModuleType, center_frequency_hz: float) -> Any:
    return module.SinglePacketSyncConfig(
        sf=SF,
        bw_hz=BW_HZ,
        sample_rate_hz=SOURCE_RATE_HZ,
        center_frequency_hz=float(center_frequency_hz),
        preamble_symbols=16,
        sync_word=0x12,
    )


def _clean_packet_worker(task: dict[str, Any]) -> dict[str, Any]:
    dataset_repo = Path(str(task["dataset_repo"]))
    metadata_path = Path(str(task["metadata_path"]))
    ota_root = Path(str(task["ota_root"]))
    symbols_per_packet = int(task["symbols_per_packet"])
    metadata = _load_json(metadata_path)
    reference_id = int(metadata["reference"]["reference_id"])
    reference = _load_json(
        ota_root.parent / "metadata" / f"{reference_id:06d}.json"
    )
    iq_path = ota_root / str(metadata["ota"]["relative_path"])
    samples = np.fromfile(iq_path, dtype=np.dtype("<c8"))
    module = _load_single_packet_sync_module(dataset_repo)
    result = module.run_single_packet_sync(
        samples,
        _sync_config(module, float(metadata["capture"]["center_frequency_hz"])),
    )
    packet_id = str(metadata["ota_id"])
    audit: dict[str, Any] = {
        "packet_id": packet_id,
        "reference_id": reference_id,
        "sync_status": str(result.status),
        "sync_valid": int(bool(result.synchronized)),
        "tested_payload_symbols": 0,
        "admitted_payload_symbols": 0,
    }
    if not result.synchronized or result.frame_sync is None:
        audit["sync_error"] = str(result.error or "")
        return {"audit": audit, "packet": None}

    frame_sync = result.frame_sync
    payload_ids = [int(value) for value in reference["symbols"]["payload_ids"]][
        :symbols_per_packet
    ]
    header_count = len(reference["symbols"]["header_ids"])
    starts = _payload_starts(
        int(frame_sync.fine_payload_start_sample),
        float(frame_sync.sfo_cum_initial),
        float(frame_sync.sfo_hat),
        header_count,
        len(payload_ids),
    )
    off_count = int(metadata["ota"]["leading_real_off_packet_samples"])
    offpacket_power = float(
        np.mean(np.abs(samples[:off_count]).astype(np.float64) ** 2)
    )
    specs: list[dict[str, Any]] = []
    signal_powers: list[float] = []
    for payload_index, (start, gt_symbol) in enumerate(zip(starts, payload_ids)):
        metrics = _savaux_metrics(
            samples, start, float(frame_sync.cfo_total_est), gt_symbol
        )
        audit["tested_payload_symbols"] += 1
        if metrics is None or not bool(metrics["correct"]):
            continue
        center_start = start + SAVAUX_ORIGIN_OFFSET_SAMPLES
        symbol = samples[center_start : center_start + SYMBOL_SAMPLES]
        signal_power = max(
            float(np.mean(np.abs(symbol).astype(np.float64) ** 2))
            - offpacket_power,
            np.finfo(np.float64).tiny,
        )
        signal_powers.append(signal_power)
        audit["admitted_payload_symbols"] += 1
        specs.append(
            {
                "payload_index": payload_index,
                "gt_symbol": gt_symbol,
                "oracle_start_sample": start,
                "clean_true_margin_db": float(metrics["true_margin_db"]),
            }
        )
    audit.update(
        {
            "fine_payload_start_sample": int(frame_sync.fine_payload_start_sample),
            "cfo_total_bins": float(frame_sync.cfo_total_est),
            "cfo_hz": float(frame_sync.cfo_hz_est),
            "sfo_hat": float(frame_sync.sfo_hat),
            "sfo_cum_initial": float(frame_sync.sfo_cum_initial),
            "signal_power": float(np.mean(signal_powers))
            if signal_powers
            else float("nan"),
            "offpacket_power": offpacket_power,
        }
    )
    packet = None
    if specs:
        packet = {
            "packet_id": packet_id,
            "reference_id": reference_id,
            "iq_path": str(iq_path),
            "center_frequency_hz": float(metadata["capture"]["center_frequency_hz"]),
            "header_count": header_count,
            "payload_count": len(payload_ids),
            "symbol_specs": specs,
            "signal_power": float(np.mean(signal_powers)),
            "clean_cfo_total_bins": float(frame_sync.cfo_total_est),
            "clean_fine_payload_start_sample": int(frame_sync.fine_payload_start_sample),
            "clean_fine_preamble_start_sample": int(frame_sync.fine_preamble_start_sample),
            "clean_sfo_hat": float(frame_sync.sfo_hat),
            "clean_sfo_cum_initial": float(frame_sync.sfo_cum_initial),
        }
    return {"audit": audit, "packet": packet}


def _noisy_trial_worker(task: dict[str, Any]) -> dict[str, Any]:
    packet = dict(task["packet"])
    dataset_repo = Path(str(task["dataset_repo"]))
    esn0_db = float(task["esn0_db"])
    seed = int(task["seed"])
    samples = np.fromfile(Path(str(packet["iq_path"])), dtype=np.dtype("<c8"))
    rng = np.random.default_rng(
        np.random.SeedSequence((seed, int(packet["reference_id"]), 73013))
    )
    noise_power = float(packet["signal_power"]) * N_BINS / (
        10.0 ** (esn0_db / 10.0)
    )
    noise = _unit_lora_band_awgn(rng, samples.size)
    noisy = np.asarray(samples + noise * math.sqrt(noise_power), dtype=np.complex64)
    del noise

    module = _load_single_packet_sync_module(dataset_repo)
    sync_config = _sync_config(module, float(packet["center_frequency_hz"]))
    sync_result = module.run_single_packet_sync(noisy, sync_config)
    frame_sync = sync_result.frame_sync
    estimate_available = frame_sync is not None
    strict_success = bool(sync_result.synchronized)
    decoder_aware_success = bool(
        sync_result.accepted("decoder_aware")
        if hasattr(sync_result, "accepted")
        else (
            sync_result.frame_location is not None
            and sync_result.frame_location.valid
            and frame_sync is not None
            and frame_sync.netid_valid
        )
    )
    trial_id = f"{packet['packet_id']}:{esn0_db:g}:{seed}"
    trial: dict[str, Any] = {
        "trial_id": trial_id,
        "packet_id": str(packet["packet_id"]),
        "reference_id": int(packet["reference_id"]),
        "esn0_db": esn0_db,
        "seed": seed,
        "sync_status": str(sync_result.status),
        "strict_sync_success": int(strict_success),
        "decoder_aware_sync_success": int(decoder_aware_success),
        "estimate_available": int(estimate_available),
        "local_sync_estimate": 0,
        "event_count": int(sync_result.event_count),
        "signal_power": float(packet["signal_power"]),
        "added_noise_power": noise_power,
        "gate_failure_reasons": "|".join(
            _sync_gate_failures(sync_result, sync_config)
        ),
    }

    frame_location = sync_result.frame_location
    if frame_location is not None:
        trial.update(
            {
                "locator_valid": int(bool(frame_location.valid)),
                "locator_score": float(frame_location.score),
                "locator_preamble_stable_count": int(
                    frame_location.preamble_stable_count
                ),
                "locator_sync1_distance": int(frame_location.sync1_distance),
                "locator_sync2_distance": int(frame_location.sync2_distance),
                "locator_sfd_distance": int(frame_location.sfd_bin_distance),
                "locator_mean_preamble_confidence_db": float(
                    frame_location.mean_preamble_confidence_db
                ),
                "locator_mean_sfd_confidence_db": float(
                    frame_location.mean_sfd_confidence_db
                ),
            }
        )

    noisy_starts: list[int] | None = None
    if frame_sync is not None:
        cfo_error_bins = float(frame_sync.cfo_total_est) - float(
            packet["clean_cfo_total_bins"]
        )
        payload_start_error = int(frame_sync.fine_payload_start_sample) - int(
            packet["clean_fine_payload_start_sample"]
        )
        preamble_start_error = int(frame_sync.fine_preamble_start_sample) - int(
            packet["clean_fine_preamble_start_sample"]
        )
        local_estimate = bool(
            abs(cfo_error_bins) <= LOCAL_CFO_LIMIT_BINS
            and abs(payload_start_error) <= LOCAL_STO_LIMIT_SAMPLES
        )
        trial.update(
            {
                "clean_cfo_total_bins": float(packet["clean_cfo_total_bins"]),
                "noisy_cfo_total_bins": float(frame_sync.cfo_total_est),
                "cfo_error_bins": cfo_error_bins,
                "cfo_error_hz": cfo_error_bins * BIN_HZ,
                "abs_cfo_error_bins": abs(cfo_error_bins),
                "clean_payload_start_sample": int(
                    packet["clean_fine_payload_start_sample"]
                ),
                "noisy_payload_start_sample": int(frame_sync.fine_payload_start_sample),
                "payload_sto_error_samples_1m": payload_start_error,
                "payload_sto_error_chips": payload_start_error / OS_FACTOR,
                "cfo_sto_coupled_residual_bins": cfo_error_bins
                - payload_start_error / OS_FACTOR,
                "abs_cfo_sto_coupled_residual_bins": abs(
                    cfo_error_bins - payload_start_error / OS_FACTOR
                ),
                "preamble_sto_error_samples_1m": preamble_start_error,
                "clean_sfo_hat": float(packet["clean_sfo_hat"]),
                "noisy_sfo_hat": float(frame_sync.sfo_hat),
                "sfo_error": float(frame_sync.sfo_hat)
                - float(packet["clean_sfo_hat"]),
                "fine_preamble_peak_max_abs_bin": int(
                    frame_sync.fine_preamble_peak_max_abs_signed_bin
                ),
                "fine_preamble_bin0_count": int(frame_sync.fine_preamble_bin0_count),
                "fine_preamble_peak_count": int(frame_sync.fine_preamble_peak_count),
                "coarse_preamble_peak_max_abs_bin": int(
                    frame_sync.preamble_peak_max_abs_signed_bin
                ),
                "coarse_preamble_bin0_count": int(frame_sync.preamble_bin0_count),
                "coarse_preamble_peak_count": int(frame_sync.preamble_peak_count),
                "netid1_est": int(frame_sync.netid1_est),
                "netid2_est": int(frame_sync.netid2_est),
                "netid_offset": int(frame_sync.netid_offset),
                "netid_valid": int(bool(frame_sync.netid_valid)),
                "branch_valid_count": int(
                    sum(bool(item.valid) for item in frame_sync.branch_sync_estimates)
                ),
                "branch_netid_valid_count": int(
                    sum(
                        bool(item.netid_valid)
                        for item in frame_sync.branch_sync_estimates
                    )
                ),
                "branch_down_val_valid_count": int(
                    sum(
                        bool(item.down_val_valid)
                        for item in frame_sync.branch_sync_estimates
                    )
                ),
                "branch_valid_phases": "|".join(
                    str(int(item.sample_phase))
                    for item in frame_sync.branch_sync_estimates
                    if bool(item.valid)
                ),
                "frame_sync_valid": int(bool(frame_sync.valid)),
                "local_sync_estimate": int(local_estimate),
            }
        )
        noisy_starts = _payload_starts(
            int(frame_sync.fine_payload_start_sample),
            float(frame_sync.sfo_cum_initial),
            float(frame_sync.sfo_hat),
            int(packet["header_count"]),
            int(packet["payload_count"]),
        )

    symbols: list[dict[str, Any]] = []
    for spec in packet["symbol_specs"]:
        payload_index = int(spec["payload_index"])
        gt_symbol = int(spec["gt_symbol"])
        oracle = _savaux_metrics(
            noisy,
            int(spec["oracle_start_sample"]),
            float(packet["clean_cfo_total_bins"]),
            gt_symbol,
        )
        if oracle is None:
            raise RuntimeError("oracle payload window exceeds the noisy packet")
        noisy_metric = None
        if frame_sync is not None and noisy_starts is not None:
            noisy_metric = _savaux_metrics(
                noisy,
                noisy_starts[payload_index],
                float(frame_sync.cfo_total_est),
                gt_symbol,
            )
        symbols.append(
            {
                "trial_id": trial_id,
                "packet_id": str(packet["packet_id"]),
                "reference_id": int(packet["reference_id"]),
                "esn0_db": esn0_db,
                "seed": seed,
                "payload_index": payload_index,
                "gt_symbol": gt_symbol,
                "strict_sync_success": int(strict_success),
                "estimate_available": int(estimate_available),
                "local_sync_estimate": int(trial["local_sync_estimate"]),
                "oracle_decision": int(oracle["decision"]),
                "oracle_correct": int(oracle["correct"]),
                "oracle_true_margin_db": float(oracle["true_margin_db"]),
                "noisy_sync_decision": ""
                if noisy_metric is None
                else int(noisy_metric["decision"]),
                "noisy_sync_correct": ""
                if noisy_metric is None
                else int(noisy_metric["correct"]),
                "noisy_sync_true_margin_db": ""
                if noisy_metric is None
                else float(noisy_metric["true_margin_db"]),
                "strict_end_to_end_correct": int(
                    strict_success
                    and noisy_metric is not None
                    and bool(noisy_metric["correct"])
                ),
            }
        )
    return {"trial": trial, "symbols": symbols}


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def _summary_by_snr(
    trials: Sequence[dict[str, Any]],
    symbols: Sequence[dict[str, Any]],
    esn0_values: Sequence[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for esn0_db in esn0_values:
        packet_rows = [
            row
            for row in trials
            if math.isclose(float(row["esn0_db"]), float(esn0_db), abs_tol=1e-12)
        ]
        symbol_rows = [
            row
            for row in symbols
            if math.isclose(float(row["esn0_db"]), float(esn0_db), abs_tol=1e-12)
        ]
        estimates = [row for row in packet_rows if int(row["estimate_available"])]
        local_estimates = [row for row in packet_rows if int(row["local_sync_estimate"])]
        cfo_abs = [float(row["abs_cfo_error_bins"]) for row in estimates]
        local_cfo_abs = [
            float(row["abs_cfo_error_bins"]) for row in local_estimates
        ]
        sto_abs = [abs(float(row["payload_sto_error_samples_1m"])) for row in estimates]
        coupled_abs = [
            abs(_cfo_sto_coupled_residual_bins(row)) for row in estimates
        ]
        noisy_available = [row for row in symbol_rows if row["noisy_sync_correct"] != ""]
        oracle_available = [row for row in symbol_rows if row["noisy_sync_correct"] != ""]
        local_symbols = [row for row in symbol_rows if int(row["local_sync_estimate"])]
        count = len(packet_rows)
        output.append(
            {
                "esn0_db": float(esn0_db),
                "packet_trials": count,
                "strict_sync_success_rate": sum(
                    int(row["strict_sync_success"]) for row in packet_rows
                )
                / count
                if count
                else float("nan"),
                "estimate_available_rate": len(estimates) / count if count else float("nan"),
                "local_estimate_rate": len(local_estimates) / count
                if count
                else float("nan"),
                "catastrophic_fraction_given_estimate": 1.0
                - len(local_estimates) / len(estimates)
                if estimates
                else float("nan"),
                "strict_success_rate_given_local_estimate": sum(
                    int(row["strict_sync_success"]) for row in local_estimates
                )
                / len(local_estimates)
                if local_estimates
                else float("nan"),
                "median_abs_cfo_error_bins": float(np.median(cfo_abs))
                if cfo_abs
                else float("nan"),
                "p90_abs_cfo_error_bins": _percentile(cfo_abs, 90.0),
                "median_abs_cfo_error_hz": float(np.median(cfo_abs)) * BIN_HZ
                if cfo_abs
                else float("nan"),
                "p90_abs_cfo_error_hz": _percentile(cfo_abs, 90.0) * BIN_HZ,
                "fraction_abs_cfo_0p25_to_0p75": sum(
                    0.25 <= value <= 0.75 for value in cfo_abs
                )
                / len(cfo_abs)
                if cfo_abs
                else float("nan"),
                "fraction_abs_cfo_gt_0p25": sum(value > 0.25 for value in cfo_abs)
                / len(cfo_abs)
                if cfo_abs
                else float("nan"),
                "fraction_abs_cfo_gt_0p5": sum(value > 0.5 for value in cfo_abs)
                / len(cfo_abs)
                if cfo_abs
                else float("nan"),
                "local_median_abs_cfo_error_bins": float(np.median(local_cfo_abs))
                if local_cfo_abs
                else float("nan"),
                "local_p90_abs_cfo_error_bins": _percentile(local_cfo_abs, 90.0),
                "local_fraction_abs_cfo_0p25_to_0p75": sum(
                    0.25 <= value <= 0.75 for value in local_cfo_abs
                )
                / len(local_cfo_abs)
                if local_cfo_abs
                else float("nan"),
                "median_abs_cfo_sto_coupled_residual_bins": float(
                    np.median(coupled_abs)
                )
                if coupled_abs
                else float("nan"),
                "p90_abs_cfo_sto_coupled_residual_bins": _percentile(
                    coupled_abs, 90.0
                ),
                "fraction_abs_cfo_sto_coupled_0p25_to_0p75": sum(
                    0.25 <= value <= 0.75 for value in coupled_abs
                )
                / len(coupled_abs)
                if coupled_abs
                else float("nan"),
                "median_abs_payload_sto_samples_1m": float(np.median(sto_abs))
                if sto_abs
                else float("nan"),
                "p90_abs_payload_sto_samples_1m": _percentile(sto_abs, 90.0),
                "oracle_sync_savaux_ser": 1.0
                - float(np.mean([int(row["oracle_correct"]) for row in symbol_rows]))
                if symbol_rows
                else float("nan"),
                "oracle_sync_savaux_ser_estimate_available": 1.0
                - float(
                    np.mean([int(row["oracle_correct"]) for row in oracle_available])
                )
                if oracle_available
                else float("nan"),
                "noisy_sync_savaux_ser_estimate_available": 1.0
                - float(
                    np.mean([int(row["noisy_sync_correct"]) for row in noisy_available])
                )
                if noisy_available
                else float("nan"),
                "conditional_savaux_ser_gap": float(
                    np.mean([int(row["oracle_correct"]) for row in oracle_available])
                    - np.mean(
                        [int(row["noisy_sync_correct"]) for row in noisy_available]
                    )
                )
                if noisy_available
                else float("nan"),
                "oracle_sync_savaux_ser_local_estimate": 1.0
                - float(np.mean([int(row["oracle_correct"]) for row in local_symbols]))
                if local_symbols
                else float("nan"),
                "noisy_sync_savaux_ser_local_estimate": 1.0
                - float(
                    np.mean([int(row["noisy_sync_correct"]) for row in local_symbols])
                )
                if local_symbols
                else float("nan"),
                "local_conditional_savaux_ser_gap": float(
                    np.mean([int(row["oracle_correct"]) for row in local_symbols])
                    - np.mean(
                        [int(row["noisy_sync_correct"]) for row in local_symbols]
                    )
                )
                if local_symbols
                else float("nan"),
                "strict_end_to_end_ser": 1.0
                - float(
                    np.mean(
                        [int(row["strict_end_to_end_correct"]) for row in symbol_rows]
                    )
                )
                if symbol_rows
                else float("nan"),
            }
        )
    return output


def _make_plots(
    output_dir: Path,
    trials: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    esn0_values: Sequence[float],
) -> None:
    import matplotlib.pyplot as plt

    columns = len(esn0_values)
    fig, axes = plt.subplots(3, columns, figsize=(3.2 * columns, 8.7), squeeze=False)
    for column, esn0_db in enumerate(esn0_values):
        rows = [
            row
            for row in trials
            if int(row["estimate_available"])
            and math.isclose(float(row["esn0_db"]), float(esn0_db), abs_tol=1e-12)
        ]
        cfo = np.asarray([float(row["cfo_error_bins"]) for row in rows])
        sto = np.asarray([float(row["payload_sto_error_samples_1m"]) for row in rows])
        coupled = np.asarray(
            [_cfo_sto_coupled_residual_bins(row) for row in rows]
        )
        clipped_cfo = np.clip(cfo, -LOCAL_CFO_LIMIT_BINS, LOCAL_CFO_LIMIT_BINS)
        clipped_sto = np.clip(
            sto, -LOCAL_STO_LIMIT_SAMPLES, LOCAL_STO_LIMIT_SAMPLES
        )
        catastrophic = int(
            np.count_nonzero(
                (np.abs(cfo) > LOCAL_CFO_LIMIT_BINS)
                | (np.abs(sto) > LOCAL_STO_LIMIT_SAMPLES)
            )
        )
        axes[0, column].hist(clipped_cfo, bins=17, color="tab:blue", alpha=0.8)
        axes[0, column].axvspan(-0.75, -0.25, color="tab:red", alpha=0.12)
        axes[0, column].axvspan(0.25, 0.75, color="tab:red", alpha=0.12)
        axes[0, column].axvline(0.0, color="black", linewidth=0.8)
        axes[0, column].set_title(
            f"{esn0_db:g} dB\ncoordinate-nonlocal={catastrophic}/{len(rows)}"
        )
        axes[0, column].set_xlabel("CFO error (LoRa bins)")
        axes[1, column].hist(clipped_sto, bins=17, color="tab:orange", alpha=0.8)
        axes[1, column].axvline(0.0, color="black", linewidth=0.8)
        axes[1, column].set_xlabel("payload STO error (1M samples)")
        clipped_coupled = np.clip(coupled, -1.0, 1.0)
        axes[2, column].hist(
            clipped_coupled, bins=17, color="tab:green", alpha=0.8
        )
        axes[2, column].axvspan(-0.75, -0.25, color="tab:red", alpha=0.12)
        axes[2, column].axvspan(0.25, 0.75, color="tab:red", alpha=0.12)
        axes[2, column].axvline(0.0, color="black", linewidth=0.8)
        axes[2, column].set_xlabel("CFO - STO/OSR residual proxy (bins)")
    axes[0, 0].set_ylabel("packet trials")
    axes[1, 0].set_ylabel("packet trials")
    axes[2, 0].set_ylabel("packet trials")
    fig.suptitle("Noisy FrameSync residual distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "noisy_framesync_error_histograms.png", dpi=180)
    plt.close(fig)

    x = [float(row["esn0_db"]) for row in summary]
    fig, axis = plt.subplots(figsize=(7.6, 4.5))
    for key, label in (
        ("oracle_sync_savaux_ser", "Savaux + clean/oracle sync"),
        (
            "noisy_sync_savaux_ser_estimate_available",
            "Savaux + noisy FrameSync estimate",
        ),
        ("strict_end_to_end_ser", "strict FrameSync gate + Savaux"),
    ):
        axis.plot(x, [float(row[key]) for row in summary], marker="o", label=label)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Es/N0 (dB)")
    axis.set_ylabel("SER")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "noisy_vs_oracle_sync_savaux_ser.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.6, 4.5))
    axis.plot(
        x,
        [float(row["strict_sync_success_rate"]) for row in summary],
        marker="o",
        label="strict success",
    )
    axis.plot(
        x,
        [float(row["estimate_available_rate"]) for row in summary],
        marker="o",
        label="estimate available",
    )
    axis.plot(
        x,
        [float(row["local_estimate_rate"]) for row in summary],
        marker="o",
        label="coordinate-local estimate",
    )
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Es/N0 (dB)")
    axis.set_ylabel("packet fraction")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "noisy_framesync_availability.png", dpi=180)
    plt.close(fig)


def _build_report(
    output_dir: Path,
    clean_audit: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# Noisy FrameSync headroom for Savaux",
        "",
        "AWGN is added to the complete 1 MS/s packet before FrameSync. Clean-sync",
        "and noisy-sync Savaux operate on the same noisy IQ. A returned estimate is",
        "reported separately from strict FrameSync validation success.",
        "",
        f"- Clean packets admitted: {sum(int(row.get('sync_valid', 0)) for row in clean_audit)}",
        f"- LoRa bin spacing: {BIN_HZ:.6f} Hz",
        "- Fatal sensitivity band from the preceding experiment: |CFO error|=0.25--0.75 bin",
        f"- Coordinate-local definition: |CFO error|<={LOCAL_CFO_LIMIT_BINS:g} bin and |STO error|<={LOCAL_STO_LIMIT_SAMPLES} samples",
        "- Coupled residual proxy: CFO error - STO error / OSR (LoRa timing/frequency ambiguity)",
        "",
        "| Es/N0 | strict sync | estimate | coord-local | strict / local | raw local median |CFO| | raw CFO in 0.25--0.75 | coupled median | coupled in 0.25--0.75 | oracle SER | noisy SER (local) | local gap | noisy SER (all) | strict E2E SER |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {float(row['esn0_db']):g} | {float(row['strict_sync_success_rate']):.3f} | "
            f"{float(row['estimate_available_rate']):.3f} | "
            f"{float(row['local_estimate_rate']):.3f} | "
            f"{float(row['strict_success_rate_given_local_estimate']):.3f} | "
            f"{float(row['local_median_abs_cfo_error_bins']):.4f} | "
            f"{float(row['local_fraction_abs_cfo_0p25_to_0p75']):.3f} | "
            f"{float(row['median_abs_cfo_sto_coupled_residual_bins']):.4f} | "
            f"{float(row['fraction_abs_cfo_sto_coupled_0p25_to_0p75']):.3f} | "
            f"{float(row['oracle_sync_savaux_ser']):.4f} | "
            f"{float(row['noisy_sync_savaux_ser_local_estimate']):.4f} | "
            f"{float(row['local_conditional_savaux_ser_gap']):.4f} | "
            f"{float(row['noisy_sync_savaux_ser_estimate_available']):.4f} | "
            f"{float(row['strict_end_to_end_ser']):.4f} |"
        )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-repo", type=Path, default=WORKSPACE_ROOT / "lora-rfsr-savaux"
    )
    parser.add_argument("--ota-root", type=Path, default=None)
    parser.add_argument("--max-packets", type=int, default=8)
    parser.add_argument("--symbols-per-packet", type=int, default=16)
    parser.add_argument("--esn0-db", default="12,13,14,15,16")
    parser.add_argument("--seeds", default="20260821,20260822")
    parser.add_argument(
        "--workers", type=int, default=min(4, max(1, os.cpu_count() or 1))
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="rebuild summary, report, and plots from existing trial CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "noisy_framesync_headroom_ota_awgn_20260821",
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
    esn0_values = _parse_list(args.esn0_db, float)
    seeds = _parse_list(args.seeds, int)
    if args.summarize_existing:
        trial_rows = _read_csv(output_dir / "packet_trials.csv")
        symbol_rows = _read_csv(output_dir / "symbol_trials.csv")
        clean_audits = _read_csv(output_dir / "clean_sync_audit.csv")
        for row in trial_rows:
            if int(row["estimate_available"]):
                coupled = _cfo_sto_coupled_residual_bins(row)
                row["cfo_sto_coupled_residual_bins"] = coupled
                row["abs_cfo_sto_coupled_residual_bins"] = abs(coupled)
            else:
                row["cfo_sto_coupled_residual_bins"] = ""
                row["abs_cfo_sto_coupled_residual_bins"] = ""
        summary = _summary_by_snr(trial_rows, symbol_rows, esn0_values)
        _write_csv(output_dir / "packet_trials.csv", trial_rows)
        _write_csv(output_dir / "summary.csv", summary)
        _make_plots(output_dir, trial_rows, summary, esn0_values)
        _build_report(output_dir, clean_audits, summary)
        print(
            json.dumps(
                {"output_dir": str(output_dir), "summary": summary}, indent=2
            )
        )
        return 0
    metadata_paths = list(_packet_metadata_paths(ota_root))[: int(args.max_packets)]

    clean_tasks = [
        {
            "dataset_repo": str(dataset_repo),
            "ota_root": str(ota_root),
            "metadata_path": str(path),
            "symbols_per_packet": int(args.symbols_per_packet),
        }
        for path in metadata_paths
    ]
    clean_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [executor.submit(_clean_packet_worker, task) for task in clean_tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            clean_results.append(future.result())
            print(
                f"completed clean FrameSync audits: {index}/{len(futures)}",
                flush=True,
            )
    clean_results.sort(key=lambda row: int(row["audit"]["reference_id"]))
    clean_audits = [row["audit"] for row in clean_results]
    packets = [row["packet"] for row in clean_results if row["packet"] is not None]
    _write_csv(output_dir / "clean_sync_audit.csv", clean_audits)
    if not packets:
        raise RuntimeError("no packets passed clean synchronization/Savaux audit")

    noisy_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        for esn0_db in esn0_values:
            noisy_tasks = [
                {
                    "dataset_repo": str(dataset_repo),
                    "packet": packet,
                    "esn0_db": float(esn0_db),
                    "seed": int(seed),
                }
                for seed in seeds
                for packet in packets
            ]
            futures = [
                executor.submit(_noisy_trial_worker, task) for task in noisy_tasks
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                noisy_results.append(future.result())
                if index % 4 == 0 or index == len(futures):
                    print(
                        f"completed {esn0_db:g} dB noisy FrameSync trials: "
                        f"{index}/{len(futures)}",
                        flush=True,
                    )

            # Checkpoint every SNR. Low-SNR FrameSync can be substantially slower
            # because more false preamble events enter the real synchronizer.
            checkpoint_results = sorted(
                noisy_results,
                key=lambda row: (
                    float(row["trial"]["esn0_db"]),
                    int(row["trial"]["seed"]),
                    int(row["trial"]["reference_id"]),
                ),
            )
            checkpoint_trials = [row["trial"] for row in checkpoint_results]
            checkpoint_symbols = [
                symbol for row in checkpoint_results for symbol in row["symbols"]
            ]
            completed_snr = tuple(
                value
                for value in esn0_values
                if any(
                    float(row["esn0_db"]) == float(value)
                    for row in checkpoint_trials
                )
            )
            _write_csv(output_dir / "packet_trials.csv", checkpoint_trials)
            _write_csv(output_dir / "symbol_trials.csv", checkpoint_symbols)
            _write_csv(
                output_dir / "summary.csv",
                _summary_by_snr(
                    checkpoint_trials, checkpoint_symbols, completed_snr
                ),
            )
            print(f"checkpointed completed SNR: {esn0_db:g} dB", flush=True)
    noisy_results.sort(
        key=lambda row: (
            float(row["trial"]["esn0_db"]),
            int(row["trial"]["seed"]),
            int(row["trial"]["reference_id"]),
        )
    )
    trial_rows = [row["trial"] for row in noisy_results]
    symbol_rows = [symbol for row in noisy_results for symbol in row["symbols"]]
    summary = _summary_by_snr(trial_rows, symbol_rows, esn0_values)
    _write_csv(output_dir / "packet_trials.csv", trial_rows)
    _write_csv(output_dir / "symbol_trials.csv", symbol_rows)
    _write_csv(output_dir / "summary.csv", summary)
    _make_plots(output_dir, trial_rows, summary, esn0_values)
    _build_report(
        output_dir,
        clean_audits,
        summary,
    )
    config = {
        "dataset_repo": str(dataset_repo),
        "ota_root": str(ota_root),
        "max_packets": int(args.max_packets),
        "clean_packets_admitted": len(packets),
        "symbols_per_packet": int(args.symbols_per_packet),
        "esn0_db": list(esn0_values),
        "seeds": list(seeds),
        "workers": int(args.workers),
        "noise_model": "complex AWGN flat in B, added once to the complete 1 MS/s packet",
        "savaux_origin_offset_samples_1m": SAVAUX_ORIGIN_OFFSET_SAMPLES,
        "bin_hz": BIN_HZ,
        "local_lock_definition": {
            "max_abs_cfo_error_bins": LOCAL_CFO_LIMIT_BINS,
            "max_abs_payload_sto_error_samples_1m": LOCAL_STO_LIMIT_SAMPLES,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
