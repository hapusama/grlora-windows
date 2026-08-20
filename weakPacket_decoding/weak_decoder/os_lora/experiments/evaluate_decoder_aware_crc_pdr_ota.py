"""Evaluate decoder-aware FrameSync gating with the complete LoRa PHY codec.

The experiment adds one realization of B-wide complex AWGN to each complete
1 MS/s OTA packet, runs the real weak-packet FrameSync, and then compares:

* strict FrameSync gate + Savaux + FEC/CRC;
* decoder-aware gate + the same Savaux + FEC/CRC;
* clean/oracle synchronization + the same noisy IQ and decoder.

The expected bytes are used only after decoding to score exact packet
delivery and CRC collisions.  They never influence synchronization, symbol
decisions, header decoding, FEC, or CRC.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
WEAK_PACKET_ROOT = SCRIPT_PATH.parents[3]
GR_LORA_ROOT = SCRIPT_PATH.parents[4]
WORKSPACE_ROOT = GR_LORA_ROOT.parent
if str(WEAK_PACKET_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_PACKET_ROOT))

from weak_decoder.os_lora.experiment_support.noisy_framesync_ota import (  # noqa: E402
    load_json,
    load_single_packet_sync_module,
    make_sync_config,
    packet_metadata_paths,
    parse_csv_list,
    read_csv_rows,
    sync_gate_failures,
    unit_lora_band_awgn,
    write_csv_rows,
)
from weak_decoder.os_lora.system.decoder_aware_crc import (  # noqa: E402
    DecoderAwarePacketResult,
    decode_savaux_sync_candidate,
)
from weak_decoder.os_lora.system.ambiguity_ridge_list import (  # noqa: E402
    arbitrate_sync_list_with_crc,
    build_ambiguity_ridge_sync_list,
)
from weak_decoder.os_lora.system.soft_hamming_crc import (  # noqa: E402
    decode_soft_hamming_sync_candidate,
)


CRC_MODE = "grlora"
LDRO_MODE = 1
SF = 12
N_BINS = 1 << SF
BW_HZ = 125_000.0
SOURCE_RATE_HZ = 1_000_000.0
OS_FACTOR = 8
SYMBOL_SAMPLES = N_BINS * OS_FACTOR
SAVAUX_ORIGIN_OFFSET_SAMPLES = -OS_FACTOR // 2
SIGNAL_POWER_SYMBOLS = 16
# Fulltrim windows end a few samples before the last SFO-adjusted Savaux
# interval on some packets.  Add noise-only tail samples, not clean signal, so
# the final 0.1% of that observation is well-defined without an SNR advantage.
DEMOD_TAIL_SAMPLES = 256


def _split_cfo_bins(total_bins: float) -> tuple[int, float]:
    integer = int(math.floor(float(total_bins) + 0.5))
    return integer, float(total_bins - integer)


def _advance_symbol_cursor(
    cursor: int, sfo_cum: float, sfo_hat: float
) -> tuple[int, float]:
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
    starts: list[int] = []
    for frame_index in range(int(header_count) + int(payload_count)):
        if frame_index >= int(header_count):
            starts.append(cursor)
        cursor, cumulative = _advance_symbol_cursor(cursor, cumulative, sfo_hat)
    return starts


def _sync_config(module: Any, center_frequency_hz: float) -> Any:
    return make_sync_config(
        module,
        center_frequency_hz,
        sf=SF,
        bw_hz=BW_HZ,
        sample_rate_hz=SOURCE_RATE_HZ,
        preamble_symbols=16,
        sync_word=0x12,
    )


def _receiver_symbols(reference: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Convert legacy modulator IDs to gr-lora_sdr hard-symbol semantics."""

    n_bins = 1 << int(reference["phy"]["sf"])
    ldro = bool(reference["phy"]["ldro"])
    header = [
        ((int(value) - 1) % n_bins) // 4
        for value in reference["symbols"]["header_ids"]
    ]
    divisor = 4 if ldro else 1
    payload = [
        ((int(value) - 1) % n_bins) // divisor
        for value in reference["symbols"]["payload_ids"]
    ]
    return header, payload


def _clean_frame_sync(packet: dict[str, Any]) -> SimpleNamespace:
    cfo_int, cfo_frac = _split_cfo_bins(float(packet["clean_cfo_total_bins"]))
    return SimpleNamespace(
        valid=True,
        fine_payload_start_sample=int(packet["clean_fine_payload_start_sample"]),
        cfo_int_est=int(cfo_int),
        cfo_frac_est=float(cfo_frac),
        sfo_hat=float(packet["clean_sfo_hat"]),
        sfo_cum_initial=float(packet["clean_sfo_cum_initial"]),
    )


def _decode(
    samples: np.ndarray,
    frame_sync: Any | None,
    *,
    allow_gate_failed_candidate: bool,
) -> DecoderAwarePacketResult:
    return decode_savaux_sync_candidate(
        samples,
        frame_sync,
        sf=SF,
        bw_hz=BW_HZ,
        os_factor=OS_FACTOR,
        ldro_mode=LDRO_MODE,
        crc_mode=CRC_MODE,
        allow_gate_failed_candidate=allow_gate_failed_candidate,
    )


def _exact_payload(result: DecoderAwarePacketResult, expected: bytes) -> bool:
    return bool(
        result.header_valid
        and result.crc_valid
        and result.payload_bytes == bytes(expected)
    )


def _symbol_errors(actual: Sequence[int], expected: Sequence[int]) -> int:
    overlap = min(len(actual), len(expected))
    return int(
        sum(int(actual[index]) != int(expected[index]) for index in range(overlap))
        + abs(len(actual) - len(expected))
    )


def _clean_worker(task: dict[str, Any]) -> dict[str, Any]:
    dataset_repo = Path(str(task["dataset_repo"]))
    ota_root = Path(str(task["ota_root"]))
    metadata = load_json(Path(str(task["metadata_path"])))
    reference_id = int(metadata["reference"]["reference_id"])
    reference_path = ota_root.parent / "metadata" / f"{reference_id:06d}.json"
    reference = load_json(reference_path)
    iq_path = ota_root / str(metadata["ota"]["relative_path"])
    samples = np.fromfile(iq_path, dtype=np.dtype("<c8"))
    module = load_single_packet_sync_module(dataset_repo)
    result = module.run_single_packet_sync(
        samples,
        _sync_config(module, float(metadata["capture"]["center_frequency_hz"])),
    )
    audit: dict[str, Any] = {
        "packet_id": str(metadata["ota_id"]),
        "reference_id": reference_id,
        "sync_status": str(result.status),
        "sync_valid": int(bool(result.synchronized)),
        "tested_payload_symbols": 0,
        "admitted_payload_symbols": 0,
    }
    if not result.synchronized or result.frame_sync is None:
        audit.update(
            {
                "sync_error": str(result.error or ""),
                "full_header_valid": 0,
                "full_crc_valid": 0,
                "full_payload_exact": 0,
            }
        )
        return {"audit": audit, "packet": None}

    frame_sync = result.frame_sync
    header_count = len(reference["symbols"]["header_ids"])
    calibration_count = min(
        SIGNAL_POWER_SYMBOLS, len(reference["symbols"]["payload_ids"])
    )
    starts = _payload_starts(
        int(frame_sync.fine_payload_start_sample),
        float(frame_sync.sfo_cum_initial),
        float(frame_sync.sfo_hat),
        header_count,
        calibration_count,
    )
    off_count = int(metadata["ota"]["leading_real_off_packet_samples"])
    offpacket_power = float(
        np.mean(np.abs(samples[:off_count]).astype(np.float64) ** 2)
    )
    signal_powers: list[float] = []
    for start_sample in starts:
        start_sample += SAVAUX_ORIGIN_OFFSET_SAMPLES
        stop_sample = start_sample + SYMBOL_SAMPLES
        audit["tested_payload_symbols"] += 1
        if start_sample < 0 or stop_sample > samples.size:
            continue
        symbol_power = float(
            np.mean(
                np.abs(samples[start_sample:stop_sample]).astype(np.float64) ** 2
            )
        )
        signal_powers.append(
            max(symbol_power - offpacket_power, np.finfo(np.float64).tiny)
        )
        audit["admitted_payload_symbols"] += 1
    if not signal_powers:
        audit.update(
            {
                "full_header_valid": 0,
                "full_crc_valid": 0,
                "full_payload_exact": 0,
            }
        )
        return {"audit": audit, "packet": None}

    signal_power = float(np.mean(signal_powers))
    packet = {
        "packet_id": str(metadata["ota_id"]),
        "reference_id": reference_id,
        "iq_path": str(iq_path),
        "center_frequency_hz": float(metadata["capture"]["center_frequency_hz"]),
        "signal_power": signal_power,
        "clean_cfo_total_bins": float(frame_sync.cfo_total_est),
        "clean_fine_payload_start_sample": int(frame_sync.fine_payload_start_sample),
        "clean_fine_preamble_start_sample": int(frame_sync.fine_preamble_start_sample),
        "clean_sfo_hat": float(frame_sync.sfo_hat),
        "clean_sfo_cum_initial": float(frame_sync.sfo_cum_initial),
    }
    audit.update(
        {
            "fine_payload_start_sample": int(frame_sync.fine_payload_start_sample),
            "cfo_total_bins": float(frame_sync.cfo_total_est),
            "cfo_hz": float(frame_sync.cfo_hz_est),
            "sfo_hat": float(frame_sync.sfo_hat),
            "sfo_cum_initial": float(frame_sync.sfo_cum_initial),
            "signal_power": signal_power,
            "offpacket_power": offpacket_power,
        }
    )
    expected = bytes.fromhex(str(reference["packet"]["frame_hex"]))
    expected_header, expected_payload = _receiver_symbols(reference)
    samples = np.fromfile(Path(str(packet["iq_path"])), dtype=np.dtype("<c8"))
    padded = np.pad(samples, (0, DEMOD_TAIL_SAMPLES)).astype(np.complex64)
    start = time.perf_counter()
    decoded = _decode(
        padded,
        _clean_frame_sync(packet),
        allow_gate_failed_candidate=False,
    )
    elapsed_ms = 1e3 * (time.perf_counter() - start)
    exact = _exact_payload(decoded, expected)
    actual_header = [item.symbol_value for item in decoded.header_symbols]
    actual_payload = [item.symbol_value for item in decoded.payload_symbols]
    audit.update(
        {
            "full_decode_status": decoded.status,
            "full_header_valid": int(decoded.header_valid),
            "full_crc_valid": int(decoded.crc_valid),
            "full_payload_exact": int(exact),
            "full_decode_ms": elapsed_ms,
            "full_header_symbol_errors": _symbol_errors(
                actual_header, expected_header
            ),
            "full_payload_symbol_errors": _symbol_errors(
                actual_payload, expected_payload
            ),
        }
    )
    if not exact:
        return {"audit": audit, "packet": None}

    packet = dict(packet)
    packet.update(
        {
            "expected_frame_hex": expected.hex(),
            "expected_header_symbols": expected_header,
            "expected_payload_symbols": expected_payload,
            "header_count": len(expected_header),
            "payload_count": len(expected_payload),
        }
    )
    # symbol_specs belong to the old SER proxy and are deliberately not carried
    # into the CRC experiment.
    packet.pop("symbol_specs", None)
    return {"audit": audit, "packet": packet}


def _decode_diagnostics(
    result: DecoderAwarePacketResult,
    packet: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    expected = bytes.fromhex(str(packet["expected_frame_hex"]))
    header_values = [item.symbol_value for item in result.header_symbols]
    payload_values = [item.symbol_value for item in result.payload_symbols]
    exact = _exact_payload(result, expected)
    return {
        f"{prefix}_decode_status": result.status,
        f"{prefix}_header_valid": int(result.header_valid),
        f"{prefix}_crc_valid": int(result.crc_valid),
        f"{prefix}_payload_exact": int(exact),
        f"{prefix}_payload_hex": result.payload_bytes.hex(),
        f"{prefix}_header_symbols_decoded": len(header_values),
        f"{prefix}_payload_symbols_decoded": len(payload_values),
        f"{prefix}_header_symbol_errors": _symbol_errors(
            header_values, packet["expected_header_symbols"]
        ),
        f"{prefix}_payload_symbol_errors": _symbol_errors(
            payload_values, packet["expected_payload_symbols"]
        ),
    }


def _noisy_worker(task: dict[str, Any]) -> dict[str, Any]:
    packet = dict(task["packet"])
    dataset_repo = Path(str(task["dataset_repo"]))
    esn0_db = float(task["esn0_db"])
    seed = int(task["seed"])
    ridge_top_k = int(task.get("ridge_top_k", 0))
    ridge_sfd_peak_pool = int(task.get("ridge_sfd_peak_pool", 32))
    ridge_soft_hamming = bool(task.get("ridge_soft_hamming", False))
    clean = np.fromfile(Path(str(packet["iq_path"])), dtype=np.dtype("<c8"))
    rng = np.random.default_rng(
        np.random.SeedSequence((seed, int(packet["reference_id"]), 73013))
    )
    noise_power = float(packet["signal_power"]) * N_BINS / (
        10.0 ** (esn0_db / 10.0)
    )
    # Generate the original-length realization first.  This exactly reproduces
    # the preceding FrameSync gate audit for the same packet/SNR/seed.
    noise = unit_lora_band_awgn(rng, clean.size, os_factor=OS_FACTOR)
    noisy = np.asarray(clean + noise * math.sqrt(noise_power), dtype=np.complex64)
    del noise
    tail = unit_lora_band_awgn(rng, DEMOD_TAIL_SAMPLES, os_factor=OS_FACTOR)
    decode_samples = np.concatenate(
        [noisy, tail * math.sqrt(noise_power)]
    ).astype(np.complex64)
    del tail

    module = load_single_packet_sync_module(dataset_repo)
    sync_config = _sync_config(module, float(packet["center_frequency_hz"]))
    sync_started = time.perf_counter()
    sync_result = module.run_single_packet_sync(noisy, sync_config)
    sync_ms = 1e3 * (time.perf_counter() - sync_started)
    frame_sync = sync_result.frame_sync
    estimate_available = frame_sync is not None
    strict_gate = bool(sync_result.synchronized)
    decoder_gate = bool(
        sync_result.accepted("decoder_aware")
        if hasattr(sync_result, "accepted")
        else (
            sync_result.frame_location is not None
            and sync_result.frame_location.valid
            and frame_sync is not None
            and frame_sync.netid_valid
        )
    )

    current_started = time.perf_counter()
    current = _decode(
        decode_samples,
        frame_sync,
        # We decode every available estimate for audit.  Policy-specific call
        # counts below include it only when that gate would admit the packet.
        allow_gate_failed_candidate=True,
    )
    current_ms = 1e3 * (time.perf_counter() - current_started)
    oracle_started = time.perf_counter()
    oracle = _decode(
        decode_samples,
        _clean_frame_sync(packet),
        allow_gate_failed_candidate=False,
    )
    oracle_ms = 1e3 * (time.perf_counter() - oracle_started)

    expected = bytes.fromhex(str(packet["expected_frame_hex"]))
    current_exact = _exact_payload(current, expected)
    oracle_exact = _exact_payload(oracle, expected)

    ridge_started = time.perf_counter()
    ridge_list = build_ambiguity_ridge_sync_list(
        noisy,
        frame_sync,
        sf=SF,
        bw_hz=BW_HZ,
        os_factor=OS_FACTOR,
        center_frequency_hz=float(packet["center_frequency_hz"]),
        preamble_symbols=16,
        sync_word=0x12,
        top_k=max(1, ridge_top_k),
        sfd_peak_pool=ridge_sfd_peak_pool,
    ) if ridge_top_k > 0 else None
    ridge_list_ms = 1e3 * (time.perf_counter() - ridge_started)
    ridge_decode_started = time.perf_counter()
    ridge_arbitration = (
        arbitrate_sync_list_with_crc(
            decode_samples,
            ridge_list.candidates,
            sf=SF,
            bw_hz=BW_HZ,
            os_factor=OS_FACTOR,
            ldro_mode=LDRO_MODE,
            crc_mode=CRC_MODE,
            require_payload_crc=True,
            stop_on_crc=True,
        )
        if ridge_list is not None
        else None
    )
    ridge_decode_ms = 1e3 * (time.perf_counter() - ridge_decode_started)
    ridge_selected = (
        None if ridge_arbitration is None else ridge_arbitration.selected
    )
    ridge_crc_accept = ridge_selected is not None
    ridge_exact = bool(
        ridge_selected is not None
        and _exact_payload(ridge_selected.decode, expected)
    )

    ridge_soft_started = time.perf_counter()
    ridge_soft_arbitration = (
        arbitrate_sync_list_with_crc(
            decode_samples,
            ridge_list.candidates,
            sf=SF,
            bw_hz=BW_HZ,
            os_factor=OS_FACTOR,
            ldro_mode=LDRO_MODE,
            crc_mode=CRC_MODE,
            require_payload_crc=True,
            stop_on_crc=True,
            decoder_mode="soft_hamming",
        )
        if ridge_list is not None and ridge_soft_hamming
        else None
    )
    ridge_soft_decode_ms = 1e3 * (
        time.perf_counter() - ridge_soft_started
    )
    ridge_soft_selected = (
        None
        if ridge_soft_arbitration is None
        else ridge_soft_arbitration.selected
    )
    ridge_soft_crc_accept = ridge_soft_selected is not None
    ridge_soft_exact = bool(
        ridge_soft_selected is not None
        and _exact_payload(ridge_soft_selected.decode, expected)
    )
    soft_oracle_started = time.perf_counter()
    soft_oracle = (
        decode_soft_hamming_sync_candidate(
            decode_samples,
            _clean_frame_sync(packet),
            sf=SF,
            bw_hz=BW_HZ,
            os_factor=OS_FACTOR,
            ldro_mode=LDRO_MODE,
            crc_mode=CRC_MODE,
            allow_gate_failed_candidate=False,
        )
        if ridge_soft_hamming
        else None
    )
    soft_oracle_ms = 1e3 * (time.perf_counter() - soft_oracle_started)
    soft_oracle_exact = bool(
        soft_oracle is not None and _exact_payload(soft_oracle, expected)
    )
    strict_delivery = bool(strict_gate and current_exact)
    decoder_delivery = bool(decoder_gate and current_exact)
    strict_crc_accept = bool(strict_gate and current.crc_valid)
    decoder_crc_accept = bool(decoder_gate and current.crc_valid)
    extra_candidate = bool(decoder_gate and not strict_gate)
    all_symbol_errors = int(
        _symbol_errors(
            [item.symbol_value for item in current.header_symbols],
            packet["expected_header_symbols"],
        )
        + _symbol_errors(
            [item.symbol_value for item in current.payload_symbols],
            packet["expected_payload_symbols"],
        )
    )

    row: dict[str, Any] = {
        "trial_id": f"{packet['packet_id']}:{esn0_db:g}:{seed}",
        "packet_id": str(packet["packet_id"]),
        "reference_id": int(packet["reference_id"]),
        "esn0_db": esn0_db,
        "seed": seed,
        "sync_status": str(sync_result.status),
        "estimate_available": int(estimate_available),
        "strict_gate": int(strict_gate),
        "decoder_aware_gate": int(decoder_gate),
        "extra_decoder_candidate": int(extra_candidate),
        "gate_failure_reasons": "|".join(
            sync_gate_failures(sync_result, sync_config)
        ),
        "strict_crc_accept": int(strict_crc_accept),
        "decoder_aware_crc_accept": int(decoder_crc_accept),
        "strict_packet_delivered": int(strict_delivery),
        "decoder_aware_packet_delivered": int(decoder_delivery),
        "oracle_packet_delivered": int(oracle_exact),
        "ridge_top_k": ridge_top_k,
        "ridge_packet_delivered": int(ridge_exact),
        "ridge_crc_accept": int(ridge_crc_accept),
        "ridge_crc_false_delivery": int(ridge_crc_accept and not ridge_exact),
        "ridge_rescue_over_decoder_aware": int(
            ridge_exact and not decoder_delivery
        ),
        "ridge_regression_vs_decoder_aware": int(
            decoder_delivery and not ridge_exact
        ),
        "ridge_candidate_count": 0
        if ridge_list is None
        else len(ridge_list.candidates),
        "ridge_decoder_attempts": 0
        if ridge_arbitration is None
        else len(ridge_arbitration.attempts),
        "ridge_selected_candidate": ""
        if ridge_arbitration is None
        or ridge_arbitration.selected_index is None
        else int(ridge_arbitration.selected_index),
        "ridge_selected_sfd_peak_rank": ""
        if ridge_selected is None
        else int(ridge_selected.candidate.sfd_peak_rank),
        "ridge_selected_delta_cfo_bins": ""
        if ridge_selected is None
        else float(ridge_selected.candidate.delta_cfo_bins),
        "ridge_selected_delta_payload_samples": ""
        if ridge_selected is None
        else int(ridge_selected.candidate.delta_payload_samples),
        "ridge_soft_hamming_enabled": int(ridge_soft_hamming),
        "ridge_soft_packet_delivered": int(ridge_soft_exact),
        "ridge_soft_crc_accept": int(ridge_soft_crc_accept),
        "ridge_soft_crc_false_delivery": int(
            ridge_soft_crc_accept and not ridge_soft_exact
        ),
        "ridge_soft_rescue_over_hard_ridge": int(
            ridge_soft_hamming and ridge_soft_exact and not ridge_exact
        ),
        "ridge_soft_regression_vs_hard_ridge": int(
            ridge_soft_hamming and ridge_exact and not ridge_soft_exact
        ),
        "ridge_soft_decoder_attempts": 0
        if ridge_soft_arbitration is None
        else len(ridge_soft_arbitration.attempts),
        "ridge_soft_selected_candidate": ""
        if ridge_soft_arbitration is None
        or ridge_soft_arbitration.selected_index is None
        else int(ridge_soft_arbitration.selected_index),
        "soft_oracle_packet_delivered": int(soft_oracle_exact),
        "strict_crc_false_delivery": int(strict_crc_accept and not current_exact),
        "decoder_aware_crc_false_delivery": int(
            decoder_crc_accept and not current_exact
        ),
        "extra_candidate_crc_rescue": int(extra_candidate and current_exact),
        "extra_candidate_crc_reject": int(
            extra_candidate and not current.crc_valid
        ),
        "extra_candidate_crc_false_accept": int(
            extra_candidate and current.crc_valid and not current_exact
        ),
        "fec_corrected_packet": int(current_exact and all_symbol_errors > 0),
        "current_all_symbol_errors": all_symbol_errors,
        "sync_ms": sync_ms,
        "current_decode_ms": current_ms,
        "oracle_decode_ms": oracle_ms,
        "ridge_list_generation_ms": ridge_list_ms,
        "ridge_decode_operational_ms": ridge_decode_ms,
        "ridge_soft_decode_operational_ms": ridge_soft_decode_ms,
        "soft_oracle_decode_ms": soft_oracle_ms,
        "strict_modeled_decode_ms": current_ms if strict_gate else 0.0,
        "decoder_aware_modeled_decode_ms": current_ms if decoder_gate else 0.0,
        "extra_modeled_decode_ms": current_ms if extra_candidate else 0.0,
        "signal_power": float(packet["signal_power"]),
        "added_noise_power": noise_power,
    }
    row.update(_decode_diagnostics(current, packet, "current"))
    row.update(_decode_diagnostics(oracle, packet, "oracle"))
    if frame_sync is not None:
        row.update(
            {
                "frame_sync_valid": int(bool(frame_sync.valid)),
                "netid_valid": int(bool(frame_sync.netid_valid)),
                "noisy_cfo_total_bins": float(frame_sync.cfo_total_est),
                "noisy_payload_start_sample": int(
                    frame_sync.fine_payload_start_sample
                ),
            }
        )
    return row


def _rate(rows: Sequence[dict[str, Any]], key: str) -> float:
    return (
        float(np.mean([int(row.get(key, 0)) for row in rows]))
        if rows
        else 0.0
    )


def _sum(rows: Sequence[dict[str, Any]], key: str) -> int:
    return int(sum(int(row.get(key, 0)) for row in rows))


def _sum_float(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key, 0.0)) for row in rows))


def _summary_by_snr(
    rows: Sequence[dict[str, Any]], esn0_values: Sequence[float]
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for esn0_db in esn0_values:
        selected = [
            row for row in rows if float(row["esn0_db"]) == float(esn0_db)
        ]
        total = len(selected)
        if not total:
            continue
        strict_calls = _sum(selected, "strict_gate")
        decoder_calls = _sum(selected, "decoder_aware_gate")
        extra_calls = _sum(selected, "extra_decoder_candidate")
        summary.append(
            {
                "esn0_db": float(esn0_db),
                "trials": total,
                "estimate_available_rate": _rate(selected, "estimate_available"),
                "strict_gate_rate": _rate(selected, "strict_gate"),
                "decoder_aware_gate_rate": _rate(selected, "decoder_aware_gate"),
                "strict_crc_pdr": _rate(selected, "strict_packet_delivered"),
                "decoder_aware_crc_pdr": _rate(
                    selected, "decoder_aware_packet_delivered"
                ),
                "oracle_crc_pdr": _rate(selected, "oracle_packet_delivered"),
                "ridge_enabled": int(
                    any(int(row.get("ridge_top_k", 0)) > 0 for row in selected)
                ),
                "ridge_top_k": max(
                    int(row.get("ridge_top_k", 0)) for row in selected
                ),
                "ridge_crc_pdr": _rate(selected, "ridge_packet_delivered"),
                "ridge_crc_accept_rate": _rate(selected, "ridge_crc_accept"),
                "ridge_crc_false_deliveries": _sum(
                    selected, "ridge_crc_false_delivery"
                ),
                "ridge_rescues_over_decoder_aware": _sum(
                    selected, "ridge_rescue_over_decoder_aware"
                ),
                "ridge_regressions_vs_decoder_aware": _sum(
                    selected, "ridge_regression_vs_decoder_aware"
                ),
                "ridge_mean_decoder_attempts": float(
                    np.mean(
                        [int(row["ridge_decoder_attempts"]) for row in selected]
                    )
                ),
                "ridge_mean_list_generation_ms": float(
                    np.mean(
                        [
                            float(row["ridge_list_generation_ms"])
                            for row in selected
                        ]
                    )
                ),
                "ridge_mean_operational_decode_ms": float(
                    np.mean(
                        [
                            float(row["ridge_decode_operational_ms"])
                            for row in selected
                        ]
                    )
                ),
                "ridge_soft_hamming_enabled": int(
                    any(
                        int(row.get("ridge_soft_hamming_enabled", 0))
                        for row in selected
                    )
                ),
                "ridge_soft_crc_pdr": _rate(
                    selected, "ridge_soft_packet_delivered"
                ),
                "ridge_soft_crc_accept_rate": _rate(
                    selected, "ridge_soft_crc_accept"
                ),
                "ridge_soft_crc_false_deliveries": _sum(
                    selected, "ridge_soft_crc_false_delivery"
                ),
                "ridge_soft_rescues_over_hard_ridge": _sum(
                    selected, "ridge_soft_rescue_over_hard_ridge"
                ),
                "ridge_soft_regressions_vs_hard_ridge": _sum(
                    selected, "ridge_soft_regression_vs_hard_ridge"
                ),
                "ridge_soft_mean_decoder_attempts": float(
                    np.mean(
                        [
                            int(row.get("ridge_soft_decoder_attempts", 0))
                            for row in selected
                        ]
                    )
                ),
                "ridge_soft_mean_operational_decode_ms": float(
                    np.mean(
                        [
                            float(
                                row.get("ridge_soft_decode_operational_ms", 0.0)
                            )
                            for row in selected
                        ]
                    )
                ),
                "soft_oracle_crc_pdr": _rate(
                    selected, "soft_oracle_packet_delivered"
                ),
                "strict_crc_accept_rate": _rate(selected, "strict_crc_accept"),
                "decoder_aware_crc_accept_rate": _rate(
                    selected, "decoder_aware_crc_accept"
                ),
                "strict_crc_false_deliveries": _sum(
                    selected, "strict_crc_false_delivery"
                ),
                "decoder_aware_crc_false_deliveries": _sum(
                    selected, "decoder_aware_crc_false_delivery"
                ),
                "strict_decoder_calls": strict_calls,
                "decoder_aware_decoder_calls": decoder_calls,
                "extra_decoder_calls": extra_calls,
                "extra_candidate_crc_rescues": _sum(
                    selected, "extra_candidate_crc_rescue"
                ),
                "extra_candidate_crc_rejects": _sum(
                    selected, "extra_candidate_crc_reject"
                ),
                "extra_candidate_crc_false_accepts": _sum(
                    selected, "extra_candidate_crc_false_accept"
                ),
                "fec_corrected_packets": _sum(selected, "fec_corrected_packet"),
                "oracle_gap_gate_rejects": sum(
                    int(row["oracle_packet_delivered"])
                    and not int(row["decoder_aware_packet_delivered"])
                    and not int(row["decoder_aware_gate"])
                    for row in selected
                ),
                "oracle_gap_bad_candidate_decodes": sum(
                    int(row["oracle_packet_delivered"])
                    and not int(row["decoder_aware_packet_delivered"])
                    and int(row["decoder_aware_gate"])
                    for row in selected
                ),
                "oracle_demod_failures": sum(
                    not int(row["oracle_packet_delivered"])
                    for row in selected
                ),
                "decoder_aware_beats_oracle": sum(
                    int(row["decoder_aware_packet_delivered"])
                    and not int(row["oracle_packet_delivered"])
                    for row in selected
                ),
                "strict_decode_total_ms": _sum_float(
                    selected, "strict_modeled_decode_ms"
                ),
                "decoder_aware_decode_total_ms": _sum_float(
                    selected, "decoder_aware_modeled_decode_ms"
                ),
                "extra_decode_total_ms": _sum_float(
                    selected, "extra_modeled_decode_ms"
                ),
                "extra_decode_ms_per_trial": _sum_float(
                    selected, "extra_modeled_decode_ms"
                )
                / total,
                "mean_sync_ms": float(
                    np.mean([float(row["sync_ms"]) for row in selected])
                ),
                "mean_candidate_decode_ms": float(
                    np.mean([float(row["current_decode_ms"]) for row in selected])
                ),
            }
        )
    return summary


def _make_plot(output_dir: Path, summary: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [float(row["esn0_db"]) for row in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(
        x,
        [float(row["strict_crc_pdr"]) for row in summary],
        "o-",
        label="strict gate",
    )
    axes[0].plot(
        x,
        [float(row["decoder_aware_crc_pdr"]) for row in summary],
        "s-",
        label="decoder-aware gate",
    )
    if any(int(row.get("ridge_enabled", 0)) for row in summary):
        ridge_k = max(int(row.get("ridge_top_k", 0)) for row in summary)
        axes[0].plot(
            x,
            [float(row["ridge_crc_pdr"]) for row in summary],
            "D-",
            label=f"ridge list K={ridge_k}",
        )
    if any(int(row.get("ridge_soft_hamming_enabled", 0)) for row in summary):
        ridge_k = max(int(row.get("ridge_top_k", 0)) for row in summary)
        axes[0].plot(
            x,
            [float(row["ridge_soft_crc_pdr"]) for row in summary],
            "P-",
            label=f"ridge K={ridge_k} + soft Hamming",
        )
    axes[0].plot(
        x,
        [float(row["oracle_crc_pdr"]) for row in summary],
        "^-",
        label="oracle sync + hard Hamming",
    )
    if any(int(row.get("ridge_soft_hamming_enabled", 0)) for row in summary):
        axes[0].plot(
            x,
            [float(row["soft_oracle_crc_pdr"]) for row in summary],
            "v--",
            label="oracle sync + soft Hamming",
        )
    axes[0].set_xlabel(r"$E_s/N_0$ (dB)")
    axes[0].set_ylabel("Exact CRC-valid packet delivery rate")
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if any(int(row.get("ridge_soft_hamming_enabled", 0)) for row in summary):
        axes[1].plot(
            x,
            [float(row["ridge_mean_decoder_attempts"]) for row in summary],
            "D-",
            label="hard ridge",
        )
        axes[1].plot(
            x,
            [
                float(row["ridge_soft_mean_decoder_attempts"])
                for row in summary
            ],
            "P-",
            label="ridge + soft Hamming",
        )
        axes[1].set_ylabel("Mean packet-decoder attempts")
        axes[1].set_ylim(0.0, max(4.1, axes[1].get_ylim()[1]))
    else:
        axes[1].plot(
            x,
            [int(row["extra_decoder_calls"]) for row in summary],
            "o-",
            label="extra candidates",
        )
        axes[1].plot(
            x,
            [int(row["extra_candidate_crc_rescues"]) for row in summary],
            "s-",
            label="CRC-valid rescues",
        )
        axes[1].plot(
            x,
            [int(row["extra_candidate_crc_rejects"]) for row in summary],
            "x-",
            label="CRC rejects",
        )
        axes[1].set_ylabel("Packets per SNR")
    axes[1].set_xlabel(r"$E_s/N_0$ (dB)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "decoder_aware_crc_pdr.png", dpi=180)
    plt.close(fig)


def _interpolated_threshold(
    summary: Sequence[dict[str, Any]], key: str, target: float
) -> float | None:
    """Return the first linearly interpolated SNR crossing of a PDR target."""

    points = sorted(
        (float(row["esn0_db"]), float(row.get(key, 0.0)))
        for row in summary
    )
    if not points:
        return None
    if points[0][1] >= float(target):
        return points[0][0]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if y0 < float(target) <= y1:
            if y1 == y0:
                return x1
            fraction = (float(target) - y0) / (y1 - y0)
            return x0 + fraction * (x1 - x0)
    return None


def _build_report(
    output_dir: Path,
    clean_audits: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
) -> None:
    clean_exact = sum(int(row.get("full_payload_exact", 0)) for row in clean_audits)
    ridge_enabled = any(int(row.get("ridge_enabled", 0)) for row in summary)
    ridge_soft_enabled = any(
        int(row.get("ridge_soft_hamming_enabled", 0)) for row in summary
    )
    ridge_k = max(
        (int(row.get("ridge_top_k", 0)) for row in summary), default=0
    )
    lines = [
        "# Decoder-aware FrameSync: full FEC/CRC OTA AWGN evaluation",
        "",
        "## Protocol",
        "",
        "- Add B-wide complex AWGN once to the complete 1 MS/s OTA packet before FrameSync.",
        "- Decode the noisy estimate with Savaux, explicit-header FEC, payload FEC, dewhitening, and PHY CRC.",
        "- Strict and decoder-aware policies reuse the identical synchronization estimate and decoded candidate; only admission differs.",
        "- The oracle branch reuses clean synchronization on the same noisy IQ.",
        "- Expected bytes are consulted only after CRC to detect collisions and score exact PDR.",
        "",
        "## Clean calibration",
        "",
        f"- Exact full-frame round trips: **{clean_exact}/{len(clean_audits)}**.",
        f"- CRC mode: `{CRC_MODE}`; demod tail: {DEMOD_TAIL_SAMPLES} noise-only samples.",
        "",
        "## Packet-level results",
        "",
        (
            "| Es/N0 (dB) | trials | strict PDR | decoder-aware PDR | "
            + (f"ridge K={ridge_k} PDR | " if ridge_enabled else "")
            + ("ridge + soft Hamming PDR | " if ridge_soft_enabled else "")
            + "hard oracle PDR | "
            + ("soft oracle PDR | " if ridge_soft_enabled else "")
            + "extra calls | rescues | CRC rejects | "
            "gate-only CRC false deliveries |"
        ),
        (
            "|---:|---:|---:|---:|"
            + ("---:|" if ridge_enabled else "")
            + ("---:|" if ridge_soft_enabled else "")
            + "---:|"
            + ("---:|" if ridge_soft_enabled else "")
            + "---:|---:|---:|---:|"
        ),
    ]
    for row in summary:
        values = [
            f"{float(row['esn0_db']):g}",
            str(int(row["trials"])),
            f"{float(row['strict_crc_pdr']):.3f}",
            f"{float(row['decoder_aware_crc_pdr']):.3f}",
        ]
        if ridge_enabled:
            values.append(f"{float(row['ridge_crc_pdr']):.3f}")
        if ridge_soft_enabled:
            values.append(f"{float(row['ridge_soft_crc_pdr']):.3f}")
        values.extend(
            [
                f"{float(row['oracle_crc_pdr']):.3f}",
            ]
        )
        if ridge_soft_enabled:
            values.append(f"{float(row['soft_oracle_crc_pdr']):.3f}")
        values.extend(
            [
                str(int(row["extra_decoder_calls"])),
                str(int(row["extra_candidate_crc_rescues"])),
                str(int(row["extra_candidate_crc_rejects"])),
                str(int(row["decoder_aware_crc_false_deliveries"])),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    total_extra = sum(int(row["extra_decoder_calls"]) for row in summary)
    total_rescues = sum(int(row["extra_candidate_crc_rescues"]) for row in summary)
    total_rejects = sum(int(row["extra_candidate_crc_rejects"]) for row in summary)
    total_false = sum(
        int(row["decoder_aware_crc_false_deliveries"]) for row in summary
    )
    lines.extend(
        [
            "",
            "## Decoder arbitration audit",
            "",
            f"- Extra candidates admitted beyond strict gating: **{total_extra}**.",
            f"- Extra candidates producing exact CRC-valid packets: **{total_rescues}**.",
            f"- Extra candidates rejected by PHY CRC: **{total_rejects}**.",
            f"- CRC-valid but wrong payload deliveries: **{total_false}**.",
            "",
            "The CSV files retain per-packet header/payload symbol errors and modeled decoder runtime for complexity analysis.",
            "",
            "## Remaining failure decomposition",
            "",
            "| Es/N0 (dB) | oracle itself fails | oracle succeeds, decoder-aware gate rejects | gate accepts, noisy candidate fails | decoder-aware beats oracle |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            "| {esn0_db:g} | {oracle_demod_failures} | "
            "{oracle_gap_gate_rejects} | {oracle_gap_bad_candidate_decodes} | "
            "{decoder_aware_beats_oracle} |".format(**row)
        )
    if ridge_enabled:
        ridge_delivered = sum(
            round(float(row["ridge_crc_pdr"]) * int(row["trials"]))
            for row in summary
        )
        soft_delivered = sum(
            round(float(row["decoder_aware_crc_pdr"]) * int(row["trials"]))
            for row in summary
        )
        oracle_delivered = sum(
            round(float(row["oracle_crc_pdr"]) * int(row["trials"]))
            for row in summary
        )
        ridge_false = sum(
            int(row["ridge_crc_false_deliveries"]) for row in summary
        )
        mean_attempts = float(
            np.average(
                [float(row["ridge_mean_decoder_attempts"]) for row in summary],
                weights=[int(row["trials"]) for row in summary],
            )
        )
        lines.extend(
            [
                "",
                "## Ambiguity-ridge list synchronization",
                "",
                (
                    f"- Ridge K={ridge_k}: {ridge_delivered} exact packets; "
                    f"soft gate: {soft_delivered}; oracle: {oracle_delivered}."
                ),
                (
                    f"- Operational mean decoder attempts per input trial: "
                    f"{mean_attempts:.3f}."
                ),
                f"- CRC-valid wrong deliveries: {ridge_false}.",
            ]
        )
    if ridge_soft_enabled:
        hard_delivered = sum(
            round(float(row["ridge_crc_pdr"]) * int(row["trials"]))
            for row in summary
        )
        soft_delivered = sum(
            round(float(row["ridge_soft_crc_pdr"]) * int(row["trials"]))
            for row in summary
        )
        soft_oracle_delivered = sum(
            round(float(row["soft_oracle_crc_pdr"]) * int(row["trials"]))
            for row in summary
        )
        soft_false = sum(
            int(row["ridge_soft_crc_false_deliveries"]) for row in summary
        )
        soft_mean_attempts = float(
            np.average(
                [
                    float(row["ridge_soft_mean_decoder_attempts"])
                    for row in summary
                ],
                weights=[int(row["trials"]) for row in summary],
            )
        )
        lines.extend(
            [
                "",
                "## Soft-Hamming decoder linkage",
                "",
                (
                    f"- Hard ridge: {hard_delivered} exact packets; ridge + "
                    f"soft Hamming: {soft_delivered}."
                ),
                f"- Soft-Hamming oracle sync: {soft_oracle_delivered}.",
                (
                    "- Operational mean soft decoder attempts per input "
                    f"trial: {soft_mean_attempts:.3f}."
                ),
                f"- CRC-valid wrong deliveries: {soft_false}.",
            ]
        )
    threshold_methods = [
        ("strict gate", "strict_crc_pdr"),
        ("decoder-aware gate", "decoder_aware_crc_pdr"),
    ]
    if ridge_enabled:
        threshold_methods.append((f"ridge K={ridge_k}", "ridge_crc_pdr"))
    if ridge_soft_enabled:
        threshold_methods.append(
            (f"ridge K={ridge_k} + soft Hamming", "ridge_soft_crc_pdr")
        )
    lines.extend(
        [
            "",
            "## Interpolated sensitivity thresholds",
            "",
            "Linear interpolation is applied only between adjacent measured SNR points.",
            "",
            "| receiver | Es/N0 at PDR50 | Es/N0 at PDR80 |",
            "|---|---:|---:|",
        ]
    )
    for label, key in threshold_methods:
        pdr50 = _interpolated_threshold(summary, key, 0.5)
        pdr80 = _interpolated_threshold(summary, key, 0.8)
        pdr50_text = "n/a" if pdr50 is None else f"{pdr50:.3f} dB"
        pdr80_text = "n/a" if pdr80 is None else f"{pdr80:.3f} dB"
        lines.append(f"| {label} | {pdr50_text} | {pdr80_text} |")
    lines.append("")
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate strict vs decoder-aware FrameSync with full FEC/CRC."
    )
    parser.add_argument(
        "--dataset-repo",
        type=Path,
        default=WORKSPACE_ROOT / "lora-rfsr-savaux",
    )
    parser.add_argument("--ota-root", type=Path, default=None)
    parser.add_argument("--max-packets", type=int, default=8)
    parser.add_argument("--esn0-db", default="12,13,14,15,16")
    parser.add_argument(
        "--seeds",
        default="20260821,20260822",
        help="defaults reproduce the preceding FrameSync gate audit",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--ridge-top-k",
        type=int,
        default=0,
        help="Enable CRC-aided ambiguity-ridge list synchronization.",
    )
    parser.add_argument("--ridge-sfd-peak-pool", type=int, default=32)
    parser.add_argument(
        "--ridge-soft-hamming",
        action="store_true",
        help="Also evaluate bounded soft-Hamming decoding on the same ridge list.",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="rebuild summary, plot, and report from existing packet_trials.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "decoder_aware_crc_pdr_ota_awgn_20260822",
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
    esn0_values = parse_csv_list(args.esn0_db, float)
    seeds = parse_csv_list(args.seeds, int)
    if int(args.ridge_top_k) < 0:
        raise ValueError("ridge-top-k must be non-negative")
    if int(args.ridge_sfd_peak_pool) <= 0:
        raise ValueError("ridge-sfd-peak-pool must be positive")

    if args.summarize_existing:
        rows = read_csv_rows(output_dir / "packet_trials.csv")
        clean_audits = read_csv_rows(output_dir / "clean_sync_audit.csv")
        summary = _summary_by_snr(rows, esn0_values)
        write_csv_rows(output_dir / "summary.csv", summary)
        _make_plot(output_dir, summary)
        _build_report(output_dir, clean_audits, summary)
        print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
        return 0

    metadata_paths = list(packet_metadata_paths(ota_root))[: int(args.max_packets)]
    clean_tasks = [
        {
            "dataset_repo": str(dataset_repo),
            "ota_root": str(ota_root),
            "metadata_path": str(path),
        }
        for path in metadata_paths
    ]
    clean_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [executor.submit(_clean_worker, task) for task in clean_tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            clean_results.append(future.result())
            print(f"completed clean full-CRC audits: {index}/{len(futures)}", flush=True)
    clean_results.sort(key=lambda item: int(item["audit"]["reference_id"]))
    clean_audits = [item["audit"] for item in clean_results]
    packets = [item["packet"] for item in clean_results if item["packet"] is not None]
    write_csv_rows(output_dir / "clean_sync_audit.csv", clean_audits)
    if not packets:
        raise RuntimeError("no packet passed clean full FEC/CRC calibration")

    all_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        for esn0_db in esn0_values:
            tasks = [
                {
                    "dataset_repo": str(dataset_repo),
                    "packet": packet,
                    "esn0_db": float(esn0_db),
                    "seed": int(seed),
                    "ridge_top_k": int(args.ridge_top_k),
                    "ridge_sfd_peak_pool": int(args.ridge_sfd_peak_pool),
                    "ridge_soft_hamming": bool(args.ridge_soft_hamming),
                }
                for seed in seeds
                for packet in packets
            ]
            futures = [executor.submit(_noisy_worker, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                all_rows.append(future.result())
                if index % 4 == 0 or index == len(futures):
                    print(
                        f"completed {esn0_db:g} dB full-CRC trials: "
                        f"{index}/{len(futures)}",
                        flush=True,
                    )
            all_rows.sort(
                key=lambda row: (
                    float(row["esn0_db"]),
                    int(row["seed"]),
                    int(row["reference_id"]),
                )
            )
            write_csv_rows(output_dir / "packet_trials.csv", all_rows)
            completed = tuple(
                value
                for value in esn0_values
                if any(float(row["esn0_db"]) == float(value) for row in all_rows)
            )
            write_csv_rows(
                output_dir / "summary.csv",
                _summary_by_snr(all_rows, completed),
            )
            print(f"checkpointed completed SNR: {esn0_db:g} dB", flush=True)

    summary = _summary_by_snr(all_rows, esn0_values)
    write_csv_rows(output_dir / "packet_trials.csv", all_rows)
    write_csv_rows(output_dir / "summary.csv", summary)
    _make_plot(output_dir, summary)
    _build_report(output_dir, clean_audits, summary)
    config = {
        "dataset_repo": str(dataset_repo),
        "ota_root": str(ota_root),
        "max_packets": int(args.max_packets),
        "clean_packets_admitted": len(packets),
        "esn0_db": list(esn0_values),
        "seeds": list(seeds),
        "workers": int(args.workers),
        "ridge_top_k": int(args.ridge_top_k),
        "ridge_sfd_peak_pool": int(args.ridge_sfd_peak_pool),
        "ridge_soft_hamming": bool(args.ridge_soft_hamming),
        "noise_model": "complex AWGN flat in B, added once to the complete 1 MS/s packet",
        "signal_power_calibration_payload_symbols": SIGNAL_POWER_SYMBOLS,
        "demod_tail_noise_samples": DEMOD_TAIL_SAMPLES,
        "decoder": (
            "Savaux likelihood -> bounded 16-codeword soft Hamming ML -> "
            "dewhitening -> PHY CRC"
            if args.ridge_soft_hamming
            else "Savaux hard symbols -> explicit header -> FEC -> dewhitening -> PHY CRC"
        ),
        "crc_mode": CRC_MODE,
        "ground_truth_usage": "post-CRC exact-delivery audit only",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
