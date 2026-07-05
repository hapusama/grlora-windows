#!/usr/bin/env python3
"""Evaluate island reconstruction DP against the v1 one-order DP baseline.

This runner is intentionally local to the experimental variant folder so the
new research branch can be benchmarked without changing the legacy evaluators.
It uses only packet-local evidence and ground-truth bins from the existing
header-first symbol CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[4]
GR_LORA_ROOT = WEAK_ROOT.parent
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.phase_line.savaux_stage1 import (  # noqa: E402
    SavauxStage1Config,
    branch_residual_sto_chips_from_sync_estimates,
    default_savaux_phase_path_config,
    payload_abs_indices,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction import (  # noqa: E402
    IslandReconstructionConfig,
    select_island_reconstruction_viterbi_path,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction.dual_evidence import (  # noqa: E402
    DualEvidenceFusionConfig,
    build_dual_savaux_stage1_packet_evidence,
)
from weak_decoder.phase_line.variants.v1_one_order_dp import select_phase_viterbi_path  # noqa: E402


DEFAULT_DATASETS = ("0_0_0_10_14_8", "0_0_0_10_14_16", "0_0_0_10_14_32")


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return int(default)
    return int(float(value))


def _as_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    return float(value)


def _as_float_vector(value: str | None) -> tuple[float, ...]:
    text = "" if value is None else str(value).strip()
    if not text:
        return ()
    parts = text.replace(",", " ").replace(";", " ").split()
    out: list[float] = []
    for part in parts:
        try:
            out.append(float(part))
        except ValueError:
            continue
    return tuple(out)


def _maybe_set_branch_vectors(packet: dict[str, Any], row: dict[str, str]) -> None:
    if not packet.get("branch_sfo_hat"):
        values = _as_float_vector(row.get("source_grlora_branch_sfo_hat"))
        if values:
            packet["branch_sfo_hat"] = values
    if not packet.get("branch_sfo_cum_initial"):
        values = _as_float_vector(row.get("source_grlora_branch_sfo_cum_initial"))
        if values:
            packet["branch_sfo_cum_initial"] = values


def _branch_residual_sto_from_packet(packet: dict[str, Any]) -> tuple[tuple[float, ...], ...] | None:
    sfo_hat = tuple(float(v) for v in packet.get("branch_sfo_hat", ()) or ())
    sfo_cum = tuple(float(v) for v in packet.get("branch_sfo_cum_initial", ()) or ())
    os_factor = int(packet.get("os_factor", 1))
    if len(sfo_hat) < os_factor or len(sfo_cum) < os_factor:
        return None
    payload_indexes = [int(item["payload_symbol_index"]) for item in packet.get("payload_symbols", [])]
    estimates = tuple(
        SimpleNamespace(sfo_hat=float(sfo_hat[idx]), sfo_cum_initial=float(sfo_cum[idx]))
        for idx in range(os_factor)
    )
    rows = branch_residual_sto_chips_from_sync_estimates(
        estimates,
        payload_indexes,
        os_factor=os_factor,
    )
    return rows or None


def _load_packets(symbol_csv: Path) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    with symbol_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("stage", "")) not in {"header", "payload"}:
                continue
            packet_index = _as_int(row.get("packet_index"))
            packet = grouped.setdefault(
                packet_index,
                {
                    "packet_index": packet_index,
                    "frame_index": _as_int(row.get("frame_index")),
                    "event_index": _as_int(row.get("event_index")),
                    "sf": _as_int(row.get("sf")),
                    "bw": _as_float(row.get("bw")),
                    "os_factor": _as_int(row.get("os_factor"), 1),
                    "cfo_int": _as_int(row.get("cfo_int")),
                    "cfo_frac": _as_float(row.get("cfo_frac")),
                    "header_valid": _parse_bool(str(row.get("header_valid", "0"))),
                    "payload_len": _as_int(row.get("payload_len")),
                    "cr": _as_int(row.get("payload_cr")),
                    "has_crc": _parse_bool(str(row.get("payload_has_crc", "0"))),
                    "ldro": _parse_bool(str(row.get("payload_ldro", "0"))),
                    "preamble_len": 8.0,
                    "header_symbols": [],
                    "payload_symbols": [],
                    "header_start_sample": None,
                    "branch_sfo_hat": (),
                    "branch_sfo_cum_initial": (),
                },
            )
            _maybe_set_branch_vectors(packet, row)
            stage = str(row.get("stage", ""))
            if stage == "header":
                if packet["header_start_sample"] is None:
                    packet["header_start_sample"] = _as_int(row.get("start_sample"))
                packet["header_symbols"].append(_as_int(row.get("symbol_value")))
            elif stage == "payload":
                packet["payload_symbols"].append(
                    {
                        "payload_symbol_index": _as_int(row.get("stage_symbol_index")),
                        "frame_symbol_index": _as_int(row.get("frame_symbol_index")),
                        "start_sample": _as_int(row.get("start_sample")),
                        "gt_bin": _as_int(row.get("raw_fft_bin")),
                        "sto_frac": _as_float(row.get("sto_frac")),
                        "sfo_hat": _as_float(row.get("sfo_hat")),
                        "sfo_cum_before": _as_float(row.get("sfo_cum_before")),
                        "sfo_sample_adjust_after": _as_int(row.get("sfo_sample_adjust_after")),
                    }
                )
    packets = [item for item in grouped.values() if item["payload_symbols"]]
    packets.sort(key=lambda item: int(item["packet_index"]))
    for packet in packets:
        if packet["header_start_sample"] is None:
            first_payload = packet["payload_symbols"][0]
            packet["header_start_sample"] = int(first_payload["start_sample"])
    return packets


def _dataset_paths(dataset: str) -> tuple[Path, Path]:
    iq = GR_LORA_ROOT / "data" / "USRP_IQ" / f"{dataset}.bin"
    symbols = WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / f"{dataset}_header_first_symbols.csv"
    return iq, symbols


def _snr_values(values: Sequence[float]) -> tuple[float | None, ...]:
    if not values:
        return (None,)
    return tuple(float(v) for v in values)


def _noise_samples(
    clean: np.ndarray,
    snr_db: float | None,
    seed: int,
    signal_reference_power: float | None,
) -> np.ndarray:
    if snr_db is None:
        return np.asarray(clean, dtype=np.complex64)
    signal_power = (
        float(signal_reference_power)
        if signal_reference_power is not None
        else float(np.mean(np.abs(clean).astype(np.float64) ** 2))
    )
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    rng = np.random.default_rng(int(seed))
    sigma = float(np.sqrt(noise_power / 2.0))
    noise = (
        rng.normal(0.0, sigma, clean.size).astype(np.float32)
        + 1j * rng.normal(0.0, sigma, clean.size).astype(np.float32)
    ).astype(np.complex64)
    return (np.asarray(clean, dtype=np.complex64) + noise).astype(np.complex64)


def _err_count(selected: Sequence[int], gt_bins: Sequence[int]) -> tuple[int, int]:
    count = min(len(selected), len(gt_bins))
    errors = sum(int(selected[idx]) != int(gt_bins[idx]) for idx in range(count))
    return int(errors), int(count)


def _compare_paths(
    candidate: Sequence[int],
    reference: Sequence[int],
    gt_bins: Sequence[int],
) -> tuple[int, int, int]:
    count = min(len(candidate), len(reference), len(gt_bins))
    rescue = 0
    break_count = 0
    changes = 0
    for idx in range(count):
        cand = int(candidate[idx])
        ref = int(reference[idx])
        gt = int(gt_bins[idx])
        if cand != ref:
            changes += 1
        if ref != gt and cand == gt:
            rescue += 1
        if ref == gt and cand != gt:
            break_count += 1
    return int(rescue), int(break_count), int(changes)


def _evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    path_config,
    island_config: IslandReconstructionConfig,
    fusion_config: DualEvidenceFusionConfig,
    use_aux_evidence: bool = False,
) -> dict[str, Any]:
    payload = list(packet["payload_symbols"])
    start_samples = [int(item["start_sample"]) for item in payload]
    gt_bins = [int(item["gt_bin"]) for item in payload]
    residual_sto_chips = [float(item.get("sfo_cum_before", 0.0)) for item in payload]
    branch_residual_sto_chips = _branch_residual_sto_from_packet(packet)
    abs_indices = payload_abs_indices(
        [int(item["payload_symbol_index"]) for item in payload],
        preamble_len=float(packet.get("preamble_len", 8.0)),
    )
    dual = build_dual_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        abs_indices=abs_indices,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=int(packet["header_start_sample"]),
        residual_sto_chips=residual_sto_chips,
        stage1_config=SavauxStage1Config(retain_dechirped_symbols=True),
        fusion_config=fusion_config,
    )
    stage1 = dual.old_stage1
    hard_bins = tuple(int(symbol.top1_bin) for symbol in stage1.symbols)
    v1 = select_phase_viterbi_path(
        center_spectra=stage1.center_spectra,
        evidence_powers=stage1.evidence_powers,
        abs_indices=stage1.abs_indices,
        config=path_config,
        offset_coherences=stage1.branch_phase_agreements,
    )
    corrected_v1 = select_phase_viterbi_path(
        center_spectra=dual.corrected_stage1.center_spectra,
        evidence_powers=dual.corrected_stage1.evidence_powers,
        abs_indices=dual.corrected_stage1.abs_indices,
        config=path_config,
        offset_coherences=dual.corrected_stage1.branch_phase_agreements,
    )
    fusion_v1 = select_phase_viterbi_path(
        center_spectra=dual.center_spectra,
        evidence_powers=dual.evidence_powers,
        abs_indices=dual.abs_indices,
        config=path_config,
        offset_coherences=dual.branch_phase_agreements,
    )
    island = select_island_reconstruction_viterbi_path(
        center_spectra=stage1.center_spectra,
        evidence_powers=stage1.evidence_powers,
        abs_indices=stage1.abs_indices,
        config=path_config,
        reconstruction_config=island_config,
        offset_coherences=stage1.branch_phase_agreements,
        branch_spectra=stage1.branch_spectra,
        dechirped_symbols=stage1.dechirped_symbols,
        os_factor=int(packet["os_factor"]),
        residual_sto_chips=residual_sto_chips,
        branch_residual_sto_chips=branch_residual_sto_chips,
    )
    fusion_island = select_island_reconstruction_viterbi_path(
        center_spectra=dual.center_spectra,
        evidence_powers=dual.evidence_powers,
        abs_indices=dual.abs_indices,
        config=path_config,
        reconstruction_config=island_config,
        offset_coherences=dual.branch_phase_agreements,
        branch_spectra=dual.branch_spectra,
        dechirped_symbols=dual.dechirped_symbols,
        os_factor=int(packet["os_factor"]),
        residual_sto_chips=residual_sto_chips,
        branch_residual_sto_chips=branch_residual_sto_chips,
        auxiliary_evidence_powers=(
            tuple(
                (
                    np.asarray(dual.old_stage1.evidence_powers[idx], dtype=np.float64),
                    np.asarray(dual.corrected_stage1.evidence_powers[idx], dtype=np.float64),
                )
                for idx in range(min(len(dual.old_stage1.evidence_powers), len(dual.corrected_stage1.evidence_powers)))
            )
            if bool(use_aux_evidence)
            else None
        ),
    )

    hard_err, compared = _err_count(hard_bins, gt_bins)
    v1_err, _ = _err_count(v1.selected_raw_bins, gt_bins)
    corrected_v1_err, _ = _err_count(corrected_v1.selected_raw_bins, gt_bins)
    fusion_v1_err, _ = _err_count(fusion_v1.selected_raw_bins, gt_bins)
    island_err, _ = _err_count(island.selected_raw_bins, gt_bins)
    fusion_island_err, _ = _err_count(fusion_island.selected_raw_bins, gt_bins)
    rescue_v1, break_v1, changes_v1 = _compare_paths(island.selected_raw_bins, v1.selected_raw_bins, gt_bins)
    rescue_corrected_v1, break_corrected_v1, changes_corrected_v1 = _compare_paths(
        corrected_v1.selected_raw_bins, v1.selected_raw_bins, gt_bins
    )
    rescue_fusion_v1, break_fusion_v1, changes_fusion_v1 = _compare_paths(
        fusion_v1.selected_raw_bins, v1.selected_raw_bins, gt_bins
    )
    rescue_fusion_island_v1, break_fusion_island_v1, changes_fusion_island_v1 = _compare_paths(
        fusion_island.selected_raw_bins, v1.selected_raw_bins, gt_bins
    )
    rescue_fusion_island_fusion, break_fusion_island_fusion, changes_fusion_island_fusion = _compare_paths(
        fusion_island.selected_raw_bins, fusion_v1.selected_raw_bins, gt_bins
    )
    rescue_hard, break_hard, changes_hard = _compare_paths(island.selected_raw_bins, hard_bins, gt_bins)
    locked_e3 = 0
    for idx, locked in enumerate(island.locked_mask):
        if not bool(locked) or idx >= len(hard_bins) or idx >= len(island.selected_raw_bins) or idx >= len(gt_bins):
            continue
        if int(hard_bins[idx]) == int(gt_bins[idx]) and int(island.selected_raw_bins[idx]) != int(hard_bins[idx]):
            locked_e3 += 1
    fusion_locked_e3 = 0
    for idx, locked in enumerate(fusion_island.locked_mask):
        if (
            not bool(locked)
            or idx >= len(hard_bins)
            or idx >= len(fusion_island.selected_raw_bins)
            or idx >= len(gt_bins)
        ):
            continue
        if int(hard_bins[idx]) == int(gt_bins[idx]) and int(fusion_island.selected_raw_bins[idx]) != int(hard_bins[idx]):
            fusion_locked_e3 += 1
    return {
        "packet_index": int(packet["packet_index"]),
        "symbol_count": int(compared),
        "hard_err": int(hard_err),
        "v1_err": int(v1_err),
        "corrected_v1_err": int(corrected_v1_err),
        "fusion_v1_err": int(fusion_v1_err),
        "island_err": int(island_err),
        "fusion_island_err": int(fusion_island_err),
        "hard_ser": float(hard_err / max(1, compared)),
        "v1_ser": float(v1_err / max(1, compared)),
        "corrected_v1_ser": float(corrected_v1_err / max(1, compared)),
        "fusion_v1_ser": float(fusion_v1_err / max(1, compared)),
        "island_ser": float(island_err / max(1, compared)),
        "fusion_island_ser": float(fusion_island_err / max(1, compared)),
        "corrected_v1_gain_vs_v1": float((v1_err - corrected_v1_err) / max(1, compared)),
        "fusion_v1_gain_vs_v1": float((v1_err - fusion_v1_err) / max(1, compared)),
        "island_gain_vs_v1": float((v1_err - island_err) / max(1, compared)),
        "fusion_island_gain_vs_v1": float((v1_err - fusion_island_err) / max(1, compared)),
        "fusion_island_gain_vs_fusion_v1": float((fusion_v1_err - fusion_island_err) / max(1, compared)),
        "island_gain_vs_hard": float((hard_err - island_err) / max(1, compared)),
        "fusion_island_gain_vs_hard": float((hard_err - fusion_island_err) / max(1, compared)),
        "rescue_corrected_v1_vs_v1": int(rescue_corrected_v1),
        "break_corrected_v1_vs_v1": int(break_corrected_v1),
        "changes_corrected_v1_vs_v1": int(changes_corrected_v1),
        "rescue_fusion_v1_vs_v1": int(rescue_fusion_v1),
        "break_fusion_v1_vs_v1": int(break_fusion_v1),
        "changes_fusion_v1_vs_v1": int(changes_fusion_v1),
        "rescue_vs_v1": int(rescue_v1),
        "break_vs_v1": int(break_v1),
        "changes_vs_v1": int(changes_v1),
        "rescue_fusion_island_vs_v1": int(rescue_fusion_island_v1),
        "break_fusion_island_vs_v1": int(break_fusion_island_v1),
        "changes_fusion_island_vs_v1": int(changes_fusion_island_v1),
        "rescue_fusion_island_vs_fusion_v1": int(rescue_fusion_island_fusion),
        "break_fusion_island_vs_fusion_v1": int(break_fusion_island_fusion),
        "changes_fusion_island_vs_fusion_v1": int(changes_fusion_island_fusion),
        "rescue_vs_hard": int(rescue_hard),
        "break_vs_hard": int(break_hard),
        "changes_vs_hard": int(changes_hard),
        "locked_e3_count": int(locked_e3),
        "fusion_locked_e3_count": int(fusion_locked_e3),
        "island_locked_count": int(island.locked_count),
        "island_uncertain_count": int(island.uncertain_count),
        "island_beam_final_size": int(island.beam_final_size),
        "fusion_island_locked_count": int(fusion_island.locked_count),
        "fusion_island_uncertain_count": int(fusion_island.uncertain_count),
        "fusion_island_beam_final_size": int(fusion_island.beam_final_size),
        "v1_trajectory_score": float(v1.trajectory_score),
        "v1_mean_amp_score": float(v1.mean_amp_score),
        "fusion_v1_trajectory_score": float(fusion_v1.trajectory_score),
        "fusion_v1_mean_amp_score": float(fusion_v1.mean_amp_score),
        "island_trajectory_score": float(island.trajectory_score),
        "island_mean_amp_score": float(island.mean_amp_score),
        "island_error": str(island.error),
        "fusion_island_trajectory_score": float(fusion_island.trajectory_score),
        "fusion_island_mean_amp_score": float(fusion_island.mean_amp_score),
        "fusion_island_error": str(fusion_island.error),
    }


def _sum(rows: Sequence[dict[str, Any]], key: str) -> int:
    return int(sum(int(row.get(key, 0)) for row in rows))


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else 0.0


def _summary(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float | None, seed: int) -> dict[str, Any]:
    symbols = _sum(rows, "symbol_count")
    hard_err = _sum(rows, "hard_err")
    v1_err = _sum(rows, "v1_err")
    corrected_v1_err = _sum(rows, "corrected_v1_err")
    fusion_v1_err = _sum(rows, "fusion_v1_err")
    island_err = _sum(rows, "island_err")
    fusion_island_err = _sum(rows, "fusion_island_err")
    return {
        "dataset": dataset,
        "snr_db": "" if snr_db is None else float(snr_db),
        "seed": int(seed),
        "packet_count": int(len(rows)),
        "symbol_count": int(symbols),
        "hard_ser": float(hard_err / max(1, symbols)),
        "v1_ser": float(v1_err / max(1, symbols)),
        "corrected_v1_ser": float(corrected_v1_err / max(1, symbols)),
        "fusion_v1_ser": float(fusion_v1_err / max(1, symbols)),
        "island_ser": float(island_err / max(1, symbols)),
        "fusion_island_ser": float(fusion_island_err / max(1, symbols)),
        "corrected_v1_gain_vs_v1": float((v1_err - corrected_v1_err) / max(1, symbols)),
        "fusion_v1_gain_vs_v1": float((v1_err - fusion_v1_err) / max(1, symbols)),
        "island_gain_vs_v1": float((v1_err - island_err) / max(1, symbols)),
        "fusion_island_gain_vs_v1": float((v1_err - fusion_island_err) / max(1, symbols)),
        "fusion_island_gain_vs_fusion_v1": float((fusion_v1_err - fusion_island_err) / max(1, symbols)),
        "island_gain_vs_hard": float((hard_err - island_err) / max(1, symbols)),
        "fusion_island_gain_vs_hard": float((hard_err - fusion_island_err) / max(1, symbols)),
        "rescue_corrected_v1_vs_v1": _sum(rows, "rescue_corrected_v1_vs_v1"),
        "break_corrected_v1_vs_v1": _sum(rows, "break_corrected_v1_vs_v1"),
        "rescue_fusion_v1_vs_v1": _sum(rows, "rescue_fusion_v1_vs_v1"),
        "break_fusion_v1_vs_v1": _sum(rows, "break_fusion_v1_vs_v1"),
        "rescue_vs_v1": _sum(rows, "rescue_vs_v1"),
        "break_vs_v1": _sum(rows, "break_vs_v1"),
        "rescue_fusion_island_vs_v1": _sum(rows, "rescue_fusion_island_vs_v1"),
        "break_fusion_island_vs_v1": _sum(rows, "break_fusion_island_vs_v1"),
        "rescue_fusion_island_vs_fusion_v1": _sum(rows, "rescue_fusion_island_vs_fusion_v1"),
        "break_fusion_island_vs_fusion_v1": _sum(rows, "break_fusion_island_vs_fusion_v1"),
        "rescue_vs_hard": _sum(rows, "rescue_vs_hard"),
        "break_vs_hard": _sum(rows, "break_vs_hard"),
        "locked_e3_count": _sum(rows, "locked_e3_count"),
        "fusion_locked_e3_count": _sum(rows, "fusion_locked_e3_count"),
        "mean_island_locked_count": _mean(rows, "island_locked_count"),
        "mean_island_uncertain_count": _mean(rows, "island_uncertain_count"),
        "mean_island_beam_final_size": _mean(rows, "island_beam_final_size"),
        "mean_fusion_island_locked_count": _mean(rows, "fusion_island_locked_count"),
        "mean_fusion_island_uncertain_count": _mean(rows, "fusion_island_uncertain_count"),
        "mean_fusion_island_beam_final_size": _mean(rows, "fusion_island_beam_final_size"),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snrs", nargs="*", type=float, default=[-25.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-packets", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=THIS_FILE.parent / "_eval" / "island_dp_compare")
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--v1-top-l", type=int, default=16)
    parser.add_argument("--island-top-l", type=int, default=40)
    parser.add_argument("--anchor-margin-db", type=float, default=1.5)
    parser.add_argument("--anchor-peak-to-median-db", type=float, default=7.0)
    parser.add_argument("--anchor-min-coherence", type=float, default=0.88)
    parser.add_argument("--branch-top-k", type=int, default=0)
    parser.add_argument("--energy-weight", type=float, default=0.24)
    parser.add_argument("--coherence-weight", type=float, default=0.06)
    parser.add_argument("--reconstruction-weight", type=float, default=0.58)
    parser.add_argument("--phase-profile-weight", type=float, default=0.04)
    parser.add_argument("--branch-profile-weight", type=float, default=0.08)
    parser.add_argument("--transition-weight", type=float, default=0.45)
    parser.add_argument("--boundary-weight", type=float, default=1.10)
    parser.add_argument("--branch-transition-weight", type=float, default=1.20)
    parser.add_argument("--baseline-bonus", type=float, default=0.0)
    parser.add_argument("--island-accept-margin", type=float, default=0.0)
    parser.add_argument("--third-bin-penalty", type=float, default=0.0)
    parser.add_argument("--allow-third-bin-when-baseline-differs", action="store_true")
    parser.add_argument("--return-to-hard-min-margin-db", type=float, default=0.80)
    parser.add_argument("--auxiliary-top-l", type=int, default=0)
    parser.add_argument("--auxiliary-extra-k", type=int, default=0)
    parser.add_argument("--auxiliary-energy-weight", type=float, default=0.0)
    parser.add_argument("--use-aux-evidence", action="store_true")
    parser.add_argument(
        "--fusion-mode",
        choices=["sum_norm", "product_norm", "max_norm", "old", "corrected"],
        default="product_norm",
    )
    parser.add_argument("--fusion-corrected-weight", type=float, default=0.50)
    parser.add_argument("--fusion-stage1-top-k", type=int, default=40)
    parser.add_argument("--fusion-sto-sign", choices=["plus", "minus"], default="plus")
    parser.add_argument("--fusion-residual-preselect-factor", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path_config = default_savaux_phase_path_config(top_l=int(args.v1_top_l))
    island_config = IslandReconstructionConfig(
        top_l=int(args.island_top_l),
        anchor_margin_db=float(args.anchor_margin_db),
        anchor_peak_to_median_db=float(args.anchor_peak_to_median_db),
        anchor_min_coherence=float(args.anchor_min_coherence),
        branch_top_k=int(args.branch_top_k),
        energy_weight=float(args.energy_weight),
        coherence_weight=float(args.coherence_weight),
        reconstruction_weight=float(args.reconstruction_weight),
        phase_profile_weight=float(args.phase_profile_weight),
        branch_profile_weight=float(args.branch_profile_weight),
        transition_weight=float(args.transition_weight),
        boundary_weight=float(args.boundary_weight),
        branch_transition_weight=float(args.branch_transition_weight),
        baseline_bonus=float(args.baseline_bonus),
        island_accept_margin=float(args.island_accept_margin),
        third_bin_penalty=float(args.third_bin_penalty),
        allow_third_bin_when_baseline_differs=bool(args.allow_third_bin_when_baseline_differs),
        return_to_hard_min_margin_db=float(args.return_to_hard_min_margin_db),
        auxiliary_top_l=int(args.auxiliary_top_l),
        auxiliary_extra_k=int(args.auxiliary_extra_k),
        auxiliary_energy_weight=float(args.auxiliary_energy_weight),
    )
    fusion_config = DualEvidenceFusionConfig(
        mode=str(args.fusion_mode),
        corrected_weight=float(args.fusion_corrected_weight),
        stage1_top_k=int(args.fusion_stage1_top_k),
        residual_sto_phase_sign=str(args.fusion_sto_sign),
        residual_sto_preselect_factor=int(args.fusion_residual_preselect_factor),
        retain_dechirped_symbols=True,
    )
    out_dir = Path(args.output_dir).resolve()
    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = _dataset_paths(str(dataset))
        if not iq_path.exists():
            raise FileNotFoundError(iq_path)
        if not symbol_path.exists():
            raise FileNotFoundError(symbol_path)
        clean = np.fromfile(iq_path, dtype=np.complex64)
        packets = _load_packets(symbol_path)
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
        for seed in args.seeds:
            for snr_db in _snr_values(args.snrs):
                samples = _noise_samples(clean, snr_db, int(seed), args.signal_reference_power)
                rows: list[dict[str, Any]] = []
                for packet in packets:
                    row = _evaluate_packet(
                        samples,
                        packet,
                        path_config,
                        island_config,
                        fusion_config,
                        use_aux_evidence=bool(args.use_aux_evidence),
                    )
                    row.update(
                        {
                            "dataset": str(dataset),
                            "snr_db": "" if snr_db is None else float(snr_db),
                            "seed": int(seed),
                        }
                    )
                    rows.append(row)
                    packet_rows.append(row)
                summary = _summary(rows, str(dataset), snr_db, int(seed))
                summary_rows.append(summary)
                print(
                    f"{dataset} snr={snr_db} seed={seed}: "
                    f"v1_ser={summary['v1_ser']:.4f} "
                    f"fusion_ser={summary['fusion_v1_ser']:.4f} "
                    f"island_ser={summary['island_ser']:.4f} "
                    f"fusion_island_ser={summary['fusion_island_ser']:.4f} "
                    f"fusion_gain={summary['fusion_v1_gain_vs_v1']:.4f} "
                    f"fusion_island_gain={summary['fusion_island_gain_vs_v1']:.4f} "
                    f"rescue={summary['rescue_fusion_island_vs_v1']} "
                    f"break={summary['break_fusion_island_vs_v1']} "
                    f"locked_e3={summary['fusion_locked_e3_count']}",
                    flush=True,
                )
    _write_csv(out_dir / "packet_metrics.csv", packet_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
