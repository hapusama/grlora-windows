#!/usr/bin/env python3
"""Evaluate Savaux Stage-1 evidence plus phase-line FFT-bin selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


THIS = Path(__file__).resolve()
PHASE_LINE_DIR = THIS.parents[1]
WEAK_ROOT = THIS.parents[3]
EXPERIMENT_DIR = WEAK_ROOT / "scripts" / "experiments"
PHASE_EXPERIMENT_DIR = EXPERIMENT_DIR / "phase_line"
for path in (str(WEAK_ROOT), str(EXPERIMENT_DIR), str(PHASE_EXPERIMENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_phase_line_threshold_sweep import _dataset_paths, _load_metadata  # noqa: E402
from run_symbol_phase_two_stage import (  # noqa: E402
    _argmax_bins,
    _decode_selected,
    _extract_payload_spectra_with_coherence,
    _packet_phase_line,
    _ser,
)
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)
from weak_decoder.phase_line import (  # noqa: E402
    PhasePathSelectorConfig,
    SavauxPhaseGuardConfig,
    SavauxStage1Config,
    build_savaux_stage1_packet_evidence,
    default_savaux_phase_path_config,
    evaluate_savaux_phase_guard,
    payload_abs_indices,
    select_phase_viterbi_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Savaux Stage-1 + phase-line selector.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--snrs", nargs="+", type=float, default=[-24.0, -25.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PHASE_LINE_DIR / "_eval" / "savaux_stage1_selector")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol", "none"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--top-l", type=int, default=16)
    parser.add_argument("--stage1-top-k", type=int, default=24)
    parser.add_argument("--branch-agreement-power", type=float, default=0.0)
    parser.add_argument("--disable-guard", action="store_true")
    parser.add_argument("--guard-max-changed-symbols", type=int, default=2)
    parser.add_argument("--guard-min-rmse-gain-pi", type=float, default=0.0)
    parser.add_argument("--guard-min-mean-phase-score", type=float, default=0.990)
    parser.add_argument("--guard-min-changed-mean-energy-drop-db", type=float, default=-2.0)
    parser.add_argument("--guard-line-trim-frac", type=float, default=0.20)
    parser.add_argument("--independent-noise", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _avg(rows: Sequence[dict[str, Any]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else 0.0


def _candidate_recall(result: Any, gt_bins: Sequence[int]) -> tuple[int, int]:
    hits = 0
    total = 0
    for idx, ev in enumerate(result.evidences):
        if idx >= len(gt_bins):
            continue
        gt = int(gt_bins[idx])
        if gt < 0:
            continue
        total += 1
        hits += int(gt in {int(v) for v in ev.top_bins})
    return int(hits), int(total)


def _hard_anchor_errors(result: Any, gt_bins: Sequence[int]) -> tuple[int, int]:
    errors = 0
    total = 0
    for idx, locked in enumerate(result.locked_mask):
        if idx >= len(gt_bins) or not bool(locked):
            continue
        gt = int(gt_bins[idx])
        if gt < 0:
            continue
        total += 1
        errors += int(int(result.selected_raw_bins[idx]) != gt)
    return int(errors), int(total)


def _payload_starts_and_abs(packet: dict[str, Any], preamble_len: float) -> tuple[list[int], tuple[float, ...], list[int]]:
    starts: list[int] = []
    payload_indexes: list[int] = []
    gt_bins: list[int] = []
    for fallback_idx, symbol in enumerate(packet["payload_symbols"]):
        payload_idx = int(symbol.get("payload_symbol_index", fallback_idx))
        starts.append(int(symbol["start_sample"]))
        payload_indexes.append(payload_idx)
        gt_bins.append(int(symbol.get("gt_bin", -1)))
    return starts, payload_abs_indices(payload_indexes, preamble_len=float(packet.get("preamble_len", preamble_len))), gt_bins


def _payload_reference_power(samples: np.ndarray, packets: dict[int, dict[str, Any]]) -> tuple[float, int]:
    chunks: list[np.ndarray] = []
    total = 0
    for packet in packets.values():
        sf = int(packet["sf"])
        os_factor = int(packet["os_factor"])
        length = (1 << sf) * os_factor
        for symbol in packet["payload_symbols"]:
            start = int(symbol["start_sample"])
            stop = start + length
            if start < 0 or stop > samples.size:
                continue
            chunk = np.asarray(samples[start:stop], dtype=np.complex64)
            chunks.append(chunk)
            total += int(chunk.size)
    if not chunks:
        return float(np.mean(np.abs(samples).astype(np.float64) ** 2)), int(samples.size)
    power_sum = 0.0
    for chunk in chunks:
        power_sum += float(np.sum(np.abs(chunk).astype(np.float64) ** 2))
    return float(power_sum / max(1, total)), int(total)


def _savaux_hard_bins(
    samples: np.ndarray,
    packet: dict[str, Any],
    starts: Sequence[int],
    stage1_config: SavauxStage1Config,
) -> tuple[int, ...]:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    origin_shift = os_factor // 2 if stage1_config.origin_shift_samples is None else int(stage1_config.origin_shift_samples)
    out: list[int] = []
    for start in starts:
        spectrum, _branches, _phase = paper_oversampled_spectrum(
            samples=samples,
            start_sample=int(start) + int(origin_shift),
            sf=sf,
            os_factor=os_factor,
            cfo_int=int(packet["cfo_int"]),
            cfo_frac=float(packet["cfo_frac"]),
            header_start_sample=int(packet["header_start_sample"]) + int(origin_shift),
            cfo_correction_mode=str(stage1_config.cfo_correction_mode),
        )
        out.append(int(np.argmax(np.abs(spectrum).astype(np.float64) ** 2)))
    return tuple(out)


def _evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    selector_config: PhasePathSelectorConfig,
    stage1_config: SavauxStage1Config,
) -> dict[str, Any]:
    starts, abs_indices, gt_bins = _payload_starts_and_abs(packet, float(args.preamble_len))
    stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=starts,
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        abs_indices=abs_indices,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=int(packet["header_start_sample"]),
        config=stage1_config,
    )
    result = select_phase_viterbi_path(
        center_spectra=stage1.center_spectra,
        evidence_powers=stage1.evidence_powers,
        abs_indices=stage1.abs_indices,
        config=selector_config,
        offset_coherences=stage1.branch_phase_agreements,
    )

    legacy_args = SimpleNamespace(
        cfo_correction_mode=str(args.cfo_correction_mode),
        preamble_len=float(args.preamble_len),
        ldro_mode=int(args.ldro_mode),
        crc_mode=str(args.crc_mode),
    )
    center_spectra, multi_spectra, _legacy_abs, _legacy_gt, coherences = _extract_payload_spectra_with_coherence(
        samples, packet, legacy_args
    )
    legacy_result = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=[np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra],
        abs_indices=_legacy_abs,
        config=PhasePathSelectorConfig(),
        offset_coherences=coherences,
    )

    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    savaux_bins = _savaux_hard_bins(samples, packet, starts, stage1_config)
    center_bins = _argmax_bins(center_spectra)
    multi_bins = _argmax_bins(multi_spectra)
    selected_bins = tuple(int(v) for v in result.selected_raw_bins)
    legacy_bins = tuple(int(v) for v in legacy_result.selected_raw_bins)
    guard = evaluate_savaux_phase_guard(
        stage1=stage1,
        phase_result=result,
        hard_bins=savaux_bins,
        config=SavauxPhaseGuardConfig(
            enabled=not bool(args.disable_guard),
            max_changed_symbols=int(args.guard_max_changed_symbols),
            min_eval_rmse_gain_pi=float(args.guard_min_rmse_gain_pi),
            min_mean_phase_score=float(args.guard_min_mean_phase_score),
            min_changed_mean_energy_drop_db=float(args.guard_min_changed_mean_energy_drop_db),
            line_trim_frac=float(args.guard_line_trim_frac),
        ),
    )
    guarded_bins = tuple(int(v) for v in guard.guarded_bins)

    center_raw_ser, center_symbol_ser, compared = _ser(center_bins, gt_bins, sf=sf, ldro=ldro)
    multi_raw_ser, multi_symbol_ser, _ = _ser(multi_bins, gt_bins, sf=sf, ldro=ldro)
    savaux_raw_ser, savaux_symbol_ser, _ = _ser(savaux_bins, gt_bins, sf=sf, ldro=ldro)
    selected_raw_ser, selected_symbol_ser, _ = _ser(selected_bins, gt_bins, sf=sf, ldro=ldro)
    guarded_raw_ser, guarded_symbol_ser, _ = _ser(guarded_bins, gt_bins, sf=sf, ldro=ldro)
    legacy_raw_ser, legacy_symbol_ser, _ = _ser(legacy_bins, gt_bins, sf=sf, ldro=ldro)
    candidate_hits, candidate_total = _candidate_recall(result, gt_bins)
    legacy_candidate_hits, legacy_candidate_total = _candidate_recall(legacy_result, gt_bins)
    hard_errors, hard_total = _hard_anchor_errors(result, gt_bins)
    decoded = _decode_selected(packet, selected_bins, args)
    guarded_decoded = _decode_selected(packet, guarded_bins, args)

    agreement_vals: list[float] = []
    for idx, raw_bin in enumerate(selected_bins):
        if idx >= len(stage1.branch_phase_agreements):
            continue
        b = int(raw_bin)
        agreement = stage1.branch_phase_agreements[idx]
        if 0 <= b < agreement.size:
            agreement_vals.append(float(agreement[b]))

    row: dict[str, Any] = {
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "payload_len": int(packet["payload_len"]),
        "cr": int(packet["cr"]),
        "has_crc": int(bool(packet["has_crc"])),
        "ldro": int(ldro),
        "symbol_count": int(len(selected_bins)),
        "gt_compared_symbols": int(compared),
        "center_argmax_raw_ser": float(center_raw_ser),
        "center_argmax_symbol_ser": float(center_symbol_ser),
        "multi_offset_argmax_raw_ser": float(multi_raw_ser),
        "multi_offset_argmax_symbol_ser": float(multi_symbol_ser),
        "legacy_phase_line_raw_ser": float(legacy_raw_ser),
        "legacy_phase_line_symbol_ser": float(legacy_symbol_ser),
        "savaux_hard_raw_ser": float(savaux_raw_ser),
        "savaux_hard_symbol_ser": float(savaux_symbol_ser),
        "savaux_phase_raw_ser": float(selected_raw_ser),
        "savaux_phase_symbol_ser": float(selected_symbol_ser),
        "savaux_guarded_raw_ser": float(guarded_raw_ser),
        "savaux_guarded_symbol_ser": float(guarded_symbol_ser),
        "savaux_phase_gain_vs_savaux": float(savaux_symbol_ser - selected_symbol_ser),
        "savaux_guarded_gain_vs_savaux": float(savaux_symbol_ser - guarded_symbol_ser),
        "savaux_guarded_gain_vs_phase": float(selected_symbol_ser - guarded_symbol_ser),
        "savaux_phase_gain_vs_legacy": float(legacy_symbol_ser - selected_symbol_ser),
        "savaux_guarded_gain_vs_legacy": float(legacy_symbol_ser - guarded_symbol_ser),
        "stage1_candidate_hits": int(candidate_hits),
        "stage1_candidate_total": int(candidate_total),
        "stage1_candidate_recall": float(candidate_hits / max(1, candidate_total)),
        "legacy_candidate_hits": int(legacy_candidate_hits),
        "legacy_candidate_total": int(legacy_candidate_total),
        "legacy_candidate_recall": float(legacy_candidate_hits / max(1, legacy_candidate_total)),
        "viterbi_candidate_recall": float(candidate_hits / max(1, candidate_total)),
        "hard_anchor_count": int(hard_total),
        "hard_anchor_error_count": int(hard_errors),
        "hard_anchor_error_rate": float(hard_errors / max(1, hard_total)),
        "mean_selected_branch_agreement": float(np.mean(agreement_vals)) if agreement_vals else 0.0,
        "trajectory_score": float(result.trajectory_score),
        "mean_phase_score": float(result.mean_phase_score),
        "mean_amp_score": float(result.mean_amp_score),
        "beam_final_size": int(result.beam_final_size),
        "guard_accept_phase": int(bool(guard.accept_phase)),
        "guard_reason": str(guard.reason),
        "guard_change_count": int(guard.change_count),
        "guard_eval_rmse_gain_pi": float(guard.eval_rmse_gain_pi),
        "guard_hard_eval_rmse_pi": float(guard.hard_eval_rmse_pi),
        "guard_phase_eval_rmse_pi": float(guard.phase_eval_rmse_pi),
        "guard_changed_mean_energy_drop_db": float(guard.changed_mean_energy_drop_db),
        "guard_mean_phase_score": float(guard.mean_phase_score),
        "error": result.error,
    }
    row.update({f"savaux_phase_{key}": value for key, value in decoded.items()})
    row.update({f"savaux_guarded_{key}": value for key, value in guarded_decoded.items()})
    return row


def _summarize(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "packet_count": int(len(rows)),
        "center_argmax_symbol_ser": _avg(rows, "center_argmax_symbol_ser"),
        "multi_offset_argmax_symbol_ser": _avg(rows, "multi_offset_argmax_symbol_ser"),
        "legacy_phase_line_symbol_ser": _avg(rows, "legacy_phase_line_symbol_ser"),
        "savaux_hard_symbol_ser": _avg(rows, "savaux_hard_symbol_ser"),
        "savaux_phase_symbol_ser": _avg(rows, "savaux_phase_symbol_ser"),
        "savaux_guarded_symbol_ser": _avg(rows, "savaux_guarded_symbol_ser"),
        "savaux_phase_gain_vs_savaux": _avg(rows, "savaux_phase_gain_vs_savaux"),
        "savaux_guarded_gain_vs_savaux": _avg(rows, "savaux_guarded_gain_vs_savaux"),
        "savaux_phase_gain_vs_legacy": _avg(rows, "savaux_phase_gain_vs_legacy"),
        "savaux_guarded_gain_vs_legacy": _avg(rows, "savaux_guarded_gain_vs_legacy"),
        "stage1_candidate_recall": _avg(rows, "stage1_candidate_recall"),
        "legacy_candidate_recall": _avg(rows, "legacy_candidate_recall"),
        "viterbi_candidate_recall": _avg(rows, "viterbi_candidate_recall"),
        "mean_hard_anchor_count": _avg(rows, "hard_anchor_count"),
        "mean_hard_anchor_error_rate": _avg(rows, "hard_anchor_error_rate"),
        "savaux_phase_crc_valid_rate": _avg(rows, "savaux_phase_crc_valid"),
        "savaux_guarded_crc_valid_rate": _avg(rows, "savaux_guarded_crc_valid"),
        "guard_accept_phase_rate": _avg(rows, "guard_accept_phase"),
        "mean_selected_branch_agreement": _avg(rows, "mean_selected_branch_agreement"),
    }


def _dataset_seed(args: argparse.Namespace, metadata: dict[str, Any], dataset_index: int) -> list[int]:
    if args.seeds:
        return [int(v) for v in args.seeds]
    return [int(metadata.get("seed", 42)) + 100000 * int(dataset_index)]


def _run_dataset(args: argparse.Namespace, dataset: str, dataset_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths = _dataset_paths(str(dataset))
    metadata = _load_metadata(paths, str(dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    packets = load_packets(paths["symbols"], args.packet)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {paths['iq']}")
    if not packets:
        raise ValueError(f"no packets loaded: {paths['symbols']}")

    if args.signal_reference_power is not None:
        signal_power = float(args.signal_reference_power)
    elif "signal_reference_power" in metadata:
        signal_power = float(metadata["signal_reference_power"])
    else:
        signal_power, ref_count = _payload_reference_power(samples, packets)
        print(
            f"{dataset}: metadata missing; using payload reference power "
            f"{signal_power:.6g} from {ref_count} samples",
            flush=True,
        )

    stage1_config = SavauxStage1Config(
        cfo_correction_mode=str(args.cfo_correction_mode),
        top_k=int(args.stage1_top_k),
        branch_agreement_power=float(args.branch_agreement_power),
    )
    selector_config = default_savaux_phase_path_config(top_l=int(args.top_l))

    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    seeds = _dataset_seed(args, metadata, dataset_index)
    for seed in seeds:
        unit_noise: np.ndarray | None = None
        if not bool(args.independent_noise):
            rng = np.random.default_rng(int(seed))
            unit_noise = (
                rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
                + 1j * rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
            ).astype(np.complex64)

        for step_index, snr_db in enumerate(args.snrs):
            noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
            sigma = math.sqrt(float(noise_power) / 2.0)
            if unit_noise is None:
                rng = np.random.default_rng(int(seed) + step_index)
                noise = (
                    rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                    + 1j * rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                ).astype(np.complex64)
                noisy = (samples + noise).astype(np.complex64, copy=False)
            else:
                noisy = (samples + sigma * unit_noise).astype(np.complex64, copy=False)

            rows: list[dict[str, Any]] = []
            for packet_index in sorted(packets):
                row = _evaluate_packet(noisy, packets[packet_index], args, selector_config, stage1_config)
                row["dataset"] = str(dataset)
                row["target_snr_db"] = float(snr_db)
                row["noise_seed"] = int(seed)
                rows.append(row)
                packet_rows.append(row)

            summary = _summarize(rows, str(dataset), float(snr_db))
            summary["noise_seed"] = int(seed)
            summary["wins_vs_savaux"] = int(
                sum(float(row["savaux_phase_symbol_ser"]) < float(row["savaux_hard_symbol_ser"]) for row in rows)
            )
            summary["ties_vs_savaux"] = int(
                sum(abs(float(row["savaux_phase_symbol_ser"]) - float(row["savaux_hard_symbol_ser"])) <= 1e-12 for row in rows)
            )
            summary["losses_vs_savaux"] = int(
                sum(float(row["savaux_phase_symbol_ser"]) > float(row["savaux_hard_symbol_ser"]) for row in rows)
            )
            summary["guarded_wins_vs_savaux"] = int(
                sum(float(row["savaux_guarded_symbol_ser"]) < float(row["savaux_hard_symbol_ser"]) for row in rows)
            )
            summary["guarded_ties_vs_savaux"] = int(
                sum(abs(float(row["savaux_guarded_symbol_ser"]) - float(row["savaux_hard_symbol_ser"])) <= 1e-12 for row in rows)
            )
            summary["guarded_losses_vs_savaux"] = int(
                sum(float(row["savaux_guarded_symbol_ser"]) > float(row["savaux_hard_symbol_ser"]) for row in rows)
            )
            summary["packet_win_rate_vs_savaux"] = float(summary["wins_vs_savaux"] / max(1, len(rows)))
            summary["guarded_packet_win_rate_vs_savaux"] = float(summary["guarded_wins_vs_savaux"] / max(1, len(rows)))
            summary_rows.append(summary)
            print(
                f"{dataset} seed={seed} snr={snr_db:>6.1f} "
                f"savaux={summary['savaux_hard_symbol_ser']:.3f} "
                f"savaux_phase={summary['savaux_phase_symbol_ser']:.3f} "
                f"guarded={summary['savaux_guarded_symbol_ser']:.3f} "
                f"legacy={summary['legacy_phase_line_symbol_ser']:.3f} "
                f"recall={summary['stage1_candidate_recall']:.3f} "
                f"phase_w/t/l={summary['wins_vs_savaux']}/"
                f"{summary['ties_vs_savaux']}/{summary['losses_vs_savaux']} "
                f"guard_w/t/l={summary['guarded_wins_vs_savaux']}/"
                f"{summary['guarded_ties_vs_savaux']}/{summary['guarded_losses_vs_savaux']}",
                flush=True,
            )
    manifest = {
        "dataset": str(dataset),
        "input_iq": str(paths["iq"]),
        "symbol_csv": str(paths["symbols"]),
        "packet_count": int(len(packets)),
        "signal_reference_power": float(signal_power),
        "seeds": [int(v) for v in seeds],
    }
    return packet_rows, summary_rows, manifest


def _aggregate_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, int | str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((float(row["target_snr_db"]), row.get("noise_seed", "all")), []).append(row)
    out: list[dict[str, Any]] = []
    for (snr_db, seed), group in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        total_packets = sum(int(row["packet_count"]) for row in group)
        wins = sum(int(row.get("wins_vs_savaux", 0)) for row in group)
        ties = sum(int(row.get("ties_vs_savaux", 0)) for row in group)
        losses = sum(int(row.get("losses_vs_savaux", 0)) for row in group)
        guarded_wins = sum(int(row.get("guarded_wins_vs_savaux", 0)) for row in group)
        guarded_ties = sum(int(row.get("guarded_ties_vs_savaux", 0)) for row in group)
        guarded_losses = sum(int(row.get("guarded_losses_vs_savaux", 0)) for row in group)
        out.append(
            {
                "dataset": "mean_of_datasets",
                "target_snr_db": float(snr_db),
                "noise_seed": seed,
                "packet_count": int(total_packets),
                "center_argmax_symbol_ser": _avg(group, "center_argmax_symbol_ser"),
                "multi_offset_argmax_symbol_ser": _avg(group, "multi_offset_argmax_symbol_ser"),
                "legacy_phase_line_symbol_ser": _avg(group, "legacy_phase_line_symbol_ser"),
                "savaux_hard_symbol_ser": _avg(group, "savaux_hard_symbol_ser"),
                "savaux_phase_symbol_ser": _avg(group, "savaux_phase_symbol_ser"),
                "savaux_guarded_symbol_ser": _avg(group, "savaux_guarded_symbol_ser"),
                "savaux_phase_gain_vs_savaux": _avg(group, "savaux_phase_gain_vs_savaux"),
                "savaux_guarded_gain_vs_savaux": _avg(group, "savaux_guarded_gain_vs_savaux"),
                "savaux_phase_gain_vs_legacy": _avg(group, "savaux_phase_gain_vs_legacy"),
                "savaux_guarded_gain_vs_legacy": _avg(group, "savaux_guarded_gain_vs_legacy"),
                "stage1_candidate_recall": _avg(group, "stage1_candidate_recall"),
                "viterbi_candidate_recall": _avg(group, "viterbi_candidate_recall"),
                "mean_hard_anchor_count": _avg(group, "mean_hard_anchor_count"),
                "mean_hard_anchor_error_rate": _avg(group, "mean_hard_anchor_error_rate"),
                "guard_accept_phase_rate": _avg(group, "guard_accept_phase_rate"),
                "wins_vs_savaux": int(wins),
                "ties_vs_savaux": int(ties),
                "losses_vs_savaux": int(losses),
                "packet_win_rate_vs_savaux": float(wins / max(1, wins + ties + losses)),
                "guarded_wins_vs_savaux": int(guarded_wins),
                "guarded_ties_vs_savaux": int(guarded_ties),
                "guarded_losses_vs_savaux": int(guarded_losses),
                "guarded_packet_win_rate_vs_savaux": float(
                    guarded_wins / max(1, guarded_wins + guarded_ties + guarded_losses)
                ),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    datasets = [str(v) for v in (args.datasets if args.datasets is not None else [args.dataset])]
    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(datasets):
        rows, summaries, manifest = _run_dataset(args, dataset, dataset_index)
        packet_rows.extend(rows)
        summary_rows.extend(summaries)
        manifests.append(manifest)

    aggregate_rows = _aggregate_summary(summary_rows) if len(datasets) > 1 else []
    all_summary_rows = summary_rows + aggregate_rows

    out_dir = Path(args.output_dir)
    _write_csv(out_dir / "per_packet_metrics.csv", packet_rows)
    _write_csv(out_dir / "snr_summary.csv", all_summary_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "datasets": datasets,
                "snrs": [float(v) for v in args.snrs],
                "manifests": manifests,
                "summary_rows": all_summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote={out_dir / 'snr_summary.csv'}")
    print(f"wrote={out_dir / 'per_packet_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
