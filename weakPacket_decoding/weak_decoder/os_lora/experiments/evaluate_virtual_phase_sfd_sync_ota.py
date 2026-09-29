"""Paired OTA+AWGN audit for virtual-phase LoRa SFD synchronization.

The only synchronization variable is the estimator used on the known second
LoRa SFD downchirp:

``chip``
    The existing gr-lora-compatible chip-rate, single-ADC-phase FFT.

``virtual_phase``
    The same SFD, but all OSR polyphase observations are coherently combined
    with Savaux's candidate-dependent LoRa chirp-wrap law.

Preamble detection, frame location, SFO/STO refinement, payload demodulation,
and all acceptance gates are otherwise identical.  The clean-packet coordinate
and reference payload symbols are used only after synchronization for audit;
they never enter either SFD decision.
"""

from __future__ import annotations

import argparse
import csv
from functools import partial
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
WEAK_PACKET_ROOT = SCRIPT_PATH.parents[3]
GR_LORA_ROOT = SCRIPT_PATH.parents[4]
WORKSPACE_ROOT = GR_LORA_ROOT.parent
if str(WEAK_PACKET_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_PACKET_ROOT))

from weak_decoder.os_lora.experiments import (  # noqa: E402
    analyze_noisy_framesync_headroom_ota as base,
)
from weak_decoder.synchronization.grlora_frame_sync import (  # noqa: E402
    run_grlora_frame_sync_validation,
)


def _parse_list(text: str, cast: type) -> tuple[Any, ...]:
    values = tuple(cast(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return values


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sync_with_sfd_mode(
    module: Any,
    samples: np.ndarray,
    config: Any,
    mode: str,
) -> Any:
    """Run the existing full packet chain with one SFD-CFO decision rule.

    ``single_packet.py`` is an existing composition wrapper in the capture
    repository.  Rebinding this imported function keeps its detector and
    locator byte-for-byte shared across the two paired conditions.
    """

    module.run_grlora_frame_sync_validation = partial(
        run_grlora_frame_sync_validation,
        sfd_cfo_mode=str(mode),
    )
    return module.run_single_packet_sync(samples, config)


def _frame_fields(
    prefix: str,
    result: Any,
    clean_frame: Any,
) -> dict[str, Any]:
    frame = result.frame_sync
    output: dict[str, Any] = {
        f"{prefix}_sync_status": str(result.status),
        f"{prefix}_strict_sync_success": int(bool(result.synchronized)),
        f"{prefix}_decoder_aware_success": int(
            bool(result.accepted("decoder_aware"))
            if hasattr(result, "accepted")
            else False
        ),
        f"{prefix}_estimate_available": int(frame is not None),
    }
    if frame is None:
        return output
    cfo_error = float(frame.cfo_total_est) - float(clean_frame.cfo_total_est)
    payload_error = int(frame.fine_payload_start_sample) - int(
        clean_frame.fine_payload_start_sample
    )
    output.update(
        {
            f"{prefix}_cfo_int": int(frame.cfo_int_est),
            f"{prefix}_cfo_total_bins": float(frame.cfo_total_est),
            f"{prefix}_cfo_error_bins": cfo_error,
            f"{prefix}_cfo_integer_matches_clean": int(
                int(frame.cfo_int_est) == int(clean_frame.cfo_int_est)
            ),
            f"{prefix}_sfd_signed_bin": int(frame.down_val_signed_bin),
            f"{prefix}_sfd_cfo_source": str(frame.sfd_cfo_source),
            f"{prefix}_sfd_peak_margin_db": float(frame.sfd_cfo_peak_margin_db),
            f"{prefix}_payload_start_sample": int(frame.fine_payload_start_sample),
            f"{prefix}_payload_start_error_samples": payload_error,
            f"{prefix}_coordinate_local": int(
                abs(cfo_error) <= base.LOCAL_CFO_LIMIT_BINS
                and abs(payload_error) <= base.LOCAL_STO_LIMIT_SAMPLES
            ),
            f"{prefix}_netid_valid": int(bool(frame.netid_valid)),
            f"{prefix}_frame_sync_valid": int(bool(frame.valid)),
        }
    )
    return output


def _payload_starts(frame: Any, packet: dict[str, Any]) -> list[int]:
    return base._payload_starts(
        int(frame.fine_payload_start_sample),
        float(frame.sfo_cum_initial),
        float(frame.sfo_hat),
        int(packet["header_count"]),
        int(packet["payload_count"]),
    )


def _evaluate_payloads(
    noisy: np.ndarray,
    packet: dict[str, Any],
    chip_result: Any,
    virtual_result: Any,
    trial_id: str,
    esn0_db: float,
    seed: int,
) -> list[dict[str, Any]]:
    chip_frame = chip_result.frame_sync
    virtual_frame = virtual_result.frame_sync
    chip_starts = _payload_starts(chip_frame, packet) if chip_frame is not None else None
    virtual_starts = (
        _payload_starts(virtual_frame, packet) if virtual_frame is not None else None
    )
    rows: list[dict[str, Any]] = []
    for spec in packet["symbol_specs"]:
        index = int(spec["payload_index"])
        gt_symbol = int(spec["gt_symbol"])
        chip_metric = (
            base._savaux_metrics(
                noisy,
                chip_starts[index],
                float(chip_frame.cfo_total_est),
                gt_symbol,
            )
            if chip_frame is not None and chip_starts is not None
            else None
        )
        virtual_metric = (
            base._savaux_metrics(
                noisy,
                virtual_starts[index],
                float(virtual_frame.cfo_total_est),
                gt_symbol,
            )
            if virtual_frame is not None and virtual_starts is not None
            else None
        )
        rows.append(
            {
                "trial_id": trial_id,
                "packet_id": str(packet["packet_id"]),
                "reference_id": int(packet["reference_id"]),
                "esn0_db": float(esn0_db),
                "seed": int(seed),
                "payload_index": index,
                "gt_symbol": gt_symbol,
                "chip_correct": "" if chip_metric is None else int(chip_metric["correct"]),
                "virtual_phase_correct": ""
                if virtual_metric is None
                else int(virtual_metric["correct"]),
                "virtual_only_rescue": int(
                    chip_metric is not None
                    and virtual_metric is not None
                    and not bool(chip_metric["correct"])
                    and bool(virtual_metric["correct"])
                ),
                "chip_only_regression": int(
                    chip_metric is not None
                    and virtual_metric is not None
                    and bool(chip_metric["correct"])
                    and not bool(virtual_metric["correct"])
                ),
            }
        )
    return rows


def _mean(values: Iterable[int | float]) -> float:
    data = list(values)
    return float(np.mean(np.asarray(data, dtype=np.float64))) if data else float("nan")


def _summary(
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
        chip_available = [
            row for row in packet_rows if int(row["chip_estimate_available"])
        ]
        virtual_available = [
            row for row in packet_rows if int(row["virtual_estimate_available"])
        ]
        paired_available = [
            row
            for row in packet_rows
            if int(row["chip_estimate_available"])
            and int(row["virtual_estimate_available"])
        ]
        paired_symbols = [
            row
            for row in symbol_rows
            if row["chip_correct"] != "" and row["virtual_phase_correct"] != ""
        ]
        output.append(
            {
                "esn0_db": float(esn0_db),
                "packet_trials": len(packet_rows),
                "chip_strict_sync_rate": _mean(
                    int(row["chip_strict_sync_success"]) for row in packet_rows
                ),
                "virtual_strict_sync_rate": _mean(
                    int(row["virtual_strict_sync_success"]) for row in packet_rows
                ),
                "chip_decoder_aware_rate": _mean(
                    int(row["chip_decoder_aware_success"]) for row in packet_rows
                ),
                "virtual_decoder_aware_rate": _mean(
                    int(row["virtual_decoder_aware_success"]) for row in packet_rows
                ),
                "chip_estimate_available_rate": len(chip_available) / len(packet_rows)
                if packet_rows
                else float("nan"),
                "virtual_estimate_available_rate": len(virtual_available)
                / len(packet_rows)
                if packet_rows
                else float("nan"),
                "chip_integer_cfo_correct_given_available": _mean(
                    int(row["chip_cfo_integer_matches_clean"])
                    for row in chip_available
                ),
                "virtual_integer_cfo_correct_given_available": _mean(
                    int(row["virtual_cfo_integer_matches_clean"])
                    for row in virtual_available
                ),
                "chip_coordinate_local_rate_given_available": _mean(
                    int(row["chip_coordinate_local"]) for row in chip_available
                ),
                "virtual_coordinate_local_rate_given_available": _mean(
                    int(row["virtual_coordinate_local"]) for row in virtual_available
                ),
                "sfd_cfo_rescues": sum(
                    int(not row["chip_cfo_integer_matches_clean"])
                    and int(row["virtual_cfo_integer_matches_clean"])
                    for row in paired_available
                ),
                "sfd_cfo_regressions": sum(
                    int(row["chip_cfo_integer_matches_clean"])
                    and int(not row["virtual_cfo_integer_matches_clean"])
                    for row in paired_available
                ),
                "median_virtual_sfd_margin_db": float(
                    np.median(
                        [
                            float(row["virtual_sfd_peak_margin_db"])
                            for row in virtual_available
                        ]
                    )
                )
                if virtual_available
                else float("nan"),
                "paired_payload_symbols": len(paired_symbols),
                "chip_savaux_ser_paired": 1.0
                - _mean(int(row["chip_correct"]) for row in paired_symbols)
                if paired_symbols
                else float("nan"),
                "virtual_savaux_ser_paired": 1.0
                - _mean(int(row["virtual_phase_correct"]) for row in paired_symbols)
                if paired_symbols
                else float("nan"),
                "virtual_only_payload_rescues": sum(
                    int(row["virtual_only_rescue"]) for row in paired_symbols
                ),
                "chip_only_payload_regressions": sum(
                    int(row["chip_only_regression"]) for row in paired_symbols
                ),
            }
        )
    return output


def _build_report(
    output_dir: Path,
    summary: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# Virtual-phase SFD synchronization: paired OTA+AWGN audit",
        "",
        "Only the second-SFD integer-CFO estimator changes between conditions. "
        "The virtual condition coherently combines OSR polyphase views using "
        "Savaux's LoRa chirp-wrap phase law. Payload truth is used only for "
        "post-hoc evaluation, never to choose a synchronization result.",
        "",
        "| Es/N0 (dB) | chip integer-CFO correct | virtual integer-CFO correct | CFO rescues / regressions | chip / virtual local | chip / virtual paired SER | payload rescues / regressions |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {esn0_db:g} | {chip_integer_cfo_correct_given_available:.3f} | "
            "{virtual_integer_cfo_correct_given_available:.3f} | "
            "{sfd_cfo_rescues} / {sfd_cfo_regressions} | "
            "{chip_coordinate_local_rate_given_available:.3f} / "
            "{virtual_coordinate_local_rate_given_available:.3f} | "
            "{chip_savaux_ser_paired:.4f} / {virtual_savaux_ser_paired:.4f} | "
            "{virtual_only_payload_rescues} / {chip_only_payload_regressions} |".format(
                **row
            )
        )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-repo", type=Path, default=WORKSPACE_ROOT / "lora-rfsr-savaux"
    )
    parser.add_argument("--ota-root", type=Path, default=None)
    parser.add_argument("--max-packets", type=int, default=8)
    parser.add_argument("--symbols-per-packet", type=int, default=16)
    parser.add_argument("--esn0-db", default="12,13,14")
    parser.add_argument("--seeds", default="20260821")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "virtual_phase_sfd_sync_ota_awgn_20260826",
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
    esn0_values = _parse_list(args.esn0_db, float)
    seeds = _parse_list(args.seeds, int)
    metadata_paths = list(base._packet_metadata_paths(ota_root))[: int(args.max_packets)]
    if not metadata_paths:
        raise RuntimeError("no OTA packets satisfy the metadata filter")

    clean_packets: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        clean_result = base._clean_packet_worker(
            {
                "dataset_repo": str(dataset_repo),
                "ota_root": str(ota_root),
                "metadata_path": str(metadata_path),
                "symbols_per_packet": int(args.symbols_per_packet),
            }
        )
        if clean_result["packet"] is not None:
            clean_packets.append(dict(clean_result["packet"]))
    if not clean_packets:
        raise RuntimeError("no packets passed the clean synchronization audit")

    module = base._load_single_packet_sync_module(dataset_repo)
    trials: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []
    for packet_index, packet in enumerate(clean_packets, start=1):
        samples = np.fromfile(Path(str(packet["iq_path"])), dtype=np.dtype("<c8"))
        config = base._sync_config(module, float(packet["center_frequency_hz"]))
        clean_result = _sync_with_sfd_mode(module, samples, config, "chip")
        clean_frame = clean_result.frame_sync
        if clean_frame is None:
            raise RuntimeError(f"clean FrameSync unexpectedly unavailable: {packet['packet_id']}")
        for esn0_db in esn0_values:
            for seed in seeds:
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        (int(seed), int(packet["reference_id"]), 73013)
                    )
                )
                noise_power = float(packet["signal_power"]) * base.N_BINS / (
                    10.0 ** (float(esn0_db) / 10.0)
                )
                noise = base._unit_lora_band_awgn(rng, samples.size)
                noisy = np.asarray(
                    samples + noise * math.sqrt(noise_power), dtype=np.complex64
                )
                trial_id = f"{packet['packet_id']}:{float(esn0_db):g}:{int(seed)}"
                chip_result = _sync_with_sfd_mode(module, noisy, config, "chip")
                virtual_result = _sync_with_sfd_mode(
                    module, noisy, config, "virtual_phase"
                )
                trial = {
                    "trial_id": trial_id,
                    "packet_id": str(packet["packet_id"]),
                    "reference_id": int(packet["reference_id"]),
                    "esn0_db": float(esn0_db),
                    "seed": int(seed),
                    "clean_cfo_int": int(clean_frame.cfo_int_est),
                    "clean_cfo_total_bins": float(clean_frame.cfo_total_est),
                    "clean_payload_start_sample": int(
                        clean_frame.fine_payload_start_sample
                    ),
                }
                trial.update(_frame_fields("chip", chip_result, clean_frame))
                trial.update(_frame_fields("virtual", virtual_result, clean_frame))
                trials.append(trial)
                symbols.extend(
                    _evaluate_payloads(
                        noisy,
                        packet,
                        chip_result,
                        virtual_result,
                        trial_id,
                        float(esn0_db),
                        int(seed),
                    )
                )
                print(
                    f"completed packet {packet_index}/{len(clean_packets)}: "
                    f"{packet['packet_id']} at {float(esn0_db):g} dB, seed {seed}",
                    flush=True,
                )

    summary = _summary(trials, symbols, esn0_values)
    _write_csv(output_dir / "packet_trials.csv", trials)
    _write_csv(output_dir / "symbol_trials.csv", symbols)
    _write_csv(output_dir / "summary.csv", summary)
    _build_report(output_dir, summary)
    config_record = {
        "dataset_repo": str(dataset_repo),
        "ota_root": str(ota_root),
        "max_packets_requested": int(args.max_packets),
        "clean_packets_admitted": len(clean_packets),
        "symbols_per_packet": int(args.symbols_per_packet),
        "esn0_db": list(esn0_values),
        "seeds": list(seeds),
        "noise_model": "complex AWGN flat in B, added once to the full packet",
        "paired_variable": "second-SFD integer CFO: chip versus virtual_phase",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config_record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
