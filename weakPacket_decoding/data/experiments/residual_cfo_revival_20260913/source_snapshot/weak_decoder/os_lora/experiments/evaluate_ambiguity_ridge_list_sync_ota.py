"""Evaluate CRC-aided ambiguity-ridge list synchronization on OTA AWGN.

This runner reproduces the exact packet/SNR/seed realizations from the formal
decoder-aware CRC experiment.  It replaces the SFD integer-CFO Argmax with a
Top-L peak pool, maps each integer CFO to its coupled payload start, retains a
ranked Top-K list, and lets the unchanged Savaux/FEC/CRC chain arbitrate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
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
    unit_lora_band_awgn,
    write_csv_rows,
)
from weak_decoder.os_lora.system.ambiguity_ridge_list import (  # noqa: E402
    arbitrate_sync_list_with_crc,
    build_ambiguity_ridge_sync_list,
)


SF = 12
N_BINS = 1 << SF
BW_HZ = 125_000.0
SOURCE_RATE_HZ = 1_000_000.0
OS_FACTOR = 8
LDRO_MODE = 1
CRC_MODE = "grlora"
PREAMBLE_SYMBOLS = 16
SYNC_WORD = 0x12
DEMOD_TAIL_SAMPLES = 256


def _sync_config(module: Any, center_frequency_hz: float) -> Any:
    return make_sync_config(
        module,
        center_frequency_hz,
        sf=SF,
        bw_hz=BW_HZ,
        sample_rate_hz=SOURCE_RATE_HZ,
        preamble_symbols=PREAMBLE_SYMBOLS,
        sync_word=SYNC_WORD,
    )


def _metadata_by_reference(ota_root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in packet_metadata_paths(ota_root):
        metadata = load_json(path)
        result.setdefault(int(metadata["reference"]["reference_id"]), path)
    return result


def _first_crc_attempt(attempts: Sequence[Any], limit: int) -> int | None:
    for index, attempt in enumerate(attempts[: int(limit)]):
        if bool(attempt.crc_accepted):
            return index
    return None


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(task["baseline"])
    dataset_repo = Path(str(task["dataset_repo"]))
    ota_root = Path(str(task["ota_root"]))
    metadata = load_json(Path(str(task["metadata_path"])))
    reference_id = int(baseline["reference_id"])
    reference = load_json(
        ota_root.parent / "metadata" / f"{reference_id:06d}.json"
    )
    expected = bytes.fromhex(str(reference["packet"]["frame_hex"]))
    samples = np.fromfile(
        ota_root / str(metadata["ota"]["relative_path"]), dtype=np.dtype("<c8")
    )
    seed = int(baseline["seed"])
    esn0_db = float(baseline["esn0_db"])
    noise_power = float(baseline["added_noise_power"])
    rng = np.random.default_rng(
        np.random.SeedSequence((seed, reference_id, 73013))
    )
    scale = math.sqrt(noise_power)
    noisy = np.asarray(
        samples
        + unit_lora_band_awgn(rng, samples.size, os_factor=OS_FACTOR) * scale,
        dtype=np.complex64,
    )
    tail = unit_lora_band_awgn(
        rng, DEMOD_TAIL_SAMPLES, os_factor=OS_FACTOR
    ) * scale
    decode_samples = np.concatenate([noisy, tail]).astype(np.complex64)

    module = load_single_packet_sync_module(dataset_repo)
    config = _sync_config(
        module, float(metadata["capture"]["center_frequency_hz"])
    )
    sync_started = time.perf_counter()
    sync_result = module.run_single_packet_sync(noisy, config)
    sync_ms = 1e3 * (time.perf_counter() - sync_started)
    frame_sync = sync_result.frame_sync

    list_started = time.perf_counter()
    if frame_sync is None:
        sync_list = build_ambiguity_ridge_sync_list(
            noisy,
            None,
            sf=SF,
            bw_hz=BW_HZ,
            os_factor=OS_FACTOR,
            center_frequency_hz=float(metadata["capture"]["center_frequency_hz"]),
            top_k=int(task["top_k"]),
            sfd_peak_pool=int(task["sfd_peak_pool"]),
        )
    else:
        sync_list = build_ambiguity_ridge_sync_list(
            noisy,
            frame_sync,
            sf=SF,
            bw_hz=BW_HZ,
            os_factor=OS_FACTOR,
            center_frequency_hz=float(metadata["capture"]["center_frequency_hz"]),
            preamble_symbols=PREAMBLE_SYMBOLS,
            sync_word=SYNC_WORD,
            top_k=int(task["top_k"]),
            sfd_peak_pool=int(task["sfd_peak_pool"]),
        )
    list_generation_ms = 1e3 * (time.perf_counter() - list_started)
    decode_started = time.perf_counter()
    arbitration = arbitrate_sync_list_with_crc(
        decode_samples,
        sync_list.candidates,
        sf=SF,
        bw_hz=BW_HZ,
        os_factor=OS_FACTOR,
        ldro_mode=LDRO_MODE,
        crc_mode=CRC_MODE,
        require_payload_crc=True,
        stop_on_crc=False,
    )
    list_decode_ms = 1e3 * (time.perf_counter() - decode_started)

    top_k_values = tuple(int(value) for value in task["top_k_values"])
    trial: dict[str, Any] = {
        "trial_id": str(baseline["trial_id"]),
        "packet_id": str(baseline["packet_id"]),
        "reference_id": reference_id,
        "esn0_db": esn0_db,
        "seed": seed,
        "estimate_available": int(frame_sync is not None),
        "strict_packet_delivered": int(baseline["strict_packet_delivered"]),
        "decoder_aware_packet_delivered": int(
            baseline["decoder_aware_packet_delivered"]
        ),
        "oracle_packet_delivered": int(baseline["oracle_packet_delivered"]),
        "strict_gate": int(baseline["strict_gate"]),
        "decoder_aware_gate": int(baseline["decoder_aware_gate"]),
        "candidate_count": len(sync_list.candidates),
        "unique_integer_cfo_pool": int(sync_list.unique_integer_cfo_pool),
        "sfd_peaks_examined": int(sync_list.sfd_peaks_examined),
        "crc_success_count_top_max": sum(
            int(attempt.crc_accepted) for attempt in arbitration.attempts
        ),
        "sync_ms": sync_ms,
        "list_generation_ms": list_generation_ms,
        "list_decode_all_ms": list_decode_ms,
        "selected_sync_cfo_reproduced": int(
            frame_sync is None
            or abs(
                float(frame_sync.cfo_total_est)
                - float(baseline.get("noisy_cfo_total_bins", frame_sync.cfo_total_est))
            )
            < 1e-9
        ),
        "selected_sync_start_reproduced": int(
            frame_sync is None
            or int(frame_sync.fine_payload_start_sample)
            == int(
                float(
                    baseline.get(
                        "noisy_payload_start_sample",
                        frame_sync.fine_payload_start_sample,
                    )
                )
            )
        ),
    }
    for k in top_k_values:
        selected_index = _first_crc_attempt(arbitration.attempts, k)
        selected = (
            None
            if selected_index is None
            else arbitration.attempts[int(selected_index)]
        )
        crc_accept = selected is not None
        exact = bool(
            selected is not None
            and selected.decode.payload_bytes == expected
            and selected.decode.crc_valid
        )
        trial.update(
            {
                f"list{k}_crc_accept": int(crc_accept),
                f"list{k}_packet_delivered": int(exact),
                f"list{k}_crc_false_delivery": int(crc_accept and not exact),
                f"list{k}_selected_candidate": ""
                if selected_index is None
                else int(selected_index),
                f"list{k}_operational_attempts": min(
                    int(k),
                    len(arbitration.attempts)
                    if selected_index is None
                    else int(selected_index) + 1,
                ),
            }
        )

    candidate_rows: list[dict[str, Any]] = []
    for attempt in arbitration.attempts:
        candidate = attempt.candidate
        decoded = attempt.decode
        candidate_rows.append(
            {
                "trial_id": str(baseline["trial_id"]),
                "reference_id": reference_id,
                "esn0_db": esn0_db,
                "seed": seed,
                "candidate_index": int(candidate.candidate_index),
                "source": str(candidate.source),
                "sfd_peak_rank": int(candidate.sfd_peak_rank),
                "sfd_peak_signed_bin": int(candidate.sfd_peak_signed_bin),
                "sfd_relative_power_db": float(candidate.sfd_relative_power_db),
                "cfo_total_est": float(candidate.cfo_total_est),
                "fine_payload_start_sample": int(
                    candidate.fine_payload_start_sample
                ),
                "delta_cfo_bins": float(candidate.delta_cfo_bins),
                "delta_payload_samples": int(candidate.delta_payload_samples),
                "ridge_residual_bins": float(candidate.ridge_residual_bins),
                "netid1_est": int(candidate.netid1_est),
                "netid2_est": int(candidate.netid2_est),
                "netid_valid": int(candidate.netid_valid),
                "netid_margin_db": float(candidate.netid_margin_db),
                "decode_status": str(decoded.status),
                "header_valid": int(decoded.header_valid),
                "crc_accepted": int(attempt.crc_accepted),
                "payload_exact": int(
                    attempt.crc_accepted and decoded.payload_bytes == expected
                ),
                "payload_hex": decoded.payload_bytes.hex(),
            }
        )
    return {"trial": trial, "candidates": candidate_rows}


def _rate(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([int(row[key]) for row in rows])) if rows else 0.0


def _summary(
    rows: Sequence[dict[str, Any]],
    esn0_values: Sequence[float],
    top_k_values: Sequence[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for esn0_db in esn0_values:
        selected = [
            row for row in rows if float(row["esn0_db"]) == float(esn0_db)
        ]
        if not selected:
            continue
        row: dict[str, Any] = {
            "esn0_db": float(esn0_db),
            "trials": len(selected),
            "strict_crc_pdr": _rate(selected, "strict_packet_delivered"),
            "decoder_aware_crc_pdr": _rate(
                selected, "decoder_aware_packet_delivered"
            ),
            "oracle_crc_pdr": _rate(selected, "oracle_packet_delivered"),
            "mean_list_generation_ms": float(
                np.mean([float(item["list_generation_ms"]) for item in selected])
            ),
            "mean_decode_all_ms": float(
                np.mean([float(item["list_decode_all_ms"]) for item in selected])
            ),
            "sync_cfo_reproduction_mismatches": sum(
                not int(item["selected_sync_cfo_reproduced"]) for item in selected
            ),
            "sync_start_reproduction_mismatches": sum(
                not int(item["selected_sync_start_reproduced"])
                for item in selected
            ),
        }
        for k in top_k_values:
            row.update(
                {
                    f"list{k}_crc_pdr": _rate(
                        selected, f"list{k}_packet_delivered"
                    ),
                    f"list{k}_crc_accept_rate": _rate(
                        selected, f"list{k}_crc_accept"
                    ),
                    f"list{k}_crc_false_deliveries": sum(
                        int(item[f"list{k}_crc_false_delivery"])
                        for item in selected
                    ),
                    f"list{k}_rescues_over_decoder_aware": sum(
                        int(item[f"list{k}_packet_delivered"])
                        and not int(item["decoder_aware_packet_delivered"])
                        for item in selected
                    ),
                    f"list{k}_regressions_vs_decoder_aware": sum(
                        int(item["decoder_aware_packet_delivered"])
                        and not int(item[f"list{k}_packet_delivered"])
                        for item in selected
                    ),
                    f"list{k}_mean_operational_attempts": float(
                        np.mean(
                            [
                                int(item[f"list{k}_operational_attempts"])
                                for item in selected
                            ]
                        )
                    ),
                }
            )
        output.append(row)
    return output


def _make_plot(
    output_dir: Path,
    summary: Sequence[dict[str, Any]],
    top_k_values: Sequence[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [float(row["esn0_db"]) for row in summary]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(
        x,
        [float(row["strict_crc_pdr"]) for row in summary],
        "o--",
        label="strict",
    )
    ax.plot(
        x,
        [float(row["decoder_aware_crc_pdr"]) for row in summary],
        "s--",
        label="soft gate",
    )
    markers = {1: "d", 2: "^", 4: "P"}
    for k in top_k_values:
        ax.plot(
            x,
            [float(row[f"list{k}_crc_pdr"]) for row in summary],
            marker=markers.get(int(k), "x"),
            label=f"ridge list K={k}",
        )
    ax.plot(
        x,
        [float(row["oracle_crc_pdr"]) for row in summary],
        "k*-",
        label="oracle sync",
    )
    ax.set_xlabel(r"$E_s/N_0$ (dB)")
    ax.set_ylabel("Exact CRC-valid packet delivery rate")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "ambiguity_ridge_list_crc_pdr.png", dpi=180)
    plt.close(fig)


def _build_report(
    output_dir: Path,
    summary: Sequence[dict[str, Any]],
    top_k_values: Sequence[int],
    trials: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> None:
    header = (
        "| Es/N0 | strict | soft gate | "
        + " | ".join(f"list K={k}" for k in top_k_values)
        + " | oracle |"
    )
    separator = "|---:|" + "---:|" * (3 + len(top_k_values))
    lines = [
        "# CRC-aided ambiguity-ridge list synchronization",
        "",
        (
            "SFD Top-L integer-CFO peaks are mapped to coupled payload "
            "boundaries; only Top-K candidates enter the unchanged "
            "Savaux/FEC/CRC decoder."
        ),
        "",
        header,
        separator,
    ]
    for row in summary:
        values = [
            f"{float(row['esn0_db']):g}",
            f"{float(row['strict_crc_pdr']):.3f}",
            f"{float(row['decoder_aware_crc_pdr']):.3f}",
        ]
        values.extend(f"{float(row[f'list{k}_crc_pdr']):.3f}" for k in top_k_values)
        values.append(f"{float(row['oracle_crc_pdr']):.3f}")
        lines.append("| " + " | ".join(values) + " |")
    trial_count = len(trials)
    strict_count = sum(int(row["strict_packet_delivered"]) for row in trials)
    soft_count = sum(
        int(row["decoder_aware_packet_delivered"]) for row in trials
    )
    oracle_count = sum(int(row["oracle_packet_delivered"]) for row in trials)
    lines.extend(
        [
            "",
            (
                f"Across {trial_count} trials: strict={strict_count}, "
                f"soft gate={soft_count}, oracle={oracle_count} exact "
                "CRC-valid packets."
            ),
        ]
    )
    oracle_headroom = max(0, oracle_count - soft_count)
    for k in top_k_values:
        delivered = sum(int(row[f"list{k}_packet_delivered"]) for row in trials)
        closed = max(0, delivered - soft_count)
        closed_text = (
            "n/a"
            if oracle_headroom == 0
            else f"{100.0 * closed / oracle_headroom:.1f}%"
        )
        mean_attempts = (
            float(
                np.mean(
                    [int(row[f"list{k}_operational_attempts"]) for row in trials]
                )
            )
            if trials
            else 0.0
        )
        lines.append(
            f"K={k} delivers {delivered}/{trial_count}, closes "
            f"{closed}/{oracle_headroom} ({closed_text}) of the "
            "soft-to-oracle gap, and needs "
            f"{mean_attempts:.3f} decoder attempts per input trial "
            "with early stopping."
        )
    lines.extend(["", "## Arbitration audit", ""])
    for k in top_k_values:
        rescues = sum(int(row[f"list{k}_rescues_over_decoder_aware"]) for row in summary)
        regressions = sum(
            int(row[f"list{k}_regressions_vs_decoder_aware"]) for row in summary
        )
        false = sum(int(row[f"list{k}_crc_false_deliveries"]) for row in summary)
        lines.append(
            f"- K={k}: {rescues} rescues over soft gate, {regressions} "
            f"regressions, {false} CRC false deliveries."
        )
    cfo_mismatch = sum(int(row["sync_cfo_reproduction_mismatches"]) for row in summary)
    start_mismatch = sum(
        int(row["sync_start_reproduction_mismatches"]) for row in summary
    )
    crc_candidate_rows = [
        row for row in candidates if int(row["crc_accepted"])
    ]
    crc_wrong_rows = [
        row for row in crc_candidate_rows if not int(row["payload_exact"])
    ]
    lines.extend(
        [
            (
                "- Reproduction mismatches versus baseline: "
                f"CFO={cfo_mismatch}, payload start={start_mismatch}."
            ),
            (
                f"- Full-list audit: {len(crc_candidate_rows)} CRC-valid "
                f"candidate decodes, {len(crc_wrong_rows)} non-exact "
                "deliveries."
            ),
            "",
            (
                "Ground-truth bytes are used only after CRC to score exact "
                "delivery and detect CRC collisions."
            ),
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CRC-aided LoRa ambiguity-ridge list synchronization."
    )
    parser.add_argument(
        "--dataset-repo",
        type=Path,
        default=WORKSPACE_ROOT / "lora-rfsr-savaux",
    )
    parser.add_argument("--ota-root", type=Path, default=None)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "decoder_aware_crc_pdr_ota_awgn_20260822",
    )
    parser.add_argument("--esn0-db", default="12,13,14,15,16")
    parser.add_argument(
        "--trial-id",
        default="",
        help="Optional comma-separated exact baseline trial IDs.",
    )
    parser.add_argument("--top-k-values", default="1,2,4")
    parser.add_argument("--sfd-peak-pool", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "ambiguity_ridge_list_crc_ota_awgn_20260822",
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
    baseline_dir = args.baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    esn0_values = parse_csv_list(args.esn0_db, float)
    top_k_values = tuple(sorted(set(parse_csv_list(args.top_k_values, int))))
    if any(value <= 0 for value in top_k_values):
        raise ValueError("top-k values must be positive")
    max_k = max(top_k_values)

    if args.summarize_existing:
        trials = read_csv_rows(output_dir / "packet_trials.csv")
        summary = _summary(trials, esn0_values, top_k_values)
        candidates = read_csv_rows(output_dir / "candidate_trials.csv")
        write_csv_rows(output_dir / "summary.csv", summary)
        _make_plot(output_dir, summary, top_k_values)
        _build_report(output_dir, summary, top_k_values, trials, candidates)
        print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
        return 0

    requested_trial_ids = {
        value.strip() for value in str(args.trial_id).split(",") if value.strip()
    }
    baseline_rows = [
        row
        for row in read_csv_rows(baseline_dir / "packet_trials.csv")
        if float(row["esn0_db"]) in {float(value) for value in esn0_values}
        and (
            not requested_trial_ids
            or str(row["trial_id"]) in requested_trial_ids
        )
    ]
    if requested_trial_ids:
        found_trial_ids = {str(row["trial_id"]) for row in baseline_rows}
        missing_trial_ids = requested_trial_ids - found_trial_ids
        if missing_trial_ids:
            raise ValueError(
                "baseline trial IDs not found after Es/N0 filtering: "
                + ", ".join(sorted(missing_trial_ids))
            )
    metadata_map = _metadata_by_reference(ota_root)
    tasks = [
        {
            "baseline": row,
            "dataset_repo": str(dataset_repo),
            "ota_root": str(ota_root),
            "metadata_path": str(metadata_map[int(row["reference_id"])]),
            "top_k": max_k,
            "top_k_values": top_k_values,
            "sfd_peak_pool": int(args.sfd_peak_pool),
        }
        for row in baseline_rows
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 4 == 0 or index == len(futures):
                print(f"completed ridge-list trials: {index}/{len(futures)}", flush=True)
            if index % 16 == 0:
                checkpoint_trials = [item["trial"] for item in results]
                checkpoint_candidates = [
                    row for item in results for row in item["candidates"]
                ]
                write_csv_rows(output_dir / "packet_trials.csv", checkpoint_trials)
                write_csv_rows(output_dir / "candidate_trials.csv", checkpoint_candidates)

    results.sort(
        key=lambda item: (
            float(item["trial"]["esn0_db"]),
            int(item["trial"]["seed"]),
            int(item["trial"]["reference_id"]),
        )
    )
    trials = [item["trial"] for item in results]
    candidates = [row for item in results for row in item["candidates"]]
    summary = _summary(trials, esn0_values, top_k_values)
    write_csv_rows(output_dir / "packet_trials.csv", trials)
    write_csv_rows(output_dir / "candidate_trials.csv", candidates)
    write_csv_rows(output_dir / "summary.csv", summary)
    _make_plot(output_dir, summary, top_k_values)
    _build_report(output_dir, summary, top_k_values, trials, candidates)
    config = {
        "dataset_repo": str(dataset_repo),
        "ota_root": str(ota_root),
        "baseline_dir": str(baseline_dir),
        "esn0_db": list(esn0_values),
        "trial_ids": sorted(requested_trial_ids),
        "top_k_values": list(top_k_values),
        "sfd_peak_pool": int(args.sfd_peak_pool),
        "workers": int(args.workers),
        "candidate_rule": (
            "SFD Top-L integer CFO; payload_start += OSR * delta_CFO; "
            "candidate-specific STO/SFO refinement"
        ),
        "arbitration": "first ranked explicit-header + payload-PHY-CRC success",
        "decode_all_for_audit": True,
        "operational_stop_on_crc": True,
        "ground_truth_usage": "post-CRC exact-delivery audit only",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
