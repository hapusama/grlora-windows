"""Paired OTA replay of fixed and preamble-ranked residual-CFO recovery.

Clean timing is used only for calibration, an explicit oracle and scoring.
All three noisy candidates share timing/SFO and the same soft PHY decoder.
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weak_decoder.os_lora.experiments import evaluate_decoder_aware_crc_pdr_ota as base
from weak_decoder.os_lora.system.residual_cfo import (
    build_residual_cfo_candidates,
    decode_residual_cfo_candidates,
    rank_residual_cfo_from_preamble,
)
from weak_decoder.synchronization.grlora_frame_sync import run_grlora_frame_sync_validation


def work(task):
    path, snrs, seeds = task
    repo = ROOT.parent.parent / "lora-rfsr-savaux"
    module = base.load_single_packet_sync_module(repo)
    module.run_grlora_frame_sync_validation = partial(
        run_grlora_frame_sync_validation, sfd_cfo_mode="chip"
    )
    clean = base._clean_worker(dict(
        dataset_repo=str(repo), ota_root=str(repo / "data/reference_phy/rfsr_db"),
        metadata_path=path,
    ))
    packet = clean["packet"]
    if packet is None:
        return clean["audit"], [], []
    samples = np.fromfile(packet["iq_path"], dtype="<c8")
    expected = bytes.fromhex(packet["expected_frame_hex"])
    config = base._sync_config(module, packet["center_frequency_hz"])
    module.run_grlora_frame_sync_validation = partial(
        run_grlora_frame_sync_validation, sfd_cfo_mode="virtual_phase"
    )
    rows, diagnostics = [], []
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence((seed, packet["reference_id"], 9122026)))
        prefix = int(rng.integers(0, base.SYMBOL_SAMPLES))
        padded = np.pad(samples, (prefix, base.DEMOD_TAIL_SAMPLES))
        noise = base.unit_lora_band_awgn(rng, len(padded), os_factor=8)
        oracle = base._clean_frame_sync(packet)
        oracle.fine_payload_start_sample += prefix
        for snr in snrs:
            noisy = (padded + noise * np.sqrt(
                packet["signal_power"] * base.N_BINS / 10 ** (snr / 10)
            )).astype(np.complex64)
            tick = time.perf_counter()
            sync = module.run_single_packet_sync(noisy, config)
            sync_seconds = time.perf_counter() - tick
            frame = sync.frame_sync
            candidates = build_residual_cfo_candidates(frame)
            tick = time.perf_counter()
            ranked = rank_residual_cfo_from_preamble(
                noisy, candidates, sf=12, os_factor=8, preamble_symbols=16,
            )
            ranking_seconds = time.perf_counter() - tick
            results, costs = {}, {}
            for candidate in candidates:
                tick = time.perf_counter()
                result = decode_residual_cfo_candidates(
                    noisy, (candidate,), sf=12, bw_hz=125000, os_factor=8,
                    max_attempts=1, ldro_mode=1, allow_gate_failed_candidate=True,
                ).attempts[0]
                results[candidate.delta_cfo_bins] = result
                costs[candidate.delta_cfo_bins] = time.perf_counter() - tick
            key = dict(packet_id=packet["packet_id"], seed=seed, esn0_db=snr)
            for candidate in ranked:
                decoded = results[candidate.delta_cfo_bins]
                diagnostics.append(dict(
                    **key, delta_cfo_bins=candidate.delta_cfo_bins,
                    preamble_score=candidate.preamble_score,
                    preamble_symbols_used=candidate.preamble_symbols_used,
                    cfo_error_bins=candidate.cfo_int_est + candidate.cfo_frac_est
                        - packet["clean_cfo_total_bins"],
                    timing_error_samples=candidate.fine_payload_start_sample - oracle.fine_payload_start_sample,
                    netid_offset=frame.netid_offset, netid_valid=int(frame.netid_valid),
                    header_valid=int(decoded.decode.header_valid),
                    crc_accept=int(decoded.crc_accepted),
                    exact=int(decoded.crc_accepted and decoded.decode.payload_bytes == expected),
                ))
            policies = {
                "single_soft": candidates[:1], "fixed_cfo3_soft": candidates,
                "preamble_single_soft": ranked[:1], "preamble_cfo3_soft": ranked,
            }
            for method, ordered in policies.items():
                selected = None
                calls, seconds = 0, 0.0
                for candidate in ordered:
                    calls += 1
                    seconds += costs[candidate.delta_cfo_bins]
                    attempt = results[candidate.delta_cfo_bins]
                    if attempt.crc_accepted:
                        selected = attempt
                        break
                accepted = selected is not None
                exact = bool(accepted and selected.decode.payload_bytes == expected)
                rows.append(dict(
                    **key, method=method, crc_accept=int(accepted), exact=int(exact),
                    false_delivery=int(accepted and not exact), attempts=calls,
                    selected_delta="" if selected is None else selected.candidate.delta_cfo_bins,
                    sync_seconds=sync_seconds, ranking_seconds=ranking_seconds if method.startswith("preamble") else 0,
                    decode_seconds=seconds,
                ))
            oracle_result = base.decode_soft_hamming_sync_candidate(
                noisy, oracle, sf=12, bw_hz=125000, os_factor=8,
                ldro_mode=1, crc_mode="grlora", allow_gate_failed_candidate=True,
            )
            accepted = bool(oracle_result.header_valid and oracle_result.header.has_crc and oracle_result.crc_valid)
            exact = bool(accepted and oracle_result.payload_bytes == expected)
            rows.append(dict(**key, method="oracle_soft", crc_accept=int(accepted),
                             exact=int(exact), false_delivery=int(accepted and not exact), attempts=1))
        print(f"finished {packet['packet_id']} seed={seed}", flush=True)
    return clean["audit"], rows, diagnostics


def summarize(rows):
    output = []
    baseline = {(r["packet_id"], r["seed"], r["esn0_db"]): r for r in rows if r["method"] == "single_soft"}
    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == method]
        deltas = {}
        wins = losses = 0
        for row in selected:
            key = row["packet_id"], row["seed"], row["esn0_db"]
            delta = row["exact"] - baseline[key]["exact"]
            deltas.setdefault(row["packet_id"], []).append(delta)
            wins += int(delta > 0)
            losses += int(delta < 0)
        clusters = np.array([np.mean(deltas[k]) for k in sorted(deltas)])
        rng = np.random.default_rng(9132026)
        boot = clusters[rng.integers(len(clusters), size=(20000, len(clusters)))].mean(axis=1)
        output.append(dict(
            method=method, correct=sum(r["exact"] for r in selected), trials=len(selected),
            false_delivery=sum(r["false_delivery"] for r in selected),
            mean_attempts=float(np.mean([r["attempts"] for r in selected])),
            wins=wins, losses=losses, physical_packets=len(clusters),
            difference=float(clusters.mean()),
            exploratory_cluster_bootstrap_95ci=np.quantile(boot, [.025, .975]).tolist(),
        ))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", type=int, default=80)
    parser.add_argument("--packets", type=int, default=12)
    parser.add_argument("--snrs", default="12,13,14")
    parser.add_argument("--seeds", default="91611,91612")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--label", required=True, help="State diagnostic replay or new same-session validation.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.offset < 0 or args.packets < 1 or args.workers < 1:
        parser.error("offset must be nonnegative; packets and workers must be positive")
    if (args.output / "config.json").exists():
        parser.error("output already contains a run; choose a new output directory")
    paths = list(base.packet_metadata_paths(ROOT.parent.parent / "lora-rfsr-savaux/data/reference_phy/rfsr_db"))[args.offset:args.offset+args.packets]
    if len(paths) != args.packets:
        parser.error("requested packet range is not available")
    snrs = [float(x) for x in args.snrs.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(metadata=[str(p) for p in paths], seeds=seeds, snrs=snrs, label=args.label,
                  selection="received preamble only; first valid header, CRC enabled, CRC pass; truth only scores",
                  scope="same-session OTA plus band-limited AWGN; eligible clean packets; not natural PDR",
                  policies="original CFO; fixed [0,-1,+1]; eight middle preamble chirps rank same candidates; soft FEC for all",
                  source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in list((ROOT / "weak_decoder").rglob("*.py")) + [Path(__file__)]})
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    audits, rows, diagnostics = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, (str(path), snrs, seeds)) for path in paths]
        for future in as_completed(futures):
            audit, result_rows, result_diagnostics = future.result()
            audits.append(audit)
            rows.extend(result_rows)
            diagnostics.extend(result_diagnostics)
            base.write_csv_rows(args.output / "clean_audit.csv", audits)
            base.write_csv_rows(args.output / "trials.csv", rows)
            base.write_csv_rows(args.output / "candidates.csv", diagnostics)
            print(f"completed {len(audits)}/{len(paths)} packets", flush=True)
    summary = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
