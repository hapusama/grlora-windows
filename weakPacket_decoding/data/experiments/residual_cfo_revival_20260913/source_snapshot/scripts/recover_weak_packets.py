"""Recover PHY packets from raw IQ and received-IQ synchronization events.

This continues run_weak_sync_chain.py through soft FEC and bounded CFO/CRC
recovery. It never reads a reference payload, clean timing, or oracle metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weak_decoder.os_lora.system.residual_cfo import (
    build_residual_cfo_candidates,
    decode_residual_cfo_candidates,
    rank_residual_cfo_from_preamble,
)
from weak_decoder.synchronization.grlora_frame_sync import run_grlora_frame_sync_validation
from weak_decoder.synchronization.preamble_detector import PreambleDetectorConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-s", "--sync-csv", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True, help="New JSONL output file.")
    parser.add_argument("--sf", type=int, required=True, choices=range(7, 13))
    parser.add_argument("--bw", type=float, default=125000)
    parser.add_argument("--samp-rate", type=float, required=True)
    parser.add_argument("--preamble-len", type=int, required=True)
    parser.add_argument("--sync-word", type=lambda x: int(x, 0), required=True)
    parser.add_argument("--center-freq", type=float, default=487.7e6)
    parser.add_argument("--ldro-mode", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--policy", choices=("fixed", "preamble"), default="fixed")
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--sfd-cfo-mode", choices=("chip", "virtual_phase"), default="virtual_phase")
    parser.add_argument("--allow-gate-failed", action="store_true",
                        help="Try located candidates even when strict sync validation fails.")
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()
    if args.bw <= 0 or args.samp_rate <= 0 or args.center_freq <= 0 or args.preamble_len < 8:
        parser.error("rates must be positive and the preamble must contain at least eight chirps")
    ratio = args.samp_rate / args.bw
    if ratio < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
        parser.error("sample rate must be an integer multiple of bandwidth")
    if args.max_events is not None and args.max_events < 1:
        parser.error("max-events must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    if args.input.stat().st_size == 0 or args.input.stat().st_size % 8:
        parser.error("input must contain nonempty raw complex64 IQ (eight bytes per sample)")
    return args


def main():
    args = parse_args()
    os_factor = int(round(args.samp_rate / args.bw))
    config = PreambleDetectorConfig(
        sf=args.sf, bw=args.bw, samp_rate=args.samp_rate,
        win_chirps=4, hop_samples=(1 << args.sf) * os_factor,
        min_periodic_peaks=max(1, args.preamble_len - 3), bin_tol=2,
    )
    iq = np.memmap(args.input, dtype="<c8", mode="r")
    with args.sync_csv.open(newline="", encoding="utf-8-sig") as handle:
        events = list(csv.DictReader(handle))
    required = ("event_index", "preamble_ref_bin", "located_preamble_start_sample")
    if events and any(field not in events[0] for field in required):
        raise ValueError("sync CSV must be produced by run_weak_sync_chain.py")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = accepted = attempts = errors = 0
    with args.output.open("x", encoding="utf-8") as output:
        for event in events[:args.max_events]:
            total += 1
            record = dict(event_index=event["event_index"], crc_accept=False, attempts=[])
            try:
                location = SimpleNamespace(**{field: int(event[field]) for field in required[:2]},
                                           preamble_start_sample=int(event[required[2]]))
                frame = run_grlora_frame_sync_validation(
                    iq, location, config, args.preamble_len, args.sync_word,
                    center_freq=args.center_freq, sfd_cfo_mode=args.sfd_cfo_mode,
                )
                candidates = build_residual_cfo_candidates(frame)
                if args.policy == "preamble":
                    candidates = rank_residual_cfo_from_preamble(
                        iq, candidates, sf=args.sf, os_factor=os_factor,
                        preamble_symbols=args.preamble_len,
                    )
                result = decode_residual_cfo_candidates(
                    iq, candidates, sf=args.sf, bw_hz=args.bw, os_factor=os_factor,
                    max_attempts=args.max_attempts, ldro_mode=args.ldro_mode,
                    crc_mode=args.crc_mode, allow_gate_failed_candidate=args.allow_gate_failed,
                )
                record["sync_valid"] = bool(frame.valid)
                record["payload_start_sample"] = frame.fine_payload_start_sample
                record["attempts"] = [dict(
                    delta_cfo_bins=item.candidate.delta_cfo_bins,
                    preamble_score=item.candidate.preamble_score,
                    status=item.decode.status, header_valid=item.decode.header_valid,
                    crc_accept=item.crc_accepted,
                ) for item in result.attempts]
                attempts += len(result.attempts)
                if result.selected is not None:
                    accepted += 1
                    record["crc_accept"] = True
                    record["payload_hex"] = result.selected.decode.payload_bytes.hex()
                    record["selected_delta_cfo_bins"] = result.selected.candidate.delta_cfo_bins
            except ValueError as exc:
                errors += 1
                record["error"] = str(exc)
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()
    summary = dict(events=total, crc_accepted=accepted, decode_attempts=attempts,
                   event_errors=errors, policy=args.policy, output=str(args.output),
                   note="CRC acceptance only; unknown transmitted count, no claim of exact delivery or PDR",
                   config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
