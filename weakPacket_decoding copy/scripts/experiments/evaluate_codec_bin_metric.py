#!/usr/bin/env python3
"""Evaluate codec-consistent bin candidate sets.

For each payload interleaver block, this script enumerates codec-valid block
candidates from full FFT evidence and projects the top block candidates back to
per-symbol raw FFT bins.  GT bins are used only to measure Top-L recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_two_stage_weak_decoder import build_config, evaluate_packet, load_packets  # noqa: E402
from weak_decoder.two_stage_weak_decoder import decode_two_stage_weak_payload  # noqa: E402
from weak_decoder.phase_guided_demod import extract_header_anchors, fit_phase_line, PhaseLine  # noqa: E402
from weak_decoder.chirp import build_downchirp  # noqa: E402
from run_two_stage_weak_decoder import extract_fft  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate codec-consistent Top-L bin recall.")
    parser.add_argument("-i", "--input-iq", type=Path, required=True)
    parser.add_argument("-s", "--symbol-csv", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--top-l", default="1,4,8,16,32,64,128")

    parser.add_argument("--top-k-metrics", type=int, default=128)
    parser.add_argument("--amplitude-floor-db", type=float, default=24.0)
    parser.add_argument("--phase-weight", type=float, default=0.0)
    parser.add_argument("--bit-metric", choices=("max", "logsumexp"), default="max")
    parser.add_argument("--nibble-candidates", type=int, default=16)
    parser.add_argument("--row-beam-width", type=int, default=4096)
    parser.add_argument("--block-candidate-limit", type=int, default=4096)
    parser.add_argument("--global-beam-width", type=int, default=1)
    parser.add_argument("--final-candidate-limit", type=int, default=1)
    parser.add_argument("--block-trim-fraction", type=float, default=0.20)
    parser.add_argument("--projection-trim-fraction", type=float, default=0.10)
    parser.add_argument("--projection-score-weight", type=float, default=1.0)
    parser.add_argument("--crc-observed-bonus", type=float, default=4.0)
    parser.add_argument("--disable-argmax-fallback", action="store_true")
    return parser.parse_args()


def _parse_int_list(text: str) -> list[int]:
    return [int(float(item.strip())) for item in str(text).split(",") if item.strip()]


def _rank_of(scores: np.ndarray, target_bin: int) -> int:
    target = int(target_bin)
    if target < 0 or target >= scores.size:
        return 0
    order = np.argsort(scores)[::-1]
    hit = np.where(order == target)[0]
    return int(hit[0] + 1) if hit.size else 0


def _phase_line(samples: np.ndarray, packet: dict[str, Any], args: argparse.Namespace) -> PhaseLine:
    header_abs, header_phases = extract_header_anchors(
        samples=samples,
        header_start_sample=int(packet["header_start_sample"]),
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        cfo_correction_mode=str(args.cfo_correction_mode),
        preamble_len=float(packet.get("preamble_len", args.preamble_len)),
        header_symbol_values=tuple(packet["header_symbols"]),
    )
    if header_abs.size >= 2:
        return fit_phase_line(header_abs, header_phases)
    return PhaseLine()


def _spectra(samples: np.ndarray, packet: dict[str, Any], args: argparse.Namespace) -> tuple[list[np.ndarray], list[int]]:
    sf = int(packet["sf"])
    cfo_total = float(packet["cfo_int"]) + float(packet["cfo_frac"])
    downchirp = build_downchirp(sf, cfo_int=packet["cfo_int"], cfo_frac=packet["cfo_frac"])
    out: list[np.ndarray] = []
    gt: list[int] = []
    for symbol in packet["payload_symbols"]:
        try:
            out.append(
                extract_fft(
                    samples=samples,
                    start_sample=int(symbol["start_sample"]),
                    sf=sf,
                    os_factor=int(packet["os_factor"]),
                    downchirp=downchirp,
                    cfo_total=cfo_total,
                    header_start_sample=int(packet["header_start_sample"]),
                    cfo_correction_mode=str(args.cfo_correction_mode),
                )
            )
            gt.append(int(symbol.get("gt_bin", -1)))
        except ValueError:
            continue
    return out, gt


def _first_unique_bins(block_candidates, symbol_offset: int, limit: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for candidate in block_candidates:
        if int(symbol_offset) >= len(candidate.raw_bins):
            continue
        raw_bin = int(candidate.raw_bins[int(symbol_offset)])
        if raw_bin in seen:
            continue
        seen.add(raw_bin)
        out.append(raw_bin)
        if len(out) >= int(limit):
            break
    return out


def evaluate_one_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    top_l: list[int],
) -> list[dict[str, Any]]:
    config = build_config(args)
    spectra, gt_bins = _spectra(samples, packet, args)
    result = decode_two_stage_weak_payload(
        payload_spectra=spectra,
        header_symbol_values=tuple(packet["header_symbols"]),
        phase_line=_phase_line(samples, packet, args),
        payload_symbol_start_abs_index=float(packet.get("preamble_len", args.preamble_len)) + 12.25,
        sf=int(packet["sf"]),
        cr=int(packet["cr"]),
        payload_len=int(packet["payload_len"]),
        has_crc=bool(packet["has_crc"]),
        ldro=bool(packet["ldro"]),
        config=config,
    )
    cw_len = int(packet["cr"]) + 4
    rows: list[dict[str, Any]] = []
    for symbol_index, likelihood in enumerate(result.likelihoods):
        block_index = symbol_index // cw_len
        symbol_offset = symbol_index % cw_len
        if block_index >= len(result.block_candidates):
            continue
        gt_bin = int(gt_bins[symbol_index])
        amp_rank = _rank_of(likelihood.raw_scores, gt_bin)
        max_l = max(top_l)
        codec_bins = _first_unique_bins(
            result.block_candidates[block_index],
            symbol_offset,
            max_l,
        )
        for cutoff in top_l:
            rows.append(
                {
                    "packet_index": int(packet["packet_index"]),
                    "payload_symbol_index": int(symbol_index),
                    "top_l": int(cutoff),
                    "gt_raw_fft_bin": int(gt_bin),
                    "amp_rank": int(amp_rank),
                    "amp_hit": int(0 < amp_rank <= int(cutoff)),
                    "codec_hit": int(gt_bin in set(codec_bins[: int(cutoff)])),
                    "codec_candidate_count": int(min(len(codec_bins), int(cutoff))),
                    "block_candidate_count": int(len(result.block_candidates[block_index])),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["top_l"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for cutoff, items in sorted(grouped.items()):
        summary.append(
            {
                "top_l": int(cutoff),
                "symbol_count": int(len(items)),
                "amp_recall": float(np.mean([int(item["amp_hit"]) for item in items])),
                "codec_recall": float(np.mean([int(item["codec_hit"]) for item in items])),
                "mean_codec_candidate_count": float(
                    np.mean([int(item["codec_candidate_count"]) for item in items])
                ),
                "mean_block_candidate_count": float(
                    np.mean([int(item["block_candidate_count"]) for item in items])
                ),
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    samples = np.fromfile(args.input_iq, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {args.input_iq}")
    packets = load_packets(args.symbol_csv, args.packet)
    if not packets:
        raise SystemExit("no header-valid packets found in symbol CSV")
    top_l = _parse_int_list(args.top_l)
    rows: list[dict[str, Any]] = []
    for packet_index in sorted(packets):
        print(f"evaluate packet {packet_index}", flush=True)
        rows.extend(evaluate_one_packet(samples, packets[packet_index], args, top_l))
    write_csv(args.output, rows)
    summary = build_summary(rows)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
