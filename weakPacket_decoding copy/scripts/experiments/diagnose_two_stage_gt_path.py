#!/usr/bin/env python3
"""GT-only diagnostics for the two-stage weak-packet decoder.

This script must not be used as part of decoding.  It uses GT raw bins from the
symbol CSV only after a normal decoder run to measure where the true PHY path is
lost: per-codeword nibble lists, per-block candidates, or global beam scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_two_stage_weak_decoder import (  # noqa: E402
    build_config,
    extract_fft,
    load_packets,
)
from weak_decoder.chirp import bin_to_grlora_symbol, build_downchirp  # noqa: E402
from weak_decoder.header_first_demod import deinterleave_hard, gray_demapping  # noqa: E402
from weak_decoder.payload_codec import hamming_decode_hard  # noqa: E402
from weak_decoder.phase_guided_demod import (  # noqa: E402
    PhaseLine,
    extract_header_anchors,
    fit_phase_line,
)
from weak_decoder.two_stage_weak_decoder import (  # noqa: E402
    TwoStageWeakConfig,
    decode_two_stage_weak_payload,
    enumerate_block_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose where the GT path is lost in two-stage decoding."
    )
    parser.add_argument("-i", "--input-iq", type=Path, required=True)
    parser.add_argument("-s", "--symbol-csv", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)

    parser.add_argument("--top-k-metrics", type=int, default=32)
    parser.add_argument("--amplitude-floor-db", type=float, default=24.0)
    parser.add_argument("--phase-weight", type=float, default=0.0)
    parser.add_argument("--bit-metric", choices=("max", "logsumexp"), default="max")
    parser.add_argument("--nibble-candidates", type=int, default=4)
    parser.add_argument("--row-beam-width", type=int, default=256)
    parser.add_argument("--block-candidate-limit", type=int, default=64)
    parser.add_argument("--global-beam-width", type=int, default=64)
    parser.add_argument("--final-candidate-limit", type=int, default=128)
    parser.add_argument("--block-trim-fraction", type=float, default=0.20)
    parser.add_argument("--projection-trim-fraction", type=float, default=0.10)
    parser.add_argument("--projection-score-weight", type=float, default=1.0)
    parser.add_argument("--crc-observed-bonus", type=float, default=4.0)
    parser.add_argument("--disable-argmax-fallback", action="store_true")
    return parser.parse_args()


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _symbol_sample_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    return int(start_sample) + int(os_value // 2) + os_value * np.arange(n_bins, dtype=np.int64)


def _packet_phase_line(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> PhaseLine:
    header_start = int(packet["header_start_sample"])
    header_abs, header_phases = extract_header_anchors(
        samples=samples,
        header_start_sample=header_start,
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


def _payload_spectra_and_gt(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[int]]:
    sf = int(packet["sf"])
    cfo_total = float(packet["cfo_int"]) + float(packet["cfo_frac"])
    downchirp = build_downchirp(sf, cfo_int=packet["cfo_int"], cfo_frac=packet["cfo_frac"])
    spectra: list[np.ndarray] = []
    gt_bins: list[int] = []
    for symbol in packet["payload_symbols"]:
        try:
            spectra.append(
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
            gt_bins.append(int(symbol.get("gt_bin", -1)))
        except ValueError:
            continue
    return spectra, gt_bins


def _gt_block_codewords(
    gt_symbol_values: list[int],
    block_index: int,
    sf: int,
    cr: int,
    ldro: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    cw_len = int(cr) + 4
    start = int(block_index) * cw_len
    block_symbols = tuple(int(v) for v in gt_symbol_values[start:start + cw_len])
    if len(block_symbols) < cw_len:
        return (), (), ()
    gray = gray_demapping(block_symbols)
    codewords = tuple(
        int(v)
        for v in deinterleave_hard(
            gray,
            sf=int(sf),
            is_header=False,
            cr=int(cr),
            ldro=bool(ldro),
        )
    )
    nibbles = tuple(
        int(v) & 0xF
        for v in hamming_decode_hard(
            codewords,
            is_header=False,
            cr=int(cr),
        )
    )
    return block_symbols, codewords, nibbles


def _candidate_rank(items: list[int], target: int) -> int:
    for idx, item in enumerate(items, start=1):
        if int(item) == int(target):
            return int(idx)
    return 0


def diagnose_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    config: TwoStageWeakConfig,
) -> list[dict[str, Any]]:
    sf = int(packet["sf"])
    cr = int(packet["cr"])
    ldro = bool(packet["ldro"])
    cw_len = cr + 4
    spectra, gt_bins = _payload_spectra_and_gt(samples, packet, args)
    phase_line = _packet_phase_line(samples, packet, args)

    result = decode_two_stage_weak_payload(
        payload_spectra=spectra,
        header_symbol_values=tuple(packet["header_symbols"]),
        phase_line=phase_line,
        payload_symbol_start_abs_index=float(packet.get("preamble_len", args.preamble_len)) + 12.25,
        sf=sf,
        cr=cr,
        payload_len=int(packet["payload_len"]),
        has_crc=bool(packet["has_crc"]),
        ldro=ldro,
        config=config,
    )
    gt_symbol_values = [
        int(bin_to_grlora_symbol(raw_bin, sf=sf, is_header=False, ldro=ldro))
        for raw_bin in gt_bins[: len(result.likelihoods)]
    ]

    wide_config = replace(
        config,
        nibble_candidates_per_codeword=16,
        block_candidate_limit=max(512, int(config.block_candidate_limit)),
        row_beam_width=max(4096, int(config.row_beam_width)),
    )
    rows: list[dict[str, Any]] = []
    for block_index in range(len(result.block_candidates)):
        block_start = block_index * cw_len
        block_likelihoods = result.likelihoods[block_start:block_start + cw_len]
        gt_block_symbols, gt_codewords, gt_nibbles = _gt_block_codewords(
            gt_symbol_values=gt_symbol_values,
            block_index=block_index,
            sf=sf,
            cr=cr,
            ldro=ldro,
        )
        if not gt_block_symbols:
            continue

        full_lists, wide_block_candidates = enumerate_block_candidates(
            block_index=block_index,
            block_likelihoods=block_likelihoods,
            sf=sf,
            cr=cr,
            ldro=ldro,
            config=wide_config,
        )
        default_blocks = list(result.block_candidates[block_index])
        gt_block_in_default = any(
            tuple(item.symbol_values) == gt_block_symbols for item in default_blocks
        )
        gt_block_in_wide = any(
            tuple(item.symbol_values) == gt_block_symbols for item in wide_block_candidates
        )
        gt_block_wide_rank = 0
        for rank, item in enumerate(wide_block_candidates, start=1):
            if tuple(item.symbol_values) == gt_block_symbols:
                gt_block_wide_rank = rank
                break

        codeword_ranks: list[int] = []
        nibble_ranks: list[int] = []
        default_hits: list[int] = []
        for row_index, codeword_list in enumerate(full_lists):
            all_codewords = [int(candidate.codeword) for candidate in codeword_list.candidates]
            all_nibbles = [int(candidate.nibble) for candidate in codeword_list.candidates]
            codeword_ranks.append(_candidate_rank(all_codewords, gt_codewords[row_index]))
            nibble_ranks.append(_candidate_rank(all_nibbles, gt_nibbles[row_index]))
            default_hits.append(
                int(
                    _candidate_rank(
                        all_codewords[: int(config.nibble_candidates_per_codeword)],
                        gt_codewords[row_index],
                    )
                    > 0
                )
            )

        rows.append(
            {
                "packet_index": int(packet["packet_index"]),
                "block_index": int(block_index),
                "gt_codewords_all_in_top_n": int(all(default_hits)),
                "gt_codeword_min_rank": min(codeword_ranks) if codeword_ranks else 0,
                "gt_codeword_max_rank": max(codeword_ranks) if codeword_ranks else 0,
                "gt_codeword_mean_rank": float(np.mean(codeword_ranks)) if codeword_ranks else 0.0,
                "gt_nibble_max_rank": max(nibble_ranks) if nibble_ranks else 0,
                "gt_block_in_default_candidates": int(gt_block_in_default),
                "gt_block_in_wide_candidates": int(gt_block_in_wide),
                "gt_block_wide_rank": int(gt_block_wide_rank),
                "default_block_candidate_count": int(len(default_blocks)),
                "wide_block_candidate_count": int(len(wide_block_candidates)),
                "selected_source": result.selected.selection_source if result.selected else "",
                "selected_crc_valid": int(result.selected.observed_crc_valid) if result.selected else 0,
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


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"block_count": 0}

    def avg(key: str) -> float:
        values = []
        for row in rows:
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else 0.0

    return {
        "block_count": len(rows),
        "gt_codewords_all_in_top_n_rate": avg("gt_codewords_all_in_top_n"),
        "gt_block_in_default_candidates_rate": avg("gt_block_in_default_candidates"),
        "gt_block_in_wide_candidates_rate": avg("gt_block_in_wide_candidates"),
        "mean_gt_codeword_max_rank": avg("gt_codeword_max_rank"),
        "mean_gt_block_wide_rank": avg("gt_block_wide_rank"),
        "mean_default_block_candidate_count": avg("default_block_candidate_count"),
        "mean_wide_block_candidate_count": avg("wide_block_candidate_count"),
    }


def main() -> int:
    args = parse_args()
    samples = np.fromfile(args.input_iq, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {args.input_iq}")
    packets = load_packets(args.symbol_csv, args.packet)
    if not packets:
        raise SystemExit("no header-valid packets found in symbol CSV")

    config = build_config(args)
    rows: list[dict[str, Any]] = []
    for packet_index in sorted(packets):
        print(f"diagnose packet {packet_index}", flush=True)
        rows.extend(diagnose_packet(samples, packets[packet_index], args, config))

    write_csv(args.output, rows)
    summary = build_summary(rows)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
