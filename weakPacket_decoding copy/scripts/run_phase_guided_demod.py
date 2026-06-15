#!/usr/bin/env python3
"""Phase-Guided FFT Demod: compare PILOT against argmax baseline.

Reads sync_chain CSV for framesync-valid packets,
runs phase-guided demod on payload symbols,
compares against header-first GT bins to evaluate accuracy.

Usage:
  python scripts/run_phase_guided_demod.py 
    -i data/USRP_IQ/0_0_0_10_14_16.bin 
    -s weakPacket_decoding/data/weak_sync_chain/sync_chain/0_0_0_10_14_16_sync_chain.csv 
    -g weakPacket_decoding/data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols.csv 
    -o weakPacket_decoding/data/phase_guided/ 
    --sf 10 --bw 125000 --samp-rate 500000 --preamble-len 16
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import product
import json
import math
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parent.parent
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.header_first_demod import (
    decode_explicit_header,
    demod_symbol_sequence,
)
from weak_decoder.payload_codec import (
    encode_explicit_frame_symbols,
    reencoded_payload_known_prefix_symbols,
)
from weak_decoder.phase_guided_demod import (
    PhaseGuidedPayloadConfig,
    PhaseLine,
    phase_guided_demod_packet,
    phase_guided_rescue_header,
    PayloadSymbolDecision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-guided FFT bin selection LoRa demod."
    )
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="complex64 IQ .bin")
    parser.add_argument("-s", "--sync-chain-csv", type=Path, required=True,
                        help="sync_chain CSV")
    parser.add_argument("-g", "--gt-symbol-csv", type=Path, default=None,
                        help="header-first symbol CSV (GT bins)")
    parser.add_argument("-o", "--output-dir", type=Path, required=True,
                        help="output directory")
    parser.add_argument("--sf", type=int, default=10,
                        help="spreading factor")
    parser.add_argument("--bw", type=float, default=125000.0,
                        help="bandwidth Hz")
    parser.add_argument("--samp-rate", type=float, default=500000.0,
                        help="IQ sample rate")
    parser.add_argument("--preamble-len", type=float, default=8.0,
                        help="preamble length in symbols")
    parser.add_argument("--os-factor", type=int, default=4,
                        help="oversampling factor")
    parser.add_argument("--packet", type=int, nargs="*", default=None,
                        help="only specific packet indices")
    parser.add_argument("--max-packets", type=int, default=None,
                        help="max packets to process")
    parser.add_argument("--top-l", type=int, default=64,
                        help="max candidates per symbol")
    parser.add_argument("--argmax-window", type=int, default=3,
                        help="argmax neighborhood radius")
    parser.add_argument("--refinement-rounds", type=int, default=3,
                        help="max refinement rounds")
    parser.add_argument("--trim-frac", type=float, default=0.25,
                        help="trim fraction for robust fit")
    parser.add_argument("--confidence-threshold", type=float, default=0.4,
                        help="pseudo-anchor confidence threshold")
    parser.add_argument("--phase-weight", type=float, default=0.85,
                        help="phase weight in combined score [0-1]")
    parser.add_argument("--enable-payload-hough", action="store_true", default=False,
                        help="enable experimental payload-native Hough phase-line bootstrap")
    parser.add_argument("--hough-slope-span-pi", type=float, default=0.25,
                        help="payload Hough slope search half-span in pi/symbol")
    parser.add_argument("--hough-slope-steps", type=int, default=81,
                        help="payload Hough slope grid size")
    parser.add_argument("--hough-intercept-bins", type=int, default=96,
                        help="payload Hough intercept grid size")
    parser.add_argument("--min-anchor-r2-for-phase", type=float, default=0.5,
                        help="minimum phase-line R2 required before payload phase correction")
    parser.add_argument("--enable-preamble-profile-score", action="store_true", default=False,
                        help="add preamble-learned in-chirp phase-profile score to FFT-bin candidates")
    parser.add_argument("--preamble-profile-weight", type=float, default=0.15,
                        help="weight of preamble phase-profile score in candidate scoring")
    parser.add_argument("--min-preamble-profile-quality", type=float, default=0.25,
                        help="minimum profile quality before applying preamble-profile score")
    parser.add_argument("--enable-block-code-search", action="store_true", default=False,
                        help="enable experimental payload interleaver-block parity-aware refinement")
    parser.add_argument("--block-candidates-per-symbol", type=int, default=4,
                        help="semantic candidates per payload symbol for block search")
    parser.add_argument("--block-parity-bonus", type=float, default=0.45,
                        help="score bonus for parity-valid payload codewords")
    parser.add_argument("--block-min-gain", type=float, default=0.05,
                        help="minimum block score gain before accepting block refinement")
    parser.add_argument("--aggressive-phase", action="store_true", default=False,
                        help="disable conservative argmax guard for phase-only experiments")
    parser.add_argument("--energy-threshold-db", type=float, default=12.0,
                        help="energy threshold for candidate expansion (dB)")
    parser.add_argument("--cfo-correction-mode",
                        choices=("symbol", "continuous"), default="continuous")
    parser.add_argument("--skip-header-anchor", action="store_true", default=False,
                        help="do not use header symbols as anchors")
    parser.add_argument("--disable-header-rescue", action="store_true", default=False,
                        help="disable phase/checksum beam rescue when argmax header fails")
    parser.add_argument("--header-candidates-per-symbol", type=int, default=10,
                        help="semantic header candidates retained per symbol")
    parser.add_argument("--header-beam-width", type=int, default=4096,
                        help="beam width for constrained header rescue")
    parser.add_argument("--header-phase-weight", type=float, default=0.55,
                        help="phase weight for header rescue candidate ranking")
    parser.add_argument("--header-max-payload-len", type=int, default=64,
                        help="max plausible explicit-header payload length")
    parser.add_argument("--expected-payload-len", type=int, default=None,
                        help="optional session prior: exact payload length")
    parser.add_argument("--expected-cr", type=int, default=None,
                        help="optional session prior: exact PHY CR field")
    parser.add_argument("--expected-has-crc", type=int, choices=(0, 1), default=None,
                        help="optional session prior: exact CRC flag")
    parser.add_argument("--expected-header-symbols", type=str, default=None,
                        help="optional session prior: comma-separated 8 raw header symbols")
    parser.add_argument("--expected-payload-symbols", type=str, default=None,
                        help="optional session prior: comma-separated payload symbols, -1 for unknown")
    parser.add_argument("--payload-template-file", type=Path, default=None,
                        help="JSON file with expected_payload_symbols")
    parser.add_argument("--byte-template-file", type=Path, default=None,
                        help="JSON file with expected_payload_bytes")
    parser.add_argument("--dynamic-byte-model-file", type=Path, default=None,
                        help="JSON file with affine dynamic byte models")
    parser.add_argument("--enable-byte-prior-symbols", action="store_true", default=False,
                        help="re-encode byte/template priors into per-packet payload symbol priors")
    parser.add_argument("--enable-byte-residual-search", action="store_true", default=False,
                        help="enumerate unknown template bytes, re-encode candidates, and select by phase-consistent FFT evidence")
    parser.add_argument("--residual-byte-index", type=int, nargs="*", default=None,
                        help="unknown byte indices to enumerate; default is all -1 template bytes")
    parser.add_argument("--residual-byte-values", type=str, default="0-255",
                        help="candidate values for enumerated bytes, e.g. 0-255 or 1,3,5")
    parser.add_argument("--residual-max-unknown-bytes", type=int, default=1,
                        help="maximum unknown byte dimensions to brute-force")
    parser.add_argument("--residual-max-candidates", type=int, default=4096,
                        help="maximum re-encoded byte-residual candidates per packet")
    parser.add_argument("--candidate-search-min-known", type=int, default=3,
                        help="minimum known symbols required to score a payload candidate")
    parser.add_argument("--candidate-search-phase-weight", type=float, default=0.25)
    parser.add_argument("--candidate-search-line-weight", type=float, default=0.50)
    parser.add_argument("--candidate-search-amp-weight", type=float, default=0.20)
    parser.add_argument("--candidate-search-profile-weight", type=float, default=0.05)
    parser.add_argument("--candidate-search-prior-weight", type=float, default=0.35,
                        help="soft weight for dynamic-byte model priors during residual search")
    parser.add_argument("--candidate-search-score-mode",
                        choices=("bounded", "map"), default="bounded",
                        help="residual candidate scoring: bounded weighted score or MAP-like log-likelihood")
    parser.add_argument("--candidate-search-phase-kappa", type=float, default=2.0,
                        help="von-Mises phase concentration used by --candidate-search-score-mode map")
    parser.add_argument("--candidate-search-adaptive-kappa", action="store_true", default=False,
                        help="estimate MAP phase kappa from packet phase-line residual quality")
    parser.add_argument("--candidate-search-kappa-source",
                        choices=("scoring_line", "initial_line"), default="scoring_line",
                        help="phase line used to estimate adaptive kappa")
    parser.add_argument("--candidate-search-kappa-min", type=float, default=1.0,
                        help="minimum adaptive kappa")
    parser.add_argument("--candidate-search-kappa-max", type=float, default=4.0,
                        help="maximum adaptive kappa")
    parser.add_argument("--candidate-search-kappa-sigma-floor-pi", type=float, default=0.12,
                        help="minimum phase sigma, in pi units, used when estimating adaptive kappa")
    parser.add_argument("--candidate-search-amp-log-floor", type=float, default=1e-6,
                        help="amplitude/profile floor before log in MAP candidate scoring")
    parser.add_argument("--residual-model-prior-sigma", type=float, default=0.5,
                        help="circular-byte Gaussian sigma for model-centered residual candidates")
    parser.add_argument("--write-residual-candidates", action="store_true", default=False,
                        help="write all residual byte candidate likelihoods to CSV")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed")
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    v = str(row.get(key, "")).strip()
    return int(float(v)) if v else int(default)


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    v = str(row.get(key, "")).strip()
    return float(v) if v else float(default)


def _flag(row: dict[str, str], key: str, default: bool = False) -> bool:
    v = str(row.get(key, "")).strip()
    return bool(int(float(v))) if v else bool(default)


def _group_key(row: dict[str, str]) -> tuple[int, int]:
    return (_int(row, "packet_index", -1), _int(row, "event_index", -1))


def read_sync_candidates(
    path: Path,
    packet_filter: Optional[set[int]],
    max_packets: Optional[int],
) -> list[dict]:
    candidates = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not _flag(row, "grlora_framesync_valid", False):
                continue
            pk = _int(row, "packet_index", -1)
            if packet_filter is not None and pk not in packet_filter:
                continue
            candidates.append(dict(row))
    candidates.sort(key=lambda r: (_int(r, "packet_index"), _int(r, "event_index")))
    if max_packets is not None:
        candidates = candidates[:int(max_packets)]
    return candidates


def read_gt_symbols(
    path: Path,
    packet_filter: Optional[set[int]],
) -> dict[tuple[int, int], dict[int, int]]:
    """Returns {(packet_idx, event_idx): {payload_symbol_index: gt_raw_fft_bin}}."""
    result: dict[tuple[int, int], dict[int, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (str(row.get("stage", "")).strip() != "payload"
                    or _int(row, "header_valid", 0) != 1):
                continue
            key = _group_key(row)
            pk = _int(row, "packet_index", -1)
            if packet_filter is not None and pk not in packet_filter:
                continue
            if key not in result:
                result[key] = {}
            sym_idx = _int(row, "stage_symbol_index", -1)
            gt_bin = _int(row, "raw_fft_bin", -1)
            if gt_bin >= 0:
                result[key][sym_idx] = gt_bin
    return result


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Optional[list[str]] = None,
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fn = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fn})


def _load_reconstruction_priors(
    byte_template_file: Optional[Path],
    dynamic_model_file: Optional[Path],
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    byte_template: list[int] = []
    models: dict[int, dict[str, Any]] = {}
    if byte_template_file:
        with byte_template_file.resolve().open("r", encoding="utf-8") as f:
            doc = json.load(f)
        byte_template = [int(v) for v in doc.get("expected_payload_bytes", [])]
    if dynamic_model_file:
        with dynamic_model_file.resolve().open("r", encoding="utf-8") as f:
            doc = json.load(f)
        models = {int(m["byte_index"]): m for m in doc.get("models", [])}
    return byte_template, models


def _reconstruct_payload_hex(
    packet_index: int,
    byte_template: list[int],
    models: dict[int, dict[str, Any]],
) -> tuple[str, int]:
    if not byte_template:
        return "", 0
    payload: list[int] = []
    unknown = 0
    for idx, value in enumerate(byte_template):
        if int(value) >= 0:
            payload.append(int(value) & 0xFF)
            continue
        model = models.get(idx)
        if model and model.get("model") == "affine_mod_256":
            payload.append(
                (
                    int(model["slope"]) * int(packet_index)
                    + int(model["intercept"])
                ) & 0xFF
            )
        else:
            payload.append(0)
            unknown += 1
    return bytes(payload).hex(), int(unknown)


def _payload_bytes_from_priors(
    packet_index: int,
    byte_template: list[int],
    models: dict[int, dict[str, Any]],
) -> tuple[bytes, int]:
    payload: list[int] = []
    unknown = 0
    for idx, value in enumerate(byte_template):
        if int(value) >= 0:
            payload.append(int(value) & 0xFF)
            continue
        model = models.get(idx)
        if model and model.get("model") == "affine_mod_256":
            payload.append(
                (
                    int(model["slope"]) * int(packet_index)
                    + int(model["intercept"])
                ) & 0xFF
            )
        else:
            payload.append(0)
            unknown += 1
    return bytes(payload), int(unknown)


def _merge_symbol_priors(
    base: Optional[tuple[int, ...]],
    override: list[int],
) -> tuple[int, ...]:
    size = max(len(base or ()), len(override))
    merged = [-1 for _ in range(size)]
    if base is not None:
        for idx, value in enumerate(base):
            merged[idx] = int(value)
    for idx, value in enumerate(override):
        if int(value) >= 0:
            merged[idx] = int(value)
    return tuple(merged)


def _parse_byte_values(spec: str) -> list[int]:
    values: list[int] = []
    for part in str(spec).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            lo_s, hi_s = item.split("-", 1)
            lo = int(lo_s.strip())
            hi = int(hi_s.strip())
            step = 1 if hi >= lo else -1
            values.extend(range(lo, hi + step, step))
        else:
            values.append(int(item))
    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        byte = int(value) & 0xFF
        if byte not in seen:
            seen.add(byte)
            deduped.append(byte)
    return deduped


def _payload_byte_candidates_from_template(
    packet_index: int,
    byte_template: list[int],
    models: dict[int, dict[str, Any]],
    residual_indices: Optional[list[int]],
    value_options: list[int],
    max_unknown_bytes: int,
    max_candidates: int,
    model_prior_sigma: float,
) -> tuple[list[bytes], int, list[float]]:
    """Enumerate complete payload byte candidates for residual search."""
    if not byte_template:
        return [], 0, []
    if not value_options:
        return [], 0, []

    requested = set(int(v) for v in residual_indices) if residual_indices else None
    payload: list[int] = []
    search_indices: list[int] = []
    for idx, value in enumerate(byte_template):
        if int(value) >= 0:
            payload.append(int(value) & 0xFF)
            continue
        if requested is None or idx in requested:
            payload.append(0)
            search_indices.append(idx)
            continue
        model = models.get(idx)
        if model and model.get("model") == "affine_mod_256":
            payload.append(
                (
                    int(model["slope"]) * int(packet_index)
                    + int(model["intercept"])
                ) & 0xFF
            )
        else:
            payload.append(0)
            search_indices.append(idx)

    if not search_indices or len(search_indices) > int(max_unknown_bytes):
        return [], int(len(search_indices)), []

    candidates: list[bytes] = []
    priors: list[float] = []
    for combo in product(value_options, repeat=len(search_indices)):
        cand = list(payload)
        prior_parts: list[float] = []
        for idx, value in zip(search_indices, combo):
            cand[int(idx)] = int(value) & 0xFF
            model = models.get(int(idx))
            if model and model.get("model") == "affine_mod_256":
                center = (
                    int(model["slope"]) * int(packet_index)
                    + int(model["intercept"])
                ) & 0xFF
                raw_dist = abs((int(value) & 0xFF) - center)
                dist = min(raw_dist, 256 - raw_dist)
                sigma = max(1e-6, float(model_prior_sigma))
                prior_parts.append(float(math.exp(-0.5 * ((dist / sigma) ** 2))))
        candidates.append(bytes(cand))
        priors.append(float(sum(prior_parts) / len(prior_parts)) if prior_parts else 0.0)
        if len(candidates) >= int(max_candidates):
            break
    return candidates, int(len(search_indices)), priors


def _encode_payload_candidate_symbols(
    payload_candidates: list[bytes],
    base_symbols: Optional[tuple[int, ...]],
    sf: int,
    cr: int,
    has_crc: bool,
    ldro: bool,
) -> tuple[list[tuple[int, ...]], int]:
    symbol_candidates: list[tuple[int, ...]] = []
    known_count = 0
    for payload in payload_candidates:
        _, encoded_payload_symbols = encode_explicit_frame_symbols(
            payload,
            sf=int(sf),
            cr=int(cr),
            has_crc=bool(has_crc),
            ldro=bool(ldro),
        )
        prefix_len = reencoded_payload_known_prefix_symbols(
            payload_len=len(payload),
            has_crc=bool(has_crc),
            sf=int(sf),
            cr=int(cr),
            ldro=bool(ldro),
        )
        encoded_prior = [
            int(v) if idx < int(prefix_len) else -1
            for idx, v in enumerate(encoded_payload_symbols)
        ]
        known_count = max(known_count, sum(1 for value in encoded_prior if int(value) >= 0))
        symbol_candidates.append(_merge_symbol_priors(base_symbols, encoded_prior))
    return symbol_candidates, int(known_count)


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        np.random.seed(int(args.seed))

    input_path = args.input.resolve()
    sync_csv = args.sync_chain_csv.resolve()
    gt_csv = args.gt_symbol_csv.resolve() if args.gt_symbol_csv else None
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    byte_template, dynamic_byte_models = _load_reconstruction_priors(
        args.byte_template_file,
        args.dynamic_byte_model_file,
    )

    samples = np.fromfile(input_path, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"Empty IQ file: {input_path}")

    sf = int(args.sf)
    os_factor = int(args.os_factor)
    n_bins = 1 << sf
    chirp_samples = n_bins * os_factor
    preamble_len = float(args.preamble_len)

    packet_filter = set(args.packet) if args.packet else None
    candidates = read_sync_candidates(
        sync_csv, packet_filter=packet_filter, max_packets=args.max_packets,
    )
    gt_map = read_gt_symbols(gt_csv, packet_filter=packet_filter) if gt_csv else {}

    expected_header_symbols = None
    if args.expected_header_symbols:
        expected_header_symbols = tuple(
            int(item.strip()) for item in str(args.expected_header_symbols).split(",")
            if item.strip()
        )
        if len(expected_header_symbols) != 8:
            raise ValueError("--expected-header-symbols needs exactly 8 integers")

    expected_payload_symbols = None
    if args.expected_payload_symbols:
        expected_payload_symbols = tuple(
            int(item.strip()) for item in str(args.expected_payload_symbols).split(",")
            if item.strip()
        )
    if args.payload_template_file:
        with args.payload_template_file.resolve().open("r", encoding="utf-8") as f:
            payload_template_doc = json.load(f)
        if isinstance(payload_template_doc, dict):
            payload_values = payload_template_doc.get("expected_payload_symbols", [])
        else:
            payload_values = payload_template_doc
        expected_payload_symbols = tuple(int(v) for v in payload_values)

    config = PhaseGuidedPayloadConfig(
        top_l=int(args.top_l),
        max_refinement_rounds=int(args.refinement_rounds),
        trim_frac=float(args.trim_frac),
        confidence_threshold=float(args.confidence_threshold),
        argmax_window_radius=int(args.argmax_window),
        energy_threshold_db=float(args.energy_threshold_db),
        use_header_anchor=not bool(args.skip_header_anchor),
        phase_weight=float(args.phase_weight),
        use_payload_hough_line=bool(args.enable_payload_hough),
        hough_slope_span_pi=float(args.hough_slope_span_pi),
        hough_slope_steps=int(args.hough_slope_steps),
        hough_intercept_bins=int(args.hough_intercept_bins),
        min_anchor_r2_for_phase=float(args.min_anchor_r2_for_phase),
        use_block_code_search=bool(args.enable_block_code_search),
        block_candidates_per_symbol=int(args.block_candidates_per_symbol),
        block_parity_bonus=float(args.block_parity_bonus),
        block_min_gain=float(args.block_min_gain),
        conservative_argmax_guard=not bool(args.aggressive_phase),
        use_preamble_profile_score=bool(args.enable_preamble_profile_score),
        preamble_profile_weight=float(args.preamble_profile_weight),
        min_preamble_profile_quality=float(args.min_preamble_profile_quality),
        candidate_search_min_known=int(args.candidate_search_min_known),
        candidate_search_phase_weight=float(args.candidate_search_phase_weight),
        candidate_search_line_weight=float(args.candidate_search_line_weight),
        candidate_search_amp_weight=float(args.candidate_search_amp_weight),
        candidate_search_profile_weight=float(args.candidate_search_profile_weight),
        candidate_search_prior_weight=float(args.candidate_search_prior_weight),
        candidate_search_score_mode=str(args.candidate_search_score_mode),
        candidate_search_phase_kappa=float(args.candidate_search_phase_kappa),
        candidate_search_adaptive_kappa=bool(args.candidate_search_adaptive_kappa),
        candidate_search_kappa_source=str(args.candidate_search_kappa_source),
        candidate_search_kappa_min=float(args.candidate_search_kappa_min),
        candidate_search_kappa_max=float(args.candidate_search_kappa_max),
        candidate_search_kappa_sigma_floor_pi=float(args.candidate_search_kappa_sigma_floor_pi),
        candidate_search_amp_log_floor=float(args.candidate_search_amp_log_floor),
        enable_header_rescue=not bool(args.disable_header_rescue),
        header_candidates_per_symbol=int(args.header_candidates_per_symbol),
        header_beam_width=int(args.header_beam_width),
        header_phase_weight=float(args.header_phase_weight),
        header_max_payload_len=int(args.header_max_payload_len),
        expected_payload_len=args.expected_payload_len,
        expected_cr=args.expected_cr,
        expected_has_crc=(
            bool(args.expected_has_crc)
            if args.expected_has_crc is not None else None
        ),
        expected_header_symbols=expected_header_symbols,
        expected_payload_symbols=expected_payload_symbols,
    )
    residual_value_options = _parse_byte_values(args.residual_byte_values)
    residual_indices = (
        [int(v) for v in args.residual_byte_index]
        if args.residual_byte_index is not None else None
    )

    summary_rows: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    residual_candidate_rows: list[dict[str, Any]] = []
    total_symbols_processed = 0
    total_correct_processed = 0
    print(f"Processing {len(candidates)} framesync-valid packets...")

    for idx, cand in enumerate(candidates):
        packet_idx = _int(cand, "packet_index")
        event_idx = _int(cand, "event_index")
        fine_payload_start = _int(cand, "grlora_fine_payload_start_sample")
        cfo_int = _int(cand, "grlora_cfo_int_est", 0)
        cfo_frac = _float(cand, "grlora_cfo_frac_est", 0.0)
        sfo_hat = _float(cand, "grlora_sfo_hat", 0.0)
        cfo_correction_mode = str(args.cfo_correction_mode)

        # ── Demod header (8 symbols, argmax) ────────────────────────────
        header_results = demod_symbol_sequence(
            samples=samples,
            header_start_sample=int(fine_payload_start),
            sf=sf,
            os_factor=os_factor,
            cfo_int=cfo_int,
            cfo_frac=cfo_frac,
            sfo_hat=sfo_hat,
            sfo_cum_initial=0.0,
            header_count=8,
            payload_count=0,
            payload_ldro=False,
            cfo_correction_mode=cfo_correction_mode,
        )

        if len(header_results) < 8:
            print(f"  Packet {packet_idx}, event {event_idx}: "
                  f"header demod incomplete, skipping")
            continue

        header_symbol_values = [r.symbol_value for r in header_results]
        header_decode = decode_explicit_header(
            header_symbol_values,
            sf=sf,
            bw=float(args.bw),
            ldro_mode=2,
        )
        header_method = "argmax"
        header_rescue = None

        if expected_header_symbols is not None:
            header_symbol_values = list(expected_header_symbols)
            header_decode = decode_explicit_header(
                header_symbol_values,
                sf=sf,
                bw=float(args.bw),
                ldro_mode=2,
            )
            header_method = "session_header_prior_forced"

        if not header_decode.header_valid:
            header_rescue = phase_guided_rescue_header(
                samples=samples,
                header_start_sample=int(fine_payload_start),
                sf=sf,
                bw=float(args.bw),
                os_factor=os_factor,
                cfo_int=cfo_int,
                cfo_frac=cfo_frac,
                sfo_hat=sfo_hat,
                preamble_len=preamble_len,
                config=config,
                cfo_correction_mode=cfo_correction_mode,
            )
            if header_rescue.success:
                header_symbol_values = list(header_rescue.symbol_values)
                header_decode = decode_explicit_header(
                    header_symbol_values,
                    sf=sf,
                    bw=float(args.bw),
                    ldro_mode=2,
                )
                header_method = str(header_rescue.method)

        if not header_decode.header_valid:
            reason = header_rescue.method if header_rescue is not None else "argmax"
            print(f"  Packet {packet_idx}, event {event_idx}: "
                  f"header invalid ({reason}), skipping")
            continue

        payload_ldro = header_decode.ldro
        payload_count = header_decode.payload_symbol_count

        # ── GT bins for this packet ──────────────────────────────────────
        gt_bins = gt_map.get((packet_idx, event_idx), {}) if gt_map else {}

        packet_payload_bytes = b""
        packet_payload_unknown_bytes = 0
        packet_byte_symbol_prior_known = 0
        packet_residual_candidate_count = 0
        packet_residual_unknown_bytes = 0
        packet_residual_symbol_known = 0
        packet_expected_payload_symbols = expected_payload_symbols
        packet_expected_payload_symbol_candidates: Optional[tuple[tuple[int, ...], ...]] = None
        packet_residual_payload_hexes: list[str] = []
        packet_residual_candidate_values: list[str] = []
        packet_residual_candidate_priors: Optional[tuple[float, ...]] = None
        if bool(args.enable_byte_prior_symbols) and byte_template:
            packet_payload_bytes, packet_payload_unknown_bytes = _payload_bytes_from_priors(
                packet_idx,
                byte_template,
                dynamic_byte_models,
            )
            if (
                packet_payload_bytes
                and packet_payload_unknown_bytes == 0
                and len(packet_payload_bytes) == int(header_decode.payload_len)
            ):
                _, encoded_payload_symbols = encode_explicit_frame_symbols(
                    packet_payload_bytes,
                    sf=sf,
                    cr=int(header_decode.cr),
                    has_crc=bool(header_decode.has_crc),
                    ldro=bool(header_decode.ldro),
                )
                prefix_len = reencoded_payload_known_prefix_symbols(
                    payload_len=int(header_decode.payload_len),
                    has_crc=bool(header_decode.has_crc),
                    sf=sf,
                    cr=int(header_decode.cr),
                    ldro=bool(header_decode.ldro),
                )
                encoded_prior = [
                    int(v) if idx < int(prefix_len) else -1
                    for idx, v in enumerate(encoded_payload_symbols)
                ]
                packet_byte_symbol_prior_known = sum(
                    1 for value in encoded_prior if int(value) >= 0
                )
                packet_expected_payload_symbols = _merge_symbol_priors(
                    expected_payload_symbols,
                    encoded_prior,
                )
        if bool(args.enable_byte_residual_search) and byte_template:
            payload_candidates, packet_residual_unknown_bytes, candidate_priors = (
                _payload_byte_candidates_from_template(
                    packet_index=packet_idx,
                    byte_template=byte_template,
                    models=dynamic_byte_models,
                    residual_indices=residual_indices,
                    value_options=residual_value_options,
                    max_unknown_bytes=int(args.residual_max_unknown_bytes),
                    max_candidates=int(args.residual_max_candidates),
                    model_prior_sigma=float(args.residual_model_prior_sigma),
                )
            )
            packet_residual_candidate_count = len(payload_candidates)
            packet_residual_payload_hexes = [payload.hex() for payload in payload_candidates]
            value_indexes = (
                residual_indices if residual_indices is not None
                else [idx for idx, value in enumerate(byte_template) if int(value) < 0]
            )
            packet_residual_candidate_values = []
            for payload in payload_candidates:
                vals = [
                    f"{int(byte_idx)}:{int(payload[int(byte_idx)]):02x}"
                    for byte_idx in value_indexes
                    if 0 <= int(byte_idx) < len(payload)
                ]
                packet_residual_candidate_values.append(";".join(vals))
            packet_residual_candidate_priors = tuple(float(v) for v in candidate_priors)
            if payload_candidates:
                symbol_candidates, packet_residual_symbol_known = (
                    _encode_payload_candidate_symbols(
                        payload_candidates=payload_candidates,
                        base_symbols=expected_payload_symbols,
                        sf=sf,
                        cr=int(header_decode.cr),
                        has_crc=bool(header_decode.has_crc),
                        ldro=bool(header_decode.ldro),
                    )
                )
                packet_expected_payload_symbol_candidates = tuple(symbol_candidates)

        packet_config = config
        if (
            packet_expected_payload_symbols is not expected_payload_symbols
            or packet_expected_payload_symbol_candidates is not None
        ):
            packet_config = replace(
                config,
                expected_payload_symbols=packet_expected_payload_symbols,
                expected_payload_symbol_candidates=packet_expected_payload_symbol_candidates,
                expected_payload_candidate_prior_scores=packet_residual_candidate_priors,
            )

        # ── Run PILOT phase-guided demod ─────────────────────────────────
        result = phase_guided_demod_packet(
            samples=samples,
            header_start_sample=int(fine_payload_start),
            sf=sf,
            os_factor=os_factor,
            cfo_int=cfo_int,
            cfo_frac=cfo_frac,
            sfo_hat=sfo_hat,
            preamble_len=preamble_len,
            header_symbol_values=header_symbol_values,
            header_payload_len=header_decode.payload_len,
            header_cr=header_decode.cr,
            header_has_crc=header_decode.has_crc,
            header_ldro=payload_ldro,
            payload_symbol_count=payload_count,
            config=packet_config,
            cfo_correction_mode=cfo_correction_mode,
            gt_bins=gt_bins,
        )

        # ── Summary row ─────────────────────────────────────────────────
        il = result.get("initial_phase_line", PhaseLine())
        fl = result.get("final_phase_line", PhaseLine())
        round_acc = result.get("round_accuracy", [])
        ser_final = result.get("final_error_rate", 1.0)
        correct_count = result.get("correct_count", 0)
        total_payload = result.get("total_payload", 0)
        prior_search = result.get("payload_prior_search")

        total_symbols_processed += total_payload
        total_correct_processed += correct_count
        reconstructed_payload_hex, reconstructed_unknown_bytes = _reconstruct_payload_hex(
            packet_idx,
            byte_template,
            dynamic_byte_models,
        )
        prior_search_payload_hex = ""
        if (
            prior_search is not None
            and bool(getattr(prior_search, "success", False))
            and 0 <= int(getattr(prior_search, "selected_index", -1)) < len(packet_residual_payload_hexes)
        ):
            prior_search_payload_hex = packet_residual_payload_hexes[
                int(getattr(prior_search, "selected_index"))
            ]
            reconstructed_payload_hex = prior_search_payload_hex
            reconstructed_unknown_bytes = 0

        summary_rows.append({
            "packet_index": packet_idx,
            "event_index": event_idx,
            "success": int(result.get("success", False)),
            "total_payload": total_payload,
            "correct_count": correct_count,
            "pre_block_correct_count": result.get("pre_block_correct_count", ""),
            "pre_template_correct_count": result.get("pre_template_correct_count", ""),
            "block_code_search_enabled": result.get("block_code_search_enabled", 0),
            "payload_template_known": result.get("payload_template_known", 0),
            "byte_symbol_prior_known": packet_byte_symbol_prior_known,
            "byte_residual_candidates": packet_residual_candidate_count,
            "byte_residual_unknown_bytes": packet_residual_unknown_bytes,
            "byte_residual_symbol_known": packet_residual_symbol_known,
            "prior_search_success": int(bool(getattr(prior_search, "success", False))),
            "prior_search_selected": getattr(prior_search, "selected_index", ""),
            "prior_search_payload_hex": prior_search_payload_hex,
            "prior_search_candidates": getattr(prior_search, "candidates_evaluated", ""),
            "prior_search_known_symbols": getattr(prior_search, "known_symbols", ""),
            "prior_search_score": (
                f"{float(getattr(prior_search, 'score', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_signal_score": (
                f"{float(getattr(prior_search, 'signal_score', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_score_mode": (
                str(getattr(prior_search, "score_mode", ""))
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_effective_kappa": (
                f"{float(getattr(prior_search, 'effective_kappa', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_kappa_source": (
                str(args.candidate_search_kappa_source)
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_kappa_source_line_rmse_pi": (
                f"{float(getattr(prior_search, 'kappa_source_line_rmse_pi', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_prior_score": (
                f"{float(getattr(prior_search, 'prior_score', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_margin": (
                f"{float(getattr(prior_search, 'margin', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_phase_score": (
                f"{float(getattr(prior_search, 'mean_phase_score', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_amp_score": (
                f"{float(getattr(prior_search, 'mean_amp_score', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_profile_score": (
                f"{float(getattr(prior_search, 'mean_profile_score', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_line_rmse_pi": (
                f"{float(getattr(prior_search, 'line_rmse_pi', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_line_r2": (
                f"{float(getattr(prior_search, 'line_r2', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_map_phase_ll": (
                f"{float(getattr(prior_search, 'map_phase_ll', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_map_line_ll": (
                f"{float(getattr(prior_search, 'map_line_ll', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "prior_search_map_amp_ll": (
                f"{float(getattr(prior_search, 'map_amp_ll', 0.0)):.6f}"
                if getattr(prior_search, "success", False) else ""
            ),
            "final_error_rate": f"{ser_final:.4f}",
            "initial_slope_pi": f"{il.slope_pi:.4f}",
            "final_slope_pi": f"{fl.slope_pi:.4f}",
            "initial_line_r2": f"{il.fit_r2:.4f}",
            "final_line_r2": f"{fl.fit_r2:.4f}",
            "preamble_anchors": result.get("preamble_anchor_count", 0),
            "header_anchors": result.get("header_anchor_count", 0),
            "header_method": header_method,
            "header_rescue_score": (
                f"{header_rescue.score:.4f}"
                if header_rescue is not None and header_rescue.success else ""
            ),
            "header_rescue_visited": (
                header_rescue.candidates_visited
                if header_rescue is not None else ""
            ),
            "preamble_line_slope_pi": f"{float(result.get('preamble_line_slope_pi', 0)):.4f}",
            "header_line_slope_pi": f"{float(result.get('header_line_slope_pi', 0)):.4f}",
            "hough_line_slope_pi": f"{float(result.get('hough_line_slope_pi', 0)):.4f}",
            "hough_score": f"{float(result.get('hough_score', 0)):.4f}",
            "preamble_profile_quality": f"{float(result.get('preamble_profile_quality', 0)):.4f}",
            "preamble_profile_anchors": result.get("preamble_profile_anchors", 0),
            "preamble_line_r2": f"{float(result.get('preamble_line_r2', 0)):.4f}",
            "header_line_r2": f"{float(result.get('header_line_r2', 0)):.4f}",
            "refinement_rounds": result.get("refinement_rounds", 0),
            "round0_acc": f"{round_acc[0]:.4f}" if len(round_acc) > 0 else "",
            "round1_acc": f"{round_acc[1]:.4f}" if len(round_acc) > 1 else "",
            "round2_acc": f"{round_acc[2]:.4f}" if len(round_acc) > 2 else "",
            "payload_len": header_decode.payload_len,
            "gt_bins_available": int(len(gt_bins)),
            "reconstructed_payload_hex": reconstructed_payload_hex,
            "reconstructed_unknown_bytes": reconstructed_unknown_bytes,
        })

        if bool(args.write_residual_candidates) and prior_search is not None:
            for cand_score in getattr(prior_search, "candidate_scores", ()):
                cand_idx = int(getattr(cand_score, "candidate_index", -1))
                residual_candidate_rows.append({
                    "packet_index": packet_idx,
                    "event_index": event_idx,
                    "candidate_index": cand_idx,
                    "selected": int(cand_idx == int(getattr(prior_search, "selected_index", -1))),
                    "candidate_values": (
                        packet_residual_candidate_values[cand_idx]
                        if 0 <= cand_idx < len(packet_residual_candidate_values) else ""
                    ),
                    "candidate_payload_hex": (
                        packet_residual_payload_hexes[cand_idx]
                        if 0 <= cand_idx < len(packet_residual_payload_hexes) else ""
                    ),
                    "score": f"{float(getattr(cand_score, 'score', 0.0)):.9f}",
                    "signal_score": f"{float(getattr(cand_score, 'signal_score', 0.0)):.9f}",
                    "prior_score": f"{float(getattr(cand_score, 'prior_score', 0.0)):.9f}",
                    "score_mode": str(getattr(cand_score, "score_mode", "")),
                    "effective_kappa": f"{float(getattr(cand_score, 'effective_kappa', 0.0)):.9f}",
                    "kappa_source": str(args.candidate_search_kappa_source),
                    "kappa_source_line_rmse_pi": (
                        f"{float(getattr(cand_score, 'kappa_source_line_rmse_pi', 0.0)):.9f}"
                    ),
                    "known_symbols": int(getattr(cand_score, "known_symbols", 0)),
                    "mean_phase_score": f"{float(getattr(cand_score, 'mean_phase_score', 0.0)):.9f}",
                    "mean_amp_score": f"{float(getattr(cand_score, 'mean_amp_score', 0.0)):.9f}",
                    "mean_profile_score": f"{float(getattr(cand_score, 'mean_profile_score', 0.0)):.9f}",
                    "line_rmse_pi": f"{float(getattr(cand_score, 'line_rmse_pi', 0.0)):.9f}",
                    "line_r2": f"{float(getattr(cand_score, 'line_r2', 0.0)):.9f}",
                    "map_phase_ll": f"{float(getattr(cand_score, 'map_phase_ll', 0.0)):.9f}",
                    "map_line_ll": f"{float(getattr(cand_score, 'map_line_ll', 0.0)):.9f}",
                    "map_amp_ll": f"{float(getattr(cand_score, 'map_amp_ll', 0.0)):.9f}",
                    "map_profile_ll": f"{float(getattr(cand_score, 'map_profile_ll', 0.0)):.9f}",
                })

        # ── Per-symbol rows ─────────────────────────────────────────────
        for dec in result.get("payload_decisions", []):
            symbol_rows.append({
                "packet_index": packet_idx,
                "event_index": event_idx,
                "symbol_index": dec.stage_symbol_index,
                "selected_bin": dec.raw_fft_bin,
                "signed_bin": dec.signed_fft_bin,
                "symbol_value": dec.symbol_value,
                "phase_at_bin": f"{dec.phase_at_bin:.6f}",
                "phase_predicted": f"{dec.phase_predicted:.6f}",
                "phase_residual_rad": f"{dec.phase_residual_rad:.6f}",
                "phase_score": f"{dec.phase_score:.6f}",
                "margin": f"{dec.margin:.6f}",
                "confidence": f"{dec.confidence:.4f}",
                "amplitude": f"{dec.amplitude:.6f}",
                "energy_ratio": f"{dec.energy_ratio:.6f}",
                "argmax_bin": dec.argmax_bin,
                "argmax_is_selected": int(dec.argmax_is_selected),
                "rank_by_amp": dec.rank_by_amp,
                "gt_bin": dec.gt_bin,
                "is_correct": int(dec.is_correct),
            })

        print(
            f"  [{idx + 1}/{len(candidates)}] "
            f"Pkt {packet_idx} E{event_idx}: "
            f"payload={total_payload:3d} "
            f"correct={correct_count:3d} "
            f"SER={ser_final:.4f} "
            f"rounds={result.get('refinement_rounds', 0):1d} "
            f"hdr-slope={fl.slope_pi:.4f}pi/sym"
        )

    # ── Write outputs ───────────────────────────────────────────────────
    summary_path = output_dir / f"{input_path.stem}_phase_guided_summary.csv"
    symbol_path = output_dir / f"{input_path.stem}_phase_guided_symbols.csv"
    residual_candidate_path = output_dir / f"{input_path.stem}_residual_candidates.csv"
    write_csv(summary_path, summary_rows)
    write_csv(symbol_path, symbol_rows)
    if residual_candidate_rows:
        write_csv(residual_candidate_path, residual_candidate_rows)

    # ── Print aggregates ────────────────────────────────────────────────
    if gt_map and summary_rows:
        overall_ser = (
            1.0 - total_correct_processed / max(1, total_symbols_processed)
        )
        rates = [
            float(str(r.get("final_error_rate", "1.0"))) for r in summary_rows
        ]
        mean_ser = sum(rates) / len(rates) if rates else 0.0
        print(f"\nAggregate: {len(summary_rows)} packets, "
              f"{total_symbols_processed} symbols")
        print(f"  Mean PER-SER = {mean_ser:.4f}, "
              f"Overall SER = {overall_ser:.4f}")
    print(f"Summary: {summary_path}")
    print(f"Symbols: {symbol_path}")
    if residual_candidate_rows:
        print(f"Residual candidates: {residual_candidate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
