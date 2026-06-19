#!/usr/bin/env python3
"""CRC/PRR probe for real IQ captures without symbol or byte ground truth.

The header-first CSV supplies timing and decoded header parameters.  Its
payload raw_fft_bin column is a center-FFT hard decision, not ground truth for
lab captures, so this script reports only packet-level decode/CRC outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = WEAK_ROOT / "scripts" / "experiments"
SAVAUX_RUNNER_DIR = EXPERIMENTS_DIR / "baselines" / "savaux_oversampled"
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(SAVAUX_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(SAVAUX_RUNNER_DIR))

from run_paper_oversampled_baseline import _decode_payload, load_packets as load_savaux_packets  # noqa: E402
from run_savaux_codec_sweep import _codec_config, _extract_savaux_spectra  # noqa: E402
from run_savaux_current_threshold_sweep import _current_default_args  # noqa: E402
from run_symbol_phase_threshold_sweep import _evaluate_packet_methods as evaluate_current_packet  # noqa: E402
from run_symbol_phase_threshold_sweep import _write_csv  # noqa: E402
from run_symbol_phase_two_stage import build_config  # noqa: E402
from run_two_stage_weak_decoder import load_packets as load_current_packets  # noqa: E402
from weak_decoder.phase_guided_demod import PhaseLine  # noqa: E402
from weak_decoder.two_stage_weak_decoder import TwoStageWeakConfig, decode_two_stage_weak_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CRC outcomes on one real IQ/header-first capture.")
    parser.add_argument("-i", "--input-iq", type=Path, required=True)
    parser.add_argument("-s", "--symbol-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "baseline_comparison" / "real_iq_crc_probe",
    )
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol", "none"), default="continuous")
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--paper-origin-shift", type=int, default=None)
    parser.add_argument("--max-packets", type=int, default=None, help="Limit paired packets after header filtering.")

    parser.add_argument("--top-k-metrics", type=int, default=64)
    parser.add_argument("--amplitude-floor-db", type=float, default=30.0)
    parser.add_argument("--phase-weight", type=float, default=0.0)
    parser.add_argument("--bit-metric", choices=("max", "logsumexp"), default="max")
    parser.add_argument("--nibble-candidates", type=int, default=8)
    parser.add_argument("--row-beam-width", type=int, default=512)
    parser.add_argument("--block-candidate-limit", type=int, default=128)
    parser.add_argument("--block-symbol-seed-top-m", type=int, default=0)
    parser.add_argument("--block-symbol-seed-deep-top-l", type=int, default=0)
    parser.add_argument("--block-symbol-seed-max-deep-positions", type=int, default=1)
    parser.add_argument("--block-symbol-seed-quota", type=int, default=0)
    parser.add_argument("--block-symbol-seed-max-combinations", type=int, default=50000)
    parser.add_argument("--global-beam-width", type=int, default=512)
    parser.add_argument("--global-rank-diverse-top-r", type=int, default=0)
    parser.add_argument("--global-rank-diverse-beam-width", type=int, default=0)
    parser.add_argument("--global-rank-cost-max", type=float, default=0.0)
    parser.add_argument("--global-rank-cost-state-limit", type=int, default=0)
    parser.add_argument("--final-candidate-limit", type=int, default=512)
    parser.add_argument("--block-trim-fraction", type=float, default=0.20)
    parser.add_argument("--projection-trim-fraction", type=float, default=0.10)
    parser.add_argument("--projection-score-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-score-weight", type=float, default=0.0)
    parser.add_argument("--crc-observed-bonus", type=float, default=6.0)
    parser.add_argument("--crc-candidate-min-evidence-margin", type=float, default=-999.0)
    parser.add_argument("--crc-candidate-max-beam-rank", type=int, default=64)
    parser.add_argument("--crc-candidate-high-rank-max-beam-rank", type=int, default=256)
    parser.add_argument("--crc-candidate-high-rank-min-evidence-margin", type=float, default=4.0)
    parser.add_argument(
        "--crc-selection-policy",
        choices=("best_score", "earliest_rank", "score_unless_low_margin"),
        default="earliest_rank",
    )
    parser.add_argument("--crc-non-earliest-min-evidence-margin", type=float, default=0.50)
    parser.add_argument(
        "--crc-failure-selection-policy",
        choices=("argmax_fallback", "best_score", "small_change_best_score"),
        default="argmax_fallback",
    )
    parser.add_argument("--crc-failure-max-symbol-changes", type=int, default=2)
    parser.add_argument("--crc-failure-min-evidence-margin", type=float, default=-0.50)
    parser.add_argument("--crc-state-search-block-top-r", type=int, default=0)
    parser.add_argument("--crc-state-search-state-limit", type=int, default=20000)
    parser.add_argument("--crc-state-search-keep-per-key", type=int, default=1)
    parser.add_argument("--crc-state-search-max-candidates", type=int, default=256)
    parser.add_argument("--crc-state-search-min-evidence-margin", type=float, default=0.0)
    parser.add_argument("--disable-argmax-fallback", action="store_true")
    return parser.parse_args()


def _avg(rows: Sequence[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def _evaluate_savaux_codec_real_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    config: TwoStageWeakConfig,
) -> dict[str, Any]:
    spectra, raw_score_overrides, _unused_bins = _extract_savaux_spectra(samples, packet, args)
    hard_bins = [int(np.argmax(score)) for score in raw_score_overrides]
    hard_decode = _decode_payload(packet, hard_bins, args.crc_mode, args.ldro_mode)

    selected_payload_hex = str(hard_decode["payload_hex"])
    selected_crc_valid = int(hard_decode["crc_valid"])
    selected_source = "savaux_hard_crc_preserved" if selected_crc_valid else "savaux_hard_crc_failed"
    selected_beam_rank = -1
    symbol_change_count = 0
    candidate_payload_count = 0
    observed_crc_valid_count = 0
    accepted_crc_valid_count = 0

    if not selected_crc_valid:
        result = decode_two_stage_weak_payload(
            payload_spectra=spectra,
            header_symbol_values=tuple(packet["header_symbols"]),
            phase_line=PhaseLine(),
            payload_symbol_start_abs_index=12.25 + 8.0,
            sf=int(packet["sf"]),
            cr=int(packet["cr"]),
            payload_len=int(packet["payload_len"]),
            has_crc=bool(packet["has_crc"]),
            ldro=bool(packet["ldro"]),
            config=config,
            raw_score_overrides=raw_score_overrides,
            trajectory_spectra=spectra,
        )
        candidate_payload_count = int(result.metrics.get("candidate_payload_count", 0))
        observed_crc_valid_count = int(result.metrics.get("observed_crc_valid_count", 0))
        accepted_crc_valid_count = int(result.metrics.get("accepted_crc_valid_count", 0))
        selected = result.selected
        if selected is not None:
            selected_payload_hex = selected.payload_bytes.hex()
            selected_crc_valid = int(selected.observed_crc_valid)
            selected_source = selected.selection_source
            selected_beam_rank = int(selected.beam_rank)
            symbol_change_count = sum(
                int(int(a) != int(b))
                for a, b in zip(hard_bins, selected.raw_bins)
            )

    return {
        "savaux_paper_crc_valid": int(hard_decode["crc_valid"]),
        "savaux_paper_payload_hex": str(hard_decode["payload_hex"]),
        "savaux_paper_decode_error": str(hard_decode["decode_error"]),
        "savaux_codec_crc_valid": int(selected_crc_valid),
        "savaux_codec_payload_hex": str(selected_payload_hex),
        "savaux_codec_selected_source": str(selected_source),
        "savaux_codec_selected_beam_rank": int(selected_beam_rank),
        "savaux_codec_symbol_change_count": int(symbol_change_count),
        "savaux_codec_candidate_payload_count": int(candidate_payload_count),
        "savaux_codec_observed_crc_valid_count": int(observed_crc_valid_count),
        "savaux_codec_accepted_crc_valid_count": int(accepted_crc_valid_count),
    }


def build_summary(
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    detected_events: int | None = None,
    framesync_valid: int | None = None,
) -> dict[str, Any]:
    packet_count = int(len(rows))
    return {
        "file_name": args.input_iq.name,
        "input_iq": str(args.input_iq),
        "symbol_csv": str(args.symbol_csv),
        "detected_events": "" if detected_events is None else int(detected_events),
        "framesync_valid": "" if framesync_valid is None else int(framesync_valid),
        "header_valid_packets": packet_count,
        "traditional_fft_crc_valid": int(sum(int(row["traditional_fft_crc_valid"]) for row in rows)),
        "traditional_fft_crc_valid_rate": _avg(rows, "traditional_fft_crc_valid"),
        "multi_offset_argmax_crc_valid": int(sum(int(row["multi_offset_argmax_crc_valid"]) for row in rows)),
        "multi_offset_argmax_crc_valid_rate": _avg(rows, "multi_offset_argmax_crc_valid"),
        "current_selected_crc_valid": int(sum(int(row["current_selected_crc_valid"]) for row in rows)),
        "current_selected_crc_valid_rate": _avg(rows, "current_selected_crc_valid"),
        "savaux_paper_crc_valid": int(sum(int(row["savaux_paper_crc_valid"]) for row in rows)),
        "savaux_paper_crc_valid_rate": _avg(rows, "savaux_paper_crc_valid"),
        "savaux_codec_crc_valid": int(sum(int(row["savaux_codec_crc_valid"]) for row in rows)),
        "savaux_codec_crc_valid_rate": _avg(rows, "savaux_codec_crc_valid"),
        "savaux_codec_extra_crc_valid_vs_paper": int(
            sum(int(row["savaux_codec_crc_valid"]) - int(row["savaux_paper_crc_valid"]) for row in rows)
        ),
        "savaux_codec_mean_symbol_change_count": _avg(rows, "savaux_codec_symbol_change_count"),
        "note": "real IQ CRC/PRR probe only; no byte/symbol GT is used",
    }


def _read_sync_counts(symbol_csv: Path) -> tuple[int | None, int | None]:
    frames_path = symbol_csv.with_name(symbol_csv.stem.replace("_symbols", "_frames") + ".csv")
    if not frames_path.exists():
        return None, None
    with frames_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), sum(int(str(row.get("source_grlora_framesync_valid", "0")).strip() or 0) for row in rows)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = np.fromfile(args.input_iq, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {args.input_iq}")
    current_packets = load_current_packets(args.symbol_csv, None)
    savaux_packets = load_savaux_packets(args.symbol_csv, packet_filter=None, max_packets=None)
    savaux_by_packet = {int(packet["packet_index"]): packet for packet in savaux_packets}
    paired_packet_ids = sorted(set(current_packets).intersection(savaux_by_packet))
    if args.max_packets is not None:
        paired_packet_ids = paired_packet_ids[: int(args.max_packets)]

    current_args = _current_default_args(args)
    current_config = build_config(current_args)
    codec_config = _codec_config(args)

    rows: list[dict[str, Any]] = []
    for packet_id in paired_packet_ids:
        current_row = evaluate_current_packet(
            samples,
            current_packets[int(packet_id)],
            current_args,
            current_config,
        )
        codec_row = _evaluate_savaux_codec_real_packet(
            samples,
            savaux_by_packet[int(packet_id)],
            args,
            codec_config,
        )
        rows.append(
            {
                "file_name": args.input_iq.name,
                "packet_index": int(packet_id),
                "frame_index": int(current_row["frame_index"]),
                "event_index": int(current_row["event_index"]),
                "payload_len": int(current_row["payload_len"]),
                "symbol_count": int(current_row["symbol_count"]),
                "traditional_fft_crc_valid": int(current_row["center_crc_valid"]),
                "multi_offset_argmax_crc_valid": int(current_row["multi_crc_valid"]),
                "current_selected_crc_valid": int(current_row["selected_crc_valid"]),
                **codec_row,
            }
        )

    detected_events, framesync_valid = _read_sync_counts(args.symbol_csv)
    summary = build_summary(rows, args, detected_events=detected_events, framesync_valid=framesync_valid)
    stem = args.input_iq.stem
    _write_csv(output_dir / f"{stem}_real_iq_crc_packets.csv", rows)
    _write_csv(output_dir / f"{stem}_real_iq_crc_summary.csv", [summary])
    (output_dir / f"{stem}_real_iq_crc_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"{args.input_iq.name}: headers={summary['header_valid_packets']} "
        f"current={summary['current_selected_crc_valid_rate']:.3f} "
        f"savaux={summary['savaux_paper_crc_valid_rate']:.3f} "
        f"codec={summary['savaux_codec_crc_valid_rate']:.3f}"
    )
    print(f"wrote={output_dir / f'{stem}_real_iq_crc_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
