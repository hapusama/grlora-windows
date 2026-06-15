#!/usr/bin/env python3
"""Sweep joint residual-byte decoding against ordinary argmax.

This harness runs ``run_phase_guided_demod.py`` in residual-candidate export
mode with *no* per-packet dynamic-byte prior, then applies
``joint_decode_residual_candidates.py`` to recover a cross-packet affine byte
trajectory.  It is meant for the argmax-centered paper question: how much weak
packet recovery remains once ordinary argmax has collapsed?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


WEAK_ROOT = Path(__file__).resolve().parent.parent
RUNNER = WEAK_ROOT / "scripts" / "run_phase_guided_demod.py"
JOINT = WEAK_ROOT / "scripts" / "joint_decode_residual_candidates.py"

DEFAULT_HEADER = "75,163,15,20,211,206,182,86"
DEFAULT_SYNC = WEAK_ROOT / "data" / "weak_sync_chain" / "sync_chain" / "0_0_0_10_14_16_sync_chain.csv"
DEFAULT_GT = WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / "0_0_0_10_14_16_header_first_symbols.csv"
DEFAULT_SYMBOL_TEMPLATE = WEAK_ROOT / "data" / "phase_guided" / "session_payload_template_excl_1_2_3_5_6_guarded.json"
DEFAULT_BYTE_TEMPLATE = WEAK_ROOT / "data" / "phase_guided" / "session_payload_byte_template_from_symbols_excl_1_2_3_5_6.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep joint residual search with ordinary argmax baseline."
    )
    parser.add_argument("--snr", type=int, nargs="*", default=[-20, -23, -25, -27])
    parser.add_argument("--max-packets", type=int, default=5)
    parser.add_argument("--output-root", type=Path,
                        default=WEAK_ROOT / "data" / "phase_guided" / "joint_residual_sweep")
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--input-stem", type=str, default="0_0_0_10_14_16",
                        help="stem used for low-SNR IQ files")
    parser.add_argument("--low-snr-dir", type=Path, default=None,
                        help="directory containing <input-stem>_snr_mXXdB.bin files")
    parser.add_argument("--sync-chain-csv", type=Path, default=DEFAULT_SYNC)
    parser.add_argument("--gt-symbol-csv", type=Path, default=DEFAULT_GT)
    parser.add_argument("--expected-header-symbols", type=str, default=DEFAULT_HEADER)
    parser.add_argument("--payload-template-file", type=Path, default=DEFAULT_SYMBOL_TEMPLATE)
    parser.add_argument("--byte-template-file", type=Path, default=DEFAULT_BYTE_TEMPLATE)
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--bw", type=float, default=125000.0)
    parser.add_argument("--samp-rate", type=float, default=500000.0)
    parser.add_argument("--preamble-len", type=float, default=16.0)
    parser.add_argument("--refinement-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--byte-index", type=int, default=6)
    parser.add_argument("--expected-affine", type=str, default="1,1")
    parser.add_argument("--outlier-penalty", type=float, default=None)
    parser.add_argument("--outlier-mode", choices=("joint", "posthoc"), default="joint")
    parser.add_argument("--max-outliers", type=int, default=None)
    parser.add_argument("--outlier-min-top1-gap", type=float, default=0.0)
    parser.add_argument("--outlier-min-score-std", type=float, default=0.0)
    parser.add_argument("--outlier-max-entropy", type=float, default=1.0)
    parser.add_argument("--candidate-search-score-mode",
                        choices=("bounded", "map"), default="bounded")
    parser.add_argument("--candidate-search-phase-kappa", type=float, default=2.0)
    parser.add_argument("--candidate-search-adaptive-kappa", action="store_true", default=False)
    parser.add_argument("--candidate-search-kappa-source",
                        choices=("scoring_line", "initial_line"), default="scoring_line")
    parser.add_argument("--candidate-search-kappa-min", type=float, default=1.0)
    parser.add_argument("--candidate-search-kappa-max", type=float, default=4.0)
    parser.add_argument("--candidate-search-kappa-sigma-floor-pi", type=float, default=0.12)
    parser.add_argument("--skip-runner", action="store_true", default=False,
                        help="reuse existing residual candidate CSVs and rerun only the joint layer")
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def snr_input_path(args: argparse.Namespace, snr_db: int) -> Path:
    stem = f"{args.input_stem}_snr_m{abs(int(snr_db))}dB.bin"
    if args.low_snr_dir is not None:
        return args.low_snr_dir.resolve() / stem
    if str(args.input_stem) != "0_0_0_10_14_16":
        return WEAK_ROOT / "data" / "low_snr_gt_bin" / str(args.input_stem) / stem
    if int(snr_db) in (-10, -15, -20):
        return WEAK_ROOT / "data" / "low_snr_gt_bin" / "0_0_0_10_14_16" / stem
    return WEAK_ROOT / "data" / "low_snr_gt_bin" / "0_0_0_10_14_16_extreme_snr" / stem


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sum_int(rows: list[dict[str, str]], key: str) -> int:
    total = 0
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value:
            total += int(float(value))
    return total


def summarize_symbols(symbol_path: Path) -> tuple[int, int]:
    total = 0
    correct = 0
    for row in _read_rows(symbol_path):
        gt = int(float(row.get("gt_bin", -1) or -1))
        argmax = int(float(row.get("argmax_bin", -1) or -1))
        if gt >= 0:
            total += 1
            correct += int(argmax == gt)
    return total, correct


def run_one(args: argparse.Namespace, snr_db: int) -> dict[str, Any]:
    input_path = snr_input_path(args, int(snr_db))
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    run_dir = args.output_root / f"snr_m{abs(int(snr_db))}dB"
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    summary_path = run_dir / f"{stem}_phase_guided_summary.csv"
    symbol_path = run_dir / f"{stem}_phase_guided_symbols.csv"
    candidate_path = run_dir / f"{stem}_residual_candidates.csv"
    joint_csv = run_dir / f"{stem}_joint_affine_decisions.csv"
    joint_json = run_dir / f"{stem}_joint_affine_summary.json"

    runner_cmd = [
        sys.executable,
        str(RUNNER),
        "-i", str(input_path),
        "-s", str(args.sync_chain_csv),
        "-g", str(args.gt_symbol_csv),
        "-o", str(run_dir),
        "--sf", str(args.sf),
        "--bw", str(args.bw),
        "--samp-rate", str(args.samp_rate),
        "--preamble-len", str(args.preamble_len),
        "--max-packets", str(args.max_packets),
        "--seed", str(args.seed),
        "--refinement-rounds", str(args.refinement_rounds),
        "--expected-header-symbols", str(args.expected_header_symbols),
        "--payload-template-file", str(args.payload_template_file),
        "--byte-template-file", str(args.byte_template_file),
        "--enable-byte-residual-search",
        "--residual-byte-index", str(args.byte_index),
        "--residual-byte-values", "0-255",
        "--residual-max-unknown-bytes", "1",
        "--residual-max-candidates", "256",
        "--candidate-search-min-known", "3",
        "--candidate-search-prior-weight", "0.0",
        "--candidate-search-score-mode", str(args.candidate_search_score_mode),
        "--candidate-search-phase-kappa", str(args.candidate_search_phase_kappa),
        "--candidate-search-kappa-source", str(args.candidate_search_kappa_source),
        "--candidate-search-kappa-min", str(args.candidate_search_kappa_min),
        "--candidate-search-kappa-max", str(args.candidate_search_kappa_max),
        "--candidate-search-kappa-sigma-floor-pi", str(args.candidate_search_kappa_sigma_floor_pi),
        "--write-residual-candidates",
    ]
    if bool(args.candidate_search_adaptive_kappa):
        runner_cmd.append("--candidate-search-adaptive-kappa")
    print(" ".join(runner_cmd), flush=True)
    if args.skip_runner and not args.dry_run and not candidate_path.exists():
        raise FileNotFoundError(
            f"--skip-runner requested but residual candidates are missing: {candidate_path}"
        )
    if not args.dry_run and not args.skip_runner:
        subprocess.run(runner_cmd, check=True)

    joint_cmd = [
        sys.executable,
        str(JOINT),
        "-c", str(candidate_path),
        "-o", str(joint_csv),
        "--summary-json", str(joint_json),
        "--score-field", "signal_score",
        "--byte-index", str(args.byte_index),
        "--expected-affine", str(args.expected_affine),
        "-g", str(args.gt_symbol_csv),
        "--sf", str(args.sf),
        "--bw", str(args.bw),
    ]
    if args.outlier_penalty is not None:
        joint_cmd.extend([
            "--outlier-penalty", str(args.outlier_penalty),
            "--outlier-mode", str(args.outlier_mode),
        ])
    if args.max_outliers is not None:
        joint_cmd.extend(["--max-outliers", str(args.max_outliers)])
    if args.outlier_penalty is not None:
        joint_cmd.extend([
            "--outlier-min-top1-gap", str(args.outlier_min_top1_gap),
            "--outlier-min-score-std", str(args.outlier_min_score_std),
            "--outlier-max-entropy", str(args.outlier_max_entropy),
        ])
    print(" ".join(joint_cmd), flush=True)
    if not args.dry_run:
        subprocess.run(joint_cmd, check=True)

    summary_rows = _read_rows(summary_path)
    total_symbols = _sum_int(summary_rows, "total_payload")
    independent_correct = _sum_int(summary_rows, "correct_count")
    argmax_total, argmax_correct = summarize_symbols(symbol_path)

    joint_summary: dict[str, Any] = {}
    if joint_json.exists():
        joint_summary = json.loads(joint_json.read_text(encoding="utf-8"))

    return {
        "snr_db": int(snr_db),
        "packets": len(summary_rows),
        "total_symbols": total_symbols,
        "argmax_correct_symbols": argmax_correct,
        "argmax_ser": (
            f"{1.0 - argmax_correct / argmax_total:.4f}"
            if argmax_total else ""
        ),
        "independent_correct_symbols": independent_correct,
        "independent_ser": (
            f"{1.0 - independent_correct / total_symbols:.4f}"
            if total_symbols else ""
        ),
        "independent_app_exact": int(joint_summary.get("independent_app_exact", 0)),
        "joint_app_exact": int(joint_summary.get("joint_app_exact", 0)),
        "app_compared": int(joint_summary.get("app_compared", 0)),
        "joint_affine_slope": int(joint_summary.get("affine_slope", -1)),
        "joint_affine_intercept": int(joint_summary.get("affine_intercept", -1)),
        "joint_outlier_count": int(joint_summary.get("outlier_count", 0)),
        "joint_outlier_penalty": (
            f"{float(joint_summary.get('outlier_penalty', 0.0)):.6f}"
            if joint_summary.get("outlier_penalty") is not None else ""
        ),
        "joint_outlier_mode": str(joint_summary.get("outlier_mode", "")),
        "joint_outlier_min_top1_gap": (
            f"{float(joint_summary.get('outlier_min_top1_gap', 0.0)):.6f}"
            if joint_summary.get("outlier_penalty") is not None else ""
        ),
        "candidate_search_score_mode": str(args.candidate_search_score_mode),
        "candidate_search_adaptive_kappa": int(bool(args.candidate_search_adaptive_kappa)),
        "candidate_search_kappa_source": str(args.candidate_search_kappa_source),
        "joint_model_margin": (
            f"{float(joint_summary.get('model_score_margin', 0.0)):.6f}"
            if joint_summary else ""
        ),
        "joint_model_margin_per_packet": (
            f"{float(joint_summary.get('model_score_margin_per_packet', 0.0)):.6f}"
            if joint_summary else ""
        ),
        "joint_top1_count": int(joint_summary.get("joint_top1_count", 0)),
        "mean_joint_candidate_rank": (
            f"{float(joint_summary.get('mean_joint_candidate_rank', 0.0)):.3f}"
            if joint_summary.get("mean_joint_candidate_rank") is not None else ""
        ),
        "mean_independent_minus_joint_score_gap": (
            f"{float(joint_summary.get('mean_independent_minus_joint_score_gap', 0.0)):.6f}"
            if joint_summary else ""
        ),
        "mean_candidate_top1_gap": (
            f"{float(joint_summary.get('mean_candidate_top1_gap', 0.0)):.6f}"
            if joint_summary else ""
        ),
        "mean_candidate_score_std": (
            f"{float(joint_summary.get('mean_candidate_score_std', 0.0)):.6f}"
            if joint_summary else ""
        ),
        "mean_candidate_entropy_norm": (
            f"{float(joint_summary.get('mean_candidate_entropy_norm', 0.0)):.6f}"
            if joint_summary else ""
        ),
        "expected_affine_ok": joint_summary.get("expected_affine_ok", ""),
        "run_dir": str(run_dir),
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "snr_db",
        "packets",
        "total_symbols",
        "argmax_correct_symbols",
        "argmax_ser",
        "independent_correct_symbols",
        "independent_ser",
        "independent_app_exact",
        "joint_app_exact",
        "app_compared",
        "joint_affine_slope",
        "joint_affine_intercept",
        "joint_outlier_count",
        "joint_outlier_penalty",
        "joint_outlier_mode",
        "joint_outlier_min_top1_gap",
        "candidate_search_score_mode",
        "candidate_search_adaptive_kappa",
        "candidate_search_kappa_source",
        "joint_model_margin",
        "joint_model_margin_per_packet",
        "joint_top1_count",
        "mean_joint_candidate_rank",
        "mean_independent_minus_joint_score_gap",
        "mean_candidate_top1_gap",
        "mean_candidate_score_std",
        "mean_candidate_entropy_norm",
        "expected_affine_ok",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    summary_csv = (
        args.summary_csv.resolve()
        if args.summary_csv else args.output_root / "joint_residual_sweep_summary.csv"
    )
    rows: list[dict[str, Any]] = []
    for snr_db in args.snr:
        row = run_one(args, int(snr_db))
        rows.append(row)
        write_summary(summary_csv, rows)
    print(f"\nSummary: {summary_csv}")
    for row in rows:
        print(
            f"  {row['snr_db']:>4} dB "
            f"argmax_SER={row['argmax_ser']} "
            f"ind_SER={row['independent_ser']} "
            f"ind_app={row['independent_app_exact']}/{row['app_compared']} "
            f"joint_app={row['joint_app_exact']}/{row['app_compared']} "
            f"outliers={row.get('joint_outlier_count', '')} "
            f"joint_rank={row.get('mean_joint_candidate_rank', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
