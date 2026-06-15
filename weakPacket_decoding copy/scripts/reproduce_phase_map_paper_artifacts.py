#!/usr/bin/env python3
"""Reproduce the Phase-MAP paper artifacts for the current weak-LoRa session.

The default command reruns the two joint residual sweeps, refreshes the
phase-kappa ablations, and rebuilds the compact paper tables.  Use
``--skip-sweeps`` when the candidate CSVs and sweep summaries already exist and
only the final tables should be regenerated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


WEAK_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = WEAK_ROOT / "scripts"
SWEEP_JOINT = SCRIPTS / "sweep_joint_residual_session.py"
SWEEP_KAPPA = SCRIPTS / "sweep_map_kappa_from_candidates.py"
MAKE_THRESHOLD = SCRIPTS / "make_joint_threshold_table.py"
MAKE_ABLATION = SCRIPTS / "make_phase_ablation_table.py"
MAKE_VALIDATION = SCRIPTS / "make_generalization_validation_table.py"

DEFAULT_THRESHOLD_ROOT = (
    WEAK_ROOT / "data" / "phase_guided" / "joint_residual_sweep_map_threshold"
)
DEFAULT_WEAK_ROOT = WEAK_ROOT / "data" / "phase_guided" / "joint_residual_sweep_map"
DEFAULT_TABLE_DIR = WEAK_ROOT / "data" / "phase_guided" / "paper_tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Phase-MAP threshold and phase-ablation artifacts."
    )
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for child scripts")
    parser.add_argument("--max-packets", type=int, default=5)
    parser.add_argument("--threshold-snr", type=int, nargs="*",
                        default=[-10, -15, -20, -23])
    parser.add_argument("--weak-snr", type=int, nargs="*",
                        default=[-20, -23, -25, -27])
    parser.add_argument("--weak-kappa-snr", type=int, nargs="*",
                        default=[-25, -27],
                        help="weak-root SNRs used for the deep kappa table rows")
    parser.add_argument("--threshold-root", type=Path,
                        default=DEFAULT_THRESHOLD_ROOT)
    parser.add_argument("--weak-root", type=Path, default=DEFAULT_WEAK_ROOT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--score-mode", choices=("bounded", "map"), default="map")
    parser.add_argument("--phase-kappa", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, nargs="*",
                        default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--skip-sweeps", action="store_true", default=False,
                        help="only rebuild final tables from existing summaries")
    parser.add_argument("--skip-kappa", action="store_true", default=False,
                        help="do not refresh kappa ablation summaries")
    parser.add_argument("--skip-tables", action="store_true", default=False,
                        help="do not rebuild CSV/Markdown paper tables")
    parser.add_argument("--include-validation-table", action="store_true", default=False,
                        help="also rebuild the extra-capture validation table")
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def _fmt(cmd: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in cmd])


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(_fmt(cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], check=True)


def _snr_args(values: list[int]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.append(str(int(value)))
    return out


def main() -> int:
    args = parse_args()
    python = str(args.python)
    threshold_root = args.threshold_root.resolve()
    weak_root = args.weak_root.resolve()
    table_dir = args.table_dir.resolve()

    if not args.skip_sweeps:
        run_cmd([
            python,
            str(SWEEP_JOINT),
            "--snr", *_snr_args(args.threshold_snr),
            "--max-packets", str(args.max_packets),
            "--output-root", str(threshold_root),
            "--candidate-search-score-mode", str(args.score_mode),
            "--candidate-search-phase-kappa", str(args.phase_kappa),
        ], dry_run=bool(args.dry_run))

        run_cmd([
            python,
            str(SWEEP_JOINT),
            "--snr", *_snr_args(args.weak_snr),
            "--max-packets", str(args.max_packets),
            "--output-root", str(weak_root),
            "--candidate-search-score-mode", str(args.score_mode),
            "--candidate-search-phase-kappa", str(args.phase_kappa),
        ], dry_run=bool(args.dry_run))

    if not args.skip_sweeps and not args.skip_kappa:
        run_cmd([
            python,
            str(SWEEP_KAPPA),
            "--candidate-root", str(threshold_root),
            "--snr", *_snr_args(args.threshold_snr),
            "--kappa", *[str(float(k)) for k in args.kappa],
            "--source-kappa", str(args.phase_kappa),
        ], dry_run=bool(args.dry_run))

        run_cmd([
            python,
            str(SWEEP_KAPPA),
            "--candidate-root", str(weak_root),
            "--snr", *_snr_args(args.weak_kappa_snr),
            "--kappa", *[str(float(k)) for k in args.kappa],
            "--source-kappa", str(args.phase_kappa),
        ], dry_run=bool(args.dry_run))

    if not args.skip_tables:
        run_cmd([
            python,
            str(MAKE_THRESHOLD),
            "--summary-csv",
            str(threshold_root / "joint_residual_sweep_summary.csv"),
            str(weak_root / "joint_residual_sweep_summary.csv"),
            "--output-dir", str(table_dir),
        ], dry_run=bool(args.dry_run))

        run_cmd([
            python,
            str(MAKE_ABLATION),
            "--summary-csv",
            str(threshold_root / "map_kappa_sweep_summary.csv"),
            str(weak_root / "map_kappa_sweep_summary.csv"),
            "--output-dir", str(table_dir),
        ], dry_run=bool(args.dry_run))

        if bool(args.include_validation_table):
            run_cmd([
                python,
                str(MAKE_VALIDATION),
                "--output-dir", str(table_dir),
            ], dry_run=bool(args.dry_run))

    print("\nArtifacts:")
    print(f"  Threshold root: {threshold_root}")
    print(f"  Weak root:      {weak_root}")
    print(f"  Paper tables:   {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
