#!/usr/bin/env python3
"""Batch compare coherence and local-window phase FFT-bin selectors.

This script stays at the FFT-bin selection layer.  It reuses
`run_symbol_phase_two_stage.evaluate_packet`, where CRC is only an evaluation
field after hard decisions, not a bin-selection input.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_symbol_phase_two_stage import (  # noqa: E402
    build_config,
    build_summary,
    evaluate_packet,
)
from run_two_stage_weak_decoder import load_packets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch compare coherence and local-window phase selectors on noisy IQ files."
    )
    parser.add_argument(
        "--iq-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "low_snr_gt_bin" / "0_0_0_10_14_16_m22_m27_sto_input",
    )
    parser.add_argument(
        "--symbol-csv",
        type=Path,
        default=WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / "0_0_0_10_14_16_header_first_symbols.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "local_window_phase_batch")
    parser.add_argument("--snr", nargs="*", type=float, default=None)
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)

    parser.add_argument("--top-l-low-confidence", type=int, default=24)
    parser.add_argument("--lock-margin-db", type=float, default=1.5)
    parser.add_argument("--lock-peak-to-median-db", type=float, default=5.0)
    parser.add_argument("--lock-phase-score", type=float, default=0.35)
    parser.add_argument("--min-locked-for-line", type=int, default=4)
    parser.add_argument("--line-trim-frac", type=float, default=0.25)
    parser.add_argument("--phase-model", choices=("linear", "quadratic"), default="linear")
    parser.add_argument("--trajectory-rmse-scale-pi", type=float, default=0.30)
    parser.add_argument("--trajectory-phase-weight", type=float, default=0.20)
    parser.add_argument("--trajectory-line-weight", type=float, default=0.00)
    parser.add_argument("--trajectory-amp-weight", type=float, default=0.80)
    parser.add_argument("--trajectory-profile-weight", type=float, default=0.00)
    parser.add_argument("--phase-override-min-gain", type=float, default=0.15)
    parser.add_argument("--phase-override-max-drop-db", type=float, default=0.60)
    parser.add_argument("--phase-override-score-margin", type=float, default=0.06)
    parser.add_argument("--phase-override-min-line-anchors", type=int, default=8)
    parser.add_argument("--phase-override-max-line-rmse-pi", type=float, default=0.25)
    parser.add_argument("--coherence-weight", type=float, default=0.0)
    parser.add_argument("--coherence-candidate-top-l", type=int, default=0)
    parser.add_argument("--lock-min-coherence", type=float, default=0.0)
    parser.add_argument("--smooth-phase-weight", type=float, default=0.05)
    parser.add_argument("--smooth-amp-weight", type=float, default=0.50)
    parser.add_argument("--smooth-coherence-weight", type=float, default=0.90)
    parser.add_argument("--smooth-slope-penalty", type=float, default=0.05)
    parser.add_argument("--smooth-curvature-penalty", type=float, default=0.10)
    parser.add_argument("--smooth-max-energy-drop-db", type=float, default=20.0)
    parser.add_argument("--smooth-min-line-anchors", type=int, default=4)
    parser.add_argument("--smooth-min-locked-ratio", type=float, default=0.0)
    parser.add_argument("--smooth-max-line-rmse-pi", type=float, default=float("inf"))
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--window-degree", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--window-phase-weight", type=float, default=0.05)
    parser.add_argument("--window-amp-weight", type=float, default=0.50)
    parser.add_argument("--window-coherence-weight", type=float, default=0.90)
    parser.add_argument("--window-slope-weight", type=float, default=0.00)
    parser.add_argument("--window-curvature-weight", type=float, default=0.00)
    parser.add_argument("--window-phase-scale-pi", type=float, default=0.25)
    parser.add_argument("--window-slope-scale-pi", type=float, default=0.45)
    parser.add_argument("--window-curvature-scale-pi", type=float, default=0.25)
    parser.add_argument("--window-recent-decay", type=float, default=0.75)
    parser.add_argument("--window-anchor-span", type=float, default=8.0)
    parser.add_argument("--window-anchor-min", type=int, default=2)
    parser.add_argument("--window-anchor-max-rmse-pi", type=float, default=0.40)
    parser.add_argument("--window-min-locked-ratio", type=float, default=0.10)
    parser.add_argument("--window-guard-min-phase-gain", type=float, default=0.10)
    parser.add_argument("--window-guard-max-energy-drop-db", type=float, default=0.75)
    parser.add_argument("--window-guard-max-coherence-drop", type=float, default=0.08)
    return parser.parse_args()


def _snr_from_name(path: Path) -> float | None:
    match = re.search(r"snr_m(\d+(?:p\d+)?)dB", path.name)
    if match is None:
        return None
    return -float(match.group(1).replace("p", "."))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
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


def _method_args(args: argparse.Namespace, mode: str) -> SimpleNamespace:
    data = vars(args).copy()
    data["selection_mode"] = str(mode)
    return SimpleNamespace(**data)


def main() -> int:
    args = parse_args()
    packets = load_packets(args.symbol_csv, args.packet)
    if not packets:
        raise SystemExit(f"no packets loaded: {args.symbol_csv}")
    allowed_snrs = None if args.snr is None else {round(float(v), 6) for v in args.snr}
    iq_files: list[tuple[float, Path]] = []
    for path in sorted(args.iq_dir.glob("*.bin")):
        snr = _snr_from_name(path)
        if snr is None:
            continue
        if allowed_snrs is not None and round(float(snr), 6) not in allowed_snrs:
            continue
        iq_files.append((float(snr), path))
    if not iq_files:
        raise SystemExit(f"no noisy IQ files found in {args.iq_dir}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    for snr, iq_path in iq_files:
        samples = np.fromfile(iq_path, dtype=np.complex64)
        if samples.size == 0:
            raise ValueError(f"empty IQ file: {iq_path}")
        for mode in ("coherence", "window", "window_guarded"):
            method_args = _method_args(args, mode)
            config = build_config(method_args)
            rows: list[dict[str, Any]] = []
            for packet_index in sorted(packets):
                row = evaluate_packet(samples, packets[packet_index], method_args, config)
                row["method"] = mode
                row["target_snr_db"] = float(snr)
                row["input_iq"] = str(iq_path)
                rows.append(row)
                packet_rows.append(row)
            summary = build_summary(rows, method_args)
            summary_row = {
                "method": mode,
                "target_snr_db": float(snr),
                "input_iq": str(iq_path),
                **{key: value for key, value in summary.items() if key != "parameters"},
            }
            summary_rows.append(summary_row)
            print(
                f"snr={snr:>5.1f} method={mode:<9} "
                f"selected_ser={summary['selected_symbol_ser']:.3f} "
                f"crc={summary['crc_valid_rate']:.3f}",
                flush=True,
            )

    _write_csv(out_dir / "per_packet_metrics.csv", packet_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"summary_rows": summary_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote={out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
