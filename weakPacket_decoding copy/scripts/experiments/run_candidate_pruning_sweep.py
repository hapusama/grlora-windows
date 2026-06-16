#!/usr/bin/env python3
"""Run candidate-pruning Recall@L sweeps and merge the summaries."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


WEAK_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = WEAK_ROOT / "scripts" / "experiments" / "evaluate_candidate_pruning_metric.py"
DEFAULT_OUT_ROOT = WEAK_ROOT / "data" / "candidate_pruning" / "sweeps"
LOW_SNR_ROOT = WEAK_ROOT / "data" / "low_snr_gt_bin"
HEADER_FIRST_ROOT = WEAK_ROOT / "data" / "weak_sync_chain" / "header_first"
SNR_RE = re.compile(r"^(?P<stem>.+)_snr_m(?P<snr>\d+)dB\.bin$")


@dataclass(frozen=True)
class DatasetCase:
    stem: str
    snr_db: int
    iq_path: Path
    symbol_csv: Path
    preamble_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run phase-aware candidate-pruning Recall@L sweeps."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="dataset stem to include, e.g. 0_0_0_10_14_16; repeatable",
    )
    parser.add_argument(
        "--snr-db",
        default="20,23,25,27",
        help="positive dB labels for *_snr_mXXdB.bin files",
    )
    parser.add_argument(
        "--trend-source",
        default="early-payload,header-offset",
        help="comma-separated phase trend sources",
    )
    parser.add_argument(
        "--preselect-mode",
        default="default,allbin",
        help="comma-separated modes: default, allbin",
    )
    parser.add_argument("--phase-bonus-weights", default="0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--phase-gate-width-pi", default="0.166667,0.25,0.333333,0.5")
    parser.add_argument("--top-l", default="1,2,4,8,16,32,64")
    parser.add_argument("--rescue-counts", default="0,1,2,4")
    parser.add_argument("--phase-evidence", choices=("center", "multi-offset"), default="multi-offset")
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip completed summary JSON files")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed run")
    return parser.parse_args()


def _parse_csv_ints(text: str) -> list[int]:
    return [int(float(item.strip())) for item in str(text).split(",") if item.strip()]


def _parse_csv_text(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _inside_weak_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(WEAK_ROOT.resolve())
        return True
    except ValueError:
        return False


def _preamble_from_stem(stem: str) -> int:
    try:
        return int(stem.split("_")[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot infer preamble length from stem: {stem}") from exc


def discover_cases(datasets: list[str], snrs: list[int]) -> list[DatasetCase]:
    wanted_datasets = set(datasets)
    wanted_snrs = set(snrs)
    cases: dict[tuple[str, int], DatasetCase] = {}
    for iq_path in LOW_SNR_ROOT.rglob("*.bin"):
        match = SNR_RE.match(iq_path.name)
        if match is None:
            continue
        stem = match.group("stem")
        snr_db = int(match.group("snr"))
        if wanted_datasets and stem not in wanted_datasets:
            continue
        if wanted_snrs and snr_db not in wanted_snrs:
            continue
        symbol_csv = HEADER_FIRST_ROOT / f"{stem}_header_first_symbols.csv"
        if not symbol_csv.exists():
            continue
        key = (stem, snr_db)
        # Prefer the shortest path when duplicate SNR files exist under control dirs.
        current = cases.get(key)
        if current is None or len(iq_path.parts) < len(current.iq_path.parts):
            cases[key] = DatasetCase(
                stem=stem,
                snr_db=snr_db,
                iq_path=iq_path,
                symbol_csv=symbol_csv,
                preamble_len=_preamble_from_stem(stem),
            )
    return [cases[key] for key in sorted(cases, key=lambda item: (item[0], item[1]))]


def run_one(
    case: DatasetCase,
    trend_source: str,
    preselect_mode: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[Path, Path, int]:
    tag = f"{case.stem}_snr_m{case.snr_db}dB_{trend_source}_{preselect_mode}"
    output_csv = out_dir / f"{tag}.csv"
    summary_json = out_dir / f"{tag}_summary.json"
    cmd = [
        sys.executable,
        str(EVALUATOR),
        "-i",
        str(case.iq_path),
        "-s",
        str(case.symbol_csv),
        "-o",
        str(output_csv),
        "--summary-json",
        str(summary_json),
        "--quiet",
        "--preamble-len",
        str(case.preamble_len),
        "--phase-trend-source",
        trend_source,
        "--phase-evidence",
        str(args.phase_evidence),
        "--phase-bonus-weights",
        str(args.phase_bonus_weights),
        "--phase-gate-width-pi",
        str(args.phase_gate_width_pi),
        "--top-l",
        str(args.top_l),
        "--rescue-counts",
        str(args.rescue_counts),
    ]
    if preselect_mode == "allbin":
        cmd.extend(["--energy-preselect-count", "0", "--energy-preselect-factor", "0"])
    elif preselect_mode != "default":
        raise ValueError(f"unknown preselect mode: {preselect_mode}")
    if args.packet is not None:
        cmd.extend(["--packet", str(int(args.packet))])

    print(
        f"[run] {case.stem} snr=-{case.snr_db}dB trend={trend_source} "
        f"preselect={preselect_mode}",
        flush=True,
    )
    if args.dry_run:
        print("      " + " ".join(cmd), flush=True)
        return output_csv, summary_json, 0
    if args.resume and summary_json.exists() and output_csv.exists():
        print("      skip existing", flush=True)
        return output_csv, summary_json, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=WEAK_ROOT)
    if result.returncode != 0:
        print(f"      failed with exit code {result.returncode}", flush=True)
    return output_csv, summary_json, int(result.returncode)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def merge_summary_json(
    records: list[dict[str, Any]],
    top_l_values: list[int],
    out_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        summary_path = Path(record["summary_json"])
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for item in payload:
            row = {
                "stem": record["stem"],
                "snr_db": record["snr_db"],
                "preamble_len": record["preamble_len"],
                "trend_source_arg": record["trend_source"],
                "preselect_mode": record["preselect_mode"],
                "per_symbol_csv": record["per_symbol_csv"],
                "summary_json": str(summary_path),
            }
            row.update(item)
            rows.append(row)
    if not rows:
        return
    fields: list[str] = []
    priority = [
        "stem",
        "snr_db",
        "preamble_len",
        "trend_source_arg",
        "preselect_mode",
        "phase_bonus_weight",
        "phase_gate_width_pi",
        "phase_evidence",
        "symbol_count",
        "packet_count",
    ]
    for key in priority:
        if any(key in row for row in rows):
            fields.append(key)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    merged_path = out_dir / "merged_summary.csv"
    with merged_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)

    best_rows: list[dict[str, Any]] = []
    for cutoff in top_l_values:
        grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[
                (
                    str(row["stem"]),
                    int(row["snr_db"]),
                    str(row["trend_source_arg"]),
                    str(row["preselect_mode"]),
                )
            ].append(row)
        for (stem, snr_db, trend, preselect), items in grouped.items():
            best = max(
                items,
                key=lambda item: (
                    float(item.get(f"phase_gated_recall@{cutoff}", 0.0)),
                    float(item.get(f"gain_vs_multi_offset@{cutoff}", 0.0)),
                    -float(item.get(f"phase_damage_count@{cutoff}", 0.0)),
                ),
            )
            best_rows.append(
                {
                    "stem": stem,
                    "snr_db": snr_db,
                    "trend_source_arg": trend,
                    "preselect_mode": preselect,
                    "top_l": cutoff,
                    "phase_bonus_weight": best.get("phase_bonus_weight", ""),
                    "phase_gate_width_pi": best.get("phase_gate_width_pi", ""),
                    "center_recall": best.get(f"center_recall@{cutoff}", ""),
                    "multi_offset_recall": best.get(f"multi_offset_recall@{cutoff}", ""),
                    "phase_gated_recall": best.get(f"phase_gated_recall@{cutoff}", ""),
                    "gain_vs_center": best.get(f"gain_vs_center@{cutoff}", ""),
                    "gain_vs_multi_offset": best.get(f"gain_vs_multi_offset@{cutoff}", ""),
                    "phase_rescue_count": best.get(f"phase_rescue_count@{cutoff}", ""),
                    "phase_damage_count": best.get(f"phase_damage_count@{cutoff}", ""),
                    "mean_phase_trend_quality": best.get("mean_phase_trend_quality", ""),
                    "mean_early_anchor_count": best.get("mean_early_anchor_count", ""),
                }
            )
    best_fields = [
        "stem",
        "snr_db",
        "trend_source_arg",
        "preselect_mode",
        "top_l",
        "phase_bonus_weight",
        "phase_gate_width_pi",
        "center_recall",
        "multi_offset_recall",
        "phase_gated_recall",
        "gain_vs_center",
        "gain_vs_multi_offset",
        "phase_rescue_count",
        "phase_damage_count",
        "mean_phase_trend_quality",
        "mean_early_anchor_count",
    ]
    with (out_dir / "best_by_top_l.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=best_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in best_fields} for row in best_rows)


def merge_per_packet(
    records: list[dict[str, Any]],
    top_l_values: list[int],
    out_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        per_symbol_path = Path(record["per_symbol_csv"])
        if not per_symbol_path.exists():
            continue
        groups: dict[tuple[int, float, float, str], list[dict[str, str]]] = defaultdict(list)
        with per_symbol_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                groups[
                    (
                        _int(row, "packet_index"),
                        _float(row, "phase_bonus_weight"),
                        _float(row, "phase_gate_width_pi"),
                        str(row.get("phase_evidence", "")),
                    )
                ].append(row)
        for (packet_index, phase_weight, gate_width_pi, phase_evidence), items in groups.items():
            out: dict[str, Any] = {
                "stem": record["stem"],
                "snr_db": record["snr_db"],
                "preamble_len": record["preamble_len"],
                "trend_source_arg": record["trend_source"],
                "preselect_mode": record["preselect_mode"],
                "packet_index": packet_index,
                "phase_bonus_weight": phase_weight,
                "phase_gate_width_pi": gate_width_pi,
                "phase_evidence": phase_evidence,
                "symbol_count": len(items),
                "mean_trend_r2": _mean([
                    _float(item, "trend_r2")
                    for item in items
                    if str(item.get("trend_r2", "")).strip()
                ]),
                "mean_residual_std_pi": _mean([
                    _float(item, "residual_std_pi")
                    for item in items
                    if str(item.get("residual_std_pi", "")).strip()
                ]),
                "mean_early_anchor_count": _mean([
                    float(_int(item, "early_anchor_count"))
                    for item in items
                ]),
                "mean_phase_trend_quality": _mean([
                    _float(item, "phase_trend_quality")
                    for item in items
                ]),
            }
            for cutoff in top_l_values:
                center = [_int(item, f"center_top{cutoff}_hit") for item in items]
                multi = [_int(item, f"multi_top{cutoff}_hit") for item in items]
                phase = [_int(item, f"phase_gated_top{cutoff}_hit") for item in items]
                rescue = [_int(item, f"phase_rescue_top{cutoff}") for item in items]
                damage = [_int(item, f"phase_damage_top{cutoff}") for item in items]
                out[f"center_recall@{cutoff}"] = _mean([float(value) for value in center])
                out[f"multi_offset_recall@{cutoff}"] = _mean([float(value) for value in multi])
                out[f"phase_gated_recall@{cutoff}"] = _mean([float(value) for value in phase])
                out[f"phase_rescue_count@{cutoff}"] = sum(rescue)
                out[f"phase_damage_count@{cutoff}"] = sum(damage)
            rows.append(out)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (out_dir / "per_packet_recall.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def write_manifest(
    args: argparse.Namespace,
    out_dir: Path,
    cases: list[DatasetCase],
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "weak_root": str(WEAK_ROOT),
        "argv": sys.argv,
        "cases": [
            {
                "stem": case.stem,
                "snr_db": case.snr_db,
                "iq_path": str(case.iq_path),
                "symbol_csv": str(case.symbol_csv),
                "preamble_len": case.preamble_len,
            }
            for case in cases
        ],
        "records": records,
        "failures": failures,
        "options": {
            "trend_source": args.trend_source,
            "preselect_mode": args.preselect_mode,
            "phase_bonus_weights": args.phase_bonus_weights,
            "phase_gate_width_pi": args.phase_gate_width_pi,
            "top_l": args.top_l,
            "rescue_counts": args.rescue_counts,
            "phase_evidence": args.phase_evidence,
            "packet": args.packet,
            "resume": args.resume,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = DEFAULT_OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    elif not out_dir.is_absolute():
        out_dir = WEAK_ROOT / out_dir
    out_dir = out_dir.resolve()
    if not _inside_weak_root(out_dir):
        raise ValueError(f"output dir must stay inside {WEAK_ROOT}: {out_dir}")

    snrs = _parse_csv_ints(args.snr_db)
    trend_sources = _parse_csv_text(args.trend_source)
    preselect_modes = _parse_csv_text(args.preselect_mode)
    top_l_values = _parse_csv_ints(args.top_l)
    cases = discover_cases(args.dataset, snrs)
    if not cases:
        raise SystemExit("no dataset cases discovered")

    print(f"Output: {out_dir}", flush=True)
    print(f"Cases: {len(cases)}", flush=True)
    for case in cases:
        print(f"  {case.stem} snr=-{case.snr_db}dB preamble={case.preamble_len}", flush=True)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        for trend_source in trend_sources:
            for preselect_mode in preselect_modes:
                output_csv, summary_json, code = run_one(
                    case=case,
                    trend_source=trend_source,
                    preselect_mode=preselect_mode,
                    args=args,
                    out_dir=out_dir,
                )
                record = {
                    "stem": case.stem,
                    "snr_db": case.snr_db,
                    "preamble_len": case.preamble_len,
                    "trend_source": trend_source,
                    "preselect_mode": preselect_mode,
                    "per_symbol_csv": str(output_csv),
                    "summary_json": str(summary_json),
                    "returncode": code,
                }
                records.append(record)
                if code != 0:
                    failures.append(record)
                    if not args.keep_going:
                        write_manifest(args, out_dir, cases, records, failures)
                        return code

    if not args.dry_run:
        merge_summary_json(records, top_l_values, out_dir)
        merge_per_packet(records, top_l_values, out_dir)
    write_manifest(args, out_dir, cases, records, failures)
    if failures:
        print(f"Completed with {len(failures)} failed runs", flush=True)
        return 1
    if args.dry_run:
        print("Dry run complete", flush=True)
        print(f"Manifest: {out_dir / 'manifest.json'}", flush=True)
        return 0
    print("Completed sweep", flush=True)
    print(f"Merged summary: {out_dir / 'merged_summary.csv'}", flush=True)
    print(f"Best table:     {out_dir / 'best_by_top_l.csv'}", flush=True)
    print(f"Per-packet:     {out_dir / 'per_packet_recall.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
