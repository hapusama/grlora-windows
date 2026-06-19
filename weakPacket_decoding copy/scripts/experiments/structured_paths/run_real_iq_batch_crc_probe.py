#!/usr/bin/env python3
"""Batch CRC/PRR comparison for real USRP IQ captures.

This runner orchestrates the existing real-IQ pipeline:

1. run_weak_sync_chain.py finds preambles and frame boundaries.
2. run_header_first_demod.py decodes the explicit LoRa header and payload bins.
3. run_real_iq_crc_probe.py compares traditional FFT, current decoder,
   Savaux paper OSR, and Savaux+codec by packet-level CRC.

Real lab captures in data/USRP_IQ do not include byte/symbol ground truth, so
the output is PRR/CRC-only rather than SER/BER.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


WEAK_ROOT = Path(__file__).resolve().parents[3]
GR_LORA_ROOT = WEAK_ROOT.parent
USRP_IQ_ROOT = GR_LORA_ROOT / "data" / "USRP_IQ"
SCRIPT_ROOT = WEAK_ROOT / "scripts"
DEFAULT_OUTPUT_DIR = WEAK_ROOT / "data" / "baseline_comparison" / "real_iq_crc_batch_methods"

DEFAULT_REPRESENTATIVE_STEMS = (
    "1_0_10_11_2_16",
    "1_1_0_11_2_16",
    "1_0_10_12_2_16",
    "1_1_0_12_2_16",
    "1_0_10_12_6_16",
    "1_1_0_12_6_16",
    "1_0_10_12_10_16",
    "1_1_0_12_10_16",
    "1_0_10_12_14_16",
    "1_1_0_12_14_16",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch real-IQ CRC/PRR comparison across traditional FFT, current, Savaux, and Savaux+codec."
    )
    parser.add_argument(
        "--iq-root",
        type=Path,
        default=USRP_IQ_ROOT,
        help="Root containing USRP IQ .bin captures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for sync/header/probe artifacts and summaries.",
    )
    parser.add_argument(
        "--captures",
        nargs="+",
        default=None,
        help="Capture stems, filenames, or paths. Default uses a representative SF/TP/location set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every .bin under --iq-root. This can take a long time.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of captures after selection.")
    parser.add_argument(
        "--per-param-limit",
        type=int,
        default=None,
        help="Optional evenly spaced limit per (SF, tx_power) group after selection.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate existing sync/header/probe outputs.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed capture.")

    parser.add_argument("--bw", type=float, default=125000.0)
    parser.add_argument("--samp-rate", type=float, default=500000.0)
    parser.add_argument("--center-freq", type=float, default=487.7e6)
    parser.add_argument("--sync-word", default="0x34")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--ldro-mode", type=int, default=2)

    parser.add_argument("--win-chirps", type=int, default=4)
    parser.add_argument("--hop-chirps", type=float, default=None, help="Sync detector hop in chirps.")
    parser.add_argument("--hop-samples", type=int, default=None, help="Sync detector hop in samples.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Only scan the first N IQ samples in sync.")
    parser.add_argument("--max-windows", type=int, default=None, help="Only scan the first N sync detector windows.")
    parser.add_argument("--min-periodic-peaks", type=int, default=None)
    parser.add_argument("--frame-min-preamble-peaks", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--align-step-samples", type=int, default=None, help="Override sync alignment search step.")
    parser.add_argument("--frame-step-samples", type=int, default=None, help="Override SFD frame search step.")
    parser.add_argument("--frame-filter", choices=("framesync-valid", "netid-valid", "frame-valid", "all"), default="framesync-valid")
    parser.add_argument("--header-max-frames", type=int, default=None, help="Maximum selected sync candidates to demod.")
    parser.add_argument("--probe-max-packets", type=int, default=None, help="Maximum paired packets to evaluate in CRC probe.")

    parser.add_argument("--nibble-candidates", type=int, default=16)
    parser.add_argument("--row-beam-width", type=int, default=2048)
    parser.add_argument("--block-candidate-limit", type=int, default=512)
    parser.add_argument("--block-symbol-seed-top-m", type=int, default=0)
    parser.add_argument("--block-symbol-seed-deep-top-l", type=int, default=0)
    parser.add_argument("--block-symbol-seed-max-deep-positions", type=int, default=1)
    parser.add_argument("--block-symbol-seed-quota", type=int, default=0)
    parser.add_argument("--block-symbol-seed-max-combinations", type=int, default=50000)
    parser.add_argument("--global-beam-width", type=int, default=2048)
    parser.add_argument("--global-rank-diverse-top-r", type=int, default=0)
    parser.add_argument("--global-rank-diverse-beam-width", type=int, default=0)
    parser.add_argument("--global-rank-cost-max", type=float, default=0.0)
    parser.add_argument("--global-rank-cost-state-limit", type=int, default=0)
    parser.add_argument("--final-candidate-limit", type=int, default=2048)
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
    return parser.parse_args()


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


def _read_one_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return dict(rows[0]) if rows else {}


def infer_capture_params(path: Path) -> dict[str, int | str]:
    parts = path.stem.split("_")
    if len(parts) < 6:
        raise ValueError(f"cannot infer LoRa params from filename: {path.name}")
    try:
        position_token = parts[2]
        return {
            "experiment_id": int(parts[0]),
            "corridor_id": int(parts[1]),
            "position_id": int(position_token) if position_token.lstrip("-").isdigit() else position_token,
            "sf": int(parts[3]),
            "tx_power_dbm": int(parts[4]),
            "preamble_len": int(parts[-1]),
        }
    except ValueError as exc:
        raise ValueError(f"cannot infer LoRa params from filename: {path.name}") from exc


def _find_by_stem(iq_root: Path, item: str) -> Path | None:
    candidate = Path(item)
    if candidate.exists():
        return candidate.resolve()
    name = candidate.name
    stem = name[:-4] if name.lower().endswith(".bin") else name
    matches = sorted(iq_root.rglob(f"{stem}.bin"))
    return matches[0].resolve() if matches else None


def select_captures(args: argparse.Namespace) -> list[Path]:
    iq_root = args.iq_root.resolve()
    if args.all:
        selected = sorted(iq_root.rglob("*.bin"))
    else:
        requested = args.captures if args.captures is not None else list(DEFAULT_REPRESENTATIVE_STEMS)
        selected = []
        missing: list[str] = []
        for item in requested:
            path = _find_by_stem(iq_root, str(item))
            if path is None:
                missing.append(str(item))
            else:
                selected.append(path)
        if missing:
            print(f"warning: skipped missing captures: {', '.join(missing)}", file=sys.stderr)
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    unique: dict[str, Path] = {}
    for path in selected:
        unique[str(path.resolve()).lower()] = path.resolve()
    selected = list(unique.values())
    if args.per_param_limit is not None:
        selected = _limit_per_param_group(selected, int(args.per_param_limit))
    return selected


def _evenly_spaced(items: Sequence[Path], limit: int) -> list[Path]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    last = len(items) - 1
    indices = sorted({int(round(i * last / (limit - 1))) for i in range(limit)})
    return [items[index] for index in indices]


def _limit_per_param_group(paths: Sequence[Path], limit: int) -> list[Path]:
    groups: dict[tuple[int, int], list[Path]] = {}
    passthrough: list[Path] = []
    for path in paths:
        try:
            params = infer_capture_params(path)
            key = (int(params["sf"]), int(params["tx_power_dbm"]))
        except Exception:
            passthrough.append(path)
            continue
        groups.setdefault(key, []).append(path)

    selected: list[Path] = []
    for key in sorted(groups):
        selected.extend(_evenly_spaced(sorted(groups[key]), limit))
    selected.extend(passthrough)
    return selected


def run_step(
    label: str,
    cmd: Sequence[str],
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path,
) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with stdout_path.open("w", encoding="utf-8", newline="") as stdout, stderr_path.open("w", encoding="utf-8", newline="") as stderr:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - start
    print(f"  {label}: rc={proc.returncode} elapsed={elapsed:.1f}s")
    return int(proc.returncode), float(elapsed)


def output_paths(base_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "sync_csv": base_dir / "sync_chain" / f"{stem}_sync_chain.csv",
        "events_csv": base_dir / "events" / f"{stem}_events.csv",
        "windows_csv": base_dir / "windows" / f"{stem}_windows.csv",
        "header_symbols_csv": base_dir / "header_first" / f"{stem}_header_first_symbols.csv",
        "header_frames_csv": base_dir / "header_first" / f"{stem}_header_first_frames.csv",
        "probe_dir": base_dir / "probe",
        "logs_dir": base_dir / "logs" / stem,
    }


def run_capture(iq_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    params = infer_capture_params(iq_path)
    sf = int(params["sf"])
    preamble_len = int(params["preamble_len"])
    stem = iq_path.stem
    paths = output_paths(args.output_dir.resolve(), stem)
    row: dict[str, Any] = {
        "file_name": iq_path.name,
        "input_iq": str(iq_path),
        **params,
        "status": "ok",
        "error": "",
    }

    sync_cmd = [
        sys.executable,
        str(SCRIPT_ROOT / "run_weak_sync_chain.py"),
        "-i",
        str(iq_path),
        "-o",
        str(paths["sync_csv"]),
        "--events-csv",
        str(paths["events_csv"]),
        "--windows-csv",
        str(paths["windows_csv"]),
        "--sf",
        str(sf),
        "--bw",
        str(args.bw),
        "--samp-rate",
        str(args.samp_rate),
        "--center-freq",
        str(args.center_freq),
        "--sync-word",
        str(args.sync_word),
        "--preamble-len",
        str(preamble_len),
        "--win-chirps",
        str(args.win_chirps),
    ]
    if args.min_periodic_peaks is not None:
        sync_cmd += ["--min-periodic-peaks", str(args.min_periodic_peaks)]
    if args.hop_chirps is not None:
        sync_cmd += ["--hop-chirps", str(args.hop_chirps)]
    if args.hop_samples is not None:
        sync_cmd += ["--hop-samples", str(args.hop_samples)]
    if args.frame_min_preamble_peaks is not None:
        sync_cmd += ["--frame-min-preamble-peaks", str(args.frame_min_preamble_peaks)]
    if args.max_events is not None:
        sync_cmd += ["--max-events", str(args.max_events)]
    if args.sample_limit is not None:
        sync_cmd += ["--sample-limit", str(args.sample_limit)]
    if args.max_windows is not None:
        sync_cmd += ["--max-windows", str(args.max_windows)]
    if args.align_step_samples is not None:
        sync_cmd += ["--align-step-samples", str(args.align_step_samples)]
    if args.frame_step_samples is not None:
        sync_cmd += ["--frame-step-samples", str(args.frame_step_samples)]

    header_cmd = [
        sys.executable,
        str(SCRIPT_ROOT / "run_header_first_demod.py"),
        "-i",
        str(iq_path),
        "-s",
        str(paths["sync_csv"]),
        "-o",
        str(paths["header_symbols_csv"]),
        "--frames-output",
        str(paths["header_frames_csv"]),
        "--sf",
        str(sf),
        "--bw",
        str(args.bw),
        "--samp-rate",
        str(args.samp_rate),
        "--ldro-mode",
        str(args.ldro_mode),
        "--frame-filter",
        str(args.frame_filter),
    ]
    if args.header_max_frames is not None:
        header_cmd += ["--max-frames", str(args.header_max_frames)]

    probe_cmd = [
        sys.executable,
        str(SCRIPT_ROOT / "experiments" / "structured_paths" / "run_real_iq_crc_probe.py"),
        "-i",
        str(iq_path),
        "-s",
        str(paths["header_symbols_csv"]),
        "--output-dir",
        str(paths["probe_dir"]),
        "--crc-mode",
        str(args.crc_mode),
        "--ldro-mode",
        str(args.ldro_mode),
        "--nibble-candidates",
        str(args.nibble_candidates),
        "--row-beam-width",
        str(args.row_beam_width),
        "--block-candidate-limit",
        str(args.block_candidate_limit),
        "--block-symbol-seed-top-m",
        str(args.block_symbol_seed_top_m),
        "--block-symbol-seed-deep-top-l",
        str(args.block_symbol_seed_deep_top_l),
        "--block-symbol-seed-max-deep-positions",
        str(args.block_symbol_seed_max_deep_positions),
        "--block-symbol-seed-quota",
        str(args.block_symbol_seed_quota),
        "--block-symbol-seed-max-combinations",
        str(args.block_symbol_seed_max_combinations),
        "--global-beam-width",
        str(args.global_beam_width),
        "--global-rank-diverse-top-r",
        str(args.global_rank_diverse_top_r),
        "--global-rank-diverse-beam-width",
        str(args.global_rank_diverse_beam_width),
        "--global-rank-cost-max",
        str(args.global_rank_cost_max),
        "--global-rank-cost-state-limit",
        str(args.global_rank_cost_state_limit),
        "--final-candidate-limit",
        str(args.final_candidate_limit),
        "--crc-candidate-max-beam-rank",
        str(args.crc_candidate_max_beam_rank),
        "--crc-candidate-high-rank-max-beam-rank",
        str(args.crc_candidate_high_rank_max_beam_rank),
        "--crc-candidate-high-rank-min-evidence-margin",
        str(args.crc_candidate_high_rank_min_evidence_margin),
        "--crc-selection-policy",
        str(args.crc_selection_policy),
        "--crc-non-earliest-min-evidence-margin",
        str(args.crc_non_earliest_min_evidence_margin),
        "--crc-failure-selection-policy",
        str(args.crc_failure_selection_policy),
        "--crc-failure-max-symbol-changes",
        str(args.crc_failure_max_symbol_changes),
        "--crc-failure-min-evidence-margin",
        str(args.crc_failure_min_evidence_margin),
        "--crc-state-search-block-top-r",
        str(args.crc_state_search_block_top_r),
        "--crc-state-search-state-limit",
        str(args.crc_state_search_state_limit),
        "--crc-state-search-keep-per-key",
        str(args.crc_state_search_keep_per_key),
        "--crc-state-search-max-candidates",
        str(args.crc_state_search_max_candidates),
        "--crc-state-search-min-evidence-margin",
        str(args.crc_state_search_min_evidence_margin),
    ]
    if args.probe_max_packets is not None:
        probe_cmd += ["--max-packets", str(args.probe_max_packets)]

    steps = (
        ("sync", sync_cmd, paths["sync_csv"]),
        ("header", header_cmd, paths["header_symbols_csv"]),
        ("probe", probe_cmd, paths["probe_dir"] / f"{stem}_real_iq_crc_summary.csv"),
    )
    for label, cmd, required_path in steps:
        if not args.force and required_path.exists():
            row[f"{label}_returncode"] = 0
            row[f"{label}_elapsed_s"] = "cached"
            continue
        rc, elapsed = run_step(
            label,
            cmd,
            stdout_path=paths["logs_dir"] / f"{label}.stdout.log",
            stderr_path=paths["logs_dir"] / f"{label}.stderr.log",
            cwd=WEAK_ROOT,
        )
        row[f"{label}_returncode"] = rc
        row[f"{label}_elapsed_s"] = f"{elapsed:.3f}"
        if rc != 0:
            row["status"] = f"{label}_failed"
            stderr = paths["logs_dir"] / f"{label}.stderr.log"
            if stderr.exists():
                text = stderr.read_text(encoding="utf-8", errors="replace").strip()
                row["error"] = text[-800:]
            return row

    summary_path = paths["probe_dir"] / f"{stem}_real_iq_crc_summary.csv"
    if summary_path.exists():
        summary = _read_one_csv(summary_path)
        for key, value in summary.items():
            row[f"probe_{key}"] = value
    else:
        row["status"] = "probe_summary_missing"
    return row


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("sf", "")), str(row.get("tx_power_dbm", "")))
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    rate_keys = (
        "probe_traditional_fft_crc_valid_rate",
        "probe_current_selected_crc_valid_rate",
        "probe_savaux_paper_crc_valid_rate",
        "probe_savaux_codec_crc_valid_rate",
    )
    for (sf, tx_power), items in sorted(groups.items(), key=lambda item: (int(item[0][0] or 0), int(item[0][1] or 0))):
        ok_items = [item for item in items if str(item.get("status", "")) == "ok"]
        total_headers = sum(int(float(item.get("probe_header_valid_packets", 0) or 0)) for item in ok_items)
        row: dict[str, Any] = {
            "sf": sf,
            "tx_power_dbm": tx_power,
            "capture_count": len(items),
            "ok_capture_count": len(ok_items),
            "header_valid_packets": total_headers,
        }
        for key in rate_keys:
            weighted = 0.0
            denom = 0
            for item in ok_items:
                packets = int(float(item.get("probe_header_valid_packets", 0) or 0))
                try:
                    rate = float(item.get(key, "nan"))
                except (TypeError, ValueError):
                    continue
                weighted += rate * packets
                denom += packets
            row[key] = weighted / denom if denom else ""
        out.append(row)
    return out


def main() -> int:
    args = parse_args()
    captures = select_captures(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not captures:
        raise SystemExit("no captures selected")

    rows: list[dict[str, Any]] = []
    for index, iq_path in enumerate(captures, start=1):
        print(f"[{index}/{len(captures)}] {iq_path.name}")
        try:
            row = run_capture(iq_path, args)
        except Exception as exc:
            row = {
                "file_name": iq_path.name,
                "input_iq": str(iq_path),
                "status": "exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        _write_csv(args.output_dir / "real_iq_batch_crc_summary.csv", rows)
        if str(row.get("status", "")) != "ok" and args.fail_fast:
            break

    group_rows = aggregate(rows)
    _write_csv(args.output_dir / "real_iq_batch_crc_group_summary.csv", group_rows)
    manifest = {
        "capture_count": len(captures),
        "ok_capture_count": sum(1 for row in rows if str(row.get("status", "")) == "ok"),
        "output_dir": str(args.output_dir.resolve()),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote={args.output_dir / 'real_iq_batch_crc_summary.csv'}")
    print(f"wrote={args.output_dir / 'real_iq_batch_crc_group_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
