#!/usr/bin/env python3
"""Build a compact paper-style threshold table from joint sweep summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


WEAK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARIES = [
    WEAK_ROOT / "data" / "phase_guided" / "joint_residual_sweep_map_threshold" / "joint_residual_sweep_summary.csv",
    WEAK_ROOT / "data" / "phase_guided" / "joint_residual_sweep_map" / "joint_residual_sweep_summary.csv",
]
DEFAULT_OUT = WEAK_ROOT / "data" / "phase_guided" / "paper_tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge argmax-vs-MAP joint sweep summaries into paper tables."
    )
    parser.add_argument("--summary-csv", type=Path, nargs="*", default=DEFAULT_SUMMARIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-stem", default="threshold_argmax_vs_map_joint")
    parser.add_argument("--ser-threshold", type=float, nargs="*", default=[0.1],
                        help="SER thresholds for rough SNR crossing estimates")
    return parser.parse_args()


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    by_snr: dict[int, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                snr = _int(row, "snr_db", 0)
                argmax_ser = _float(row, "argmax_ser", 0.0)
                independent_ser = _float(row, "independent_ser", 0.0)
                app_compared = _int(row, "app_compared", 0)
                joint_app = _int(row, "joint_app_exact", 0)
                independent_app = _int(row, "independent_app_exact", 0)
                by_snr[snr] = {
                    "snr_db": snr,
                    "argmax_ser": argmax_ser,
                    "map_independent_ser": independent_ser,
                    "ser_reduction": argmax_ser - independent_ser,
                    "argmax_correct_symbols": _int(row, "argmax_correct_symbols", 0),
                    "independent_correct_symbols": _int(row, "independent_correct_symbols", 0),
                    "total_symbols": _int(row, "total_symbols", 0),
                    "independent_app_exact": f"{independent_app}/{app_compared}",
                    "joint_app_exact": f"{joint_app}/{app_compared}",
                    "joint_model_margin": _float(row, "joint_model_margin", 0.0),
                    "mean_joint_candidate_rank": _float(row, "mean_joint_candidate_rank", 0.0),
                }
    return [by_snr[key] for key in sorted(by_snr, reverse=True)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "snr_db",
        "argmax_ser",
        "map_independent_ser",
        "ser_reduction",
        "argmax_correct_symbols",
        "independent_correct_symbols",
        "total_symbols",
        "independent_app_exact",
        "joint_app_exact",
        "joint_model_margin",
        "mean_joint_candidate_rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Argmax vs Phase-MAP Joint Threshold Table",
        "",
        "| SNR | Argmax SER | MAP residual SER | SER reduction | Independent app | Joint app | Joint margin |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {snr_db} dB | {argmax_ser:.4f} | {map_independent_ser:.4f} | "
            "{ser_reduction:.4f} | {independent_app_exact} | {joint_app_exact} | "
            "{joint_model_margin:.6f} |".format(**row)
        )
    lines.extend([
        "",
        "Notes:",
        "- MAP residual SER is the independent codec-projected residual search result.",
        "- Joint app is the hard affine session trajectory result.",
        "- The target claim should emphasize robust 3 dB+ improvement over ordinary argmax, not the lowest SNR point alone.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def estimate_crossing(rows: list[dict[str, Any]], key: str, target: float) -> float | None:
    ordered = sorted(rows, key=lambda row: float(row["snr_db"]), reverse=True)
    for hi, lo in zip(ordered, ordered[1:]):
        y_hi = float(hi[key])
        y_lo = float(lo[key])
        if y_hi <= target <= y_lo and y_lo != y_hi:
            x_hi = float(hi["snr_db"])
            x_lo = float(lo["snr_db"])
            frac = (float(target) - y_hi) / (y_lo - y_hi)
            return x_hi + frac * (x_lo - x_hi)
        if y_hi == target:
            return float(hi["snr_db"])
    if ordered and float(ordered[-1][key]) == target:
        return float(ordered[-1]["snr_db"])
    return None


def threshold_report(rows: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for target in thresholds:
        argmax_cross = estimate_crossing(rows, "argmax_ser", float(target))
        map_cross = estimate_crossing(rows, "map_independent_ser", float(target))
        gain = (
            float(argmax_cross - map_cross)
            if argmax_cross is not None and map_cross is not None else None
        )
        report.append({
            "ser_threshold": float(target),
            "argmax_crossing_snr_db": argmax_cross,
            "map_residual_crossing_snr_db": map_cross,
            "estimated_gain_db": gain,
            "note": "linear interpolation over sparse SNR sweep points",
        })
    return report


def append_threshold_markdown(path: Path, report: list[dict[str, Any]]) -> None:
    lines = [
        "",
        "## Rough Threshold Estimates",
        "",
        "| SER target | Argmax crossing | MAP residual crossing | Estimated gain |",
        "|---:|---:|---:|---:|",
    ]
    for row in report:
        def fmt(value: Any) -> str:
            return "" if value is None else f"{float(value):.2f} dB"
        lines.append(
            f"| {float(row['ser_threshold']):.3f} | "
            f"{fmt(row['argmax_crossing_snr_db'])} | "
            f"{fmt(row['map_residual_crossing_snr_db'])} | "
            f"{fmt(row['estimated_gain_db'])} |"
        )
    lines.extend([
        "",
        "These are rough linear interpolations over sparse SNR sweep points.",
    ])
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    rows = read_rows([p.resolve() for p in args.summary_csv])
    if not rows:
        raise ValueError("no summary rows found")
    out_dir = args.output_dir.resolve()
    csv_path = out_dir / f"{args.output_stem}.csv"
    md_path = out_dir / f"{args.output_stem}.md"
    json_path = out_dir / f"{args.output_stem}_thresholds.json"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    report = threshold_report(rows, [float(v) for v in args.ser_threshold])
    append_threshold_markdown(md_path, report)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Threshold JSON: {json_path}")
    for row in rows:
        print(
            f"  {row['snr_db']:>4} dB "
            f"argmax_SER={row['argmax_ser']:.4f} "
            f"map_SER={row['map_independent_ser']:.4f} "
            f"joint_app={row['joint_app_exact']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
