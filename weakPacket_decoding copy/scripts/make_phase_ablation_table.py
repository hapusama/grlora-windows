#!/usr/bin/env python3
"""Build paper-style phase-kappa ablation tables from kappa sweep summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


WEAK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARIES = [
    WEAK_ROOT / "data" / "phase_guided" / "joint_residual_sweep_map_threshold" / "map_kappa_sweep_summary.csv",
    WEAK_ROOT / "data" / "phase_guided" / "joint_residual_sweep_map" / "map_kappa_sweep_summary.csv",
]
DEFAULT_OUT = WEAK_ROOT / "data" / "phase_guided" / "paper_tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge MAP kappa sweeps into a phase-ablation table."
    )
    parser.add_argument("--summary-csv", type=Path, nargs="*", default=DEFAULT_SUMMARIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-stem", default="phase_kappa_ablation")
    return parser.parse_args()


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                app_compared = _int(row, "app_compared", 0)
                joint_app = _int(row, "joint_app_exact", 0)
                independent_app = _int(row, "independent_app_exact", 0)
                rows.append({
                    "snr_db": _int(row, "snr_db", 0),
                    "kappa": _float(row, "kappa", 0.0),
                    "joint_app_exact": f"{joint_app}/{app_compared}",
                    "independent_app_exact": f"{independent_app}/{app_compared}",
                    "model_margin": _float(row, "model_margin", 0.0),
                    "mean_joint_candidate_rank": _float(row, "mean_joint_candidate_rank", 0.0),
                    "expected_affine_ok": _int(row, "expected_affine_ok", 0),
                })
    rows.sort(key=lambda item: (item["snr_db"], item["kappa"]), reverse=False)
    return rows


def pivot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_snr: dict[int, dict[str, Any]] = {}
    kappas = sorted({float(row["kappa"]) for row in rows})
    for row in rows:
        snr = int(row["snr_db"])
        rec = by_snr.setdefault(snr, {"snr_db": snr})
        k = f"{float(row['kappa']):g}"
        rec[f"k{k}_joint_app"] = row["joint_app_exact"]
        rec[f"k{k}_margin"] = f"{float(row['model_margin']):.6f}"
        rec[f"k{k}_rank"] = f"{float(row['mean_joint_candidate_rank']):.3f}"
    return [by_snr[snr] for snr in sorted(by_snr, reverse=True)], kappas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = ["snr_db"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: Path, rows: list[dict[str, Any]], kappas: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    k_labels = [f"{k:g}" for k in kappas]
    lines = [
        "# Phase Kappa Ablation",
        "",
        "| SNR | " + " | ".join(f"k={k} app" for k in k_labels) + " |",
        "|---:|" + "|".join("---:" for _ in k_labels) + "|",
    ]
    for row in rows:
        cells = [str(row.get(f"k{k}_joint_app", "")) for k in k_labels]
        lines.append(f"| {row['snr_db']} dB | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Margins:",
        "",
        "| SNR | " + " | ".join(f"k={k} margin" for k in k_labels) + " |",
        "|---:|" + "|".join("---:" for _ in k_labels) + "|",
    ])
    for row in rows:
        cells = [str(row.get(f"k{k}_margin", "")) for k in k_labels]
        lines.append(f"| {row['snr_db']} dB | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Notes:",
        "- k=0 removes the phase likelihood term.",
        "- App exactness is joint affine session decoding.",
        "- The deeper weak points show when phase becomes necessary rather than only confidence-improving.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = read_rows([p.resolve() for p in args.summary_csv])
    if not rows:
        raise ValueError("no kappa summary rows found")
    table_rows, kappas = pivot(rows)
    out_dir = args.output_dir.resolve()
    csv_path = out_dir / f"{args.output_stem}.csv"
    md_path = out_dir / f"{args.output_stem}.md"
    write_csv(csv_path, table_rows)
    write_markdown(md_path, table_rows, kappas)
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    for row in table_rows:
        print(f"  {row['snr_db']:>4} dB " + " ".join(
            f"k={k:g}:{row.get(f'k{k:g}_joint_app', '')}" for k in kappas
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
