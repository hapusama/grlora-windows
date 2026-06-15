#!/usr/bin/env python3
"""Build a compact table for extra-capture Phase-MAP validation sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


WEAK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARIES = [
    WEAK_ROOT / "data" / "phase_guided" / "joint_residual_sweep_map_preamble8_validation" / "joint_residual_sweep_summary.csv",
    WEAK_ROOT / "data" / "phase_guided" / "joint_residual_sweep_map_preamble32_validation" / "joint_residual_sweep_summary.csv",
]
DEFAULT_OUT = WEAK_ROOT / "data" / "phase_guided" / "paper_tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge extra-capture validation sweep summaries."
    )
    parser.add_argument("--summary-csv", type=Path, nargs="*", default=DEFAULT_SUMMARIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-stem", default="generalization_capture_validation")
    return parser.parse_args()


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def _label_from_path(path: Path) -> str:
    text = str(path.parent.name)
    if "preamble8" in text:
        return "0_0_0_10_14_8"
    if "preamble32" in text:
        return "0_0_0_10_14_32"
    return text


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        capture = _label_from_path(path)
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                app_compared = _int(row, "app_compared", 0)
                independent_app = _int(row, "independent_app_exact", 0)
                joint_app = _int(row, "joint_app_exact", 0)
                argmax_ser = _float(row, "argmax_ser", 0.0)
                independent_ser = _float(row, "independent_ser", 0.0)
                rows.append({
                    "capture": capture,
                    "snr_db": _int(row, "snr_db", 0),
                    "packets": _int(row, "packets", 0),
                    "argmax_ser": argmax_ser,
                    "map_independent_ser": independent_ser,
                    "ser_reduction": argmax_ser - independent_ser,
                    "independent_app_exact": f"{independent_app}/{app_compared}",
                    "joint_app_exact": f"{joint_app}/{app_compared}",
                    "joint_model_margin": _float(row, "joint_model_margin", 0.0),
                    "mean_joint_candidate_rank": _float(row, "mean_joint_candidate_rank", 0.0),
                })
    rows.sort(key=lambda item: (item["capture"], -int(item["snr_db"])))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "capture",
        "snr_db",
        "packets",
        "argmax_ser",
        "map_independent_ser",
        "ser_reduction",
        "independent_app_exact",
        "joint_app_exact",
        "joint_model_margin",
        "mean_joint_candidate_rank",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Extra-Capture Phase-MAP Validation",
        "",
        "| Capture | SNR | Argmax SER | MAP residual SER | SER reduction | Independent app | Joint app | Joint rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {capture} | {snr_db} dB | {argmax_ser:.4f} | "
            "{map_independent_ser:.4f} | {ser_reduction:.4f} | "
            "{independent_app_exact} | {joint_app_exact} | "
            "{mean_joint_candidate_rank:.3f} |".format(**row)
        )
    lines.extend([
        "",
        "Notes:",
        "- These are sanity/generalization checks on additional USRP_IQ captures.",
        "- They reuse the same Phase-MAP residual candidate scoring and hard affine joint layer.",
        "- The preamble-32 -23 dB point is intentionally kept as a boundary case: symbol SER improves, but app-level joint recovery is not perfect.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = read_rows([path.resolve() for path in args.summary_csv])
    if not rows:
        raise ValueError("no validation summary rows found")
    out_dir = args.output_dir.resolve()
    csv_path = out_dir / f"{args.output_stem}.csv"
    md_path = out_dir / f"{args.output_stem}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    for row in rows:
        print(
            f"  {row['capture']} {row['snr_db']:>4} dB "
            f"argmax_SER={row['argmax_ser']:.4f} "
            f"map_SER={row['map_independent_ser']:.4f} "
            f"joint_app={row['joint_app_exact']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
