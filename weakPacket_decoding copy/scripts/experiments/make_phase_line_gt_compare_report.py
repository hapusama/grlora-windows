#!/usr/bin/env python3
"""Build compact tables and plots for selector/GT/argmax phase-line comparison."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize phase-line comparison CSV by SNR.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="phase_line_gt_compare_summary.csv produced by compare_phase_line_to_gt_payload.py",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def mean(rows: Sequence[dict[str, str]], key: str) -> float:
    values = [value for row in rows if (value := finite_float(row.get(key))) is not None]
    return float(np.mean(values)) if values else float("nan")


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_pi(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}pi"


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    headers = [
        "SNR",
        "selected SER",
        "center FFT argmax SER",
        "lock ratio",
        "selector line R2",
        "GT noisy payload R2",
        "center FFT argmax line R2",
        "selector vs GT mean resid",
        "argmax vs GT mean resid",
    ]
    keys = [
        "snr_db",
        "selected_ser",
        "center_argmax_ser",
        "lock_ratio",
        "selector_line_r2",
        "gt_noisy_payload_r2",
        "center_argmax_line_r2",
        "selector_vs_gt_mean_resid_pi",
        "center_argmax_vs_gt_mean_resid_pi",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" for _ in headers) + " |",
    ]
    for row in rows:
        formatted: list[str] = []
        for key in keys:
            if key == "snr_db":
                formatted.append(str(int(round(float(row[key])))))
            elif key.endswith("_resid_pi"):
                formatted.append(fmt_pi(float(row[key])))
            else:
                formatted.append(fmt(float(row[key])))
        lines.append("| " + " | ".join(formatted) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg_plot(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    width = 920
    height = 520
    margin_left = 80
    margin_right = 34
    margin_top = 42
    margin_bottom = 72
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    snr_values = [float(row["snr_db"]) for row in rows]
    min_snr = min(snr_values)
    max_snr = max(snr_values)

    def x_pos(snr: float) -> float:
        if max_snr == min_snr:
            return margin_left + plot_w / 2.0
        return margin_left + (float(snr) - min_snr) * plot_w / (max_snr - min_snr)

    def y_pos(value: float) -> float:
        return margin_top + (1.0 - max(0.0, min(1.0, float(value)))) * plot_h

    series = [
        ("selector line R2", "selector_line_r2", "#2764b3"),
        ("GT noisy payload R2", "gt_noisy_payload_r2", "#2f8a46"),
        ("center FFT argmax line R2", "center_argmax_line_r2", "#b35424"),
    ]
    sorted_rows = sorted(rows, key=lambda row: float(row["snr_db"]))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:15px;fill:#20242a}.small{font-size:13px;fill:#4b5563}.title{font-size:20px;font-weight:700}.axis{stroke:#222;stroke-width:1.4}.grid{stroke:#d7dce2;stroke-width:1}.line{fill:none;stroke-width:3}.tick{stroke:#222;stroke-width:1}</style>',
        f'<text class="title" x="{margin_left}" y="28">Phase-line consistency vs SNR</text>',
    ]

    for tick in np.linspace(0.0, 1.0, 6):
        y = y_pos(float(tick))
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}"/>')
        parts.append(f'<line class="tick" x1="{margin_left - 5}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}"/>')
        parts.append(f'<text class="small" text-anchor="end" x="{margin_left - 10}" y="{y + 4:.2f}">{tick:.1f}</text>')

    for snr in sorted(snr_values):
        x = x_pos(float(snr))
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_h}"/>')
        parts.append(f'<line class="tick" x1="{x:.2f}" y1="{margin_top + plot_h}" x2="{x:.2f}" y2="{margin_top + plot_h + 5}"/>')
        parts.append(f'<text class="small" text-anchor="middle" x="{x:.2f}" y="{margin_top + plot_h + 25}">{int(round(snr))}</text>')

    parts.extend(
        [
            f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}"/>',
            f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}"/>',
            f'<text text-anchor="middle" x="{margin_left + plot_w / 2:.2f}" y="{height - 22}">SNR (dB)</text>',
            f'<text transform="translate(24 {margin_top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">phase-line R2</text>',
        ]
    )

    legend_x = margin_left + plot_w - 270
    legend_y = margin_top + 18
    for idx, (label, key, color) in enumerate(series):
        points = " ".join(
            f'{x_pos(float(row["snr_db"])):.2f},{y_pos(float(row[key])):.2f}'
            for row in sorted_rows
        )
        parts.append(f'<polyline class="line" stroke="{color}" points="{points}"/>')
        for row in sorted_rows:
            x = x_pos(float(row["snr_db"]))
            y = y_pos(float(row[key]))
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}"/>')
        y_legend = legend_y + idx * 24
        parts.append(f'<line x1="{legend_x}" y1="{y_legend}" x2="{legend_x + 30}" y2="{y_legend}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="small" x="{legend_x + 40}" y="{y_legend + 4}">{label}</text>')

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def maybe_plot(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional local dependency
        svg_path = path.with_suffix(".svg")
        print(f"matplotlib unavailable ({exc}); writing SVG fallback")
        write_svg_plot(svg_path, rows)
        return svg_path

    snr = np.asarray([float(row["snr_db"]) for row in rows], dtype=np.float64)
    order = np.argsort(snr)
    snr = snr[order]

    series = [
        ("selector line R2", "selector_line_r2", "#2764b3"),
        ("GT noisy payload R2", "gt_noisy_payload_r2", "#2f8a46"),
        ("center FFT argmax line R2", "center_argmax_line_r2", "#b35424"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for label, key, color in series:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)[order]
        ax.plot(snr, values, marker="o", linewidth=2.0, label=label, color=color)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("phase-line R2")
    ax.set_title("Phase-line consistency vs SNR")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def build_rows(summary_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in summary_rows:
        snr = finite_float(row.get("target_snr_db"))
        if snr is None:
            continue
        grouped.setdefault(float(snr), []).append(row)

    out: list[dict[str, Any]] = []
    for snr, rows in sorted(grouped.items(), reverse=True):
        out.append(
            {
                "snr_db": float(snr),
                "dataset_count": int(len(rows)),
                "packet_count": int(round(mean(rows, "packet_count") * len(rows))),
                "selected_ser": mean(rows, "mean_selected_symbol_ser"),
                "center_argmax_ser": mean(rows, "mean_center_symbol_ser"),
                "multi_offset_argmax_ser": mean(rows, "mean_multi_symbol_ser"),
                "lock_ratio": mean(rows, "mean_locked_ratio"),
                "selector_line_r2": mean(rows, "mean_selector_line_r2"),
                "gt_noisy_payload_r2": mean(rows, "mean_gt_noisy_payload_line_r2"),
                "center_argmax_line_r2": mean(rows, "mean_center_argmax_line_r2"),
                "selector_vs_gt_mean_resid_pi": mean(
                    rows, "mean_selector_vs_gt_noisy_resid_mean_abs_pi"
                ),
                "center_argmax_vs_gt_mean_resid_pi": mean(
                    rows, "mean_center_argmax_vs_gt_noisy_resid_mean_abs_pi"
                ),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    summary_csv = args.summary_csv.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else summary_csv.parent
    rows = build_rows(read_csv(summary_csv))
    csv_path = output_dir / "phase_line_gt_compare_by_snr.csv"
    md_path = output_dir / "phase_line_gt_compare_by_snr.md"
    plot_path = maybe_plot(output_dir / "phase_line_r2_by_snr.png", rows)
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"wrote={csv_path}")
    print(f"wrote={md_path}")
    print(f"wrote={plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
