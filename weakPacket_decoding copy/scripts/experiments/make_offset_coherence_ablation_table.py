#!/usr/bin/env python3
"""Merge offset-coherence ablation sweeps into paper-style tables."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WEAK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = WEAK_ROOT / "data" / "ablation_offset_coherence_summary"


@dataclass(frozen=True)
class Variant:
    variant_id: str
    name: str
    output_dir: str
    method: str
    sweep: str
    multi_offset_energy: str
    top_l: str
    locking: str
    offset_coherence: str
    packet_line: str
    candidate_expansion: str = ""
    include_thresholds: bool = True


FULL_VARIANTS = [
    Variant("A0", "center argmax", "ablation_current_default", "center", "full", "no", "1", "no", "no", "no"),
    Variant("A1", "multi-offset argmax", "ablation_current_default", "multi", "full", "yes", "1", "no", "no", "no"),
    Variant("A2", "energy-only selected Top-24", "ablation_energy_only_top24", "selected", "full", "yes", "24", "yes", "no", "no"),
    Variant("A3", "offset coherence only Top-24", "ablation_coherence_only_top24", "selected", "full", "yes", "24", "yes", "yes", "no"),
    Variant("A4", "energy + offset coherence, no line", "ablation_amp_coherence_no_line", "selected", "full", "yes", "24", "yes", "yes", "no"),
    Variant("A5", "current default", "ablation_current_default", "selected", "full", "yes", "24", "yes", "yes", "small"),
    Variant("A6", "packet-line phase only Top-24", "ablation_packet_line_only", "selected", "full", "yes", "24", "yes", "no", "yes"),
    Variant("A7-L8", "Top-L 8 default weights", "ablation_topL_8", "selected", "full", "yes", "8", "yes", "yes", "small"),
    Variant("A7-L16", "Top-L 16 default weights", "ablation_topL_16", "selected", "full", "yes", "16", "yes", "yes", "small"),
    Variant("A7-L24", "Top-L 24 default weights", "ablation_topL_24", "selected", "full", "yes", "24", "yes", "yes", "small"),
    Variant("A7-L32", "Top-L 32 default weights", "ablation_topL_32", "selected", "full", "yes", "32", "yes", "yes", "small"),
    Variant("A8", "no high-confidence lock", "ablation_no_high_conf_lock", "selected", "full", "yes", "24", "no", "yes", "small"),
]

PROBE_VARIANTS = [
    Variant("A5", "current default", "ablation_current_default", "selected", "full", "yes", "24", "yes", "yes", "small", include_thresholds=False),
    Variant("A9-C32", "coherence candidate Top-32", "ablation_coherence_candidate_top32", "selected", "probe -20..-23", "yes", "24", "yes", "yes", "small", "32", False),
    Variant("A9-C64", "coherence candidate Top-64", "ablation_coherence_candidate_top64", "selected", "probe -20..-23", "yes", "24", "yes", "yes", "small", "64", False),
    Variant("A9-C128", "coherence candidate Top-128", "ablation_coherence_candidate_top128", "selected", "probe -20..-23", "yes", "24", "yes", "yes", "small", "128", False),
    Variant("A10", "smooth trajectory beam probe", "ablation_smooth_beam_probe", "selected", "probe -20..-23", "yes", "24", "yes", "yes", "smooth", "", False),
]

METRICS = ["SER<=10%", "accuracy>=90%", "CRC/PRR>=90%", "CRC/PRR>=80%", "CRC/PRR>=50%"]
SNR_POINTS = [-20.0, -21.0, -22.0, -23.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offset-coherence ablation summary tables.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def fmt(value: float | None, digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def threshold_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], float | None]:
    out: dict[tuple[str, str], float | None] = {}
    for row in rows:
        if row.get("dataset") != "mean_of_datasets":
            continue
        out[(row.get("method", ""), row.get("metric", ""))] = as_float(row.get("threshold_snr_db", ""))
    return out


def mean_curve_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("dataset") == "mean_of_datasets"]


def curve_by_snr(rows: list[dict[str, str]]) -> dict[float, dict[str, str]]:
    out: dict[float, dict[str, str]] = {}
    for row in mean_curve_rows(rows):
        snr = as_float(row.get("target_snr_db", ""))
        if snr is not None:
            out[snr] = row
    return out


def avg_curve_metric(curve: dict[float, dict[str, str]], key: str) -> float | None:
    values = [as_float(curve[snr].get(key, "")) for snr in SNR_POINTS if snr in curve]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def method_curve_value(row: dict[str, str], method: str, suffix: str) -> float | None:
    return as_float(row.get(f"{method}_{suffix}", ""))


def threshold_summary_row(variant: Variant) -> dict[str, Any]:
    base = WEAK_ROOT / "data" / variant.output_dir
    thresholds = threshold_map(read_csv(base / "threshold_table.csv"))
    curve = curve_by_snr(read_csv(base / "snr_curve_summary.csv"))

    center_ser = thresholds.get(("center", "SER<=10%"))
    multi_ser = thresholds.get(("multi", "SER<=10%"))
    method_ser = thresholds.get((variant.method, "SER<=10%"))

    row: dict[str, Any] = {
        "variant_id": variant.variant_id,
        "variant": variant.name,
        "output_dir": f"data/{variant.output_dir}",
        "sweep": variant.sweep,
        "method": variant.method,
        "multi_offset_energy": variant.multi_offset_energy,
        "top_l": variant.top_l,
        "locking": variant.locking,
        "offset_coherence": variant.offset_coherence,
        "packet_line": variant.packet_line,
        "candidate_expansion": variant.candidate_expansion,
    }

    for metric in METRICS:
        method_thr = thresholds.get((variant.method, metric))
        center_thr = thresholds.get(("center", metric))
        multi_thr = thresholds.get(("multi", metric))
        key = {
            "SER<=10%": "ser10",
            "accuracy>=90%": "acc90",
            "CRC/PRR>=90%": "crc90",
            "CRC/PRR>=80%": "crc80",
            "CRC/PRR>=50%": "crc50",
        }[metric]
        row[f"{key}_threshold_snr_db"] = fmt(method_thr)
        row[f"{key}_gain_vs_center_db"] = fmt(None if method_thr is None or center_thr is None else center_thr - method_thr)
        if variant.method == "center":
            row[f"{key}_gain_vs_multi_db"] = ""
        else:
            row[f"{key}_gain_vs_multi_db"] = fmt(None if method_thr is None or multi_thr is None else multi_thr - method_thr)

    for snr in SNR_POINTS:
        curve_row = curve.get(snr, {})
        label = f"m{abs(int(snr))}"
        row[f"{label}_ser"] = fmt(method_curve_value(curve_row, variant.method, "symbol_ser"), 3)
        row[f"{label}_crc_prr"] = fmt(method_curve_value(curve_row, variant.method, "crc_valid_rate"), 3)

    if variant.method == "selected":
        row["mean_locked_ratio_m20_m23"] = fmt(avg_curve_metric(curve, "mean_locked_ratio"), 3)
        row["mean_false_lock_rate_m20_m23"] = fmt(avg_curve_metric(curve, "mean_false_lock_rate"), 3)
        row["mean_uncertain_candidate_recall_m20_m23"] = fmt(avg_curve_metric(curve, "mean_uncertain_candidate_recall"), 3)
        row["mean_selected_offset_coherence_m20_m23"] = fmt(avg_curve_metric(curve, "mean_selected_offset_coherence"), 3)
    else:
        row["mean_locked_ratio_m20_m23"] = ""
        row["mean_false_lock_rate_m20_m23"] = ""
        row["mean_uncertain_candidate_recall_m20_m23"] = ""
        row["mean_selected_offset_coherence_m20_m23"] = ""
    row["ser_gain_over_multi_db"] = "" if variant.method == "center" else fmt(None if method_ser is None or multi_ser is None else multi_ser - method_ser)
    row["ser_gain_over_center_db"] = fmt(None if method_ser is None or center_ser is None else center_ser - method_ser)
    return row


def probe_curve_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    default_curve = curve_by_snr(read_csv(WEAK_ROOT / "data" / "ablation_current_default" / "snr_curve_summary.csv"))
    for variant in PROBE_VARIANTS:
        curve = curve_by_snr(read_csv(WEAK_ROOT / "data" / variant.output_dir / "snr_curve_summary.csv"))
        for snr in SNR_POINTS:
            row = curve.get(snr, {})
            default_row = default_curve.get(snr, {})
            selected_ser = method_curve_value(row, variant.method, "symbol_ser")
            default_ser = method_curve_value(default_row, "selected", "symbol_ser")
            selected_crc = method_curve_value(row, variant.method, "crc_valid_rate")
            default_crc = method_curve_value(default_row, "selected", "crc_valid_rate")
            rows.append(
                {
                    "variant_id": variant.variant_id,
                    "variant": variant.name,
                    "output_dir": f"data/{variant.output_dir}",
                    "sweep": variant.sweep,
                    "snr_db": int(snr),
                    "selected_ser": fmt(selected_ser, 3),
                    "selected_crc_prr": fmt(selected_crc, 3),
                    "ser_delta_vs_default": fmt(None if selected_ser is None or default_ser is None else selected_ser - default_ser, 3),
                    "crc_delta_vs_default": fmt(None if selected_crc is None or default_crc is None else selected_crc - default_crc, 3),
                    "mean_locked_ratio": fmt(as_float(row.get("mean_locked_ratio", "")), 3),
                    "mean_false_lock_rate": fmt(as_float(row.get("mean_false_lock_rate", "")), 3),
                    "mean_uncertain_candidate_recall": fmt(as_float(row.get("mean_uncertain_candidate_recall", "")), 3),
                    "mean_selected_offset_coherence": fmt(as_float(row.get("mean_selected_offset_coherence", "")), 3),
                }
            )
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(path: Path, threshold_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]]) -> None:
    main_ids = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A8"]
    main_rows = [row for row in threshold_rows if row["variant_id"] in main_ids]
    top_rows = [row for row in threshold_rows if str(row["variant_id"]).startswith("A7")]
    probe_summary: list[dict[str, Any]] = []
    for variant_id in ["A5", "A9-C32", "A9-C64", "A9-C128", "A10"]:
        group = [row for row in probe_rows if row["variant_id"] == variant_id]
        if not group:
            continue
        probe_summary.append(
            {
                "variant_id": variant_id,
                "variant": group[0]["variant"],
                "mean_ser_m20_m23": fmt(sum(float(row["selected_ser"]) for row in group) / len(group), 3),
                "mean_crc_m20_m23": fmt(sum(float(row["selected_crc_prr"]) for row in group) / len(group), 3),
                "mean_ser_delta_vs_default": fmt(sum(float(row["ser_delta_vs_default"]) for row in group) / len(group), 3),
                "mean_crc_delta_vs_default": fmt(sum(float(row["crc_delta_vs_default"]) for row in group) / len(group), 3),
            }
        )

    lines: list[str] = [
        "# Offset-Coherence Ablation Summary",
        "",
        "Scope: `gr-lora_sdr/weakPacket_decoding copy`",
        "",
        "Formal full sweeps use `--snr-start -12 --snr-stop -26 --snr-step -1` on the three validation datasets.",
        "Probe rows use only `-20..-23 dB`, so they should be compared by curve values rather than full threshold gains.",
        "",
        "## Main Full-Sweep Ablation",
        "",
    ]
    lines.extend(
        markdown_table(
            main_rows,
            ["variant_id", "variant", "multi_offset_energy", "top_l", "locking", "offset_coherence", "packet_line", "ser10_gain_vs_center_db", "crc90_gain_vs_center_db", "ser10_gain_vs_multi_db", "m22_ser", "mean_uncertain_candidate_recall_m20_m23"],
            ["ID", "Variant", "Multi", "Top-L", "Lock", "Coherence", "Line", "SER Gain", "CRC90 Gain", "SER Gain vs Multi", "SER@-22", "Recall"],
        )
    )
    lines.extend(["", "## Top-L Full-Sweep Ablation", ""])
    lines.extend(
        markdown_table(
            top_rows,
            ["variant_id", "top_l", "ser10_gain_vs_center_db", "crc90_gain_vs_center_db", "ser10_gain_vs_multi_db", "m20_ser", "m21_ser", "m22_ser", "m23_ser", "mean_uncertain_candidate_recall_m20_m23"],
            ["ID", "Top-L", "SER Gain", "CRC90 Gain", "SER Gain vs Multi", "SER@-20", "SER@-21", "SER@-22", "SER@-23", "Recall"],
        )
    )
    lines.extend(["", "## Quick Probes (-20..-23 dB)", ""])
    lines.extend(
        markdown_table(
            probe_summary,
            ["variant_id", "variant", "mean_ser_m20_m23", "mean_crc_m20_m23", "mean_ser_delta_vs_default", "mean_crc_delta_vs_default"],
            ["ID", "Variant", "Mean SER", "Mean CRC/PRR", "SER Delta vs Default", "CRC Delta vs Default"],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Energy-only selected is identical to multi-offset argmax, so the selected-path machinery alone does not create extra gain.",
            "- Offset coherence adds a clear threshold gain over multi-offset energy; amplitude protection is important because coherence-only is weaker.",
            "- Packet-line phase alone is not competitive. The current default uses it only as a small auxiliary term.",
            "- Top-24 is the best full-sweep low-complexity setting in this matrix; Top-32 increases recall but does not improve the formal SER/CRC thresholds.",
            "- Disabling high-confidence locks degrades the threshold, supporting the claim that locks protect already-reliable symbols from over-reranking.",
            "- Coherence-candidate expansion and smooth beam probes do not justify replacing the default on the -20..-23 dB probe range.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    threshold_rows = [threshold_summary_row(variant) for variant in FULL_VARIANTS]
    probe_rows = probe_curve_rows()
    write_csv(out_dir / "ablation_threshold_summary.csv", threshold_rows)
    write_csv(out_dir / "ablation_probe_curve_m20_m23.csv", probe_rows)
    write_report(out_dir / "ablation_report.md", threshold_rows, probe_rows)
    print(f"wrote={out_dir / 'ablation_threshold_summary.csv'}")
    print(f"wrote={out_dir / 'ablation_probe_curve_m20_m23.csv'}")
    print(f"wrote={out_dir / 'ablation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
