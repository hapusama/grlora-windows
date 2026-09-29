"""Assemble the HANDOFF_20260820 Step-1 ablation panel from run directories.

Combines the official K=4 run, the K=1 candidate-0 run, and the
upstream-soft runs into the six-arm decomposition the handoff asks for:

    A  candidate-0 hard        (strict / ridge K=1 hard)
    B  candidate-0 upstream soft
    C  candidate-0 current soft
    D  ridge K=4 upstream soft
    E  ridge K=4 current soft
    F  oracle-sync versions

Also cross-audits strict/relaxed columns across runs sharing seeds: they must
agree trial-by-trial because the AWGN realization is seeded per trial.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def by_trial(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row["trial_id"]): row for row in rows}


def rate(rows: Sequence[dict[str, str]], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(int(row.get(key, 0)) for row in rows) / len(rows)


def interpolate_threshold(points: Sequence[tuple[float, float]], target: float) -> float:
    ordered = sorted(points)
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return float("nan")


def collect(rows: Sequence[dict[str, str]], key: str) -> list[tuple[float, float]]:
    snrs = sorted({float(row["esn0_db"]) for row in rows})
    return [(snr, rate([r for r in rows if float(r["esn0_db"]) == snr], key)) for snr in snrs]


def cross_audit(
    label_a: str,
    rows_a: Sequence[dict[str, str]],
    label_b: str,
    rows_b: Sequence[dict[str, str]],
    columns: Sequence[str],
) -> list[str]:
    trials_a = by_trial(rows_a)
    trials_b = by_trial(rows_b)
    shared = sorted(set(trials_a) & set(trials_b))
    problems: list[str] = []
    for column in columns:
        diffs = [
            trial_id
            for trial_id in shared
            if trials_a[trial_id].get(column, "") != trials_b[trial_id].get(column, "")
        ]
        if diffs:
            problems.append(
                f"{label_a} vs {label_b}: {column} differs on "
                f"{len(diffs)}/{len(shared)} shared trials "
                f"(first: {diffs[0]})"
            )
    if not shared:
        problems.append(f"{label_a} vs {label_b}: no shared trials")
    return problems


ARMS: list[tuple[str, str, str]] = [
    ("A0 strict (cand-0 hard, strict gates)", "strict_packet_delivered", "k4"),
    ("A1 relaxed hard (cand-0)", "decoder_aware_packet_delivered", "k4"),
    ("A2 ridge K=1 hard (cand-0 + refine)", "ridge_packet_delivered", "k1"),
    ("B  ridge K=1 upstream soft (cand-0)", "ridge_upstream_packet_delivered", "k1_up"),
    ("C  ridge K=1 current soft (cand-0)", "ridge_soft_packet_delivered", "k1"),
    ("D  ridge K=4 upstream soft", "ridge_upstream_packet_delivered", "k4_up"),
    ("E  ridge K=4 current soft", "ridge_soft_packet_delivered", "k4"),
    ("F1 hard oracle", "oracle_packet_delivered", "k4"),
    ("F2 current-soft oracle", "soft_oracle_packet_delivered", "k4"),
    ("F3 upstream-soft oracle", "upstream_oracle_packet_delivered", "k4_up"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k4", type=Path, required=True, help="official K=4 10-seed run")
    parser.add_argument("--k1", type=Path, required=True, help="K=1 candidate-0 run")
    parser.add_argument(
        "--k4-upstream",
        type=Path,
        required=True,
        help="K=4 run with --ridge-upstream-soft",
    )
    parser.add_argument(
        "--k1-upstream",
        type=Path,
        default=None,
        help="optional K=1 run with --ridge-upstream-soft",
    )
    parser.add_argument("--output", type=Path, default=None, help="write report here")
    args = parser.parse_args()

    sources: dict[str, list[dict[str, str]]] = {
        "k4": read_rows(args.k4 / "packet_trials.csv"),
        "k1": read_rows(args.k1 / "packet_trials.csv"),
        "k4_up": read_rows(args.k4_upstream / "packet_trials.csv"),
    }
    if args.k1_upstream is not None:
        sources["k1_up"] = read_rows(args.k1_upstream / "packet_trials.csv")

    lines: list[str] = []
    lines.append("# Step-1 ablation panel (auto-generated)\n")

    lines.append("## Cross-run determinism audit (shared seeds must agree)\n")
    audit_columns = ["strict_packet_delivered", "decoder_aware_packet_delivered"]
    problems: list[str] = []
    problems += cross_audit("k4", sources["k4"], "k1", sources["k1"], audit_columns)
    problems += cross_audit(
        "k4", sources["k4"], "k4_up", sources["k4_up"], audit_columns
    )
    if "k1_up" in sources:
        problems += cross_audit(
            "k1", sources["k1"], "k1_up", sources["k1_up"], audit_columns
        )
    lines.append("- " + "\n- ".join(problems) if problems else "- all shared trials agree")
    lines.append("")

    lines.append("## Per-arm PDR by Es/N0\n")
    header = ["arm", "n"] + [
        f"{snr:g}dB" for snr in sorted({float(r['esn0_db']) for r in sources['k4']})
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for arm_label, column, source in ARMS:
        rows = sources.get(source)
        if rows is None:
            continue
        snrs = sorted({float(r["esn0_db"]) for r in rows})
        cells = [f"{rate([r for r in rows if float(r['esn0_db']) == s], column) * 100:.1f}%"
                 for s in snrs]
        if snrs != sorted({float(r['esn0_db']) for r in sources['k4']}):
            cells = [f"{s:g}dB:{c}" for s, c in zip(snrs, cells)]
        lines.append(f"| {arm_label} | {len(rows)} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Exact-delivery totals and thresholds\n")
    lines.append("| arm | exact/total | PDR50 dB | PDR80 dB |")
    lines.append("|---|---|---|---|")
    for arm_label, column, source in ARMS:
        rows = sources.get(source)
        if rows is None:
            continue
        exact = sum(int(r.get(column, 0)) for r in rows)
        points = collect(rows, column)
        pdr50 = interpolate_threshold(points, 0.5)
        pdr80 = interpolate_threshold(points, 0.8)
        lines.append(
            f"| {arm_label} | {exact}/{len(rows)} | "
            f"{pdr50:.3f} | {pdr80:.3f} |"
        )
    lines.append("")

    lines.append("## False deliveries (CRC-valid wrong packets)\n")
    for label, column, source in [
        ("K=1 current soft", "ridge_soft_crc_false_delivery", "k1"),
        ("K=4 upstream soft", "ridge_upstream_crc_false_delivery", "k4_up"),
        ("K=1 upstream soft", "ridge_upstream_crc_false_delivery", "k1_up"),
    ]:
        rows = sources.get(source)
        if rows is None:
            continue
        total = sum(int(r.get(column, 0)) for r in rows)
        lines.append(f"- {label}: {total}")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
