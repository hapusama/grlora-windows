"""Audit FrameSync hard gates and evaluate a decoder-aware soft acceptance path."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence


SCRIPT_PATH = Path(__file__).resolve()
WEAK_PACKET_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_INPUT = (
    WEAK_PACKET_ROOT
    / "data"
    / "experiments"
    / "framesync_gate_audit_ota_awgn_20260822"
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _decoder_aware_accept(row: dict[str, Any]) -> bool:
    """Keep frame-location and netID structure, but defer bin0 purity to decode."""

    return bool(
        _truth(row.get("estimate_available", 0))
        and _truth(row.get("locator_valid", 0))
        and _truth(row.get("netid_valid", 0))
    )


def _augment_packets(
    packets: Sequence[dict[str, Any]],
    symbols: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in symbols:
        by_trial[str(row["trial_id"])].append(row)

    output: list[dict[str, Any]] = []
    for source in packets:
        row = dict(source)
        trial_symbols = by_trial[str(row["trial_id"])]
        correct = sum(
            _truth(item.get("noisy_sync_correct", 0)) for item in trial_symbols
        )
        total = len(trial_symbols)
        decodable = bool(total and correct == total)
        strict = _truth(row.get("strict_sync_success", 0))
        estimate = _truth(row.get("estimate_available", 0))
        if strict:
            gate_class = "strict_valid"
        elif not estimate:
            gate_class = "no_estimate"
        elif decodable:
            gate_class = "strict_invalid_decodable"
        else:
            gate_class = "strict_invalid_nondecode"
        row.update(
            {
                "payload_correct_symbols": correct,
                "payload_total_symbols": total,
                "payload_ser": 1.0 - correct / total if total else 1.0,
                "payload_packet_decodable": int(decodable),
                "gate_class": gate_class,
                "decoder_aware_accept": int(_decoder_aware_accept(row)),
            }
        )
        output.append(row)
    return output


def _gate_reason_counts(packets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in packets:
        gate_class = str(row["gate_class"])
        if not gate_class.startswith("strict_invalid"):
            continue
        for reason in str(row.get("gate_failure_reasons", "")).split("|"):
            if reason:
                counts[(gate_class, reason)] += 1
    return [
        {"gate_class": key[0], "gate_failure_reason": key[1], "count": value}
        for key, value in sorted(counts.items())
    ]


def _policy_summary(
    packets: Sequence[dict[str, Any]],
    esn0_values: Sequence[float],
) -> list[dict[str, Any]]:
    policies: tuple[tuple[str, Callable[[dict[str, Any]], bool]], ...] = (
        ("strict", lambda row: _truth(row["strict_sync_success"])),
        ("decoder_aware", _decoder_aware_accept),
        ("estimate_available", lambda row: _truth(row["estimate_available"])),
        ("oracle_acceptance", lambda row: _truth(row["payload_packet_decodable"])),
    )
    output: list[dict[str, Any]] = []
    for esn0_db in esn0_values:
        rows = [
            row
            for row in packets
            if math.isclose(float(row["esn0_db"]), float(esn0_db), abs_tol=1e-12)
        ]
        total_packets = len(rows)
        total_symbols = sum(int(row["payload_total_symbols"]) for row in rows)
        for name, predicate in policies:
            accepted = [row for row in rows if predicate(row)]
            correct_symbols = sum(
                int(row["payload_correct_symbols"]) for row in accepted
            )
            delivered = sum(
                _truth(row["payload_packet_decodable"]) for row in accepted
            )
            false_rejects = sum(
                _truth(row["payload_packet_decodable"]) and not predicate(row)
                for row in rows
            )
            accepted_nondecode = sum(
                not _truth(row["payload_packet_decodable"]) for row in accepted
            )
            output.append(
                {
                    "esn0_db": float(esn0_db),
                    "policy": name,
                    "packet_trials": total_packets,
                    "accepted_packets": len(accepted),
                    "acceptance_rate": len(accepted) / total_packets,
                    "delivered_packets": delivered,
                    "packet_delivery_rate": delivered / total_packets,
                    "end_to_end_ser": 1.0 - correct_symbols / total_symbols,
                    "decodable_false_rejects": false_rejects,
                    "accepted_nondecode_packets": accepted_nondecode,
                }
            )
    return output


def _make_plots(
    output_dir: Path,
    policy_rows: Sequence[dict[str, Any]],
    reason_rows: Sequence[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "strict": ("original hard gate", "tab:blue"),
        "decoder_aware": ("decoder-aware gate", "tab:orange"),
        "estimate_available": ("accept any estimate", "tab:green"),
        "oracle_acceptance": ("oracle acceptance", "black"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for policy, (label, color) in styles.items():
        rows = sorted(
            (row for row in policy_rows if row["policy"] == policy),
            key=lambda row: float(row["esn0_db"]),
        )
        x = [float(row["esn0_db"]) for row in rows]
        if policy != "oracle_acceptance":
            axes[0].plot(
                x,
                [float(row["end_to_end_ser"]) for row in rows],
                marker="o",
                color=color,
                label=label,
            )
        axes[1].plot(
            x,
            [float(row["packet_delivery_rate"]) for row in rows],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].set_ylabel("end-to-end SER")
    axes[1].set_ylabel("packet delivery rate (16/16 symbols)")
    for axis in axes:
        axis.set_xlabel("Es/N0 (dB)")
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "framesync_gate_policy_comparison.png", dpi=180)
    plt.close(fig)

    reasons = sorted({str(row["gate_failure_reason"]) for row in reason_rows})
    classes = ("strict_invalid_decodable", "strict_invalid_nondecode")
    labels = ("decodable false rejects", "non-decodable rejects")
    positions = list(range(len(reasons)))
    width = 0.38
    fig, axis = plt.subplots(figsize=(10.0, 4.8))
    for offset, (gate_class, label) in enumerate(zip(classes, labels)):
        counts = [
            next(
                (
                    int(row["count"])
                    for row in reason_rows
                    if row["gate_class"] == gate_class
                    and row["gate_failure_reason"] == reason
                ),
                0,
            )
            for reason in reasons
        ]
        axis.bar(
            [value + (offset - 0.5) * width for value in positions],
            counts,
            width=width,
            label=label,
        )
    axis.set_xticks(positions, [item.replace("framesync_", "") for item in reasons])
    axis.tick_params(axis="x", rotation=25)
    axis.set_ylabel("packet trials")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "framesync_gate_failure_reasons.png", dpi=180)
    plt.close(fig)


def _build_report(
    output_dir: Path,
    packets: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    reason_rows: Sequence[dict[str, Any]],
    esn0_values: Sequence[float],
) -> None:
    false_rejects = [
        row for row in packets if row["gate_class"] == "strict_invalid_decodable"
    ]
    bin0_false_rejects = sum(
        "framesync_preamble_all_bin0"
        in str(row.get("gate_failure_reasons", "")).split("|")
        for row in false_rejects
    )
    lines = [
        "# Decoder-aware FrameSync gate audit",
        "",
        f"- Packet trials: {len(packets)}",
        f"- Strict-invalid but 16/16-symbol decodable: {len(false_rejects)}",
        f"- Those failing the all-preamble-bin0 gate: {bin0_false_rejects}/{len(false_rejects)}",
        "- Decoder-aware policy: require a valid frame location and valid netID; defer the all-bin0 check to downstream FEC/CRC.",
        "",
        "| Es/N0 | strict SER | decoder-aware SER | any-estimate SER | strict PDR | decoder-aware PDR | oracle-accept PDR | decoder-aware false rejects | decoder-aware nondecode accepts |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for esn0_db in esn0_values:
        by_policy = {
            str(row["policy"]): row
            for row in summary
            if math.isclose(float(row["esn0_db"]), float(esn0_db), abs_tol=1e-12)
        }
        strict = by_policy["strict"]
        soft = by_policy["decoder_aware"]
        available = by_policy["estimate_available"]
        oracle = by_policy["oracle_acceptance"]
        lines.append(
            f"| {float(esn0_db):g} | {float(strict['end_to_end_ser']):.4f} | "
            f"{float(soft['end_to_end_ser']):.4f} | "
            f"{float(available['end_to_end_ser']):.4f} | "
            f"{float(strict['packet_delivery_rate']):.3f} | "
            f"{float(soft['packet_delivery_rate']):.3f} | "
            f"{float(oracle['packet_delivery_rate']):.3f} | "
            f"{int(soft['decodable_false_rejects'])} | "
            f"{int(soft['accepted_nondecode_packets'])} |"
        )
    lines.extend(
        [
            "",
            "`accepted_nondecode_packets` is a compute/latency cost, not an undetected payload delivery when CRC remains mandatory.",
            "",
            "## Failure counts",
            "",
        ]
    )
    for row in reason_rows:
        lines.append(
            f"- {row['gate_class']} / {row['gate_failure_reason']}: {row['count']}"
        )
    (output_dir / "GATE_AUDIT_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--esn0-db", default="12,13,14,15,16")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    input_dir = args.input_dir.resolve()
    esn0_values = tuple(
        float(item.strip()) for item in str(args.esn0_db).split(",") if item.strip()
    )
    packets = _augment_packets(
        _read_csv(input_dir / "packet_trials.csv"),
        _read_csv(input_dir / "symbol_trials.csv"),
    )
    reason_rows = _gate_reason_counts(packets)
    policy_rows = _policy_summary(packets, esn0_values)
    _write_csv(input_dir / "packet_gate_audit.csv", packets)
    _write_csv(input_dir / "gate_failure_counts.csv", reason_rows)
    _write_csv(input_dir / "gate_policy_summary.csv", policy_rows)
    _make_plots(input_dir, policy_rows, reason_rows)
    _build_report(input_dir, packets, policy_rows, reason_rows, esn0_values)
    print(
        json.dumps(
            {"input_dir": str(input_dir), "policy_summary": policy_rows},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
