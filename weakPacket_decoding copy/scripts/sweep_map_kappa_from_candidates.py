#!/usr/bin/env python3
"""Offline kappa sweep for MAP-style residual candidate likelihoods.

The MAP residual scorer writes per-candidate likelihood components to CSV.
This script reuses those exported candidates and rescales only the phase
concentration parameter ``kappa``:

    phase_ll(kappa) = kappa * mean(cos(phase_residual))

No IQ is reread and no FFTs are recomputed.  The goal is to calibrate whether
the phase likelihood is under- or over-weighted before rerunning expensive
front-end experiments.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

WEAK_ROOT = Path(__file__).resolve().parent.parent

import sys

if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.payload_codec import decode_explicit_frame_symbols


DEFAULT_ROOT = WEAK_ROOT / "data" / "phase_guided" / "joint_residual_sweep_map"
DEFAULT_GT = WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / "0_0_0_10_14_16_header_first_symbols.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore exported MAP residual candidates over kappa."
    )
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--snr", type=int, nargs="*", default=[-20, -23, -25, -27])
    parser.add_argument("--kappa", type=float, nargs="*",
                        default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--source-kappa", type=float, default=2.0,
                        help="kappa used when the candidate CSV was generated")
    parser.add_argument("--phase-weight", type=float, default=0.25)
    parser.add_argument("--line-weight", type=float, default=0.50)
    parser.add_argument("--amp-weight", type=float, default=0.20)
    parser.add_argument("--profile-weight", type=float, default=0.05)
    parser.add_argument("--byte-index", type=int, default=6)
    parser.add_argument("--expected-affine", type=str, default="1,1")
    parser.add_argument("-g", "--gt-symbol-csv", type=Path, default=DEFAULT_GT)
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--bw", type=float, default=125000.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    return parser.parse_args()


def snr_stem(snr_db: int) -> str:
    return f"0_0_0_10_14_16_snr_m{abs(int(snr_db))}dB"


def candidate_csv_for(root: Path, snr_db: int) -> Path:
    stem = snr_stem(snr_db)
    return root / f"snr_m{abs(int(snr_db))}dB" / f"{stem}_residual_candidates.csv"


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def _candidate_byte(row: dict[str, str], byte_index: int) -> int | None:
    value = str(row.get("candidate_values", "")).strip()
    for part in value.split(";"):
        item = part.strip()
        if not item or ":" not in item:
            continue
        idx_s, byte_s = item.split(":", 1)
        try:
            if int(idx_s) == int(byte_index):
                return int(byte_s, 16)
        except ValueError:
            continue
    return None


def load_expected_payloads_from_symbols(
    path: Path,
    sf: int,
    bw: float,
    ldro_mode: int,
) -> dict[int, str]:
    if not path.exists():
        return {}
    grouped: dict[tuple[int, int], dict[str, list[int]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if _int(row, "header_valid", 0) != 1:
                continue
            stage = str(row.get("stage", "")).strip()
            if stage not in {"header", "payload"}:
                continue
            key = (_int(row, "packet_index", -1), _int(row, "event_index", -1))
            value = _int(row, "symbol_value", -1)
            if key[0] < 0 or key[1] < 0 or value < 0:
                continue
            grouped.setdefault(key, {"header": [], "payload": []})[stage].append(value)
    result: dict[int, str] = {}
    for (packet_index, _event_index), symbols in sorted(grouped.items()):
        if len(symbols["header"]) != 8 or not symbols["payload"]:
            continue
        decoded = decode_explicit_frame_symbols(
            symbols["header"],
            symbols["payload"],
            sf=int(sf),
            bw=float(bw),
            ldro_mode=int(ldro_mode),
        )
        if decoded.payload.crc_valid:
            result.setdefault(int(packet_index), decoded.payload.payload_bytes.hex())
    return result


def load_candidates(
    path: Path,
    byte_index: int,
    kappa: float,
    source_kappa: float,
    weights: tuple[float, float, float, float],
) -> dict[int, dict[int, dict[str, Any]]]:
    w_phase, w_line, w_amp, w_profile = weights
    w_sum = max(1e-12, w_phase + w_line + w_amp + w_profile)
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            packet = _int(row, "packet_index", -1)
            value = _candidate_byte(row, byte_index=int(byte_index))
            if packet < 0 or value is None:
                continue
            source_phase_ll = _float(row, "map_phase_ll", 0.0)
            mean_cos = source_phase_ll / max(1e-12, float(source_kappa))
            phase_ll = float(kappa) * mean_cos
            line_ll = _float(row, "map_line_ll", 0.0)
            amp_ll = _float(row, "map_amp_ll", 0.0)
            profile_ll = _float(row, "map_profile_ll", 0.0)
            score = (
                w_phase * phase_ll
                + w_line * line_ll
                + w_amp * amp_ll
                + w_profile * profile_ll
            ) / w_sum
            rec = dict(row)
            rec["_score"] = float(score)
            rec["_byte_value"] = int(value)
            rec["_rescored_phase_ll"] = float(phase_ll)
            old = grouped.setdefault(packet, {}).get(int(value))
            if old is None or float(score) > float(old["_score"]):
                grouped[packet][int(value)] = rec
    return grouped


def best_affine(
    grouped: dict[int, dict[int, dict[str, Any]]],
) -> tuple[float, tuple[int, int], float, tuple[int, int], dict[int, int]]:
    packets = sorted(grouped)
    best_score = float("-inf")
    second_score = float("-inf")
    best_model = (0, 0)
    second_model = (0, 0)
    best_values: dict[int, int] = {}
    for slope in range(256):
        for intercept in range(256):
            total = 0.0
            values: dict[int, int] = {}
            ok = True
            for packet in packets:
                value = (int(slope) * int(packet) + int(intercept)) & 0xFF
                rec = grouped[packet].get(value)
                if rec is None:
                    ok = False
                    break
                total += float(rec["_score"])
                values[packet] = int(value)
            if not ok:
                continue
            if total > best_score:
                second_score = best_score
                second_model = best_model
                best_score = float(total)
                best_model = (int(slope), int(intercept))
                best_values = values
            elif total > second_score:
                second_score = float(total)
                second_model = (int(slope), int(intercept))
    return best_score, best_model, second_score, second_model, best_values


def summarize_one(
    grouped: dict[int, dict[int, dict[str, Any]]],
    expected_payloads: dict[int, str],
    expected_affine: tuple[int, int] | None,
) -> dict[str, Any]:
    best_score, best_model, second_score, second_model, best_values = best_affine(grouped)
    packets = sorted(grouped)
    joint_exact = 0
    independent_exact = 0
    compared = 0
    rank_sum = 0.0
    rank_count = 0
    top1 = 0
    for packet in packets:
        value = int(best_values.get(packet, -1))
        rec = grouped[packet].get(value, {})
        ranked = sorted(
            grouped[packet].values(),
            key=lambda item: float(item.get("_score", 0.0)),
            reverse=True,
        )
        independent = ranked[0]
        joint_rank = None
        for pos, item in enumerate(ranked, start=1):
            if int(item.get("_byte_value", -1)) == value:
                joint_rank = int(pos)
                break
        if joint_rank is not None:
            rank_count += 1
            rank_sum += float(joint_rank)
            top1 += int(joint_rank == 1)
        expected = expected_payloads.get(packet, "")
        if expected:
            compared += 1
            joint_exact += int(str(rec.get("candidate_payload_hex", "")).lower() == expected)
            independent_exact += int(
                str(independent.get("candidate_payload_hex", "")).lower() == expected
            )
    return {
        "packets": len(packets),
        "affine_slope": int(best_model[0]),
        "affine_intercept": int(best_model[1]),
        "second_affine_slope": int(second_model[0]),
        "second_affine_intercept": int(second_model[1]),
        "model_score": float(best_score),
        "model_margin": float(best_score - second_score),
        "model_margin_per_packet": float((best_score - second_score) / max(1, len(packets))),
        "joint_top1_count": int(top1),
        "mean_joint_candidate_rank": float(rank_sum / rank_count) if rank_count else "",
        "app_compared": int(compared),
        "joint_app_exact": int(joint_exact),
        "independent_app_exact": int(independent_exact),
        "expected_affine_ok": (
            int(best_model == expected_affine) if expected_affine is not None else ""
        ),
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "snr_db",
        "kappa",
        "packets",
        "affine_slope",
        "affine_intercept",
        "second_affine_slope",
        "second_affine_intercept",
        "model_score",
        "model_margin",
        "model_margin_per_packet",
        "joint_top1_count",
        "mean_joint_candidate_rank",
        "app_compared",
        "joint_app_exact",
        "independent_app_exact",
        "expected_affine_ok",
        "candidate_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    args = parse_args()
    root = args.candidate_root.resolve()
    summary_csv = (
        args.summary_csv.resolve()
        if args.summary_csv else root / "map_kappa_sweep_summary.csv"
    )
    expected_payloads = load_expected_payloads_from_symbols(
        args.gt_symbol_csv.resolve(),
        sf=int(args.sf),
        bw=float(args.bw),
        ldro_mode=int(args.ldro_mode),
    )
    expected_affine = None
    if args.expected_affine:
        a_s, b_s = str(args.expected_affine).split(",", 1)
        expected_affine = (int(a_s) & 0xFF, int(b_s) & 0xFF)
    weights = (
        max(0.0, float(args.phase_weight)),
        max(0.0, float(args.line_weight)),
        max(0.0, float(args.amp_weight)),
        max(0.0, float(args.profile_weight)),
    )

    rows: list[dict[str, Any]] = []
    for snr_db in args.snr:
        candidate_csv = candidate_csv_for(root, int(snr_db))
        if not candidate_csv.exists():
            raise FileNotFoundError(candidate_csv)
        for kappa in args.kappa:
            grouped = load_candidates(
                candidate_csv,
                byte_index=int(args.byte_index),
                kappa=float(kappa),
                source_kappa=float(args.source_kappa),
                weights=weights,
            )
            row = summarize_one(grouped, expected_payloads, expected_affine)
            row.update({
                "snr_db": int(snr_db),
                "kappa": float(kappa),
                "candidate_csv": str(candidate_csv),
            })
            rows.append(row)
            write_summary(summary_csv, rows)
    print(f"Summary: {summary_csv}")
    for row in rows:
        print(
            f"  {row['snr_db']:>4} dB kappa={float(row['kappa']):>4.1f} "
            f"margin={float(row['model_margin']):.6f} "
            f"rank={row['mean_joint_candidate_rank']} "
            f"joint_app={row['joint_app_exact']}/{row['app_compared']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
