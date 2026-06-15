#!/usr/bin/env python3
"""Jointly decode dynamic residual bytes from candidate likelihood CSVs.

The per-packet residual search in ``run_phase_guided_demod.py`` emits a score
for every codec-projected byte candidate.  This script adds a session-level
layer: search a simple affine byte trajectory across packets and select the
candidate value on each packet from that trajectory.  It is deliberately small
and offline; it exists to test the paper idea that phase evidence is stronger
when used as a session-consistent likelihood instead of independent argmax-like
decisions.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any


WEAK_ROOT = Path(__file__).resolve().parent.parent
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.payload_codec import decode_explicit_frame_symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint affine residual-byte decoding from candidate likelihoods."
    )
    parser.add_argument("-c", "--candidate-csv", type=Path, required=True,
                        help="CSV written by run_phase_guided_demod --write-residual-candidates")
    parser.add_argument("-o", "--output-csv", type=Path, required=True,
                        help="output per-packet joint decisions")
    parser.add_argument("--summary-json", type=Path, default=None,
                        help="optional JSON summary path")
    parser.add_argument("--score-field", default="signal_score",
                        choices=("signal_score", "score"),
                        help="candidate likelihood to use for joint decoding")
    parser.add_argument("--slope", type=int, nargs="*", default=None,
                        help="allowed affine slopes modulo 256; default 0..255")
    parser.add_argument("--intercept", type=int, nargs="*", default=None,
                        help="allowed affine intercepts modulo 256; default 0..255")
    parser.add_argument("--byte-index", type=int, default=6,
                        help="residual byte index encoded in candidate_values")
    parser.add_argument("--expected-affine", type=str, default=None,
                        help="optional expected affine a,b for reporting, e.g. 1,1")
    parser.add_argument("--outlier-penalty", type=float, default=None,
                        help="optional score penalty for letting a packet use its independent top candidate")
    parser.add_argument("--outlier-mode", choices=("joint", "posthoc"), default="joint",
                        help="joint searches affine and outliers together; posthoc fixes affine first")
    parser.add_argument("--max-outliers", type=int, default=None,
                        help="maximum packets allowed to leave the affine trajectory")
    parser.add_argument("--outlier-min-top1-gap", type=float, default=0.0,
                        help="minimum packet top1-top2 score gap required to allow an outlier")
    parser.add_argument("--outlier-min-score-std", type=float, default=0.0,
                        help="minimum packet candidate score std required to allow an outlier")
    parser.add_argument("--outlier-max-entropy", type=float, default=1.0,
                        help="maximum normalized candidate entropy required to allow an outlier")
    parser.add_argument("-g", "--gt-symbol-csv", type=Path, default=None,
                        help="optional header-first symbol CSV for app-payload exactness")
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--bw", type=float, default=125000.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    return parser.parse_args()


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


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


def read_candidates(path: Path, byte_index: int, score_field: str) -> dict[int, dict[int, dict[str, Any]]]:
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            packet_index = _int(row, "packet_index", -1)
            value = _candidate_byte(row, byte_index=int(byte_index))
            if packet_index < 0 or value is None:
                continue
            score = _float(row, str(score_field), 0.0)
            rec = dict(row)
            rec["_score"] = float(score)
            rec["_byte_value"] = int(value)
            old = grouped.setdefault(packet_index, {}).get(int(value))
            if old is None or float(score) > float(old["_score"]):
                grouped[packet_index][int(value)] = rec
    return grouped


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def candidate_score_stats(ranked: list[dict[str, Any]]) -> dict[str, float]:
    """Return scale-aware confidence diagnostics for one packet's candidates."""
    scores = [float(item.get("_score", 0.0)) for item in ranked]
    if not scores:
        return {
            "top1_gap": 0.0,
            "score_std": 0.0,
            "entropy_norm": 1.0,
        }
    top1_gap = float(scores[0] - scores[1]) if len(scores) > 1 else 0.0
    mean = float(sum(scores) / len(scores))
    var = float(sum((s - mean) ** 2 for s in scores) / max(1, len(scores)))
    std = math.sqrt(max(0.0, var))
    scale = std if std > 1e-12 else 1.0
    max_s = max(scores)
    exps = [math.exp(max(-80.0, min(80.0, (s - max_s) / scale))) for s in scores]
    denom = sum(exps)
    if denom <= 0.0 or len(scores) <= 1:
        entropy_norm = 0.0
    else:
        probs = [v / denom for v in exps]
        entropy = -sum(p * math.log(max(p, 1e-300)) for p in probs)
        entropy_norm = float(entropy / math.log(len(scores)))
    return {
        "top1_gap": top1_gap,
        "score_std": float(std),
        "entropy_norm": float(entropy_norm),
    }


def main() -> int:
    args = parse_args()
    grouped = read_candidates(
        args.candidate_csv.resolve(),
        byte_index=int(args.byte_index),
        score_field=str(args.score_field),
    )
    if not grouped:
        raise ValueError(f"no candidates found in {args.candidate_csv}")

    packets = sorted(grouped)
    slopes = [int(v) & 0xFF for v in args.slope] if args.slope else list(range(256))
    intercepts = (
        [int(v) & 0xFF for v in args.intercept]
        if args.intercept else list(range(256))
    )

    best_score = float("-inf")
    second_score = float("-inf")
    best_model = (0, 0)
    second_model = (0, 0)
    best_values: dict[int, int] = {}
    best_affine_values: dict[int, int] = {}
    best_outliers: dict[int, int] = {}
    best_outlier_count = 0
    ranked_by_packet = {
        packet: sorted(
            grouped[packet].values(),
            key=lambda item: float(item.get("_score", 0.0)),
            reverse=True,
        )
        for packet in packets
    }
    stats_by_packet = {
        packet: candidate_score_stats(ranked_by_packet[packet])
        for packet in packets
    }
    independent_by_packet = {
        packet: ranked_by_packet[packet][0]
        for packet in packets
    }
    for slope in slopes:
        for intercept in intercepts:
            total = 0.0
            values: dict[int, int] = {}
            affine_values: dict[int, int] = {}
            outliers: dict[int, int] = {}
            ok = True
            for packet in packets:
                value = (int(slope) * int(packet) + int(intercept)) & 0xFF
                rec = grouped[packet].get(value)
                if rec is None:
                    ok = False
                    break
                chosen_value = int(value)
                chosen_score = float(rec["_score"])
                affine_values[packet] = int(value)
                if args.outlier_penalty is not None and args.outlier_mode == "joint":
                    independent = independent_by_packet[packet]
                    independent_value = int(independent.get("_byte_value", -1)) & 0xFF
                    independent_score = (
                        float(independent.get("_score", 0.0))
                        - float(args.outlier_penalty)
                    )
                    stats = stats_by_packet[packet]
                    confidence_ok = (
                        float(stats["top1_gap"]) >= float(args.outlier_min_top1_gap)
                        and float(stats["score_std"]) >= float(args.outlier_min_score_std)
                        and float(stats["entropy_norm"]) <= float(args.outlier_max_entropy)
                    )
                    if (
                        confidence_ok
                        and independent_value != int(value)
                        and independent_score > chosen_score
                    ):
                        chosen_value = int(independent_value)
                        chosen_score = float(independent_score)
                        outliers[packet] = 1
                total += float(chosen_score)
                values[packet] = int(chosen_value)
            if (
                args.max_outliers is not None
                and len(outliers) > int(args.max_outliers)
            ):
                ok = False
            if not ok:
                continue
            if total > best_score:
                second_score = best_score
                second_model = best_model
                best_score = float(total)
                best_model = (int(slope), int(intercept))
                best_values = values
                best_affine_values = affine_values
                best_outliers = outliers
                best_outlier_count = len(outliers)
            elif total > second_score:
                second_score = float(total)
                second_model = (int(slope), int(intercept))

    if not best_values or not math.isfinite(best_score):
        raise ValueError(
            "no affine trajectory satisfied the outlier constraints; "
            "increase --max-outliers, increase --outlier-penalty, or disable outliers"
        )

    if args.outlier_penalty is not None and args.outlier_mode == "posthoc":
        improvements: list[tuple[float, int, int]] = []
        for packet in packets:
            affine_value = int(best_affine_values.get(packet, best_values.get(packet, -1)))
            rec = grouped[packet].get(affine_value)
            if rec is None:
                continue
            independent = independent_by_packet[packet]
            independent_value = int(independent.get("_byte_value", -1)) & 0xFF
            if independent_value == affine_value:
                continue
            stats = stats_by_packet[packet]
            confidence_ok = (
                float(stats["top1_gap"]) >= float(args.outlier_min_top1_gap)
                and float(stats["score_std"]) >= float(args.outlier_min_score_std)
                and float(stats["entropy_norm"]) <= float(args.outlier_max_entropy)
            )
            if not confidence_ok:
                continue
            improvement = (
                float(independent.get("_score", 0.0))
                - float(args.outlier_penalty)
                - float(rec.get("_score", 0.0))
            )
            if improvement > 0.0:
                improvements.append((float(improvement), int(packet), int(independent_value)))
        improvements.sort(reverse=True)
        max_outliers = (
            len(improvements)
            if args.max_outliers is None else max(0, int(args.max_outliers))
        )
        best_outliers = {}
        for improvement, packet, value in improvements[:max_outliers]:
            best_values[int(packet)] = int(value)
            best_outliers[int(packet)] = 1
            best_score += float(improvement)
        best_outlier_count = len(best_outliers)

    expected_payloads = (
        load_expected_payloads_from_symbols(
            args.gt_symbol_csv.resolve(),
            sf=int(args.sf),
            bw=float(args.bw),
            ldro_mode=int(args.ldro_mode),
        )
        if args.gt_symbol_csv else {}
    )

    rows: list[dict[str, Any]] = []
    joint_exact = 0
    independent_exact = 0
    compared = 0
    joint_top1 = 0
    rank_sum = 0.0
    rank_count = 0
    score_gap_sum = 0.0
    top1_gap_sum = 0.0
    entropy_sum = 0.0
    score_std_sum = 0.0
    for packet in packets:
        value = int(best_values.get(packet, -1))
        affine_value = int(best_affine_values.get(packet, value))
        used_outlier = int(packet in best_outliers)
        rec = grouped[packet].get(value, {})
        ranked = sorted(
            grouped[packet].values(),
            key=lambda item: float(item.get("_score", 0.0)),
            reverse=True,
        )
        stats = candidate_score_stats(ranked)
        top1_gap_sum += float(stats["top1_gap"])
        entropy_sum += float(stats["entropy_norm"])
        score_std_sum += float(stats["score_std"])
        independent = ranked[0]
        joint_rank = ""
        for pos, item in enumerate(ranked, start=1):
            if int(item.get("_byte_value", -1)) == value:
                joint_rank = pos
                break
        if joint_rank != "":
            rank_count += 1
            rank_sum += float(joint_rank)
            joint_top1 += int(joint_rank == 1)
        expected_payload = expected_payloads.get(packet, "")
        joint_payload = str(rec.get("candidate_payload_hex", "")).strip().lower()
        independent_payload = str(independent.get("candidate_payload_hex", "")).strip().lower()
        score_gap = float(independent.get("_score", 0.0)) - float(rec.get("_score", 0.0))
        score_gap_sum += float(score_gap)
        if expected_payload:
            compared += 1
            joint_exact += int(joint_payload == expected_payload)
            independent_exact += int(independent_payload == expected_payload)
        rows.append({
            "packet_index": packet,
            "joint_byte_value": value,
            "joint_byte_hex": f"{value:02x}" if value >= 0 else "",
            "affine_byte_value": affine_value,
            "affine_byte_hex": f"{affine_value & 0xFF:02x}" if affine_value >= 0 else "",
            "used_outlier": used_outlier,
            "joint_payload_hex": joint_payload,
            "joint_score": f"{float(rec.get('_score', 0.0)):.9f}",
            "joint_candidate_rank": joint_rank,
            "joint_candidate_count": len(ranked),
            "candidate_top1_gap": f"{float(stats['top1_gap']):.9f}",
            "candidate_score_std": f"{float(stats['score_std']):.9f}",
            "candidate_entropy_norm": f"{float(stats['entropy_norm']):.9f}",
            "independent_byte_value": int(independent.get("_byte_value", -1)),
            "independent_byte_hex": f"{int(independent.get('_byte_value', -1)) & 0xFF:02x}",
            "independent_payload_hex": independent_payload,
            "independent_score": f"{float(independent.get('_score', 0.0)):.9f}",
            "score_gap_independent_minus_joint": f"{score_gap:.9f}",
            "expected_payload_hex": expected_payload,
            "joint_app_exact": int(joint_payload == expected_payload) if expected_payload else "",
            "independent_app_exact": int(independent_payload == expected_payload) if expected_payload else "",
            "affine_slope": int(best_model[0]),
            "affine_intercept": int(best_model[1]),
        })
    write_csv(args.output_csv.resolve(), rows)

    expected_ok = ""
    if args.expected_affine:
        a_s, b_s = str(args.expected_affine).split(",", 1)
        expected_ok = int((int(a_s) & 0xFF, int(b_s) & 0xFF) == best_model)

    summary = {
        "candidate_csv": str(args.candidate_csv),
        "packets": len(packets),
        "score_field": str(args.score_field),
        "affine_slope": int(best_model[0]),
        "affine_intercept": int(best_model[1]),
        "total_score": float(best_score),
        "second_affine_slope": int(second_model[0]),
        "second_affine_intercept": int(second_model[1]),
        "second_total_score": float(second_score),
        "model_score_margin": float(best_score - second_score),
        "model_score_margin_per_packet": (
            float((best_score - second_score) / max(1, len(packets)))
        ),
        "outlier_penalty": (
            float(args.outlier_penalty) if args.outlier_penalty is not None else None
        ),
        "outlier_mode": str(args.outlier_mode),
        "max_outliers": args.max_outliers,
        "outlier_min_top1_gap": float(args.outlier_min_top1_gap),
        "outlier_min_score_std": float(args.outlier_min_score_std),
        "outlier_max_entropy": float(args.outlier_max_entropy),
        "outlier_count": int(best_outlier_count),
        "expected_affine_ok": expected_ok,
        "app_compared": int(compared),
        "joint_app_exact": int(joint_exact),
        "independent_app_exact": int(independent_exact),
        "joint_top1_count": int(joint_top1),
        "mean_joint_candidate_rank": (
            float(rank_sum / rank_count) if rank_count else None
        ),
        "mean_independent_minus_joint_score_gap": (
            float(score_gap_sum / max(1, len(packets)))
        ),
        "mean_candidate_top1_gap": float(top1_gap_sum / max(1, len(packets))),
        "mean_candidate_score_std": float(score_std_sum / max(1, len(packets))),
        "mean_candidate_entropy_norm": float(entropy_sum / max(1, len(packets))),
        "output_csv": str(args.output_csv),
    }
    if args.summary_json:
        import json
        args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.resolve().write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    print(
        "Joint affine: "
        f"value = ({best_model[0]}*packet + {best_model[1]}) mod 256; "
        f"packets={len(packets)} score={best_score:.6f} "
        f"margin={best_score - second_score:.6f} "
        f"outliers={best_outlier_count}"
    )
    if compared:
        print(
            f"App exact: joint={joint_exact}/{compared}, "
            f"independent={independent_exact}/{compared}"
        )
    print(f"Decisions: {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
