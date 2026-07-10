#!/usr/bin/env python3
"""Evaluate ChirpSVD demodulation evidence against Savaux OSR evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.common import (  # noqa: E402
    dataset_paths,
    err_count,
    load_packets,
    noise_samples,
    signal_reference_power,
    snr_values,
    write_csv,
)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)
from weak_decoder.os_lora.chirp_svd import chirp_svd_spectra  # noqa: E402
from weak_decoder.os_lora.nonuniform_sampling import prepare_dechirped_symbol  # noqa: E402


METHODS = (
    "savaux",
    "svd_left",
    "svd_rank1_savaux",
    "svd_rank2_savaux",
)


def _power_metrics(power: np.ndarray, gt_bin: int, guard_bins: int = 1) -> dict[str, float | int]:
    values = np.asarray(power, dtype=np.float64)
    if values.size == 0:
        return {
            "selected_bin": 0,
            "gt_rank": 0,
            "gt_power": 0.0,
            "gt_margin_db": 0.0,
            "gt_floor_db": 0.0,
        }
    gt = int(gt_bin) % int(values.size)
    selected = int(np.argmax(values))
    gt_power = float(values[gt])
    rank = int(np.sum(values > gt_power) + 1)
    other = values.copy()
    other[gt] = -np.inf
    next_power = float(np.max(other))
    mask = np.ones(values.size, dtype=bool)
    for delta in range(-int(guard_bins), int(guard_bins) + 1):
        mask[(gt + delta) % values.size] = False
    floor = values[mask] if np.any(mask) else values
    floor_median = float(np.median(floor)) if floor.size else 0.0
    return {
        "selected_bin": selected,
        "gt_rank": rank,
        "gt_power": gt_power,
        "gt_margin_db": float(10.0 * math.log10((gt_power + 1e-30) / (next_power + 1e-30))),
        "gt_floor_db": float(10.0 * math.log10((gt_power + 1e-30) / (floor_median + 1e-30))),
    }


def _evaluate_symbol(
    samples: np.ndarray,
    packet: dict[str, Any],
    symbol: dict[str, Any],
    origin_shift: int,
    cfo_correction_mode: str,
) -> dict[str, Any]:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    start = int(symbol["start_sample"]) + int(origin_shift)
    header_start = int(packet["header_start_sample"]) + int(origin_shift)
    gt_bin = int(symbol["gt_bin"])

    savaux_spectrum, _branches, _phase = paper_oversampled_spectrum(
        samples=samples,
        start_sample=start,
        sf=sf,
        os_factor=os_factor,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=header_start,
        cfo_correction_mode=cfo_correction_mode,
    )
    dechirped = prepare_dechirped_symbol(
        samples=samples,
        start_sample=start,
        sf=sf,
        os_factor=os_factor,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=header_start,
        cfo_correction_mode=cfo_correction_mode,
    )
    svd = chirp_svd_spectra(dechirped=dechirped, sf=sf, os_factor=os_factor, ranks=(1, 2))

    spectra_power: dict[str, np.ndarray] = {
        "savaux": np.abs(savaux_spectrum).astype(np.float64) ** 2,
        "svd_left": np.abs(svd.svd_left_spectrum).astype(np.float64) ** 2,
        "svd_rank1_savaux": np.abs(svd.rank_savaux_spectra[1]).astype(np.float64) ** 2,
        "svd_rank2_savaux": np.abs(svd.rank_savaux_spectra[2]).astype(np.float64) ** 2,
    }

    out: dict[str, Any] = {
        "packet_index": int(packet["packet_index"]),
        "payload_symbol_index": int(symbol["payload_symbol_index"]),
        "gt_bin": int(gt_bin),
        "rank1_ratio": float(svd.rank1_ratio),
        "rank2_ratio": float(svd.rank2_ratio),
        "sigma1": float(svd.singular_values[0]) if svd.singular_values else 0.0,
        "sigma2": float(svd.singular_values[1]) if len(svd.singular_values) > 1 else 0.0,
    }
    for method, power in spectra_power.items():
        metrics = _power_metrics(power, gt_bin)
        out[f"{method}_selected_bin"] = int(metrics["selected_bin"])
        out[f"{method}_ok"] = int(metrics["selected_bin"]) == int(gt_bin)
        out[f"{method}_gt_rank"] = int(metrics["gt_rank"])
        out[f"{method}_gt_power"] = float(metrics["gt_power"])
        out[f"{method}_gt_margin_db"] = float(metrics["gt_margin_db"])
        out[f"{method}_gt_floor_db"] = float(metrics["gt_floor_db"])
    return out


def _evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    os_factor = int(packet["os_factor"])
    origin_shift = int(args.origin_shift) if args.origin_shift is not None else os_factor // 2
    rows: list[dict[str, Any]] = []
    for symbol in packet["payload_symbols"]:
        rows.append(
            _evaluate_symbol(
                samples=samples,
                packet=packet,
                symbol=symbol,
                origin_shift=origin_shift,
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
        )
    return rows


def _summary(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float | None, seed: int) -> dict[str, Any]:
    symbols = int(len(rows))
    gt_bins = [int(row["gt_bin"]) for row in rows]
    out: dict[str, Any] = {
        "dataset": dataset,
        "snr_db": "" if snr_db is None else float(snr_db),
        "seed": int(seed),
        "symbol_count": symbols,
        "mean_rank1_ratio": float(np.mean([float(row["rank1_ratio"]) for row in rows])) if rows else 0.0,
        "mean_rank2_ratio": float(np.mean([float(row["rank2_ratio"]) for row in rows])) if rows else 0.0,
    }
    savaux_selected = [int(row["savaux_selected_bin"]) for row in rows]
    for method in METHODS:
        selected = [int(row[f"{method}_selected_bin"]) for row in rows]
        err, compared = err_count(selected, gt_bins)
        out[f"{method}_ser"] = float(err / max(1, compared))
        out[f"{method}_mean_gt_rank"] = (
            float(np.mean([int(row[f"{method}_gt_rank"]) for row in rows])) if rows else 0.0
        )
        out[f"{method}_mean_gt_margin_db"] = (
            float(np.mean([float(row[f"{method}_gt_margin_db"]) for row in rows])) if rows else 0.0
        )
        if method != "savaux":
            fix = 0
            break_count = 0
            changes = 0
            for idx in range(min(len(selected), len(savaux_selected), len(gt_bins))):
                cand = int(selected[idx])
                sav = int(savaux_selected[idx])
                gt = int(gt_bins[idx])
                if cand != sav:
                    changes += 1
                if sav != gt and cand == gt:
                    fix += 1
                if sav == gt and cand != gt:
                    break_count += 1
            out[f"{method}_fix_vs_savaux"] = int(fix)
            out[f"{method}_break_vs_savaux"] = int(break_count)
            out[f"{method}_changes_vs_savaux"] = int(changes)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_16"])
    parser.add_argument("--snrs", nargs="*", type=float, default=[-22.0, -23.0, -24.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-packets", type=int, default=3)
    parser.add_argument("--origin-shift", type=int, default=None)
    parser.add_argument("--cfo-correction-mode", choices=("none", "symbol", "continuous"), default="continuous")
    parser.add_argument(
        "--signal-reference-mode",
        choices=("packet", "payload", "header_payload", "whole"),
        default="packet",
    )
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "os_lora" / "chirp_svd",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    symbol_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = dataset_paths(str(dataset))
        clean = np.fromfile(iq_path, dtype=np.complex64)
        packets = load_packets(symbol_path)
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
        reference_power, reference_samples, reference_packets = signal_reference_power(
            samples=clean,
            packets=packets,
            mode=str(args.signal_reference_mode),
            explicit_power=args.signal_reference_power,
        )
        print(
            f"{dataset}: reference_power={reference_power:.6g} "
            f"samples={reference_samples} packets={reference_packets}",
            flush=True,
        )
        for seed in args.seeds:
            for snr_db in snr_values(args.snrs):
                samples = noise_samples(clean, snr_db, int(seed), reference_power)
                rows: list[dict[str, Any]] = []
                for packet in packets:
                    rows.extend(_evaluate_packet(samples=samples, packet=packet, args=args))
                for row in rows:
                    row.update(
                        {
                            "dataset": str(dataset),
                            "snr_db": "" if snr_db is None else float(snr_db),
                            "seed": int(seed),
                        }
                    )
                    symbol_rows.append(row)
                summary = _summary(rows, str(dataset), snr_db, int(seed))
                summary_rows.append(summary)
                print(
                    f"{dataset} snr={snr_db} seed={seed}: "
                    f"savaux={summary['savaux_ser']:.4f} "
                    f"svd_left={summary['svd_left_ser']:.4f} "
                    f"rank1={summary['svd_rank1_savaux_ser']:.4f} "
                    f"rank2={summary['svd_rank2_savaux_ser']:.4f} "
                    f"r1ratio={summary['mean_rank1_ratio']:.3f} "
                    f"r2ratio={summary['mean_rank2_ratio']:.3f}",
                    flush=True,
                )

    write_csv(out_dir / "symbol_metrics.csv", symbol_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
