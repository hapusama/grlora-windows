#!/usr/bin/env python3
"""Estimate empirical pattern-output noise covariance from off-packet IQ windows."""

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

from weak_decoder.baselines.common import dataset_paths, load_packets, write_csv  # noqa: E402
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import _oversampled_downchirp  # noqa: E402
from weak_decoder.chirp import build_upchirp  # noqa: E402
from weak_decoder.os_lora.nonuniform_sampling import (  # noqa: E402
    NonuniformPatternBank,
    build_pattern_bank,
    pattern_bin_values,
    pattern_noise_signatures,
)


def _active_intervals(packets: Sequence[dict[str, Any]], guard_samples: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for packet in packets:
        sf = int(packet["sf"])
        os_factor = int(packet["os_factor"])
        symbol_samples = (1 << sf) * os_factor
        for symbol in list(packet.get("header_symbols", [])) + list(packet.get("payload_symbols", [])):
            start = int(symbol["start_sample"]) - int(guard_samples)
            stop = int(symbol["start_sample"]) + symbol_samples + int(guard_samples)
            intervals.append((max(0, start), max(0, stop)))
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for start, stop in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, stop))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
    return merged


def _overlaps(intervals: Sequence[tuple[int, int]], start: int, stop: int) -> bool:
    lo = 0
    hi = len(intervals)
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals[mid][1] <= start:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(intervals):
        return False
    return intervals[lo][0] < stop


def _off_packet_starts(
    sample_count: int,
    window_len: int,
    intervals: Sequence[tuple[int, int]],
    max_windows: int,
    seed: int,
) -> tuple[int, ...]:
    rng = np.random.default_rng(int(seed))
    starts: list[int] = []
    seen: set[int] = set()
    max_start = int(sample_count) - int(window_len)
    attempts = 0
    max_attempts = max(10000, int(max_windows) * 200)
    while len(starts) < int(max_windows) and attempts < max_attempts:
        attempts += 1
        start = int(rng.integers(0, max_start + 1))
        start = (start // int(window_len)) * int(window_len)
        if start in seen:
            continue
        stop = start + int(window_len)
        if stop > int(sample_count) or _overlaps(intervals, start, stop):
            continue
        seen.add(start)
        starts.append(start)
    starts.sort()
    return tuple(starts)


def _subset_bank(bank: NonuniformPatternBank, max_patterns: int) -> NonuniformPatternBank:
    if int(max_patterns) <= 0 or len(bank.offsets) <= int(max_patterns):
        return bank
    keep = int(max_patterns)
    return NonuniformPatternBank(
        names=tuple(bank.names[:keep]),
        offsets=tuple(bank.offsets[:keep]),
        os_factor=int(bank.os_factor),
        sf=int(bank.sf),
        kind=f"{bank.kind}_first{keep}",
    )


def _empirical_covariance(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.complex128)
    values = values - np.mean(values, axis=0, keepdims=True)
    denom = max(1, values.shape[0] - 1)
    cov = values.T @ values.conj() / float(denom)
    return (cov + cov.conj().T) * 0.5


def _corr_stats(covariance: np.ndarray) -> tuple[float, float, float]:
    cov = np.asarray(covariance, dtype=np.complex128)
    if cov.size == 0:
        return 0.0, 0.0, 0.0
    diag = np.maximum(np.real(np.diag(cov)), 1e-30)
    diag_cv = float(np.std(diag) / max(float(np.mean(diag)), 1e-30))
    if cov.shape[0] <= 1:
        return diag_cv, 0.0, 0.0
    scale = np.sqrt(diag[:, None] * diag[None, :])
    corr = cov / scale
    mask = ~np.eye(cov.shape[0], dtype=bool)
    off = np.abs(corr[mask])
    return diag_cv, float(np.mean(off)), float(np.max(off))


def _range_residuals(covariance: np.ndarray, target: np.ndarray, rconds: Sequence[float]) -> dict[str, float | int]:
    cov = np.asarray(covariance, dtype=np.complex128)
    a = np.asarray(target, dtype=np.complex128)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(np.real(eigvals), 0.0)
    max_eig = float(np.max(eigvals)) if eigvals.size else 0.0
    out: dict[str, float | int] = {"eig_max": max_eig}
    norm_a = max(float(np.linalg.norm(a)), 1e-30)
    for rcond in rconds:
        keep = eigvals > max(max_eig, 1e-300) * float(rcond)
        if np.any(keep):
            basis = eigvecs[:, keep]
            proj = basis @ (basis.conj().T @ a)
            residual = float(np.linalg.norm(a - proj) / norm_a)
            rank = int(np.sum(keep))
        else:
            residual = 1.0
            rank = 0
        key = f"{float(rcond):.0e}".replace("-", "m")
        out[f"rank_{key}"] = rank
        out[f"range_residual_{key}"] = residual
    return out


def _effective_replicas(covariance: np.ndarray, target: np.ndarray, n_bins: int, rcond: float) -> float:
    cov = np.asarray(covariance, dtype=np.complex128)
    diag_mean = float(np.mean(np.maximum(np.real(np.diag(cov)), 1e-30))) if cov.size else 1.0
    cov_norm = cov / max(diag_mean, 1e-30)
    pinv = np.linalg.pinv(cov_norm, rcond=float(rcond))
    value = complex(target.conj().T @ pinv @ target)
    return float(np.real(value) / float(n_bins))


def _effective_replicas_with_noise_power(
    covariance: np.ndarray,
    target: np.ndarray,
    n_bins: int,
    noise_power: float,
    rcond: float,
) -> float:
    cov = np.asarray(covariance, dtype=np.complex128)
    pinv = np.linalg.pinv(cov, rcond=float(rcond))
    value = complex(target.conj().T @ pinv @ target)
    return float(float(noise_power) * np.real(value) / float(n_bins))


def _float_key(value: float) -> str:
    return f"{float(value):.3g}".replace("-", "m").replace(".", "p")


def _raw_noise_stats(windows: np.ndarray, os_factor: int, max_lag: int) -> dict[str, float]:
    values = np.asarray(windows, dtype=np.complex128)
    values = values - np.mean(values)
    power = float(np.mean(np.abs(values) ** 2))
    out: dict[str, float] = {"raw_noise_power": power}
    offset_vars = []
    flat = values.reshape(-1, values.shape[-1])
    for q in range(int(os_factor)):
        offset_vars.append(float(np.mean(np.abs(flat[:, q:: int(os_factor)]) ** 2)))
    if offset_vars:
        out["offset_var_mean"] = float(np.mean(offset_vars))
        out["offset_var_cv"] = float(np.std(offset_vars) / max(float(np.mean(offset_vars)), 1e-30))
    lag_corrs = []
    concat = flat.reshape(-1)
    concat = concat - np.mean(concat)
    denom = max(float(np.mean(np.abs(concat) ** 2)), 1e-30)
    for lag in range(1, min(int(max_lag), concat.size - 1) + 1):
        corr = np.mean(concat[lag:] * np.conj(concat[:-lag])) / denom
        lag_corrs.append(abs(complex(corr)))
    if lag_corrs:
        out["lag_corr_mean_abs"] = float(np.mean(lag_corrs))
        out["lag_corr_max_abs"] = float(np.max(lag_corrs))
    return out


def _analyze_one(
    samples: np.ndarray,
    starts: Sequence[int],
    sf: int,
    os_factor: int,
    raw_bin: int,
    bank: NonuniformPatternBank,
    downchirp: np.ndarray,
    raw_noise_power: float,
    rconds: Sequence[float],
    shrinkages: Sequence[float],
) -> dict[str, Any]:
    n_bins = 1 << int(sf)
    window_len = n_bins * int(os_factor)
    vectors: list[np.ndarray] = []
    for start in starts:
        chunk = np.asarray(samples[int(start): int(start) + window_len], dtype=np.complex64)
        dechirped = (chunk * downchirp).astype(np.complex64)
        vectors.append(pattern_bin_values(dechirped=dechirped, raw_fft_bin=int(raw_bin), bank=bank))
    matrix = np.asarray(vectors, dtype=np.complex128)
    cov = _empirical_covariance(matrix)
    diag_cv, mean_abs_corr, max_abs_corr = _corr_stats(cov)
    ideal = build_upchirp(sf=int(sf), symbol_id=int(raw_bin), os_factor=int(os_factor)) * downchirp
    target = pattern_bin_values(dechirped=ideal, raw_fft_bin=int(raw_bin), bank=bank)
    signatures = pattern_noise_signatures(bank, int(raw_bin))
    white_cov = signatures @ signatures.conj().T
    residuals = _range_residuals(cov, target, rconds)
    out: dict[str, Any] = {
        "raw_bin": int(raw_bin),
        "bank_kind": str(bank.kind),
        "pattern_count": int(len(bank.names)),
        "window_count": int(len(starts)),
        "diag_cv": diag_cv,
        "mean_abs_corr": mean_abs_corr,
        "max_abs_corr": max_abs_corr,
        "target_amp_mean": float(np.mean(np.abs(target))) if target.size else 0.0,
        "target_phase_std": float(np.std(np.unwrap(np.angle(target)))) if target.size else 0.0,
        "effective_replicas_empirical_rcond_1e_m6": _effective_replicas(cov, target, n_bins, 1e-6),
        "effective_replicas_empirical_rcond_1e_m3": _effective_replicas(cov, target, n_bins, 1e-3),
        "effective_replicas_rawpower_rcond_1e_m6": _effective_replicas_with_noise_power(
            cov, target, n_bins, raw_noise_power, 1e-6
        ),
        "effective_replicas_rawpower_rcond_1e_m3": _effective_replicas_with_noise_power(
            cov, target, n_bins, raw_noise_power, 1e-3
        ),
        "effective_replicas_white_model": _effective_replicas_with_noise_power(
            float(raw_noise_power) * white_cov, target, n_bins, raw_noise_power, 1e-10
        ),
    }
    for shrinkage in shrinkages:
        lam = min(max(float(shrinkage), 0.0), 1.0)
        shrunk = (1.0 - lam) * cov + lam * float(raw_noise_power) * white_cov
        key = _float_key(lam)
        out[f"effective_replicas_shrink_{key}_rcond_1e_m3"] = _effective_replicas_with_noise_power(
            shrunk, target, n_bins, raw_noise_power, 1e-3
        )
        out[f"effective_replicas_shrink_{key}_rcond_1e_m6"] = _effective_replicas_with_noise_power(
            shrunk, target, n_bins, raw_noise_power, 1e-6
        )
    out.update(residuals)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_16"])
    parser.add_argument("--raw-bins", nargs="+", type=int, default=[61, 301, 653, 900])
    parser.add_argument(
        "--bank-kinds",
        nargs="+",
        default=["fixed", "basic_only", "dither_only", "random_only", "balanced_random_only"],
    )
    parser.add_argument("--random-count", type=int, default=64)
    parser.add_argument("--bank-seed", type=int, default=0)
    parser.add_argument("--max-patterns", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--window-seed", type=int, default=1234)
    parser.add_argument("--guard-symbols", type=float, default=2.0)
    parser.add_argument("--max-lag", type=int, default=64)
    parser.add_argument("--shrinkages", nargs="*", type=float, default=[0.05, 0.2, 0.5])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "os_lora" / "empirical_noise_covariance",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = dataset_paths(str(dataset))
        samples = np.fromfile(iq_path, dtype=np.complex64)
        packets = load_packets(symbol_path)
        if not packets:
            continue
        sf = int(packets[0]["sf"])
        os_factor = int(packets[0]["os_factor"])
        n_bins = 1 << sf
        window_len = n_bins * os_factor
        guard_samples = int(round(float(args.guard_symbols) * window_len))
        intervals = _active_intervals(packets, guard_samples=guard_samples)
        starts = _off_packet_starts(
            sample_count=int(samples.size),
            window_len=window_len,
            intervals=intervals,
            max_windows=int(args.max_windows),
            seed=int(args.window_seed),
        )
        if not starts:
            raise RuntimeError(f"no off-packet windows found for {dataset}")
        downchirp = _oversampled_downchirp(sf=sf, os_factor=os_factor, cfo_int=0, cfo_frac=0.0)
        raw_windows = np.asarray(
            [(samples[start: start + window_len] * downchirp).astype(np.complex64) for start in starts],
            dtype=np.complex64,
        )
        raw_row: dict[str, Any] = {
            "dataset": str(dataset),
            "window_count": int(len(starts)),
            "window_len": int(window_len),
            "guard_samples": int(guard_samples),
            "active_interval_count": int(len(intervals)),
        }
        raw_row.update(_raw_noise_stats(raw_windows, os_factor=os_factor, max_lag=int(args.max_lag)))
        raw_rows.append(raw_row)
        print(
            f"{dataset}: off_windows={len(starts)} raw_power={raw_row['raw_noise_power']:.6g} "
            f"offset_cv={raw_row.get('offset_var_cv', 0.0):.4f} "
            f"lag_max={raw_row.get('lag_corr_max_abs', 0.0):.4f}",
            flush=True,
        )
        for bank_kind in args.bank_kinds:
            bank = build_pattern_bank(
                sf,
                os_factor,
                kind=str(bank_kind),
                random_count=int(args.random_count),
                seed=int(args.bank_seed),
            )
            bank = _subset_bank(bank, int(args.max_patterns))
            for raw_bin in args.raw_bins:
                row = _analyze_one(
                    samples=samples,
                    starts=starts,
                    sf=sf,
                    os_factor=os_factor,
                    raw_bin=int(raw_bin),
                    bank=bank,
                    downchirp=downchirp,
                    raw_noise_power=float(raw_row["raw_noise_power"]),
                    rconds=(1e-2, 1e-3, 1e-4, 1e-6),
                    shrinkages=tuple(float(v) for v in args.shrinkages),
                )
                row["dataset"] = str(dataset)
                rows.append(row)
                print(
                    f"{dataset} {bank.kind:24s} k={int(raw_bin):4d} "
                    f"corr_max={row['max_abs_corr']:.3f} "
                    f"res1e-3={row['range_residual_1em03']:.3e} "
                    f"eff1e-3={row['effective_replicas_empirical_rcond_1e_m3']:.3f}",
                    flush=True,
                )
    write_csv(out_dir / "raw_noise_metrics.csv", raw_rows)
    write_csv(out_dir / "pattern_covariance_metrics.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"raw_noise": raw_rows, "pattern_covariance": rows}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
