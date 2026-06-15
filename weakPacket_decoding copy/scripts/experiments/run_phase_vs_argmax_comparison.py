#!/usr/bin/env python3
"""Phase-Guided vs Argmax: comprehensive SER comparison across SNRs.

Takes clean IQ with known GT bins, adds AWGN at multiple SNR levels,
runs both argmax baseline and PILOT phase-guided demodulation,
and produces SER comparison curves for INFOCOM evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.header_first_demod import (
    decode_explicit_header,
    demod_symbol_sequence,
)
from weak_decoder.phase_guided_demod import (
    PhaseGuidedPayloadConfig,
    PhaseLine,
    phase_guided_demod_packet,
)
from weak_decoder.chirp import build_downchirp, dechirp_fft


@dataclass
class SnrTrialResult:
    snr_db: float
    trial: int
    packet_index: int
    event_index: int
    total_symbols: int
    argmax_correct: int
    argmax_ser: float
    pilot_correct: int
    pilot_ser: float
    header_valid: bool


def _int(row, key, default=0):
    v = str(row.get(key, "")).strip()
    return int(float(v)) if v else int(default)


def _float_val(row, key, default=float("nan")):
    v = str(row.get(key, "")).strip()
    return float(v) if v else float(default)


def _flag(row, key, default=False):
    v = str(row.get(key, "")).strip()
    return bool(int(float(v))) if v else bool(default)


def _group_key(row):
    return (_int(row, "packet_index", -1), _int(row, "event_index", -1))


def add_awgn(samples, snr_db, signal_power=None):
    if signal_power is None:
        signal_power = float(np.mean(np.abs(samples) ** 2))
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2.0) * (
        np.random.randn(samples.size) + 1j * np.random.randn(samples.size)
    )
    return (samples + noise.astype(np.complex64)).astype(np.complex64)


def argmax_demod_payload(
    samples, payload_start_sample, sf, os_factor,
    cfo_int, cfo_frac, sfo_hat, payload_count, payload_ldro, cfo_correction_mode,
):
    n_bins = 1 << sf
    chirp_samples = float(n_bins * os_factor)
    downchirp = build_downchirp(sf, cfo_int=int(cfo_int), cfo_frac=float(cfo_frac))
    cfo_total = float(cfo_int) + float(cfo_frac)
    cursor = int(payload_start_sample) + 8 * int(chirp_samples)
    sfo_cum = 0.0
    results = []
    for _ in range(payload_count):
        indexes = int(cursor) + int(os_factor // 2) + os_factor * np.arange(n_bins, dtype=np.int64)
        if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
            break
        symbol = np.asarray(samples[indexes], dtype=np.complex64)
        if cfo_correction_mode == "continuous":
            rcs = float(cursor - int(payload_start_sample)) / float(os_factor)
            ccp = float(2.0 * math.pi * cfo_total * rcs / n_bins)
            symbol = (symbol * np.exp(-1j * ccp)).astype(np.complex64)
        spectrum = dechirp_fft(symbol, downchirp)
        argmax_bin = int(np.argmax(np.abs(spectrum) ** 2))
        results.append(argmax_bin)
        step = int(chirp_samples)
        threshold = 0.5 / os_factor
        if abs(sfo_cum) > threshold:
            sign = -1 if sfo_cum < 0 else 1
            step -= sign
            sfo_cum -= sign * (1.0 / os_factor)
        sfo_cum += float(sfo_hat)
        cursor += step
    return results


def read_sync_candidates(path_obj, packet_filter, max_packets):
    cands = []
    with path_obj.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not _flag(row, "grlora_framesync_valid", False):
                continue
            pk = _int(row, "packet_index", -1)
            if packet_filter is not None and pk not in packet_filter:
                continue
            cands.append(dict(row))
    cands.sort(key=lambda r: (_int(r, "packet_index"), _int(r, "event_index")))
    if max_packets is not None:
        cands = cands[:int(max_packets)]
    return cands


def read_gt_symbols(path_obj, packet_filter):
    result = {}
    with path_obj.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("stage", "")).strip() != "payload":
                continue
            if _int(row, "header_valid", 0) != 1:
                continue
            key = _group_key(row)
            pk = _int(row, "packet_index", -1)
            if packet_filter is not None and pk not in packet_filter:
                continue
            if key not in result:
                result[key] = {}
            sym_idx = _int(row, "stage_symbol_index", -1)
            gt_bin = _int(row, "raw_fft_bin", -1)
            if gt_bin >= 0:
                result[key][sym_idx] = gt_bin
    return result


def run_single_trial(
    clean_samples, cand, gt_bins, snr_db, sf, os_factor,
    preamble_len, bw, pilot_config, cfo_correction_mode, signal_power,
):
    noisy = add_awgn(clean_samples, snr_db, signal_power)
    cfo_int = _int(cand, "grlora_cfo_int_est", 0)
    cfo_frac = _float_val(cand, "grlora_cfo_frac_est", 0.0)
    sfo_hat = _float_val(cand, "grlora_sfo_hat", 0.0)
    fine_ps = _int(cand, "grlora_fine_payload_start_sample")

    header_results = demod_symbol_sequence(
        samples=noisy, header_start_sample=int(fine_ps),
        sf=sf, os_factor=os_factor,
        cfo_int=cfo_int, cfo_frac=cfo_frac,
        sfo_hat=sfo_hat, sfo_cum_initial=0.0,
        header_count=8, payload_count=0, payload_ldro=False,
        cfo_correction_mode=cfo_correction_mode,
    )
    if len(header_results) < 8:
        return SnrTrialResult(snr_db=snr_db, trial=0,
            packet_index=_int(cand, "packet_index"),
            event_index=_int(cand, "event_index"),
            total_symbols=len(gt_bins), argmax_correct=0, argmax_ser=1.0,
            pilot_correct=0, pilot_ser=1.0, header_valid=False)

    header_sv = [r.symbol_value for r in header_results]
    header_decode = decode_explicit_header(header_sv, sf=sf, bw=bw, ldro_mode=2)
    if not header_decode.header_valid:
        return SnrTrialResult(snr_db=snr_db, trial=0,
            packet_index=_int(cand, "packet_index"),
            event_index=_int(cand, "event_index"),
            total_symbols=len(gt_bins), argmax_correct=0, argmax_ser=1.0,
            pilot_correct=0, pilot_ser=1.0, header_valid=False)

    payload_count = header_decode.payload_symbol_count
    payload_ldro = header_decode.ldro
    total = min(payload_count, max(gt_bins.keys()) + 1 if gt_bins else payload_count)

    argmax_bins = argmax_demod_payload(
        noisy, int(fine_ps), sf, os_factor,
        cfo_int, cfo_frac, sfo_hat, total, payload_ldro, cfo_correction_mode)
    argmax_correct = sum(1 for i, b in enumerate(argmax_bins) if gt_bins.get(i, -1) == b)

    pilot_result = phase_guided_demod_packet(
        samples=noisy, header_start_sample=int(fine_ps),
        sf=sf, os_factor=os_factor,
        cfo_int=cfo_int, cfo_frac=cfo_frac, sfo_hat=sfo_hat,
        preamble_len=preamble_len, header_symbol_values=header_sv,
        header_payload_len=header_decode.payload_len,
        header_cr=header_decode.cr, header_has_crc=header_decode.has_crc,
        header_ldro=payload_ldro, payload_symbol_count=total,
        config=pilot_config, cfo_correction_mode=cfo_correction_mode,
        gt_bins=gt_bins)
    pilot_correct = pilot_result.get("correct_count", 0) if pilot_result.get("success") else 0

    return SnrTrialResult(
        snr_db=snr_db, trial=0,
        packet_index=_int(cand, "packet_index"),
        event_index=_int(cand, "event_index"),
        total_symbols=total, argmax_correct=argmax_correct,
        argmax_ser=1.0 - argmax_correct / max(1, total),
        pilot_correct=pilot_correct,
        pilot_ser=1.0 - pilot_correct / max(1, total),
        header_valid=True)


def plot_ser_curves(results, output_path, dpi=220):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    snr_groups = {}
    for r in results:
        snr_groups.setdefault(r.snr_db, []).append(r)
    snrs = sorted(snr_groups.keys())
    argmax_sers, pilot_sers = [], []
    for snr in snrs:
        group = snr_groups[snr]
        total_sym = sum(r.total_symbols for r in group)
        a_corr = sum(r.argmax_correct for r in group)
        p_corr = sum(r.pilot_correct for r in group)
        argmax_sers.append(1.0 - a_corr / max(1, total_sym))
        pilot_sers.append(1.0 - p_corr / max(1, total_sym))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=int(dpi))
    ax.semilogy(snrs, argmax_sers, "o-", color="#d62728", linewidth=2,
                markersize=6, label="Argmax Baseline")
    ax.semilogy(snrs, pilot_sers, "s-", color="#1f77b4", linewidth=2,
                markersize=6, label="PILOT (Phase-Guided)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Symbol Error Rate (SER)")
    ax.set_title("LoRa SF10 Payload SER: Argmax vs PILOT Phase-Guided")
    ax.grid(True, color="#dddddd", linewidth=0.6, which="both")
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=3e-3, top=1.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PILOT vs Argmax SER comparison")
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-s", "--sync-chain-csv", type=Path, required=True)
    parser.add_argument("-g", "--gt-symbol-csv", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--bw", type=float, default=125000.0)
    parser.add_argument("--samp-rate", type=float, default=500000.0)
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--os-factor", type=int, default=4)
    parser.add_argument("--packet", type=int, nargs="*", default=None)
    parser.add_argument("--max-packets", type=int, default=None)
    parser.add_argument("--snr-db", type=float, nargs="+",
                        default=[-16, -18, -20, -22, -24, -26, -28])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--phase-weight", type=float, default=0.85)
    parser.add_argument("--top-l", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip-plot", action="store_true", default=False)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    input_path = args.input.resolve()
    sync_csv = args.sync_chain_csv.resolve()
    gt_csv = args.gt_symbol_csv.resolve()
    output_dir = (args.output_dir or Path("data/phase_vs_argmax")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_samples = np.fromfile(input_path, dtype=np.complex64)
    signal_power = float(np.mean(np.abs(clean_samples) ** 2))
    sf, os_factor = int(args.sf), int(args.os_factor)
    preamble_len, bw = float(args.preamble_len), float(args.bw)

    packet_filter = set(args.packet) if args.packet else None
    candidates = read_sync_candidates(sync_csv, packet_filter, args.max_packets)
    gt_map = read_gt_symbols(gt_csv, packet_filter)

    pilot_config = PhaseGuidedPayloadConfig(
        top_l=int(args.top_l), max_refinement_rounds=3, trim_frac=0.25,
        confidence_threshold=0.4, argmax_window_radius=3,
        energy_threshold_db=12.0, use_header_anchor=True,
        phase_weight=float(args.phase_weight))

    all_results = []
    total_work = len(args.snr_db) * len(candidates) * args.trials
    done = 0
    print(f"Running {len(args.snr_db)} SNRs x {len(candidates)} packets x {args.trials} trials = {total_work}")

    for snr_db in args.snr_db:
        for cand in candidates:
            key = (_int(cand, "packet_index"), _int(cand, "event_index"))
            gt_bins = gt_map.get(key, {})
            if not gt_bins:
                continue
            for trial in range(args.trials):
                r = run_single_trial(
                    clean_samples, cand, gt_bins, snr_db,
                    sf, os_factor, preamble_len, bw,
                    pilot_config, "continuous", signal_power)
                r.trial = trial
                all_results.append(r)
                done += 1
                if done % max(1, total_work // 10) == 0:
                    print(f"  {done}/{total_work} ({100*done/total_work:.0f}%)")

    csv_path = output_dir / f"{input_path.stem}_ser_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["snr_db", "trial", "packet_index", "event_index",
                  "total_symbols", "header_valid",
                  "argmax_correct", "argmax_ser", "pilot_correct", "pilot_ser"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: getattr(r, k) for k in fields})
    print(f"CSV: {csv_path}")

    if not args.skip_plot:
        plot_path = output_dir / f"{input_path.stem}_ser_comparison.png"
        plot_ser_curves(all_results, plot_path, dpi=int(args.dpi))

    print("\n=== SER Summary ===")
    snr_groups = {}
    for r in all_results:
        snr_groups.setdefault(r.snr_db, []).append(r)
    for snr in sorted(snr_groups.keys()):
        group = snr_groups[snr]
        total_sym = sum(r.total_symbols for r in group)
        a_corr = sum(r.argmax_correct for r in group)
        p_corr = sum(r.pilot_correct for r in group)
        print(f"  SNR={snr:5.1f} dB: Argmax SER={1.0-a_corr/max(1,total_sym):.4f}, PILOT SER={1.0-p_corr/max(1,total_sym):.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
