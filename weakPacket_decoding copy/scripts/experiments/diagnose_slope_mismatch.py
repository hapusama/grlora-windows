#!/usr/bin/env python3
"""Diagnose preamble-header-payload slope mismatch.

Validate the header-bridge strategy by comparing phase slopes
across preamble, header, and payload segments on clean IQ data.
"""

from __future__ import annotations

import argparse, csv, math, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.header_first_demod import (
    decode_explicit_header, demod_symbol_sequence,
)
from weak_decoder.phase_guided_demod import (
    extract_preamble_anchors, extract_header_anchors,
    fit_phase_line, PhaseLine,
)
from weak_decoder.chirp import build_downchirp, positive_mod, dechirp_fft


@dataclass
class SlopeDiagnosis:
    packet_index: int; event_index: int
    preamble_slope_pi: float; header_slope_pi: float; payload_slope_pi: float
    pre_to_payload_delta_pi: float; hdr_to_payload_delta_pi: float
    preamble_r2: float; header_r2: float; payload_r2: float
    preamble_anchors: int; header_anchors: int; payload_symbols: int
    header_is_better: bool

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

def extract_payload_anchors_from_gt(samples, gt_rows, sf, os_factor, cfo_int, cfo_frac, cfo_mode, preamble_len):
    n_bins = 1 << sf
    chirp_samples = float(n_bins * os_factor)
    downchirp = build_downchirp(sf, cfo_int=int(cfo_int), cfo_frac=float(cfo_frac))
    cfo_total = float(cfo_int) + float(cfo_frac)
    abs_i, phs = [], []
    for row in gt_rows:
        start_sample = _int(row, "start_sample")
        gt_bin = _int(row, "raw_fft_bin", -1)
        header_start = _int(row, "header_start_sample", start_sample - _int(row, "frame_symbol_index", -1) * chirp_samples)
        if gt_bin < 0: continue
        indexes = int(start_sample) + int(os_factor // 2) + os_factor * np.arange(n_bins, dtype=np.int64)
        if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size: continue
        symbol = np.asarray(samples[indexes], dtype=np.complex64)
        if cfo_mode == "continuous":
            rcs = float(start_sample - int(header_start)) / float(os_factor)
            symbol = (symbol * np.exp(-1j * float(2.0 * math.pi * cfo_total * rcs / n_bins))).astype(np.complex64)
        spectrum = dechirp_fft(symbol, downchirp)
        vb = spectrum[positive_mod(gt_bin, n_bins)]
        abs_i.append(float(preamble_len) + 12.25 + float(_int(row, "stage_symbol_index", -1)))
        phs.append(float(math.atan2(vb.imag, vb.real)))
    if len(abs_i) < 2: return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return np.array(abs_i, dtype=np.float64), np.unwrap(np.array(phs, dtype=np.float64))

def main():
    parser = argparse.ArgumentParser(description="Slope mismatch diagnosis")
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
    parser.add_argument("--max-packets", type=int, default=10)
    parser.add_argument("--cfo-correction-mode", choices=("symbol","continuous"), default="continuous")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    samples = np.fromfile(args.input.resolve(), dtype=np.complex64)
    sf, os_factor = int(args.sf), int(args.os_factor)
    preamble_len = float(args.preamble_len)
    cfo_mode = str(args.cfo_correction_mode)
    packet_filter = set(args.packet) if args.packet else None

    candidates = []
    with args.sync_chain_csv.resolve().open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not _flag(row, "grlora_framesync_valid"): continue
            pk = _int(row, "packet_index", -1)
            if packet_filter is not None and pk not in packet_filter: continue
            candidates.append(dict(row))
    candidates.sort(key=lambda r: (_int(r, "packet_index"), _int(r, "event_index")))
    if args.max_packets: candidates = candidates[:int(args.max_packets)]

    gt_by_key: Dict[Tuple[int, int], List[dict]] = {}
    with args.gt_symbol_csv.resolve().open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("stage","")).strip() != "payload": continue
            if _int(row, "header_valid", 0) != 1: continue
            key = _group_key(row)
            pk = _int(row, "packet_index", -1)
            if packet_filter is not None and pk not in packet_filter: continue
            gt_by_key.setdefault(key, []).append(row)

    out_dir = (args.output_dir or Path("data/slope_diagnosis")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnoses = []
    print(f"Analyzing {len(candidates)} packets...")

    for idx, cand in enumerate(candidates):
        key = (_int(cand, "packet_index"), _int(cand, "event_index"))
        gt_rows = gt_by_key.get(key, [])
        if not gt_rows: continue
        fine_ps = _int(cand, "grlora_fine_payload_start_sample")
        cfo_int = _int(cand, "grlora_cfo_int_est", 0)
        cfo_frac = _float_val(cand, "grlora_cfo_frac_est", 0.0)
        sfo_hat = _float_val(cand, "grlora_sfo_hat", 0.0)

        pre_abs, pre_phases, _, _ = extract_preamble_anchors(
            samples, int(fine_ps), sf, os_factor, preamble_len,
            cfo_int, cfo_frac, cfo_mode, max_anchors=int(round(preamble_len)))

        hdr_abs, hdr_phases = np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        hdr_results = demod_symbol_sequence(samples=samples, header_start_sample=int(fine_ps),
            sf=sf, os_factor=os_factor, cfo_int=cfo_int, cfo_frac=cfo_frac,
            sfo_hat=sfo_hat, sfo_cum_initial=0.0,
            header_count=8, payload_count=0, payload_ldro=False, cfo_correction_mode=cfo_mode)
        hdr_sv = [r.symbol_value for r in hdr_results] if len(hdr_results) >= 8 else []
        if hdr_sv:
            hdr_decode = decode_explicit_header(hdr_sv, sf=sf, bw=float(args.bw), ldro_mode=2)
            if hdr_decode.header_valid:
                hdr_abs, hdr_phases = extract_header_anchors(
                    samples, int(fine_ps), sf, os_factor,
                    cfo_int, cfo_frac, cfo_mode, preamble_len, hdr_sv)

        pay_abs, pay_phases = extract_payload_anchors_from_gt(
            samples, gt_rows, sf, os_factor, cfo_int, cfo_frac, cfo_mode, preamble_len)

        pre_line = fit_phase_line(pre_abs, pre_phases) if pre_abs.size >= 2 else PhaseLine()
        hdr_line = fit_phase_line(hdr_abs, hdr_phases) if hdr_abs.size >= 2 else PhaseLine()
        pay_line = fit_phase_line(pay_abs, pay_phases) if pay_abs.size >= 2 else PhaseLine()

        pre_delta = pre_line.slope_pi - pay_line.slope_pi if pay_line.anchor_count >= 2 else 0
        hdr_delta = hdr_line.slope_pi - pay_line.slope_pi if pay_line.anchor_count >= 2 and hdr_line.anchor_count >= 2 else 0

        d = SlopeDiagnosis(
            packet_index=key[0], event_index=key[1],
            preamble_slope_pi=pre_line.slope_pi, header_slope_pi=hdr_line.slope_pi,
            payload_slope_pi=pay_line.slope_pi,
            pre_to_payload_delta_pi=pre_delta, hdr_to_payload_delta_pi=hdr_delta,
            preamble_r2=pre_line.fit_r2, header_r2=hdr_line.fit_r2,
            payload_r2=pay_line.fit_r2,
            preamble_anchors=int(pre_abs.size), header_anchors=int(hdr_abs.size),
            payload_symbols=int(pay_abs.size),
            header_is_better=abs(hdr_delta) < abs(pre_delta))
        diagnoses.append(d)
        print(f"  Pkt {key[0]}: pre={pre_line.slope_pi:.4f}pi hdr={hdr_line.slope_pi:.4f}pi pay={pay_line.slope_pi:.4f}pi | p-p={abs(pre_delta):.4f}pi h-p={abs(hdr_delta):.4f}pi | hdr_better={d.header_is_better}")

    csv_path = out_dir / f"{args.input.stem}_slope_diagnosis.csv"
    fields = ["packet_index","event_index","preamble_slope_pi","header_slope_pi","payload_slope_pi",
              "pre_to_payload_delta_pi","hdr_to_payload_delta_pi",
              "preamble_r2","header_r2","payload_r2",
              "preamble_anchors","header_anchors","payload_symbols","header_is_better"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for d in diagnoses: w.writerow({k: getattr(d, k) for k in fields})

    if diagnoses:
        avg_pre = np.mean([abs(d.pre_to_payload_delta_pi) for d in diagnoses])
        avg_hdr = np.mean([abs(d.hdr_to_payload_delta_pi) for d in diagnoses])
        pct = 100 * sum(1 for d in diagnoses if d.header_is_better) / len(diagnoses)
        print(f"\n=== Summary ===")
        print(f"  Mean |pre-payload| = {avg_pre:.4f} pi/sym")
        print(f"  Mean |hdr-payload|  = {avg_hdr:.4f} pi/sym")
        print(f"  Header better in {pct:.0f}% of packets")
        if avg_hdr < avg_pre:
            print(f"  => VALIDATED: header slope {(avg_pre-avg_hdr)/avg_pre*100:.0f}% closer")
    print(f"\nCSV: {csv_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())