# -*- coding: utf-8 -*-
"""E1 冒烟 v2（修正网格后）：窗结构 / c_nom 核实 / 相位轨迹粗看。"""
import sys
import csv
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_common import (DS, align_seg, snr_parts, extract_symbol, freeze_gt,
                       make_slots, c_nominals, SF10_SOURCES, SF11_SOURCES)


def smoke(cap, bin_path, csv_path, sf, P, ldro, fi=0):
    ds = DS(sf)
    iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
            if r.get("header_valid") == "1" and int(r.get("payload_len") or 0) > 0]
    r = rows[fi]
    psym = int(r["payload_symbol_count"])
    gt_hdr, gt = freeze_gt(iq, r, ds, psym, ldro)
    seg, i0, boff = align_seg(iq, r, ds, P, psym)
    S, N0 = snr_parts(seg[boff * ds.nf:], ds.nf)
    print("== %s P=%d ldro=%d  GT hdr=%s" % (cap, P, ldro, gt_hdr))
    print("   native SNR=%.1f dB (S=%.3e N0=%.3e)" % (10 * np.log10(S / N0), S, N0))
    slots = make_slots(P, psym)
    cs = c_nominals(P, psym, ldro, gt_hdr, gt)
    fit_rows = [s for s in slots if s[3]]
    # 每窗实测 vs 名义
    bad = 0
    th_seq, kk_seq, snr_seq = [], [], []
    for (kind, k, off, _u), c in zip(slots, [None] * (len(slots) - len(fit_rows)) + cs) if False else []:
        pass
    # 上式复杂，直接分两类循环
    nom_iter = iter(cs)
    for kind, k, off, use in slots:
        ref = ds.ref_for(kind)
        c = next(nom_iter) if use else None
        m = extract_symbol(seg, off, ds, ref, c_nom=c)
        if m is None:
            print("%-5s%-3d off=%7.2f  OUT-OF-RANGE" % (kind, k, off))
            continue
        tag = "%-5s%-3d off=%7.2f" % (kind, k, off)
        if use:
            flag = "" if m["ok"] else "  <-- 离名义 bin 过远"
            if not m["ok"]:
                bad += 1
            print("%s c=%5d b=%5d κ=%+.3f snr=%9.1f θ=%+.3f%s"
                  % (tag, c, m["b"], m["kappa"], m["snr_eff"], m["theta_tone"], flag))
            th_seq.append(m["theta_tone"]); kk_seq.append(m["kappa"]); snr_seq.append(m["snr_eff"])
        else:
            print("%s b=%5d κ=%+.3f snr=%9.1f" % (tag, m["b"], m["kappa"], m["snr_eff"]))
    th = np.array(th_seq)
    d = np.diff(th)
    print("\n轨迹粗查: n=%d  Δθ std=%.3f rad  Δθ lag1 ρ=%.3f  κ range=[%+.3f,%+.3f]  坏窗=%d"
          % (len(th), np.std(d), np.corrcoef(d[:-1], d[1:])[0, 1],
             min(kk_seq), max(kk_seq), bad))
    amps = np.array(snr_seq)
    print("peak-snr: min=%.0f  p10=%.0f  median=%.0f" % (amps.min(), np.percentile(amps, 10), np.median(amps)))
    del iq


if __name__ == "__main__":
    for src in (SF10_SOURCES[0], SF10_SOURCES[2], SF11_SOURCES[0]):
        smoke(*src, fi=0)
        print()
