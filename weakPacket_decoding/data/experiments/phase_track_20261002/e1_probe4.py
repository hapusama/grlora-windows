# -*- coding: utf-8 -*-
"""E1 探针4：单帧肉眼看 + 全符号 fine-ν 扫描判 ν̂ 伪影。

对子集 capture 逐符号 fine 扫描（±0.9 bin 内 181 点 DTFT argmax）：
若 payload 残差塌缩 → 抛物线 ν̂ 伪影；若保持 → 真实逐符号相位结构。
"""
import sys
import os
import csv
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_common import (DS, align_seg, freeze_gt, measure_frame, extract_raw,
                       make_slots, c_nominals, fine_scan_dr, wrap,
                       SF10_SOURCES, SF11_SOURCES)


def gn_fit(th, B, iters=8):
    x = np.zeros(B.shape[1])
    for _ in range(iters):
        r = wrap(th - B @ x)
        dx, *_ = np.linalg.lstsq(B, r, rcond=None)
        x += dx
    return x, wrap(th - B @ x)


def frame_fine(iq, r, ds, P, psym, ldro, gt_hdr, gt):
    """全符号 fine 扫描版本测量。"""
    seg, i0, boff = align_seg(iq, r, ds, P, psym)
    syms, track0, cl = measure_frame(seg, P, psym, ldro, gt_hdr, gt, ds, fine=False)
    for s in syms:
        # 用 stored 网格测量的 ν 作中心重扫
        pass
    return syms, track0, seg


def fine_all(iq, r, ds, P, psym, ldro, gt_hdr, gt):
    seg, i0, boff = align_seg(iq, r, ds, P, psym)
    if seg is None:
        return None
    syms, track0, cl = measure_frame(seg, P, psym, ldro, gt_hdr, gt, ds, fine=False)
    # 对每个 fit 符号在 c±1.5 范围 fine 扫（独立于跟踪 ν̂）
    slots = make_slots(P, psym)
    it = iter(cl)
    for (kind, k, off, use), s in zip([sl for sl in slots if sl[3]], syms):
        c0 = s["c"]
        off_nf = dict(((kk, kk2), o) for kk, kk2, o in
                      [(x[0], x[1], x[2]) for x in slots])[(kind, k)]
        w = seg[int(round(off_nf * ds.nf)):int(round(off_nf * ds.nf)) + ds.nf]
        W = np.fft.fft(w.astype(np.complex128)) * ds.mask
        d = np.fft.ifft(W)
        dr = (d[::ds.os] * ds.ref_for(kind)).astype(np.complex128)
        fr = c0 + np.linspace(-1.5, 1.5, 301)
        V = np.exp(-2j * np.pi * np.outer(fr, np.arange(len(dr))) / ds.n) @ dr
        j = int(np.argmax(np.abs(V)))
        s["nu_fine2"] = float(fr[j])
        s["theta_fine2"] = float(np.angle(V[j]))
        s["amp_fine2"] = float(np.abs(V[j]))
    return syms, track0


def resid(syms, theta_key, nb, terms=("1", "i", "i2", "c", "syn", "hdr", "pay")):
    th = np.array([s[theta_key] for s in syms])
    c = np.array([s["c"] for s in syms], dtype=float) / nb
    kinds = np.array([s["kind"] for s in syms])
    i = np.arange(len(syms), dtype=float)
    cols = {"1": np.ones_like(i), "i": i, "i2": i ** 2, "c": c,
            "syn": (kinds == "sync").astype(float),
            "hdr": (kinds == "hdr").astype(float),
            "pay": (kinds == "pay").astype(float)}
    B = np.stack([cols[t] for t in terms], 1)
    x, rr = gn_fit(th, B)
    return rr, c, kinds


def main():
    srcs = [SF10_SOURCES[0], SF10_SOURCES[1], SF11_SOURCES[0], SF11_SOURCES[1]]
    # 肉眼:第一帧
    cap, bp, cp_, sf, P, ldro = SF10_SOURCES[0]
    ds = DS(sf)
    iq = np.memmap(bp, dtype=np.complex64, mode="r")
    rows = [rr for rr in csv.DictReader(open(cp_, encoding="utf-8"))
            if rr.get("header_valid") == "1" and int(rr.get("payload_len") or 0) > 0]
    r0 = rows[0]
    psym = int(r0["payload_symbol_count"])
    gt_hdr, gt = freeze_gt(iq, r0, ds, psym, ldro)
    syms, tr0, seg = frame_fine(iq, r0, ds, P, psym, ldro, gt_hdr, gt)
    print("== 肉眼 %s 帧0 (P=%d psym=%d): pre θ 序列" % (cap, P, psym))
    print("   " + " ".join("%+.2f" % s["theta"] for s in syms if s["kind"] == "pre"))
    print("   sync θ: " + " ".join("%+.2f" % s["theta"] for s in syms if s["kind"] == "sync"))
    pay = [s for s in syms if s["kind"] == "pay"]
    print("   pay θ (前 20): " + " ".join("%+.2f" % s["theta"] for s in pay[:20]))
    print("   pay ν−c (前 20): " + " ".join("%+.2f" % (s["nu"] - s["c"]) for s in pay[:20]))
    print("   track0:", {k: round(v, 4) if isinstance(v, float) else v for k, v in tr0.items()})
    del iq

    # fine 全扫
    agg = {}
    for cap, bp, cp_, sf, P, ldro in srcs:
        ds = DS(sf)
        nb = 1 << sf
        iq = np.memmap(bp, dtype=np.complex64, mode="r")
        rows = [rr for rr in csv.DictReader(open(cp_, encoding="utf-8"))
                if rr.get("header_valid") == "1" and int(rr.get("payload_len") or 0) > 0]
        for r0 in rows[:6]:
            psym = int(r0["payload_symbol_count"])
            try:
                gt_hdr, gt = freeze_gt(iq, r0, ds, psym, ldro)
            except Exception:
                continue
            out = fine_all(iq, r0, ds, P, psym, ldro, gt_hdr, gt)
            if out is None:
                continue
            syms, _ = out
            for key in ("theta", "theta_fine2"):
                rr_, c, kinds = resid(syms, key, nb)
                for kk in ("pre", "sync", "hdr", "pay"):
                    agg.setdefault((key, kk), []).extend(rr_[kinds == kk])
        del iq
    print("\n== 全符号 fine 扫描（θ@网格跟踪ν̂ vs θ@fine argmaxν̂, 4 capture × ≤6 帧）:")
    for key in ("theta", "theta_fine2"):
        line = "   %-11s" % key
        for kk in ("pre", "sync", "hdr", "pay"):
            v = np.array(agg.get((key, kk), []))
            line += "  %s σ=%.3f (n=%d)" % (kk, v.std(), len(v))
        print(line)


if __name__ == "__main__":
    main()
