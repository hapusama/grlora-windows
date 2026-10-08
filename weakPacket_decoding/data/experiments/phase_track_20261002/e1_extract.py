# -*- coding: utf-8 -*-
"""E1 native 全帧提取 → e1_native.jsonl（协议 §1 GT 冻结，measure_frame 两级测量）。"""
import sys
import csv
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_common import (ROOT, align_seg, snr_parts, freeze_gt, measure_frame,
                       SF10_SOURCES, SF11_SOURCES)

OUT = os.path.join(ROOT, "e1_native.jsonl")


def first_of(x):
    return float((x or "0").strip().split("|")[0] or 0)


def main():
    t0 = time.time()
    stats = dict(gid=0, ok=0, gt_reject=0, lead_reject=0)
    with open(OUT, "w", encoding="utf-8") as fout:
        for cap, bin_path, csv_path, sf, P, ldro in SF10_SOURCES + SF11_SOURCES:
            print("== %s" % cap, flush=True)
            ds = DS_ = __import__("e1_common").DS(sf)
            iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
            rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                    if r.get("header_valid") == "1" and int(r.get("payload_len") or 0) > 0]
            for ri, r in enumerate(rows):
                psym = int(r["payload_symbol_count"])
                try:
                    gt_hdr, gt = freeze_gt(iq, r, ds, psym, ldro)
                except Exception as e:
                    stats["gt_reject"] += 1
                    print("  [%s#%d] GT 剔除: %s" % (cap, ri, e), flush=True)
                    continue
                seg, i0, boff = align_seg(iq, r, ds, P, psym)
                if seg is None:
                    stats["lead_reject"] += 1
                    continue
                S, N0 = snr_parts(seg[boff * ds.nf:], ds.nf)
                syms, track0, _cl = measure_frame(seg, P, psym, ldro, gt_hdr, gt,
                                                  ds, fine=True)
                dlt = [(s["b"] - s["c"]) % ds.n for s in syms if s["kind"] == "pay"]
                delta = int(np.bincount(np.array(dlt), minlength=ds.n).argmax()) if dlt else None
                rec = dict(gid=stats["gid"], cap=cap, sf=sf, P=P, ldro=ldro,
                           psym=psym, plen=int(r["payload_len"]), cr=int(r["cr"]),
                           hs=int(r["header_start_sample"]),
                           cfo_int=int(r["source_grlora_cfo_int"]),
                           cfo_frac=first_of(r.get("source_grlora_cfo_frac")),
                           sto_frac=first_of(r.get("source_grlora_payload_sto_frac")),
                           sfo_hat=first_of(r.get("source_grlora_sfo_hat")),
                           S=S, N0=N0, snr_native=10 * np.log10(S / N0),
                           track0=track0, delta=delta, gt_hdr=gt_hdr, gt=gt,
                           syms=syms)
                fout.write(json.dumps(rec) + "\n")
                stats["gid"] += 1
                stats["ok"] += 1
            del iq
    print("完成: %d 帧入库, GT 剔除 %d, 越界剔除 %d (%.0fs)"
          % (stats["ok"], stats["gt_reject"], stats["lead_reject"], time.time() - t0))


if __name__ == "__main__":
    main()
