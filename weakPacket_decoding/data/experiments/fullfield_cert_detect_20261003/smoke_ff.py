# -*- coding: utf-8 -*-
"""冒烟：帧模板 + native/−24dB 的 H1/H0 分离度与走动量级检查。"""
import sys
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
                 r"\experiments\fullfield_cert_detect_20261003")
import ff_runner as R

R.FRAMES = R.build_frames()
print("frames:", len(R.FRAMES))
for fi in (0, 27):
    f = R.FRAMES[fi]
    t = R.frame_template(f)
    print("frame%02d %s P=%d walk=%+.4f bin/sym conj_dn=%d k_pre=%d "
          "k_sync=%s k_dn=%d" %
          (fi, f["cap"], f["pre"], t["walk"], t["conj_dn"], t["k_pre"],
           t["k_j"][f["pre"]:f["pre"] + 2], t["k_j"][-1]))
    for lv in (None, -24):
        recs = R.eval_unit(f, t, lv, 0)
        h1 = recs[0]["scores"]
        h0 = [r["scores"] for r in recs[1:]]
        print("  [%6s] H1 A=%.1f FF=%.1f TMPL=%.1f dB | H0 A=%.1f±%.1f "
              "FF=%.1f±%.1f TMPL=%.1f±%.1f dB" %
              ("native" if lv is None else lv,
               *(10 * np.log10(h1[v]) for v in ("A_P", "FF_lin", "FF_tmpl")),
               *(10 * np.log10(np.mean([h[v] for h in h0])) for v in
                 ("A_P", "FF_lin", "FF_tmpl")),
               *(10 * np.log10(np.std([h[v] for h in h0])) for v in
                 ("A_P", "FF_lin", "FF_tmpl"))))
