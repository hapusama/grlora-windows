# -*- coding: utf-8 -*-
"""fullfield_cert_detect_20261003 汇总 v2。

指标：
  - 门限：分档经验门限（Pfa=1e-2）+ 池化档位不变性（CFAR 性证据）；
  - Pd（分档门限）→ 检测门限 SNR@Pd=0.5；
  - 裕度 = 中位(H1 score) − 分档门限（dB）——变体 dB 差的直接读数，
    不依赖 Pd 落 0.5；
  - 分 capture 组（P=8/16/32）+ 池化。
"""
import json
import os

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")
VARIANTS = ["A_P", "FF_up", "FF_lin", "FF_q", "FF_ml", "FF_tmpl"]
PFA = 1e-2
LEVEL_ORDER = [None, -10, -14, -17, -20, -22, -24, -26, -28, -30, -32]


def lv_key(lv):
    return 99 if lv is None else -lv


def main():
    rows = [json.loads(l) for l in open(CKPT, encoding="utf-8")]
    h1 = [r for r in rows if r["kind"] == "h1"]
    h0 = [r for r in rows if r["kind"] == "h0"]
    noisy0 = [r for r in h0 if r["level"] is not None]
    lvs = sorted(set(r["level"] for r in noisy0), key=lv_key)
    print("记录：H1 %d，H0 %d（噪声档 H0 %d）" % (len(h1), len(h0), len(noisy0)))

    walks = sorted(set((r["cap"], r["frame"], round(r["walk"], 4))
                       for r in h1 if r["level"] is None))
    wv = np.array([w for _, _, w in walks])
    print("真实走动（bin/符，帧级）：中位 %+.4f，范围 [%+.4f, %+.4f]，"
          "负向 %d/28（frame17 卷绕边界帧除外）"
          % (np.median(wv), wv.min(), wv.max(), int((wv < -0.001).sum())))

    # ---- 门限：分档 + 池化不变性 ----
    thr_lv, thr_pool = {}, {}
    for v in VARIANTS:
        pooled = np.array([r["scores"][v] for r in noisy0])
        thr_pool[v] = float(np.quantile(pooled, 1 - PFA))
        for lv in lvs:
            x = np.array([r["scores"][v] for r in noisy0 if r["level"] == lv])
            thr_lv[(v, lv)] = float(np.quantile(x, 1 - PFA))
    print("\nH0 分档门限 dB（99 分位；池化极差 = CFAR 稳定性）：")
    for v in VARIANTS:
        per = [10 * np.log10(thr_lv[(v, lv)]) for lv in lvs]
        print("  %-8s 池化 %5.2f | %s | 极差 %.2f dB | H0均值 %.2f dB"
              % (v, 10 * np.log10(thr_pool[v]),
                 " ".join("%.1f" % p for p in per), max(per) - min(per),
                 10 * np.log10(np.mean(
                     [r["scores"][v] for r in noisy0]))))

    # ---- 分组表：Pd（分档门限）+ 裕度 ----
    caps = sorted(set(r["cap"] for r in h1))
    for cap in caps:
        sub = [r for r in h1 if r["cap"] == cap]
        pre = sub[0]["pre"]
        print("\n== [%s P=%d] 分档门限下的 Pd 与裕度(dB) ==" % (cap, pre))
        print("  [%6s] n=%-4d" % ("level", 0)
              + " | ".join("%13s" % v for v in VARIANTS))
        nat = [r for r in sub if r["level"] is None]
        cells = []
        for v in VARIANTS:
            mg = np.median([10 * np.log10(r["scores"][v]) for r in nat])
            cells.append("%5s/%+5.1f" % ("nat", mg))
        print("  [%6s] n=%-4d %s" % ("native", len(nat), " | ".join(cells)))
        for lv in sorted((x for x in set(r["level"] for r in sub)
                          if x is not None), key=lv_key):
            s = [r for r in sub if r["level"] == lv]
            cells = []
            for v in VARIANTS:
                pd = np.mean([r["scores"][v] > thr_lv[(v, lv)] for r in s])
                mg = np.median([10 * np.log10(r["scores"][v])
                                - 10 * np.log10(thr_lv[(v, lv)]) for r in s])
                cells.append("%5.2f/%+5.1f" % (pd, mg))
            print("  [%6s] n=%-4d %s" % ("native" if lv is None else lv,
                                         len(s), " | ".join(cells)))
        # 检测门限 SNR@Pd=0.5（分档门限）
        line = []
        for v in VARIANTS:
            pts = []
            for lv in sorted((x for x in set(r["level"] for r in sub)
                              if x is not None), key=lv_key):
                s = [r for r in sub if r["level"] == lv]
                pts.append(((30 if lv is None else lv),
                            np.mean([r["scores"][v] > thr_lv[(v, lv)]
                                     for r in s])))
            sn = np.array([p for p, _ in pts], dtype=float)
            p = np.array([q for _, q in pts])
            cross = np.nan
            for i in range(len(p) - 1):
                if p[i] >= 0.5 > p[i + 1]:
                    cross = sn[i] + (0.5 - p[i]) * (sn[i + 1] - sn[i]) \
                        / max(p[i] - p[i + 1], 1e-9)
                    break
            line.append((v, cross))
        base = dict(line)["A_P"]
        for v, c in line:
            print("    门限 %-8s %s%s"
                  % (v, "n/a" if np.isnan(c) else "%+.1f dB" % c,
                     "" if np.isnan(c) or np.isnan(base)
                     else "  (vs A_P %+.2f dB)" % (c - base)))


if __name__ == "__main__":
    main()
