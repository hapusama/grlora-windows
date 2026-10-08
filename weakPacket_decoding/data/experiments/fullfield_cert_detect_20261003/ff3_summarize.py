# -*- coding: utf-8 -*-
"""ff3 汇总：H0 闭式性检验 + 门限 + Pd + 净胜表（vs A_P）+ κ̂ 正确率。"""
import json
import os

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "checkpoint_ff3.jsonl")
PFA = 1e-2
V = ["A_P", "FF_split_pre", "FF_split_syn", "FF_split_model", "FF_tmpl"]
EXP_QUANT = -np.log(PFA)          # 精确指数分布的分位数（σ²=1）


def main():
    rows = [json.loads(l) for l in open(CKPT, encoding="utf-8")]
    for ke in (4, 8):
        h1 = [r for r in rows if r["kind"] == "h1" and r.get("ke", 4) == ke]
        h0 = [r for r in rows if r["kind"] == "h0" and r.get("ke", 4) == ke]
        if not h1:
            continue
        vv = V if ke == 4 else ["A_P", "FF_split_model", "FF_tmpl"]
        noisy0 = [r for r in h0 if r["level"] is not None]
        lvs = sorted(set(r["level"] for r in noisy0), key=lambda x: -x)
        print("\n========== K_e=%d（H1 %d / H0 %d）==========" %
              (ke, len(h1), len(h0)))
        # ---- H0：门限 + 闭式性 ----
        thr = {}
        print("H0 分档门限 dB（99 分位）| 精确指数参考 %.2f dB + σ̂² 中位数偏差"
              % (10 * np.log10(EXP_QUANT)))
        for v in vv:
            per = []
            for lv in lvs:
                x = np.array([r["scores"][v] for r in noisy0
                              if r["level"] == lv])
                thr[(v, lv)] = float(np.quantile(x, 1 - PFA))
                per.append(10 * np.log10(thr[(v, lv)]))
            print("  %-14s %s | 工作段极差 %.2f dB"
                  % (v, " ".join("%5.1f" % p for p in per),
                     max(per[3:]) - min(per[3:])))
        # 指数性检验：H0 分位比（q_1e-1/q_1e-2 应 = ln0.1/ln0.01=0.5）
        for v in vv:
            x = np.array([r["scores"][v] for r in noisy0])
            q1 = np.quantile(x, 0.9)
            q2 = np.quantile(x, 0.99)
            print("  %-14s H0 分位比 q90/q99=%.3f（指数=0.500；越低越重尾）"
                  % (v, q1 / q2))
        # ---- Pd + 裕度 + κ̂ ----
        caps = sorted(set(r["cap"] for r in h1))
        print("检测统计（分档门限；裕度=中位H1−门限 dB）：")
        print("  [%6s]" % "level" + "".join("%18s" % v for v in vv)
              + "   κ̂ok%")
        for lv in lvs:
            s = [r for r in h1 if r["level"] == lv]
            cells = []
            for v in vv:
                mg = np.median([10 * np.log10(r["scores"][v])
                                - 10 * np.log10(thr[(v, lv)]) for r in s])
                pd = np.mean([r["scores"][v] > thr[(v, lv)] for r in s])
                cells.append("%6.1f/%4.2f" % (mg, pd))
            okr = np.mean([abs(r["scores"]["dkap"]) < 1.0 / (2 * 256)
                           for r in s])
            print("  [%6d]" % lv + "".join("%18s" % c for c in cells)
                  + "   %5.2f" % okr)
        # ---- 净胜表：等 Pfa 下相对 A_P 的裕度差（−22~−28 主战带） ----
        print("净胜 vs A_P（裕度差 dB，正=胜）：")
        for lv in lvs:
            s = [r for r in h1 if r["level"] == lv]
            line = []
            for v in vv[1:]:
                mgA = np.median([10 * np.log10(r["scores"]["A_P"])
                                 - 10 * np.log10(thr[("A_P", lv)])
                                 for r in s])
                mgV = np.median([10 * np.log10(r["scores"][v])
                                 - 10 * np.log10(thr[(v, lv)])
                                 for r in s])
                line.append("%s%+.2f" % (v.replace("FF_split_", ""), mgV - mgA))
            print("  [%6d] %s" % (lv, "  ".join(line)))


if __name__ == "__main__":
    main()
