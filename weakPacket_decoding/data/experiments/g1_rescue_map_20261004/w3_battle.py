# -*- coding: utf-8 -*-
r"""W3-battle：弱点 3 第一阶段——payload 段候选调度的同表对照（2026-10-04）。

问题（doc/DeRa攻击面_弱点356_20261004.md）：DeRa Stage 3 = 检测排序候选
重跑 Stages 1-2，上限 5 个。我方主张：判决引导排序 + 更深预算。

臂（全部同一相干引擎 = DERA port v2 两段合并，同一份带噪段/候选网格）：
  A0     单候选（前导点）= DeRa Stages 1-2
  STAGE3 DeRa Stage 3 忠实代理：锚点起按 |net| 序前 5 个候选重跑
         （声明：真 Stage 3 按检测峰排序；此处以 net 距离序代理，
          对 DeRa 偏保守有利——net 序是地图实测最优序之一）
  BLIND@K (0,0) 优先 + 随机序，K ∈ {5,13,33,65}
  RIDGE@K net 等价类静态序（判决引导排序 v1），K ∈ {5,13,33,65}
指标：整包 CRC 恢复率（对 A0-fail 单元）、期望试验数、单包耗时
（每候选 = demod_payload 全重跑，实测 ~0.17s，复用引擎 v2 另报）。

数据来源：g1 checkpoint_dera_m24.jsonl（−24 dB 相干网格，84 单元 65 点，
每点已含 CRC 与 err_mask）——本脚本为查表评估，无新解码计算；
网格生成时的铁律遵守见 g1_dera_grid.py 头注与 RESULTS 增补 2。
"""
import json
import os
import numpy as np

EXP = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\g1_rescue_map_20261004"
NUS = [round(-0.30 + 0.05 * i, 2) for i in range(13)]
TAUS = [-0.5, -0.25, 0.0, 0.25, 0.5]
SEC_PER_CAND = 0.17   # g1_dera_grid 实测均值（5460 单元 928s / 12 workers 标定）


def load():
    maps = {}
    for line in open(os.path.join(EXP, "checkpoint_dera_m24.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        maps.setdefault((r["seed"], r["frame"]), {})[(r["nu"], r["tau"])] = r
    return maps


def stage3_order(g):
    """DeRa Stage 3 代理：锚点 + 按 |net=ν−τ| 距离的前 5 候选。"""
    cls = {}
    for p in g:
        cls.setdefault(round(p[0] - p[1], 2), []).append(p)
    reps = {e: min(v, key=lambda p: abs(p[0]) + abs(p[1])) for e, v in cls.items()}
    seq = [(0.0, 0.0)]
    for e in sorted(cls, key=lambda x: abs(x)):
        seq.append(reps[e])
    return seq[:6]     # 锚点 + 5 候选


def ridge_order(g):
    cls = {}
    for p in g:
        cls.setdefault(round(p[0] - p[1], 2), []).append(p)
    reps = {e: min(v, key=lambda p: abs(p[0]) + abs(p[1])) for e, v in cls.items()}
    seq = []
    for e in sorted(cls, key=lambda x: (abs(x), abs(reps[x][0]))):
        seq.append(reps[e])
        seq += sorted([p for p in cls[e] if p != reps[e]],
                      key=lambda p: abs(p[0]) + abs(p[1]))
    return seq


def blind_order(rng):
    pts = [(nu, tau) for nu in NUS for tau in TAUS if (nu, tau) != (0.0, 0.0)]
    rng.shuffle(pts)
    return [(0.0, 0.0)] + pts


def rescue_at(g, order, K):
    for i, p in enumerate(order[:K]):
        r = g.get(p)
        if r is not None and r["dera"]["crc"]:
            return i + 1
    return 0


def main():
    maps = load()
    fails, all_pk = [], 0
    a0_pass = 0
    for g in maps.values():
        all_pk += 1
        a = g.get((0.0, 0.0))
        if a is None:
            continue
        if a["dera"]["crc"]:
            a0_pass += 1
        else:
            fails.append(g)
    rng = np.random.default_rng(20261004)
    blind = blind_order(rng)

    print("[W3-battle @−24dB] 单元=%d | A0(单候选=DeRa S1-2) 通过 %d (%.1f%%) | "
          "A0-fail n=%d" % (all_pk, a0_pass, 100 * a0_pass / all_pk, len(fails)))

    def report(name, seq_fn, Ks):
        print("  %-10s %s" % (name, " ".join("%8s" % ("K=%d" % k) for k in Ks)))
        rows, ts = [], []
        for K in Ks:
            rs = [rescue_at(g, seq_fn(g), K) for g in fails]
            rows.append(100 * sum(1 for r in rs if r) / len(fails))
            ts.append(float(np.mean([r or K for r in
                                     [rescue_at(g, seq_fn(g), K) for g in fails]])))
        print("     恢复率    %s" % " ".join("%7.0f%%" % v for v in rows))
        print("     E[试验]   %s   单包耗时@K: %s" %
              (" ".join("%8.1f" % t for t in ts),
               " ".join("%.2fs" % (t * SEC_PER_CAND) for t in ts)))

    print("\n对 A0-fail 的救援（=DeRa 框架内可多恢复的整包）：")
    report("STAGE3", stage3_order, [6])
    report("BLIND", lambda g: blind, [5, 13, 33, 65])
    report("RIDGE", ridge_order, [5, 13, 33, 65])

    # 系统级汇总：总整包恢复率（A0 直接过 + 救援）
    print("\n总整包恢复率（A0 通过 + 救援）：")
    for name, seq_fn, K in [("A0（单候选=DeRa S1-2）", None, 1),
                            ("STAGE3(5候选)", stage3_order, 6),
                            ("BLIND K=13", lambda g: blind, 13),
                            ("RIDGE K=13", ridge_order, 13),
                            ("RIDGE K=65", ridge_order, 65)]:
        if seq_fn is None:
            tot = 100.0 * a0_pass / all_pk
        else:
            n_resc = sum(1 for g in fails if rescue_at(g, seq_fn(g), K))
            tot = 100.0 * (a0_pass + n_resc) / all_pk
        print("  %-38s %.1f%%" % (name, tot))


if __name__ == "__main__":
    main()
