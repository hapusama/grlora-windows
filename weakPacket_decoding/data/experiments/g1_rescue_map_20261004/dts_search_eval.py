# -*- coding: utf-8 -*-
r"""DTS-search 策略扩展评估（2026-10-04，纯查表，零新计算）。

背景：NET-ladder（纯 ν 轴）小预算区优但渐近只有 45%（−24 档 27/64 救援点
仅在 τ≠0 处）。策略 v1：
  HIER-ladder : ν 轴 ladder 全扫 → 逐行 τ=±0.25,±0.5 各扫 ν ladder
  RIDGE-class : 按 net=ν−τ 等价类分组；每类先探代表点，类代表 CRC 过才
                类内细化（"按对解码影响等价合并候选"的直接实现）
  BLIND-raster: (0,0) 优先 + 光栅序（DeRa 式盲枚举的保守代理）
对照：FIXED / BLIND-rand / NET-ladder（analyze_g1.py 已出）。
"""
import json
import os
import numpy as np

EXP = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\g1_rescue_map_20261004"
NUS = [round(-0.30 + 0.05 * i, 2) for i in range(13)]
TAUS = [-0.5, -0.25, 0.0, 0.25, 0.5]


def load_maps():
    maps = {}
    for ck, lv in [("checkpoint.jsonl", -20), ("checkpoint_m-22.jsonl", -22),
                   ("checkpoint_m-24.jsonl", -24)]:
        p = os.path.join(EXP, ck)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            maps.setdefault((lv, r["seed"], r["frame"]), {})[(r["nu"], r["tau"])] = r
    return maps


def nu_ladder():
    eps = sorted([e for e in NUS if e != 0.0], key=lambda x: (abs(x), x))
    return [(0.0, 0.0)] + [(e, 0.0) for e in eps]


def hier_order():
    """ν 轴全扫（13 点）→ τ 行逐行（按 |τ| 递增）扫 ν ladder。"""
    o = nu_ladder()
    for tau in sorted([t for t in TAUS if t != 0.0], key=abs):
        eps = sorted(NUS, key=lambda x: (abs(x), x))
        o += [(e, tau) for e in eps]
    return o


def ridge_class_order(g):
    """net 等价类代表优先：类按代表点 |net| 排序；代表点=(该类中最接近
    (0,0) 的格点)。代表 CRC 过 → 该类其余点紧随其后；不过 → 类内全部跳过。
    顺序依赖 g（自适应：用 CRC 反馈）——模拟时逐类判定。返回"评估点序列"
    与每点的类别标签，供 rescue_at 使用（只按评估顺序查 CRC）。"""
    classes = {}
    for p in g:
        classes.setdefault(round(p[0] - p[1], 2), []).append(p)
    reps = {}
    for e, pts in classes.items():
        reps[e] = min(pts, key=lambda p: abs(p[0]) + abs(p[1]))
    eorder = sorted(classes, key=lambda e: (abs(reps[e][0] - reps[e][1])
                                            if False else abs(e), abs(reps[e][0])))
    seq = []
    for e in eorder:
        seq.append(reps[e])
        # 代表点成功才展开类内（rescue_at 只看 CRC；这里保守模拟：
        # 展开条件未知时全部展开会退化为盲搜——按类代表 SER 谱决定）
        rest = sorted([p for p in classes[e] if p != reps[e]],
                      key=lambda p: abs(p[0]) + abs(p[1]))
        seq += rest
    return seq


def ridge_class_order_adaptive(g):
    """真自适应版：代表点 CRC 通过才展开该类（模拟在线决策）。
    返回的序列在"代表点失败"时跳过该类——与盲搜的区别即在此。"""
    classes = {}
    for p in g:
        classes.setdefault(round(p[0] - p[1], 2), []).append(p)
    reps = {e: min(pts, key=lambda p: abs(p[0]) + abs(p[1]))
            for e, pts in classes.items()}
    eorder = sorted(classes, key=lambda e: (abs(e), abs(reps[e][0])))
    seq = []
    for e in eorder:
        rep = reps[e]
        seq.append(rep)
        r = g.get(rep)
        if r is not None and r["chains"]["trel"]["crc"]:
            rest = sorted([p for p in classes[e] if p != rep],
                          key=lambda p: abs(p[0]) + abs(p[1]))
            seq += rest
    return seq


def rescue_at(g, order, K):
    for i, p in enumerate(order[:K]):
        r = g.get(p)
        if r is not None and r["chains"]["trel"]["crc"]:
            return i + 1
    return 0


def main():
    maps = load_maps()
    budgets = [1, 2, 3, 5, 9, 13, 17, 33, 65]
    hier = hier_order()
    raster = [(0.0, 0.0)]
    for tau in TAUS:
        for nu in NUS:
            if (nu, tau) != (0.0, 0.0):
                raster.append((nu, tau))

    for lv in (-24, -22, -20):
        fails = [g for k, g in maps.items()
                 if k[0] == lv and (0.0, 0.0) in g
                 and not g[(0.0, 0.0)]["chains"]["trel"]["crc"]]
        if not fails:
            continue
        print("\n[%4d] A-fail n=%d" % (lv, len(fails)))
        print("      %-18s %s" % ("预算K", " ".join("%6d" % k for k in budgets)))
        for name, mk in [("HIER-ladder", lambda g: hier),
                         ("RIDGE-class", ridge_class_order),
                         ("RIDGE-adaptive", ridge_class_order_adaptive),
                         ("BLIND-raster", lambda g: raster)]:
            rows = []
            for K in budgets:
                rs = [rescue_at(g, mk(g), K) for g in fails]
                rows.append(100 * sum(1 for r in rs if r) / len(fails))
            ts = [rescue_at(g, mk(g), 65) or 65 for g in fails]
            print("      %-18s %s   E[试验]=%.1f" %
                  (name, " ".join("%5.0f%%" % v for v in rows), float(np.mean(ts))))


if __name__ == "__main__":
    main()
