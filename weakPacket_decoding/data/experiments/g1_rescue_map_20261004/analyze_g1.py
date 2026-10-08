# -*- coding: utf-8 -*-
r"""G1 分析器 + DTS-search v0 等预算评估（2026-10-04）。

输入：g1 checkpoint（多档：-20/-22/-24），输出四块证据：
  A) 三臂统计（每档 A fail / B rescue / C potential）
  B) 成功区域各向异性：SER 按 net=ν−τ 分箱 vs 按 anti=ν+τ 分箱的宽度比
     （ridge 沿 (1,1) 方向 ⇒ SER 应随 net 快变、随 anti 慢变）
  C) 救援点坐标结构：A-fail 单元的 CRC pass 点在 ν 轴可达率 + net 分布；
     ν* 谷底探针：每帧跨种子 SER_net 谷底一致性
  D) 等预算救援曲线（杀伤力表）：
     - FIXED     K=1（=A 臂）
     - BLIND-rand (0,0) 优先 + 随机序（DeRa 式盲枚举代理）
     - NET-ladder (0,0) 优先 + 沿 ν 轴 |ε| 递增（结构引导 1D）
     - VALLEY    谷底引导 ladder 顺序（oracle 标签：用了同帧信息）
     指标：rescue@K、期望试验次数。
近似声明：候选等价用 err_mask 相等代理（hard 判决未存）。
"""
import json
import os
import numpy as np

EXP = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\g1_rescue_map_20261004"
NUS = [round(-0.30 + 0.05 * i, 2) for i in range(13)]
TAUS = [-0.5, -0.25, 0.0, 0.25, 0.5]


def load_maps():
    maps = {}   # (level, seed, frame) -> {(nu,tau): rec}
    for ck, lv in [("checkpoint.jsonl", -20), ("checkpoint_m-22.jsonl", -22),
                   ("checkpoint_m-24.jsonl", -24)]:
        p = os.path.join(EXP, ck)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            maps.setdefault((lv, r["seed"], r["frame"]), {})[(r["nu"], r["tau"])] = r
    return maps


def ser_net_curve(g):
    """SER 按 net=ν−τ 分箱（TREL）。"""
    acc = {}
    for (nu, tau), r in g.items():
        k = round(nu - tau, 2)
        acc.setdefault(k, []).append(r["chains"]["trel"]["ser"])
    return {k: float(np.mean(v)) for k, v in sorted(acc.items())}


def ser_anti_curve(g):
    acc = {}
    for (nu, tau), r in g.items():
        k = round(nu + tau, 2)
        acc.setdefault(k, []).append(r["chains"]["trel"]["ser"])
    return {k: float(np.mean(v)) for k, v in sorted(acc.items())}


def curve_width(c):
    """加权宽度：SER 谷之上的质量展宽（1−min 归一）。"""
    ks = np.array(sorted(c)); vs = np.array([c[k] for k in ks])
    v = 1.0 - (vs - vs.min())          # 成功度
    if v.sum() <= 0:
        return 0.0
    mu = (ks * v).sum() / v.sum()
    return float(np.sqrt((((ks - mu) ** 2) * v).sum() / v.sum()))


def ladder_order():
    """(0,0) 优先，然后沿 ν 轴 |ε| 递增、负侧先（DTS-1 不对称：负侧温和）。"""
    order = [(0.0, 0.0)]
    eps = sorted([e for e in NUS if e != 0.0], key=lambda x: (abs(x), x))
    order += [(e, 0.0) for e in eps]
    return order


BLIND_RNG = np.random.default_rng(20261004)


def blind_order():
    pts = [(nu, tau) for nu in NUS for tau in TAUS if (nu, tau) != (0.0, 0.0)]
    BLIND_RNG.shuffle(pts)
    return [(0.0, 0.0)] + pts


def rescue_at(g, order, K):
    for i, p in enumerate(order[:K]):
        r = g.get(p)
        if r is not None and r["chains"]["trel"]["crc"]:
            return i + 1
    return 0


def main():
    maps = load_maps()
    levels = sorted({k[0] for k in maps})
    print("载入 %d 单元，档 %s\n" % (len(maps), levels))

    # ---- A) 三臂 + B) 各向异性 + C) 谷底 ----
    print("=" * 78)
    print("A) 三臂统计 / B) 各向异性 / C) 救援点结构")
    valley_by_frame = {}
    aniso_all = []
    for lv in levels:
        sel = {k: g for k, g in maps.items() if k[0] == lv}
        a_fail = 0; b_res = 0; c_pot = 0
        res_on_nu_axis = 0; res_need_tau = 0
        res_net = []
        for k, g in sel.items():
            a = g.get((0.0, 0.0))
            if a is None or a["chains"]["trel"]["crc"]:
                continue
            a_fail += 1
            bpts = [p for p, r in g.items() if r["chains"]["trel"]["crc"]]
            if bpts:
                b_res += 1
                if any(tau == 0.0 for (nu, tau) in bpts):
                    res_on_nu_axis += 1
                else:
                    res_need_tau += 1
                res_net += [round(nu - tau, 2) for (nu, tau) in bpts]
            ckey = min(g, key=lambda p: g[p]["chains"]["trel"]["ser"])
            if g[ckey]["chains"]["trel"]["ser"] < 0.5:
                c_pot += 1
            cn = ser_net_curve(g)
            kmin = min(cn, key=lambda x: cn[x])
            valley_by_frame.setdefault((k[1], k[2]), []).append(kmin)
            aniso_all.append(curve_width(ser_anti_curve(g)) /
                             max(curve_width(cn), 1e-9))
        print("[%4d] A fail %3d/%3d | B rescue %3d (%3d) | C pot %3d | "
              "rescue@ν轴 %d 仅τ %d | rescue点net中位 %s | aniso(anti/net) med %.2f" %
              (lv, a_fail, len(sel), b_res,
               100 * b_res // max(a_fail, 1), c_pot,
               res_on_nu_axis, res_need_tau,
               ("%.2f" % np.median(res_net)) if res_net else "NA",
               float(np.median(aniso_all))))
        if res_net:
            hist = {}
            for v in res_net:
                hist[v] = hist.get(v, 0) + 1
            print("       rescue 点 net 直方图: %s" % dict(sorted(hist.items())))

    # 谷底一致性
    cons = [float(np.std(v)) for v in valley_by_frame.values() if len(v) >= 2]
    meds = [float(np.mean(v)) for v in valley_by_frame.values() if len(v) >= 2]
    print("\nC) ν* 谷底探针：帧内跨档谷底 std 中位 %.3f，谷底均值中位 %+.3f（%d 帧有≥2档）"
          % (np.median(cons), np.median(meds), len(cons)))

    # ---- D) 等预算曲线 ----
    print("\n" + "=" * 78)
    print("D) 等预算救援曲线（对每档全部 A-fail 单元）")
    lad = ladder_order()
    bl = blind_order()
    # valley oracle 序：每单元自己的谷底优先（作弊标注）
    budgets = [1, 2, 3, 5, 9, 17, 33, 65]
    strat = {"FIXED": [[(0.0, 0.0)]],
             "BLIND-rand": [bl],
             "NET-ladder": [lad]}
    for lv in levels:
        fails = []
        for k, g in maps.items():
            if k[0] != lv:
                continue
            a = g.get((0.0, 0.0))
            if a is not None and not a["chains"]["trel"]["crc"]:
                fails.append(g)
        if not fails:
            print("[%4d] 无 A-fail 单元" % lv)
            continue
        # valley 序（oracle）
        valseq = []
        for g in fails:
            cn = ser_net_curve(g)
            kmin = min(cn, key=lambda x: cn[x])
            order = [(0.0, 0.0)]
            pts = [(nu, 0.0) for nu in NUS if nu != 0.0]
            pts.sort(key=lambda p: abs((p[0]) - kmin))
            valseq.append(order + pts)
        print("\n[%4d] A-fail n=%d" % (lv, len(fails)))
        print("      %-11s %s" % ("预算K", " ".join("%6d" % k for k in budgets)))
        for name, seqs in list(strat.items()) + [("VALLEY-oracle", valseq)]:
            row = []
            for K in budgets:
                rs = [rescue_at(g, s if isinstance(seqs, list) and len(seqs) > 1 else seqs[0], K)
                      for g, s in zip(fails, seqs)] if name == "VALLEY-oracle" else \
                     [rescue_at(g, seqs[0], K) for g in fails]
                row.append(100 * sum(1 for r in rs if r) / len(fails))
            print("      %-11s %s" % (name, " ".join("%5.0f%%" % v for v in row)))
        # 期望试验数（上限 65）
        for name, seqs in list(strat.items()) + [("VALLEY-oracle", valseq)]:
            ts = []
            for g, s in zip(fails, seqs if name == "VALLEY-oracle" else [seqs[0]] * len(fails)):
                r = rescue_at(g, s, 65)
                ts.append(r if r else 65)
            print("      %-11s 期望试验数(≤65): %.1f" % (name, float(np.mean(ts))))

    # 决策等价近似（err_mask 代理）
    eq = []
    for k, g in maps.items():
        a = g.get((0.0, 0.0))
        if a is None or not a["chains"]["trel"]["crc"]:
            continue
        m0 = tuple(a["chains"]["trel"]["err_mask"])
        same = sum(1 for r in g.values()
                   if tuple(r["chains"]["trel"]["err_mask"]) == m0)
        eq.append(same / len(g))
    print("\n候选等价近似（与锚点 err_mask 全同的点占比，仅 A-pass 单元）："
          "中位 %.2f（n=%d）" % (float(np.median(eq)), len(eq)))


if __name__ == "__main__":
    main()
