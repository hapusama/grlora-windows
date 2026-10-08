# -*- coding: utf-8 -*-
"""M3+ 终战汇总（2026-10-04）：m3p_checkpoint.jsonl → PER 战表 + 门限插值
+ 失效分解 + 配对 w/l + join 覆盖率。只打 stdout（落盘走 shell 重定向；
     python m3p_sum.py > m3p_sum_results.json.txt）。"""
import collections
import json

DELTAS = (0.0, 0.02, 0.082)
LEVELS = [-18, -20, -22, -24, -26]
N_TOTAL = 28 * 20
CKPT = "m3p_checkpoint.jsonl"


def thr_at(per_by_lv, target):
    """线性插值：PER 曲线跨 target 的 SNR（越低越好）。"""
    lvs = sorted(per_by_lv, reverse=True)          # −18 → −26
    p = [per_by_lv[l] for l in lvs]
    for i in range(len(p) - 1):
        if p[i] <= target <= p[i + 1] or p[i + 1] <= target <= p[i]:
            if p[i + 1] == p[i]:
                return float(lvs[i])
            f = (target - p[i]) / (p[i + 1] - p[i])
            return float(lvs[i] + f * (lvs[i + 1] - lvs[i]))
    return None


def main():
    agg = collections.defaultdict(
        lambda: dict(n=0, u=0, d=0, w=0, l=0, join=0,
                     fc=collections.Counter()))
    nat = collections.defaultdict(lambda: dict(n=0, u=0, d=0))
    for line in open(CKPT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["level"] is None:
            a = nat[r["delta"]]
            a["n"] += 1
            a["u"] += r.get("u_per", 1)
            a["d"] += r.get("dera_per", 1)
            continue
        a = agg[(r["delta"], r["level"])]
        a["n"] += 1
        a["u"] += r.get("u_per", 1)
        a["d"] += r.get("dera_per", 1)
        a["join"] += int(r.get("dera_join", 0))
        a["fc"][r.get("fail_u")] += 1
        if "u_per" in r and "dera_per" in r:
            if r["u_per"] < r["dera_per"]:
                a["w"] += 1
            elif r["u_per"] > r["dera_per"]:
                a["l"] += 1
    res = {}
    print("== M3+ 终战（SF10 OTA 28帧，纯 AWGN，20 种子满额=%d/档）==" % N_TOTAL)
    print("delta  lev    n | OURS+ | dera  | w:l  | join | 主失效")
    for dd in DELTAS:
        per_u, per_d = {}, {}
        for lv in LEVELS:
            a = agg[(dd, lv)]
            if not a["n"]:
                continue
            pu, pd = a["u"] / a["n"], a["d"] / a["n"]
            per_u[lv], per_d[lv] = pu, pd
            top = a["fc"].most_common(2)
            print("%-5s %5d %4d | %.3f | %.3f | %d:%d | %4d | %s" % (
                dd, lv, a["n"], pu, pd, a["w"], a["l"], a["join"],
                ",".join("%s×%d" % t for t in top)))
            res["d%g_lv%d" % (dd, lv)] = dict(
                n=a["n"], ours=pu, dera=pd, w=a["w"], l=a["l"],
                join=a["join"], fail=dict(a["fc"]))
        t10u, t10d = thr_at(per_u, 0.1), thr_at(per_d, 0.1)
        t50u, t50d = thr_at(per_u, 0.5), thr_at(per_d, 0.5)
        print("  → 10%%PER 门限：OURS+ %s vs dera %s；50%%PER：%s vs %s" % (
            t10u, t10d, t50u, t50d))
        res["thr_d%g" % dd] = dict(t10_ours=t10u, t10_dera=t10d,
                                   t50_ours=t50u, t50_dera=t50d)
        if dd in nat and nat[dd]["n"]:
            res["native_d%g" % dd] = dict(
                n=nat[dd]["n"], ours=nat[dd]["u"] / nat[dd]["n"],
                dera=nat[dd]["d"] / nat[dd]["n"])
    print("JSON:")
    print(json.dumps(res, default=str))


if __name__ == "__main__":
    main()
