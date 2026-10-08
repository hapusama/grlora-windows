# -*- coding: utf-8 -*-
"""M3+2 终战汇总（2026-10-05）：m3p2_checkpoint.jsonl → PER 战表 + 门限
插值 + 配对 w/l + 成功 slope 分布。只打 stdout（落盘走 shell 重定向）。

⚠️ method_m3p_20261005 档案副本。在 04_results/ 内运行：
     cd 04_results && python ../03_battle/m3p2_sum.py
"""
import collections
import json

DELTAS = (0.0, 0.02)
LEVELS = [-18, -20, -22, -24, -26]
N_TOTAL = 28 * 20
CKPT = "m3p2_checkpoint.jsonl"


def thr_at(per_by_lv, target):
    lvs = sorted(per_by_lv, reverse=True)
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
        lambda: dict(n=0, u=0, d=0, w=0, l=0, sl=collections.Counter()))
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
        if r.get("u_per") == 0:
            a["sl"][r.get("u_slope", 0.0)] += 1
        if "u_per" in r and "dera_per" in r:
            if r["u_per"] < r["dera_per"]:
                a["w"] += 1
            elif r["u_per"] > r["dera_per"]:
                a["l"] += 1
    print("== M3+2 终战（斜率梯；SF10 OTA 28帧，纯 AWGN，满额=%d/档）=="
          % N_TOTAL)
    print("delta  lev    n | OURS+2| dera  | w:l   | 成功slope分布")
    for dd in DELTAS:
        per_u, per_d = {}, {}
        for lv in LEVELS:
            a = agg[(dd, lv)]
            if not a["n"]:
                continue
            pu, pd = a["u"] / a["n"], a["d"] / a["n"]
            per_u[lv], per_d[lv] = pu, pd
            print("%-5s %5d %4d | %.3f | %.3f | %d:%-4d| %s" % (
                dd, lv, a["n"], pu, pd, a["w"], a["l"], dict(a["sl"])))
        print("  → 10%%PER：OURS+2 %s vs dera %s；50%%PER：%s vs %s" % (
            thr_at(per_u, 0.1), thr_at(per_d, 0.1),
            thr_at(per_u, 0.5), thr_at(per_d, 0.5)))
    for dd in DELTAS:
        if dd in nat and nat[dd]["n"]:
            print("native d%g: ours %.3f dera %.3f (n=%d)"
                  % (dd, nat[dd]["u"] / nat[dd]["n"],
                     nat[dd]["d"] / nat[dd]["n"], nat[dd]["n"]))


if __name__ == "__main__":
    main()
