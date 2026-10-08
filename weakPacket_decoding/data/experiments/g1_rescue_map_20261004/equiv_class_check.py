# -*- coding: utf-8 -*-
r"""等价类真实性检验（用户警示 2026-10-04："不能预先把 CFO-STO 斜线上的点
当等价"）。查表 G1 三档 checkpoint，零新计算。

问题：net=ν−τ 等价类内部，判决层（err_mask / CRC）的一致率是多少？
分层预期：
  - err_mask 层：STO 相位周期重置 vs CFO 连续（Xhonneux 定理）⇒ 类内
    τ≠0 与 τ=0 点的符号轨迹可能不同 ⇒ 等价在轨迹层被破坏；
  - CRC 层：若类内 CRC 结果仍高度一致 ⇒ 类作为"停机/剪枝判据"有效，
    作为"计算复用单位"无效。
输出：类内唯一 err_mask 模式率、类内 CRC 一致率、τ 混合类 vs 纯 ν 类
的分层对比、随机对照组。
"""
import json
import os
import numpy as np

EXP = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\g1_rescue_map_20261004"


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


def main():
    maps = load_maps()
    rng = np.random.default_rng(0)
    pat_uniq, crc_unanim = [], []
    pat_uniq_mix, pat_uniq_pure = [], []   # τ 混合类 vs 纯单 τ 类
    rand_uniq = []
    for g in maps.values():
        if (0.0, 0.0) not in g or len(g) < 20:
            continue
        classes = {}
        for p, r in g.items():
            classes.setdefault(round(p[0] - p[1], 2), []).append((p, r))
        pts_all = list(g.values())
        for e, mem in classes.items():
            if len(mem) < 3:
                continue
            masks = {tuple(r["chains"]["trel"]["err_mask"]) for _, r in mem}
            crcs = {r["chains"]["trel"]["crc"] for _, r in mem}
            pat_uniq.append(len(masks) / len(mem))
            crc_unanim.append(len(crcs) == 1)
            taus = {p[1] for p, _ in mem}
            (pat_uniq_mix if len(taus) > 1 else pat_uniq_pure).append(len(masks) / len(mem))
        # 随机对照组：同大小随机点组
        k = 5
        for _ in range(4):
            sel = rng.choice(len(pts_all), size=k, replace=False)
            masks = {tuple(pts_all[i]["chains"]["trel"]["err_mask"]) for i in sel}
            rand_uniq.append(len(masks) / k)

    print("net 等价类真实性（n=%d 类，≥3 点）：" % len(pat_uniq))
    print("  类内唯一 err_mask 模式率: 中位 %.2f 均值 %.2f（1.0=完全不等价, <1=有等价）"
          % (np.median(pat_uniq), np.mean(pat_uniq)))
    print("  其中 τ 混合类: 中位 %.2f（n=%d） | 纯单τ类: 中位 %.2f（n=%d）"
          % (np.median(pat_uniq_mix), len(pat_uniq_mix),
             np.median(pat_uniq_pure), len(pat_uniq_pure)))
    print("  类内 CRC 全一致率: %.1f%%" % (100 * np.mean(crc_unanim)))
    print("  随机对照组模式率: 中位 %.2f（无等价参照）" % np.median(rand_uniq))


if __name__ == "__main__":
    main()
