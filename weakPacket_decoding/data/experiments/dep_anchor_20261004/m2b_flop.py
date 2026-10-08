# -*- coding: utf-8 -*-
"""M2b 复杂度记账：每检测窗复乘数——dep3a / dep3b（vs M2 的 dep2 表，同口径）。

dep3 增量：双段采集（Kp 前导 + 2 sync 联合搜索，行数同 dep2 的 K_a）；
确认场两组结构（dep3b：A 组 3 锚组合 + B 组 21 组合（conj 冗余消除））。
→ m2b_flop.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C
import m2b_core as W

HERE = os.path.dirname(os.path.abspath(__file__))
NF, NFFT, NFIN, NC = 4096, 8192, 256, 2048
FARSUB = 256


def count(pre):
    kp = W.ACQ_PRE[pre]
    ka = kp + 2                                   # E 行数（前导+sync）
    L = W.confirm_layout(pre)
    npre = len(L["pre_sel"])
    nsfd = len(L["sfd_sel"])
    nd = len(W.DGRID)
    nanc, nsfd_b = 3, 7
    # ---- 采集（双段）----
    acq = dict(
        row_fft=ka * int(NFFT * np.log2(NFFT) / 2),
        acq_search=7 * NFIN * ka * NC,            # 7 walk bank × (256,ka)@(ka,2048)
        acq_refine=11 * ka * 2 * NF,
        sigma2=ka * NF * FARSUB,
    )
    # ---- 确认 dep3a：A 组（3 锚）+ B 组（2 conj × 3 锚 × 7 sfd）----
    conf_dtft_a = nanc * npre * (NF + NF * nd)
    conf_dtft_b = 2 * nanc * nsfd_b * nsfd * (NF + NF * nd)
    conf_fft_a = nanc * NFIN * npre * nd
    conf_fft_b = 2 * nanc * nsfd_b * NFIN * nsfd * nd
    dep3a = dict(conf_dtft_pre=conf_dtft_a, conf_dtft_sfd=conf_dtft_b,
                 conf_fft_pre=conf_fft_a, conf_fft_sfd=conf_fft_b,
                 sigma2=(npre + nsfd) * NF * FARSUB)
    # ---- dep3b：conj 冗余消除 ⇒ B 组 3×7；另有 κs 子 FFT 同 B 组 ----
    dep3b = dict(conf_dtft_pre=conf_dtft_a,
                 conf_dtft_sfd=nanc * nsfd_b * nsfd * (NF + NF * nd),
                 conf_fft_pre=conf_fft_a,
                 conf_fft_sfd=nanc * nsfd_b * NFIN * nsfd * nd,
                 sigma2=(npre + nsfd) * NF * FARSUB)
    out = dict(acq=acq)
    out["dep3a"] = {**dep3a, "acq_total": sum(acq.values())}
    out["dep3b"] = {**dep3b, "acq_total": sum(acq.values())}
    return {a: {k: int(v) for k, v in d.items()} for a, d in out.items()}


def main():
    res = {}
    for pre in (8, 16, 32):
        c = count(pre)
        res["P=%d" % pre] = {
            a: dict(parts=d,
                    total=int(sum(v for k, v in d.items()
                                  if k != "acq_total")))
            for a, d in c.items()}
    json.dump(res, open(os.path.join(HERE, "m2b_flop.json"), "w"), indent=1)
    for pre in (8, 16, 32):
        r = res["P=%d" % pre]
        print("P=%2d: dep3a %.1fM | dep3b %.1fM（dep2 46.5-107M，dera 8.9-35.7M）"
              % (pre, r["dep3a"]["total"] / 1e6, r["dep3b"]["total"] / 1e6))
    print("→ m2b_flop.json")


if __name__ == "__main__":
    main()
