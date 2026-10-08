# -*- coding: utf-8 -*-
"""M2 复杂度记账（任务 #5）：每检测窗复乘数（complex mults）——cert / dep-old /
dep2 / dera，公式透明计数（as-implemented）。→ m2_flop.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C

HERE = os.path.dirname(os.path.abspath(__file__))
NF, NFFT, NFIN, NC = 4096, 8192, 256, 2048
NFIN = 256
FARSUB = 256                     # _far_subgrid 列数（2048/8）


def fft_mults(n, nonzeros=None):
    if nonzeros is None:
        return int(n * np.log2(n) / 2 * 3)      # 复乘≈3 real-mult 等效口径统一为复乘计数
    return int(nonzeros * n)                    # 稀疏 DFT：nnz×n


def count(pre):
    ka = 6
    nw_conf = pre + 5 - ka
    nd = 41
    nanc, nsfd, nconj = 3, 7, 2
    sig2_cert = 12 * NF * FARSUB
    cert = dict(
        row_dtft=(pre + 5) * NF,
        sigma2=sig2_cert,
        fft=NFIN * (pre + 5),
    )
    depold = dict(
        row_fft=(pre + 5) * int(NFFT * np.log2(NFFT) / 2),
        coarse_bank=11 * pre * 2 * NC // 2 + pre * 2 * NC,   # 移位功率和
        sigma2=6 * NF * FARSUB,
        dtft_grid=(pre + 5) * nd * NF,
        fft=nd * NFIN * (pre + 5),
    )
    dep2 = dict(
        acq_row_fft=ka * int(NFFT * np.log2(NFFT) / 2),
        acq_search=NFIN * ka * 2 * NC // 2 * 2,   # einsum 256×ka×2048
        acq_refine=11 * ka * 2 * NF,
        conf_dtft=nconj * nanc * (1 + nsfd) * nw_conf * (NF + NF * nd) / (
            nanc * nsfd) if False else
        nconj * (nanc * ((pre - ka + 2) * (NF + NF * nd))
                 + nanc * nsfd * 3 * (NF + NF * nd)),
        conf_fft=nconj * nanc * nsfd * NFIN * nw_conf * nd,
        sigma2=(ka + (pre - ka + 2)) * NF * FARSUB,
    )
    dera = dict(
        row_fft=pre * int(NFFT * np.log2(NFFT) / 2),
        sigma2=pre * NF * FARSUB,
        dtft=3 * pre * NF,
        fft=3 * NFIN * pre,
    )
    return {a: {k: int(v) for k, v in d.items()} for a, d in
            dict(cert=cert, depold=depold, dep2=dep2, dera=dera).items()}


def main():
    res = {}
    for pre in (8, 16, 32):
        c = count(pre)
        res["P=%d" % pre] = {a: dict(parts=d, total=int(sum(d.values())))
                             for a, d in c.items()}
    json.dump(res, open(os.path.join(HERE, "m2_flop.json"), "w"), indent=1)
    for pre in (8, 16, 32):
        r = res["P=%d" % pre]
        print("P=%2d: cert %.1fM | dep-old %.1fM | dep2 %.1fM | dera %.1fM"
              % (pre, r["cert"]["total"] / 1e6, r["depold"]["total"] / 1e6,
                 r["dep2"]["total"] / 1e6, r["dera"]["total"] / 1e6))
    print("→ m2_flop.json")


if __name__ == "__main__":
    main()
