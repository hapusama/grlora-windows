# -*- coding: utf-8 -*-
"""M2c 复杂度记账（as-implemented 每窗复乘，与 m2_flop/m2b_flop 同口径）。

dep4 = 提名（pre×FFT8192）+ σ̂²（全窗行×远子格）+ K×验证
（A 组 nA×3×(l×nd) + B 组 (2l+l/4)×15×nd + FFT(256)）。
→ m2c_flop.json
"""
import json
import os

import numpy as np

NF = 4096
ND = 21           # 验证 δ 格
NA = 3            # aoff bank
NOS = 5           # osfd bank
FFT8192 = 8192 * 13 / 2.0     # radix-2 复乘
FFT256 = 256 * 8 / 2.0
N_SUB = 150       # 远子格掩蔽后平均格数（实测 116~190 取中值）


def count(pre, k_nom):
    nA_rows = pre + 2
    n_all = pre + 4
    nom = pre * FFT8192 + pre * 8192            # FFT + |X|² 累加
    sig2 = n_all * NF * N_SUB
    per_nom_A = nA_rows * NA * NF * ND
    per_nom_B = (2 * NF + NF // 4) * NA * NOS * ND
    per_nom_fft = (NA + NA * NOS) * ND * FFT256   # 行轴 FFT（逐组合×δ 格）
    per_nom_base = (n_all + 3) * NF             # 基带/相位乘
    ver = k_nom * (per_nom_A + per_nom_B + per_nom_fft + per_nom_base)
    return dict(nominat=round(nom / 1e6, 2), sigma2=round(sig2 / 1e6, 2),
                verify_per_nom=round((per_nom_A + per_nom_B
                                      + per_nom_fft + per_nom_base) / 1e6, 2),
                total_M=round((nom + sig2 + ver) / 1e6, 1), K=k_nom)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for pre in (8, 16, 32):
        out["P%d" % pre] = dict(dep4k8=count(pre, 8), dep4k16=count(pre, 16))
    old = json.load(open(os.path.join(here, "m2b_flop.json"),
                         encoding="utf-8"))
    out["reference_m2b"] = old if isinstance(old, dict) else \
        {"note": "see m2b_flop.json"}
    json.dump(out, open(os.path.join(here, "m2c_flop.json"), "w"),
              indent=1)
    for pre in (8, 16, 32):
        d = out["P%d" % pre]
        print("P%d: dep4-K8 %.1fM (nom %.2f + sig %.1f + %d×%.2f)"
              " | dep4-K16 %.1fM"
              % (pre, d["dep4k8"]["total_M"], d["dep4k8"]["nominat"],
                 d["dep4k8"]["sigma2"], 8, d["dep4k8"]["verify_per_nom"],
                 d["dep4k16"]["total_M"]))
    print("参照（m2b_flop）：cert 12.6-12.7M | dep3b 21-50M | dera 8.9-35.7M")


if __name__ == "__main__":
    main()
