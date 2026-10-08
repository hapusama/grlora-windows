# -*- coding: utf-8 -*-
"""m1_unit.py — OURS-Dir 击杀开关单元测试（无噪合成）。

(a) 能量回收：synthetic x = upchirp(s)·e^{j2πκn/NF}（OTA 真实 fold 几何：
    DDS 带边卷绕在 TX 整数格 4(N−s)，κ 只平移音频率），genie κ̂=κ 下
    OURS-Dir 的 |S(真候选)|/(A·NF) 必须 ≥ −0.05dB，κ∈{0,.2,.3,.4,.5}。
    满能量参考 = genie 分段模板 MF = A·NF（DFT-sum 口径，与定律4同标尺）。
(b) 定律4复现：同输入 DeRa port V1/V2（整数中心）+ ML 相位合并的能量
    亏损，应复现 20log10 幅度比 −1.01dB@κ=0.3 / −2.78dB@κ=0.5（±0.2dB，
    c=100）；臂④ DeraDirTaps（分数读出）应 ~0dB。
(c) 附加：φ̂/θ_p 支路相位模型自检 + OTA 干净帧 κ 标定与 native 冒烟。

输出 m1_unit_results.json。
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.chirp import build_upchirp
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator

from m1_dirichlet import DirichletDemodulator, DeraDirTapsDemodulator

SF, N, OS, NF = 10, 1024, 4, 4096
DIR = DirichletDemodulator(SF, OS, half_taps=48, mask_aware=False)
DERA = DeRaDemodulator(SF, OS)
DDT = DeraDirTapsDemodulator(SF, OS)


def synth_symbol(c, kappa, phi=0.7, amp=1.0):
    """OTA 真实几何合成：整数 fold 4(N−c)，分数 κ 平移音频率。"""
    n = np.arange(NF)
    x = build_upchirp(SF, symbol_id=c, os_factor=OS).astype(np.complex128)
    return amp * x * np.exp(2j * np.pi * kappa * n / NF + 1j * phi)


def ours_stat(x, kappa):
    """单符号 OURS-Dir 复统计 S(b)（不入 φ̂）。"""
    Xp = [np.fft.fft(x[p::OS] * DIR._ref_p[p]) for p in range(OS)]
    return DIR.symbol_S(np.stack(Xp), kappa)


def main():
    t0 = time.time()
    rep = {"sf": SF, "os": OS, "half_taps": DIR.T, "cases": []}
    kappas = [0.0, 0.2, 0.3, 0.4, 0.5]
    print("== (a)+(b) 无噪合成：能量回收 / DeRa 定律4 亏损 / 臂④ 分数读出 ==")
    hdr = "%5s %5s | %10s %8s | %10s %10s | %10s %8s"
    print(hdr % ("kappa", "c", "OURS_rec", "OURS_arg", "DeRa_loss", "DeRaFB",
                 "DT4_rec", "DT4_arg"))
    for kap in kappas:
        for c in (100, 512, 900):
            x = synth_symbol(c, kap)
            S = ours_stat(x, kap)
            b_hat = int(np.argmax(np.abs(S)))
            # 满能量参考：genie 分段模板 Σ_p Σ_j|H|² = 4·N²（A=1，DFT-sum 口径）
            rec_db = 20.0 * np.log10(abs(S[c]) / (4.0 * N * N))
            # DeRa port V1/V2 整数中心读出 + 噪声less ML 相位
            f_proj, t_proj, _s1, _nc = DERA._project(
                x.astype(np.complex64))
            phi_opt = np.angle(t_proj[c] * np.conj(f_proj[c]))
            comb = abs(f_proj[c] + np.exp(-1j * phi_opt) * t_proj[c])
            dera_db = 20.0 * np.log10(comb / NF)
            fb = 10.0 * np.log10((abs(f_proj[c])**2 + abs(t_proj[c])**2)
                                 / NF**2)                  # 非相干对照（功率 dB）
            # 臂④ 分数读出（真 κ̂）
            nn = np.arange(NF)
            ramp = np.exp(-2j * np.pi * kap * nn / NF)
            fr = DDT._front @ (x * ramp).astype(np.complex64)
            tr = DDT._tail @ (x * ramp).astype(np.complex64)
            phi4 = np.angle(tr[c] * np.conj(fr[c]))
            comb4 = abs(fr[c] + np.exp(-1j * phi4) * tr[c])
            dt4_db = 20.0 * np.log10(comb4 / NF)
            b4 = int(np.argmax(np.abs(fr + np.exp(-1j * phi4) * tr) ** 2))
            row = dict(kappa=kap, c=c, ours_rec_db=float(rec_db),
                       ours_argmax=b_hat, dera_loss_db=float(dera_db),
                       dera_fb_powdb=float(fb), dt4_rec_db=float(dt4_db),
                       dt4_argmax=b4)
            rep["cases"].append(row)
            print("%5.1f %5d | %9.4f %8d | %9.4f %9.4f | %9.4f %8d"
                  % (kap, c, rec_db, b_hat, dera_db, fb, dt4_db, b4))

    # 击杀判定
    ok_a = all(r["ours_rec_db"] >= -0.05 for r in rep["cases"])
    law4 = {0.3: -1.01, 0.5: -2.78}
    ok_b = all(any(abs(r["kappa"] - k) < 1e-9 and r["c"] == 100
                   and abs(r["dera_loss_db"] - v) <= 0.2 for r in rep["cases"])
               for k, v in law4.items())
    rep["gate_a_pass"] = bool(ok_a)
    rep["gate_b_pass"] = bool(ok_b)
    print("\n击杀开关：(a) 能量回收≥−0.05dB = %s ; (b) 定律4复现±0.2dB = %s"
          % (ok_a, ok_b))

    # ---- (c) κ̂ 误差敏感性（能量回收 vs κ̂ 失配）----
    print("\n== (c) κ̂ 失配敏感性（c=512, κ=0.4）==")
    sens = []
    x = synth_symbol(512, 0.4)
    for kerr in (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10):
        S = ours_stat(x, 0.4 + kerr)
        sens.append(dict(kerr=kerr,
                         rec_db=float(20 * np.log10(abs(S[512]) / (4.0 * N * N))),
                         argmax=int(np.argmax(np.abs(S)))))
        print("  κ̂ err %+0.2f : rec %+.3f dB, argmax %d"
              % (kerr, sens[-1]["rec_db"], sens[-1]["argmax"]))
    rep["kappa_sens"] = sens

    rep["elapsed_s"] = time.time() - t0
    with open("m1_unit_results.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print("saved m1_unit_results.json (%.1fs)" % rep["elapsed_s"])
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
