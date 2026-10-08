# -*- coding: utf-8 -*-
"""E2 单元测试（击杀开关 0）：合成 SF10 帧 + det 相位模型的精确性检验。

生成模型（= 探针1-5 在 OTA native 上实证的生成律，域=混叠 N 域）：
  窗内信号 w[m] = A·e^{j(Φ_i + 2πτ·ν_i/N)}·upchirp(v_i)[m]·e^{j2π·4κ·m/NF}
  ν_i = (v_i + κ) mod N；Φ_i = Wiener 游走 σ_w=0.08（帧初相随机）。
物理位移变体（τ∈{0,1,2} 整数）：符号提前 τ·OS 个 NF 样本放置（窗口晚 τ），
  det 增补 πτ(τ−1)/N（wrap 型 TX 推导项），音位自动移到 v+κ+τ。
条件化真值 (v,τ,κ) 提取 Φ̂_i = ∠Z(ν_i) − det；判据：增量 σ/√2（等效游走
σ）<0.1 rad（注入 σ_w=0.08 ⇒ 期望 ≈0.08 + 管线残差）。
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C
from weak_decoder.chirp import build_upchirp

OUT = os.path.join(C.ROOT, "e2_unit_results.json")


def synth_frame(ds, P, psym, vals_pre, vals_hdr, vals_pay, kap, tau, sigma_w,
                seed, mode="phase", snr_db=None):
    """合成 NF 域帧段（槽位与 e1_common.make_slots 同构）。

    mode="phase"：直接相位注入 e^{j2πτν/N}（κ 与 τ 独立，OTA 实证生成律）。
    mode="delay"：符号整体位移 τ·OS 样本（物理整数时延；音位 v+κ+τ）。
    返回 (seg complex64, phi_true[(P+2+8+psym),], nu_true_measured[])。
    """
    rng = np.random.default_rng(seed)
    n, nf, os_ = ds.n, ds.nf, ds.os
    seg = np.zeros((P + 5 + 8 + psym + 2) * nf, dtype=np.complex128)
    phi = float(rng.uniform(-np.pi, np.pi))
    phis = []
    slots = [(1.0 + j - 0.25, int(v)) for j, v in enumerate(vals_pre)]
    slots += [(P + 0.75, 23), (P + 1.75, 31)]
    slots += [(P + 5.0 + j, int(v)) for j, v in enumerate(vals_hdr)]
    slots += [(P + 13.0 + k, int(v)) for k, v in enumerate(vals_pay)]
    up_cache = {}
    kramp = np.exp(2j * np.pi * (4.0 * kap) * np.arange(nf) / nf)
    for si, (off, v) in enumerate(slots):
        m = int(round(off * nf))
        if mode == "delay":
            m -= int(round(tau * os_))
        if v not in up_cache:
            up_cache[v] = build_upchirp(ds.sf, v % n, os_).astype(np.complex128)
        phi = phi + (float(rng.normal(0, sigma_w)) if si else 0.0)
        nu = (v + kap + (tau if mode == "delay" else 0.0)) % n
        w = up_cache[v] * kramp
        if mode == "phase":
            w = w * np.exp(1j * (phi + 2 * np.pi * tau * nu / n))
        else:
            w = w * np.exp(1j * phi)
        seg[m:m + nf] += w
        phis.append(phi)
    return seg.astype(np.complex64), np.array(phis)


def main():
    ds = C.DS(10)
    n = ds.n
    rng = np.random.default_rng(7)
    results = []
    for kap in (0.0, 0.2, 0.4):
        for tau in (0.0, 0.2, 0.4, 1.0, 2.0):
            for mode in ("phase", "delay"):
                if mode == "delay" and tau != int(tau):
                    continue
                P, psym = 8, 24
                vals_pre = [n - 1] * P
                vals_hdr = list(rng.integers(0, n, 8))
                vals_pay = list(rng.integers(0, n, psym))
                seg, phis = synth_frame(
                    ds, P, psym, vals_pre, vals_hdr, vals_pay, kap, tau, 0.08,
                    seed=1000 + int(kap * 10) * 7 + int(tau * 10), mode=mode)
                # 测量位置：pre(P) + sync(2) + pay(psym)；真值 Φ 对应下标
                offs = [1.0 + j - 0.25 for j in range(P)] + [P + 0.75, P + 1.75] \
                    + [P + 13.0 + k for k in range(psym)]
                nus = [(n - 1 + kap) % n] * P + [(23 + kap) % n, (31 + kap) % n] \
                    + [(v + kap) % n for v in vals_pay]
                if mode == "delay":
                    nus = [(x + tau) % n for x in nus]
                phi_idx = list(range(P)) + [P, P + 1] + list(range(P + 10, P + 10 + psym))
                ths, resid = [], []
                for off, nu, pi in zip(offs, nus, phi_idx):
                    m = C.extract_raw(seg, off, ds, ds.ref_dn, c_nom=0)
                    z = np.dot(m["dr"], np.exp(-2j * np.pi * (nu % n)
                                               * np.arange(n) / n))
                    det = 2 * np.pi * tau * nu / n + \
                        (np.pi * tau * (tau - 1) / n if mode == "delay" else 0.0)
                    ths.append(np.angle(z) - det)
                    resid.append(C.wrap(ths[-1] - phis[pi]))
                ths = np.array(ths)
                resid = np.array(resid)
                inc = C.wrap(np.diff(ths))
                inc_pay = inc[P + 1:]                     # payload 段增量
                results.append(dict(
                    mode=mode, kappa=kap, tau=tau,
                    inc_sigma_all=float(np.std(inc)),
                    inc_sigma_pay=float(np.std(inc_pay)),
                    walk_sigma_eq_pay=float(np.std(inc_pay) / np.sqrt(2)),
                    resid_sigma=float(np.std(resid)),
                    resid_max_abs=float(np.max(np.abs(resid))),
                ))
                r = results[-1]
                print("%-6s κ=%.1f τ=%4.1f  inc_all=%.4f inc_pay=%.4f "
                      "walk_eq=%.4f residσ=%.4f max=%.4f" % (
                          mode, kap, tau, r["inc_sigma_all"], r["inc_sigma_pay"],
                          r["walk_sigma_eq_pay"], r["resid_sigma"],
                          r["resid_max_abs"]), flush=True)
    ok = all(r["walk_sigma_eq_pay"] < 0.1 and r["resid_sigma"] < 0.1
             for r in results)
    print("\n击杀开关0：%s（全部等效游走 σ 与残差 σ <0.1 rad）" % ("通过" if ok else "失败"))
    json.dump(dict(results=results, kill_switch_pass=bool(ok)),
              open(OUT, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
