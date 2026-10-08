# -*- coding: utf-8 -*-
"""E2 探针（2026-10-02）：GT 条件化"真音处 DTFT 相位"在 OTA native 上是否为
Wiener 游走（σ≈0.08 rad），还是被确定性数据依赖项 det(v) 支配。

域声明（物理律定律9）：本实验全程在**混叠 N 域**（e1_common 抽取：MASK 带限
→ OS 抽取 p=0 → ×ref_dn → N 点 FFT；音在 ν mod N，DTFT 转向矢量
e^{-j2πνn/N} 对 ν 以 N 为周期，折返按 mod N 处理）。

候选 det（精确复数实现，不近似）：
  D0  纯音匹配滤波：θ_i = Φ_i（测量在真音处，Dirichlet 核相位=0）
  D1  相邻差 π 奇偶跳变律：Δθ 含 −π·Δv·(N−1)/N mod 2π
      ⇔ 逐符号 det(v) = −π·v·(N−1)/N mod 2π（任务书给的形态）
检验：条件化 GT (v_i, κ₀=前导拟合) 下取 θ_i = ∠Z_i(v_i+κ₀)，看增量 σ。
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C

OUT = os.path.join(C.ROOT, "e2_probe_results.json")


def zdtft(dr, nu, n):
    return complex(np.dot(dr, np.exp(-2j * np.pi * (nu % n) * np.arange(len(dr)) / n)))


def kappa_from_pre(pre_drs, n, span=0.7, npts=561):
    """前导联合 DTFT 扫描 κ₀（音在 C_PRE+κ）。"""
    kk = np.linspace(-span, span, npts)
    tot = np.zeros(npts)
    for dr in pre_drs:
        V = np.exp(-2j * np.pi * (((C.C_PRE % n) + kk)[:, None]) * np.arange(n)[None, :] / n) @ dr
        tot += np.abs(V) ** 2
    j = int(np.argmax(tot))
    if 0 < j < npts - 1:
        y0, y1, y2 = tot[j - 1], tot[j], tot[j + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30)
        kk0 = kk[j] + np.clip(d, -0.5, 0.5) * (kk[1] - kk[0])
    else:
        kk0 = kk[j]
    return float(kk0)


def extract_dr(seg, off_nf, ds, kind):
    m = C.extract_raw(seg, off_nf, ds, ds.ref_for(kind), c_nom=0)
    return None if m is None else m["dr"]


def main():
    frames = [json.loads(l) for l in open(os.path.join(C.ROOT, "e1_native.jsonl"),
                                          encoding="utf-8")]
    frames = [f for f in frames if f["sf"] == 10]
    per_frame = []
    for f in frames:
        src = next(s for s in C.SF10_SOURCES if s[0] == f["cap"])
        ds = C.DS(f["sf"])
        iq = np.memmap(src[1], dtype=np.complex64, mode="r")
        r = dict(header_start_sample=f["hs"], source_grlora_cfo_int=f["cfo_int"],
                 source_grlora_cfo_frac=str(f["cfo_frac"]),
                 source_grlora_payload_sto_frac=str(f["sto_frac"]))
        seg, _i0, _bo = C.align_seg(iq, r, ds, f["P"], f["psym"])
        del iq
        n = ds.n
        pre_drs = [extract_dr(seg, 1.0 + j - 0.25, ds, "pre") for j in range(f["P"])]
        syn_drs = [extract_dr(seg, f["P"] + 0.75, ds, "sync"),
                   extract_dr(seg, f["P"] + 1.75, ds, "sync")]
        pay_drs = [extract_dr(seg, f["P"] + 13.0 + k, ds, "pay")
                   for k in range(f["psym"])]
        kap = kappa_from_pre(pre_drs, n)
        # 相位测量（真音处 DTFT）
        th_pre = np.array([np.angle(zdtft(dr, C.C_PRE + kap, n)) for dr in pre_drs])
        th_syn = np.array([np.angle(zdtft(dr, cs + kap, n)) for dr, cs
                           in zip(syn_drs, C.C_SYNC)])
        th_pay = np.array([np.angle(zdtft(dr, v + kap, n)) for dr, v in zip(pay_drs, f["gt"])])
        amp_pay = np.array([abs(zdtft(dr, v + kap, n)) for dr, v in zip(pay_drs, f["gt"])])
        amp_pre = np.array([abs(zdtft(dr, C.C_PRE + kap, n)) for dr in pre_drs])
        v = np.array(f["gt"])
        dv = np.diff(v)
        d_raw = C.wrap(np.diff(th_pay))
        d_par = C.wrap(np.diff(th_pay) + np.pi * dv * (n - 1) / n)   # D1 奇偶律
        d_pre = C.wrap(np.diff(th_pre))
        # 音位自检：payload argmax − (v+κ) 应≈0（整数锁定 g_hp≈0）
        nu_err = []
        for dr, vv in zip(pay_drs, f["gt"]):
            b = int(np.argmax(np.abs(np.fft.fft(dr))))
            nu_err.append((b - vv) % n - (1 if (b - vv) % n > n / 2 else 0))
        nu_err = np.array(nu_err, dtype=float)
        per_frame.append(dict(
            gid=f["gid"], kap0=kap, P=f["P"], psym=f["psym"],
            pre_inc_sigma=float(np.std(d_pre)),
            pay_inc_sigma_raw=float(np.std(d_raw)),
            pay_inc_sigma_parity=float(np.std(d_par)),
            pay_inc_mean_abs=float(np.mean(np.abs(d_raw))),
            pay_amp_cv=float(np.std(amp_pay) / max(np.mean(amp_pay), 1e-30)),
            pay_over_pre_amp=float(np.mean(amp_pay) / max(np.mean(amp_pre), 1e-30)),
            sync_dtheta=float(C.wrap(th_syn[1] - th_syn[0])),
            th_pre_last=float(th_pre[-1]), th_syn2=float(th_syn[1]),
            th_pay0=float(th_pay[0]),
            prepay_gap=float(C.wrap(th_pay[0] - th_syn[1])),
            nu_err_mode=float(np.bincount(nu_err.astype(int)).argmax()) if len(nu_err) else None,
            nu_err_other=float(np.mean(np.abs(nu_err))),
        ))
        print("gid%3d κ=%.3f pre_inc=%.3f pay_raw=%.3f pay_par=%.3f |d|=%.2f ampCV=%.3f gap=%.2f" % (
            f["gid"], kap, np.std(d_pre), np.std(d_raw), np.std(d_par),
            np.mean(np.abs(d_raw)), per_frame[-1]["pay_amp_cv"],
            per_frame[-1]["prepay_gap"]), flush=True)

    def stat(k):
        x = np.array([p[k] for p in per_frame])
        return dict(med=float(np.median(x)), p25=float(np.percentile(x, 25)),
                    p75=float(np.percentile(x, 75)))

    summary = {k: stat(k) for k in ("pre_inc_sigma", "pay_inc_sigma_raw",
                                    "pay_inc_sigma_parity", "pay_inc_mean_abs",
                                    "pay_amp_cv", "pay_over_pre_amp", "prepay_gap",
                                    "nu_err_other")}
    json.dump(dict(summary=summary, per_frame=per_frame),
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
