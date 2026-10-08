# -*- coding: utf-8 -*-
"""E2 探针2：payload 相位增量 ~均匀的来源 = 残余分数 STO 的音位滑动项
det(v) = 2π(v+κ)Δτ/N mod 2π（任务书 STO 滑动项形态）还是二次项？

方法：对每帧扫描 Δτ，取 φ_i = wrap(θ_i − 2π v_i Δτ/N)，圆域线性拟合
φ ≈ a+b·i（Wiener 游走的均值斜率吸收），看残差 σ 是否塌缩到 ~0.1 rad。
对照 det 族：D0=0、D1=π 奇偶、T1=STO 线性 v、T2=二次 v²、T1+D1 组合。
同时用 header（8 符号已知值，音在 4·v_hdr）做跨段一致性检验。
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C
from e2_probe import zdtft, kappa_from_pre, extract_dr

OUT = os.path.join(C.ROOT, "e2_probe2_results.json")


def circfit_lin(phi, i, iters=6):
    B = np.stack([np.ones_like(i), i], 1)
    x = np.zeros(2)
    for _ in range(iters):
        r = C.wrap(phi - B @ x)
        dx, *_ = np.linalg.lstsq(B, r, rcond=None)
        x += dx
    r = C.wrap(phi - B @ x)
    return float(np.sqrt(np.mean(r ** 2))), x


def main():
    frames = [json.loads(l) for l in open(os.path.join(C.ROOT, "e1_native.jsonl"),
                                          encoding="utf-8")]
    frames = [f for f in frames if f["sf"] == 10]
    taus = np.linspace(-0.8, 0.8, 161)
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
        pay_drs = [extract_dr(seg, f["P"] + 13.0 + k, ds, "pay")
                   for k in range(f["psym"])]
        hdr_drs = [extract_dr(seg, f["P"] + 5.0 + j, ds, "hdr") for j in range(8)]
        kap = kappa_from_pre(pre_drs, n)
        th_pay = np.array([np.angle(zdtft(dr, v + kap, n)) for dr, v in zip(pay_drs, f["gt"])])
        th_hdr = np.array([np.angle(zdtft(dr, 4 * h + kap, n)) for dr, h in zip(hdr_drs, f["gt_hdr"])])
        v = np.array(f["gt"], dtype=float)
        vh = 4.0 * np.array(f["gt_hdr"], dtype=float)
        i = np.arange(f["psym"], dtype=float)
        ih = np.arange(8, dtype=float)
        best = None
        for tau in taus:
            sig, _ = circfit_lin(C.wrap(th_pay - 2 * np.pi * v * tau / n), i)
            if best is None or sig < best[1]:
                best = (float(tau), float(sig))
        s0, _ = circfit_lin(th_pay, i)
        sp, _ = circfit_lin(C.wrap(th_pay + np.pi * v * (n - 1) / n), i)
        # 二次 det
        sq, _ = circfit_lin(C.wrap(th_pay - np.pi * v * v / n), i)
        # header 用最佳 τ 的跨段检验（header 与 payload 相隔 8 符号，独立拟合）
        sh0, _ = circfit_lin(th_hdr, ih)
        sht, _ = circfit_lin(C.wrap(th_hdr - 2 * np.pi * vh * best[0] / n), ih)
        per_frame.append(dict(gid=f["gid"], kap0=kap,
                              sigma_d0=float(s0), sigma_parity=float(sp),
                              sigma_quad=float(sq), sigma_sto=best[1], tau_hat=best[0],
                              hdr_sigma_d0=float(sh0), hdr_sigma_sto=float(sht)))
        print("gid%3d κ=%+.3f σ: D0=%.3f parity=%.3f quad=%.3f STO=%.3f(τ̂=%+.3f) | hdr D0=%.3f STO=%.3f" % (
            f["gid"], kap, s0, sp, sq, best[1], best[0], sh0, sht), flush=True)

    def stat(k):
        x = np.array([p[k] for p in per_frame])
        return dict(med=float(np.median(x)), p25=float(np.percentile(x, 25)),
                    p75=float(np.percentile(x, 75)))

    keys = ("sigma_d0", "sigma_parity", "sigma_quad", "sigma_sto", "tau_hat",
            "hdr_sigma_d0", "hdr_sigma_sto")
    summary = {k: stat(k) for k in keys}
    # τ̂ 与 κ 的相关性（同一物理来源？）
    ta = np.array([p["tau_hat"] for p in per_frame])
    ka = np.array([p["kap0"] for p in per_frame])
    summary["corr_tau_kappa"] = float(np.corrcoef(ta, ka)[0, 1])
    json.dump(dict(summary=summary, per_frame=per_frame),
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
