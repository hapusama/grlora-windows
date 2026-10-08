# -*- coding: utf-8 -*-
"""E2 探针3：φ_H(ν) 逐 bin 固定相位响应的跨帧可学习性（LOO 检验）。

E1 probe C 发现残差按 c 池化解释 77-83% 方差 → 固定 φ_H(c)。检验：
  1. 帧内去趋势（圆域 a+b·i）后残差 r_i 与音位 ν_i=v_i+κ₀ 池化；
  2. φ_H 用连续 ν 轴分箱（bin=16）+ 圆滑估计，per-capture 与 global 两种；
  3. Leave-one-frame-out：用其他帧估的 φ_H 修正当帧 → 增量 σ 是否塌缩
     到 Wiener 水平（~0.12 = √2×0.083）。
塌缩 → 机制可行，det(v)=φ_H(v+κ) 查表项进 HMM；不塌缩 → 如实证伪。
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C
from e2_probe import zdtft, kappa_from_pre, extract_dr

OUT = os.path.join(C.ROOT, "e2_probe3_results.json")
BINW = 16          # φ_H 分箱宽度（bin）
SMOOTH = 2         # 箱间圆滑半径（箱）


def circfit_lin(phi, i, iters=6):
    B = np.stack([np.ones_like(i), i], 1)
    x = np.zeros(2)
    for _ in range(iters):
        r = C.wrap(phi - B @ x)
        dx, *_ = np.linalg.lstsq(B, r, rcond=None)
        x += dx
    return C.wrap(phi - B @ x)


def phiH_est(nu, r, n):
    """连续 ν 轴分箱圆均值 + 邻箱平滑。返回 (nb,) 数组与箱中心。"""
    nb = n // BINW
    cnt = np.zeros(nb)
    acc = np.zeros(nb, dtype=complex)
    idx = np.clip((nu % n).astype(int) // BINW, 0, nb - 1)
    for j, rr in zip(idx, r):
        cnt[j] += 1
        acc[j] += np.exp(1j * rr)
    ph = np.full(nb, np.nan)
    ok = cnt > 0
    ph[ok] = np.angle(acc[ok])
    # 圆滑（缺失箱跳过）
    ext = np.concatenate([ph[-SMOOTH:], ph, ph[:SMOOTH]])
    okx = np.concatenate([ok[-SMOOTH:], ok, ok[:SMOOTH]])
    out = np.full(nb, np.nan)
    for j in range(nb):
        w = okx[j:j + 2 * SMOOTH + 1]
        v = ext[j:j + 2 * SMOOTH + 1]
        if w.any() and not np.isnan(v[w]).any():
            out[j] = np.angle(np.sum(np.exp(1j * v[w])))
        elif ok[j]:
            out[j] = ph[j]
    return out, np.arange(nb) * BINW + BINW / 2


def apply_phiH(nu, phiH, n):
    nb = n // BINW
    j = np.clip((nu % n).astype(int) // BINW, 0, nb - 1)
    v = phiH[j]
    return np.where(np.isnan(v), 0.0, v)


def main():
    frames = [json.loads(l) for l in open(os.path.join(C.ROOT, "e1_native.jsonl"),
                                          encoding="utf-8")]
    frames = [f for f in frames if f["sf"] == 10]
    data = {}      # gid -> dict(nu, th, inc_raw, ...)
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
        kap = kappa_from_pre(pre_drs, n)
        th = np.array([np.angle(zdtft(dr, v + kap, n)) for dr, v in zip(pay_drs, f["gt"])])
        nu = (np.array(f["gt"], dtype=float) + kap) % n
        i = np.arange(f["psym"], dtype=float)
        res = circfit_lin(th, i)
        data[f["gid"]] = dict(cap=f["cap"], nu=nu.tolist(), th=th.tolist(),
                              res=res.tolist(), kap=kap)
    gids = sorted(data)
    caps = sorted(set(data[g]["cap"] for g in gids))

    # ---- per-capture LOO ----
    loo_rows = []
    for g in gids:
        d = data[g]
        th = np.array(d["th"])
        n = 1024
        inc_raw = float(np.std(C.wrap(np.diff(th))))
        train_g = [h for h in gids if h != g and data[h]["cap"] == d["cap"]]
        train_all = [h for h in gids if h != g]
        row = dict(gid=g, cap=d["cap"], inc_raw=inc_raw)
        for lab, tr in (("loo_cap", train_g), ("loo_global", train_all)):
            if not tr:
                continue
            nu_tr = np.concatenate([data[h]["nu"] for h in tr])
            r_tr = np.concatenate([data[h]["res"] for h in tr])
            phiH, _ = phiH_est(nu_tr, r_tr, n)
            thc = C.wrap(th - apply_phiH(np.array(d["nu"]), phiH, n))
            row["inc_%s" % lab] = float(np.std(C.wrap(np.diff(thc))))
            row["sig_fit_%s" % lab] = float(np.std(circfit_lin(thc, np.arange(len(thc)))))
        loo_rows.append(row)
        print("gid%3d inc_raw=%.3f  " % (g, inc_raw) + " ".join(
            "%s:%.3f" % (k[4:], v) for k, v in row.items() if k.startswith("inc_")),
            flush=True)

    def stat(vals):
        v = np.array(vals)
        return dict(med=float(np.median(v)), p25=float(np.percentile(v, 25)),
                    p75=float(np.percentile(v, 75)))

    summary = dict(
        inc_raw=stat([r["inc_raw"] for r in loo_rows]),
        inc_loo_cap=stat([r["inc_loo_cap"] for r in loo_rows if "inc_loo_cap" in r]),
        inc_loo_global=stat([r["inc_loo_global"] for r in loo_rows if "inc_loo_global" in r]),
        sig_fit_loo_cap=stat([r["sig_fit_loo_cap"] for r in loo_rows if "sig_fit_loo_cap" in r]),
        sig_fit_loo_global=stat([r["sig_fit_loo_global"] for r in loo_rows
                                 if "sig_fit_loo_global" in r]),
    )
    json.dump(dict(summary=summary, loo=loo_rows), open(OUT, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
