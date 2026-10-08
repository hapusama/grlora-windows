# -*- coding: utf-8 -*-
"""E1 轨迹模型拟合（native）：经验滑动定律 + 增量统计 + 幅度 CV → e1_track_results.json。

模型族（θ 定义 = 分数 bin ν̂ 处 DTFT 相位，ν̂ = 抛物线；精扫 θ_fine 用于交叉核验）：
  M0: ψ                         M1: ψ+ω·i
  M2: ψ+ω·i+α·i²                M2c: ψ+ω·i+α·i²+β·c_i     M3: ψ+ω·i+α·i²+γ·i³+β·c_i
拟合 = 卷绕残差 Gauss-Newton（圆域 LS），R² = 1 − σ_res²/σ_raw²（σ_raw 对圆均值）。
另报：ε_θ（相位推进等效 bin/符，帧首/中/末）与 ε_ν（ν̂ 滑动 bin/符）的解耦比；
增量 d_i = wrap(θ_{i+1}−θ_i−主项) 的 σ 与 lag-1 自相关（Wiener/OU/白 判据）；
前导段单独线性拟合（与 A3 可比）。
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_common import ROOT, wrap

IN = os.path.join(ROOT, "e1_native.jsonl")
OUT = os.path.join(ROOT, "e1_track_results.json")


def gn_fit(th, basis, iters=4):
    """圆域 LS：min Σ wrap(θ − B·x)²。返回 (x, σ_res, conv, res)。"""
    B = np.asarray(basis, dtype=float)
    x, *_ = np.linalg.lstsq(B, np.unwrap(th - np.median(th)), rcond=None)
    conv = False
    for _ in range(iters):
        r = wrap(th - B @ x)
        dx, *_ = np.linalg.lstsq(B, r, rcond=None)
        x = x + dx
        r2 = wrap(th - B @ x)
        if np.max(np.abs(dx)) < 1e-10:
            conv = True
            break
    res = wrap(th - B @ x)
    return x, float(np.sqrt(np.mean(res ** 2))), conv, res


def fit_frame(syms, n_bins):
    fit_syms = [s for s in syms]
    th = np.array([s["theta"] for s in fit_syms])
    c = np.array([s["c"] for s in fit_syms], dtype=float)
    i = np.arange(len(th), dtype=float)
    th_cm = th - np.angle(np.mean(np.exp(1j * th)))
    sigma_raw = float(np.sqrt(np.mean(wrap(th - np.angle(np.mean(np.exp(1j * th)))) ** 2)))
    out = {"n_sym": len(th)}
    models = {}
    for name, basis in (
            ("M0", np.stack([np.ones_like(i)], 1)),
            ("M1", np.stack([np.ones_like(i), i], 1)),
            ("M2", np.stack([np.ones_like(i), i, i ** 2], 1)),
            ("M2c", np.stack([np.ones_like(i), i, i ** 2, c / n_bins], 1)),
            ("M3", np.stack([np.ones_like(i), i, i ** 2, i ** 3, c / n_bins], 1))):
        x, sig, conv, res = gn_fit(th, basis)
        models[name] = dict(coef=[float(v) for v in x], sigma=sig,
                            r2=float(1 - sig ** 2 / max(sigma_raw ** 2, 1e-12)),
                            conv=bool(conv))
    out["models"] = models
    # 模型选择：σ 不再下降 >3% 的最简模型
    order = ["M0", "M1", "M2", "M2c", "M3"]
    sel = "M0"
    for a, b in zip(order, order[1:]):
        if models[b]["sigma"] < 0.97 * models[a]["sigma"]:
            sel = b
    out["sel"] = sel
    x = np.array(models[sel]["coef"])
    out["omega"] = float(x[1]) if len(x) > 1 else 0.0
    out["alpha"] = float(x[2]) if sel in ("M2", "M2c", "M3") else 0.0
    out["beta"] = float(x[3]) if sel in ("M2c", "M3") else 0.0
    out["kappa_ph"] = float(out["beta"] * n_bins / (2 * np.pi))   # β = 2πκ/N → κ
    out["sigma_res"] = models[sel]["sigma"]
    out["r2"] = models[sel]["r2"]
    # 等效频率偏移（bin/符）：ε(i) = (ω + 2α·i + 3γ·i²)/(2π)
    ns = len(th)
    eps = []
    for j, if_ in enumerate((0.0, (ns - 1) / 2.0, ns - 1.0)):
        e = out["omega"] / (2 * np.pi)
        if sel in ("M2", "M2c", "M3"):
            e += out["alpha"] * if_ / np.pi
        if sel == "M3":
            e += models[sel]["coef"][3] * if_ ** 2 * 3 / (2 * np.pi)
        eps.append(float(e))
    out["eps_bin_start"], out["eps_bin_mid"], out["eps_bin_end"] = eps
    # 前导段单独线性（A3 可比）
    pre_n = sum(1 for s in fit_syms if s["kind"] == "pre")
    if pre_n >= 4:
        thp = th[:pre_n]
        xp, sigp, _, _ = gn_fit(thp, np.stack([np.ones(pre_n), np.arange(pre_n, dtype=float)], 1))
        out["pre_eps"] = float(xp[1] / (2 * np.pi))
        out["pre_sigma"] = float(sigp)
    # ν̂ 滑动（payload 段，抛物线 κ̂ 相对 c）
    pay = [(k, s["kappa"]) for k, s in enumerate(fit_syms) if s["kind"] == "pay"]
    if len(pay) >= 8:
        kp = np.array([p[1] for p in pay])
        ip = np.arange(len(kp), dtype=float)
        cf = np.polyfit(ip, kp, 1)
        out["nu_slide_pay"] = float(cf[0])          # bin/符
        out["nu_scatter_pay"] = float(np.std(kp - np.polyval(cf, ip)))
    # 增量统计（去主项 M2c 后）
    if sel in ("M2c", "M3"):
        main = models[sel]["coef"][1] + 2 * models[sel]["coef"][2] * (i[:-1] + 0.5) \
            + out["beta"] * np.diff(c) / n_bins
        if sel == "M3":
            main += 3 * models[sel]["coef"][3] * (i[:-1] + 0.5) ** 2
    elif sel == "M2":
        main = models[sel]["coef"][1] + 2 * models[sel]["coef"][2] * (i[:-1] + 0.5)
    else:
        main = np.full(max(len(th) - 1, 0), out["omega"])
    d = wrap(np.diff(th) - main)
    out["d_sigma"] = float(np.std(d))
    out["d_lag1"] = float(np.corrcoef(d[:-1], d[1:])[0, 1]) if len(d) > 2 else None
    out["d_mean_abs"] = float(np.mean(np.abs(d)))
    # 幅度 CV（ok 符号）
    amp = np.array([s["amp"] for s in fit_syms])
    out["amp_cv"] = float(amp.std() / amp.mean())
    out["amp_cv_pay"] = float(
        (lambda a: a.std() / a.mean())(np.array([s["amp"] for s in fit_syms if s["kind"] == "pay"])))
    # 精扫 vs 抛物线 θ 的偏差（测量器自洽性）
    tf = [(s["theta_fine"], s["theta"]) for s in syms if "theta_fine" in s]
    if len(tf) >= 4:
        dd = wrap(np.array([a - b for a, b in tf]))
        out["theta_fine_minus_par_med"] = float(np.median(dd))
        out["theta_fine_minus_par_iqr"] = float(np.percentile(dd, 75) - np.percentile(dd, 25))
    # 半窗频率斜坡（符号内滑动的直接观测）
    rh = [s["nu_h2"] - s["nu_h1"] for s in syms if "nu_h1" in s]
    if len(rh) >= 4:
        out["half_ramp_med"] = float(np.median(rh))
        out["half_ramp_p10"], out["half_ramp_p90"] = [float(v) for v in np.percentile(rh, [10, 90])]
    return out


def main():
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    per_frame = []
    for fr in frames:
        r = fit_frame(fr["syms"], 1 << fr["sf"])
        r["gid"] = fr["gid"]; r["cap"] = fr["cap"]; r["sf"] = fr["sf"]
        r["P"] = fr["P"]; r["psym"] = fr["psym"]; r["snr_native"] = fr["snr_native"]
        per_frame.append(r)

    def agg(key, sel=None):
        v = np.array([r[key] for r in per_frame
                      if r.get(key) is not None and r[key] == r[key]
                      and (sel is None or sel(r))])
        if len(v) == 0:
            return None
        return dict(n=len(v), med=float(np.median(v)),
                    p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                    p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))

    keys = ["omega", "alpha", "beta", "kappa_ph", "sigma_res", "r2",
            "eps_bin_start", "eps_bin_mid", "eps_bin_end", "pre_eps", "pre_sigma",
            "nu_slide_pay", "nu_scatter_pay", "d_sigma", "d_lag1", "amp_cv", "amp_cv_pay",
            "theta_fine_minus_par_med", "half_ramp_med"]
    summary = {k: agg(k) for k in keys}
    summary["d_lag1"] = agg("d_lag1")
    sel_counts = {}
    for r in per_frame:
        sel_counts[r["sel"]] = sel_counts.get(r["sel"], 0) + 1
    by_sf = {}
    for sf in (10, 11):
        sub = [r for r in per_frame if r["sf"] == sf]
        by_sf[str(sf)] = {k: agg(k, sel=lambda r, sf=sf: r["sf"] == sf) for k in keys}
        by_sf[str(sf)]["n"] = len(sub)
        by_sf[str(sf)]["sel_counts"] = {}
        for r in sub:
            by_sf[str(sf)]["sel_counts"][r["sel"]] = by_sf[str(sf)]["sel_counts"].get(r["sel"], 0) + 1
    res = dict(n_frames=len(per_frame), sel_counts=sel_counts, summary=summary,
               by_sf=by_sf, per_frame=per_frame)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f)
    print("== 模型选择计数: %s" % sel_counts)
    for k in keys:
        s = summary.get(k)
        if s:
            print("%-26s med=%+.5f  p25=%+.5f p75=%+.5f  (n=%d)"
                  % (k, s["med"], s["p25"], s["p75"], s["n"]))
    for sf in ("10", "11"):
        print("-- SF%s (n=%d, sel=%s)" % (sf, by_sf[sf]["n"], by_sf[sf]["sel_counts"]))
        for k in ("eps_bin_mid", "alpha", "nu_slide_pay", "sigma_res", "d_sigma", "d_lag1",
                  "amp_cv_pay", "half_ramp_med", "r2"):
            s = by_sf[sf].get(k)
            if s:
                print("   %-18s med=%+.5f p25=%+.5f p75=%+.5f" % (k, s["med"], s["p25"], s["p75"]))


if __name__ == "__main__":
    main()
