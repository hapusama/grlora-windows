# -*- coding: utf-8 -*-
"""E1 残差结构探针：全帧 σ_res≈1.4 rad 是真实相位结构还是测量伪影？

D: theta_fine（精扫 ν̂）替代抛物线 ν̂ 后残差是否塌缩 → ν̂ 伪影检验
A: 分 kind 残差 σ（pre/sync/hdr/pay）→ 散布来自哪类符号
B: 残差对 c 的二次依赖（β·c 已被 M2c 吸收，看剩余 c²/c³ 结构）
C: 跨帧同 c 残差一致性（φ_H(c) 固定逐 bin 通道/TX 相位 → 可学习）：
   同 capture 内按 c 池化，frame 去均值后组间方差解释比
"""
import sys
import os
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_common import ROOT, wrap

IN = os.path.join(ROOT, "e1_native.jsonl")


def resid_set(syms, theta_key="theta", deg_c=1):
    th = np.array([s[theta_key] for s in syms])
    c = np.array([s["c"] for s in syms], dtype=float)
    i = np.arange(len(th), dtype=float)
    n = 1024.0 * (2 if len(c) and max(abs(c)) > 700 else 1)  # 粗判 SF 不重要，用 c 归一
    B = [np.ones_like(i), i, i ** 2, c / 1024.0, (c / 1024.0) ** 2, (c / 1024.0) ** 3][:2 + 1 + deg_c]
    B = np.stack(B, 1)
    x = np.zeros(B.shape[1])
    for _ in range(6):
        r = wrap(th - B @ x)
        dx, *_ = np.linalg.lstsq(B, r, rcond=None)
        x += dx
    return wrap(th - B @ x), c, np.array([s["kind"] for s in syms]), x


def main():
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    # A: 分 kind 残差（M2c 主项）
    per_kind = defaultdict(list)
    sig_fine_par = []
    for fr in frames:
        r, c, kinds, _ = resid_set(fr["syms"])
        for kk, rr in zip(kinds, r):
            per_kind[kk].append(rr)
        # D: fine 子集
        sub = [s for s in fr["syms"] if "theta_fine" in s]
        if len(sub) >= 6:
            rf, cf, kf, _ = resid_set(sub, "theta_fine")
            rp, cp, kp, _ = resid_set(sub, "theta")
            common = set(s["k"] for s in sub)
            sig_fine_par.append((float(np.std(rf)), float(np.std(rp))))
    print("== A. 分 kind 残差 σ (rad, M2c+c²+c³ 主项):")
    for kk in ("pre", "sync", "hdr", "pay"):
        v = np.array(per_kind[kk])
        print("   %-5s n=%5d  σ=%+.4f  circσ=%.4f" % (
            kk, len(v), np.std(v), np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * v)))))))
    print("\n== D. fine ν̂ vs 抛物线 ν̂（子集帧内 σ, n=%d 帧）:" % len(sig_fine_par))
    sf = np.array(sig_fine_par)
    print("   σ(fine θ) med=%.4f   σ(par θ) med=%.4f  （fine<par 帧占比 %.2f）"
          % (np.median(sf[:, 0]), np.median(sf[:, 1]), np.mean(sf[:, 0] < sf[:, 1])))

    # B: c 依赖阶数
    print("\n== B. 主项复杂度扫描（全帧 σ, n=148）:")
    for deg_c, lab in ((0, "i,i² only"), (1, "+c (M2c)"), (2, "+c²"), (3, "+c³")):
        sigs = []
        for fr in frames:
            r, *_ = resid_set(fr["syms"], deg_c=deg_c)
            sigs.append(np.std(r))
        print("   %-10s σ med=%.4f  p25=%.4f p75=%.4f" % (
            lab, np.median(sigs), np.percentile(sigs, 25), np.percentile(sigs, 75)))

    # C: 跨帧逐 c 一致性（同 capture 池化，frame 去均值）
    print("\n== C. 跨帧逐 c 残差一致性（φ_H(c) 检验）:")
    for sf_ in (10, 11):
        by_cap = defaultdict(list)
        for fr in frames:
            if fr["sf"] != sf_:
                continue
            r, c, kinds, _ = resid_set(fr["syms"])
            r = r - np.mean(np.exp(1j * r)) * 0
            r = r - np.angle(np.mean(np.exp(1j * r)))  # 去帧圆均值
            for cc, rr, kk in zip(c, r, kinds):
                by_cap[fr["cap"]].append((int(cc) % (1 << sf_), rr, kk))
        tot_var, exp_var, n_tot, n_grp = [], [], 0, 0
        for cap, lst in by_cap.items():
            bc = defaultdict(list)
            for cc, rr, kk in lst:
                bc[cc].append(rr)
            allr = np.array([x[1] for x in lst])
            tot_var.append(np.var(allr))
            # 组间方差解释比：E_c[ (mean_c)² ] / var
            mus = np.array([np.angle(np.mean(np.exp(1j * np.array(v))))for v in bc.values() if len(v) >= 3])
            cnts = np.array([len(v) for v in bc.values() if len(v) >= 3])
            if len(mus) >= 4:
                exp_var.append(np.sum(cnts * mus ** 2) / np.sum(cnts))
                n_grp += len(mus)
            n_tot += len(lst)
        print("   SF%d: 符号=%d, ≥3 样本 c 组=%d, 组间方差/总方差 = %.3f（n_cap=%d）"
              % (sf_, n_tot, n_grp,
                 np.sum(exp_var) / max(np.sum(tot_var), 1e-9), len(tot_var)))

    # 补：θ_pk（峰 bin 相位，无 ν̂ 校正）版本 σ —— 与 θ(跟踪 ν̂) 对照
    sigs_pk, sigs_tr = [], []
    for fr in frames:
        rp, *_ = resid_set(fr["syms"], "theta_pk")
        rt, *_ = resid_set(fr["syms"], "theta")
        sigs_pk.append(np.std(rp)); sigs_tr.append(np.std(rt))
    print("\n== 补. σ(θ_pk @argmax bin) med=%.4f vs σ(θ @跟踪ν̂) med=%.4f"
          % (np.median(sigs_pk), np.median(sigs_tr)))


if __name__ == "__main__":
    main()
