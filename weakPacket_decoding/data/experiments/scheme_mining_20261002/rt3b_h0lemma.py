# -*- coding: utf-8 -*-
"""rt3b 任务二：A1 负定律的 H0 修正引理——闭式 + 数值实例化（对表 rt_d 0.21dB）。

引理（Gumbel 域有效自由度版）：
  设序列相干 GLRT 统计量 T = max_{s∈S} G_s²（G_s = |Σ_n ζ_{n,s_n}|，sqrt 单位）。
  条件：(C1) H0 下单候选尾为指数型（红队已证；本脚本用 3 档 FAR 分位数
  的局部斜率 s 复核）；(C2) 码约束 S_code ⊂ S_full，有效候选数比
  r_eff = N_eff^code/N_eff^full（极值相关使 N_eff ≪ |S| 名义值）。
  结论：
   (i)  thr_i(q) ≈ b + s·ln(N_i^eff·C/q)  ⇒ Δ_thr = s·ln(1/r_eff)
        （sqrt 单位，同 FAR）；
   (ii) H1 下 T_i = max(S+n0, W_i)，W_i=错候选极值，E W_i 同样平移
        s·ln(1/r_eff) ⇒ 有效门限释放 Δ_eff = Δ_thr − Δ_overlap，
        Δ_overlap := E[(G_full−G_S)⁺] − E[(G_cod−G_S)⁺] ≥ 0；
   (iii) frontier 位移 ε = −20·log10(1 − Δ_eff/marg*)，
        marg* = √thr_full + z_{0.9}·σ_H1 − b_full（z=1.2816；a,b,σ 数值测）。
  实例化：SF8 CR4/5 全实测对表；CR4/8、SF12 用 η 塌缩因子闭式外推（标注近似）。

运行：py -3.12 rt3b_h0lemma.py
"""
import numpy as np
import json
import time

M, NP, NS, K, NB = 256, 8, 30, 16, 6          # 6 blocks x 5 symbols (CR4/5)
BLK = 5
kap = np.arange(K) / K
mm = np.arange(M)
TW = np.exp(-2j * np.pi * np.outer(kap, mm) / M)
XOR = np.array([[t ^ u for u in range(256)] for t in range(256)])


def cumfold(v):
    v = int(v)
    g = v
    for sh in range(1, 8):
        g ^= v >> sh
    return g


def make_luts():
    luts = []
    for i in range(BLK):
        lut = np.zeros(256, dtype=np.int64)
        for x in range(256):
            w = 0
            for m_ in range(8):
                if (x >> m_) & 1:
                    w |= 1 << (7 - ((i - 1 - m_) % 8))
            lut[x] = (cumfold(w) + 1) & 255
        luts.append(lut)
    return luts


LUT = make_luts()


def encode54(d):
    p4 = d[..., 0] ^ d[..., 1] ^ d[..., 2] ^ d[..., 3]
    return (d @ (1 << np.array([3, 2, 1, 0]))) | p4


def gen_payload(rng, n):
    out = np.zeros((n, NS), dtype=np.int64)
    for b in range(NB):
        d = rng.integers(0, 2, size=(n, 8, 4))
        cw = encode54(d)
        xb = ((cw[:, :, None] >> np.arange(4, -1, -1)) & 1)
        for i in range(BLK):
            x = 0
            for m_ in range(8):
                x |= xb[:, m_, i].astype(np.int64) << m_
            out[:, b * BLK + i] = LUT[i][x]
    return out


def gen_noise(rng, n, sigma):
    return sigma * ((rng.standard_normal((n, NP + NS, M))
                     + 1j * rng.standard_normal((n, NP + NS, M))) / np.sqrt(2))


def xorconv(A, B):
    return (A[:, None, :] + B[:, XOR]).max(axis=2)


def stats(y, sigma, bins_true=None, J=12, dphi=np.pi / 12):
    """返回 (T1, T3c, T3cod[, S_true])。bins_true 给出时附真序列统计量."""
    n = y.shape[0]
    sc = 1.0 / (sigma * np.sqrt(M))
    T1 = np.zeros(n)
    T3c = np.zeros(n)
    T3d = np.zeros(n)
    S = np.zeros(n)
    fi = np.arange(n)[:, None]
    si = np.arange(NS)[None, :]
    for g in range(K):
        zp = np.einsum('psm,gm->psg', y[:, :NP, :], TW[g:g + 1])[:, :, 0] * sc
        T1 = np.maximum(T1, np.abs(zp.sum(1)) ** 2)
        Z = np.fft.fft(y[:, NP:, :] * TW[g], axis=-1) * sc
        mag = np.abs(Z)
        idx = np.argpartition(mag, -J, axis=-1)[..., -J:]
        Zt = np.take_along_axis(Z, idx, axis=-1)
        ph = np.angle((Zt.mean(axis=1)).sum(axis=1))
        if bins_true is not None and g == 0:
            zt = Z[fi, si, bins_true[fi, si]]
            for k in (-1, 0, 1):
                e = np.exp(-1j * (ph + k * dphi))
                S = np.maximum(S, (zt * e[:, None]).real.sum(axis=1) ** 2)
        for k in (-1, 0, 1):
            e = np.exp(-1j * (ph + k * dphi))
            proj = (Zt * e[:, None, None]).real
            T3c = np.maximum(T3c, proj.max(axis=-1).sum(axis=1) ** 2)
            gg = (Z * e[:, None, None]).real.astype(np.float32)
            tot = np.zeros(n, np.float32)
            for b in range(NB):
                h = [gg[:, b * BLK + i, LUT[i]] for i in range(BLK)]
                blk = xorconv(xorconv(h[0], h[1]), xorconv(h[2], h[3]))
                tot += (blk + h[4]).max(axis=1)
            T3d = np.maximum(T3d, tot ** 2)
    if bins_true is not None:
        return T1, T3c, T3d, S
    return T1, T3c, T3d


def run_h0(n, seed):
    rng = np.random.default_rng(seed)
    a1, a2, a3 = [], [], []
    for i in range(0, n, 32):
        c = min(32, n - i)
        w = gen_noise(rng, c, 1.0)
        t1, t2, t3 = stats(w, 1.0)
        a1.append(t1); a2.append(t2); a3.append(t3)
    return (np.concatenate(a1), np.concatenate(a2), np.concatenate(a3))


def run_h1(gamma_db, n, seed):
    rng = np.random.default_rng(seed)
    sigma = 10.0 ** (-gamma_db / 20.0)
    acc = [[], [], [], []]
    for i in range(0, n, 32):
        c = min(32, n - i)
        w = gen_noise(rng, c, sigma)
        bins = gen_payload(rng, c)
        nu = np.concatenate([np.zeros((c, NP), int), bins], axis=1)
        phi = rng.uniform(0, 2 * np.pi, c)
        y = w + np.exp(2j * np.pi * nu[..., None] * mm[None, None, :] / M
                       + 1j * phi[:, None, None])
        tt = stats(y, sigma, bins_true=bins)
        for j in range(4):
            acc[j].append(tt[j])
    return [np.concatenate(a) for a in acc]


def main():
    t0 = time.time()
    far = 1e-2
    z90 = 1.2816
    # ---- H0 ----
    h0 = run_h0(2400, 777)
    thr = {}
    for f in (3e-2, 1e-2, 3e-3):
        thr[f] = [float(np.quantile(h0[i], 1 - f)) for i in (1, 2)]  # T3c,T3cod
    q32, q12, q33 = (np.sqrt(np.array(thr[f])) for f in (3e-2, 1e-2, 3e-3))
    s_full = (q32[0] - q33[0]) / np.log(10.0)
    s_cod = (q32[1] - q33[1]) / np.log(10.0)
    d_thr = q12[0] - q12[1]
    print("H0 sqrt-thr @FAR 3e-2/1e-2/3e-3  T3c: %.2f/%.2f/%.2f  T3cod: %.2f/%.2f/%.2f"
          % (q32[0], q12[0], q33[0], q32[1], q12[1], q33[1]))
    print("Gumbel 斜率 s: full=%.3f cod=%.3f sqrt-nat ; Δ_thr(1e-2)=%.3f"
          % (s_full, s_cod, d_thr))
    ln_reff = d_thr / s_full
    eta = ln_reff / 48.0
    print("有效自由度: ln(1/r_eff)=%.2f (名义 48 bit) -> 塌缩因子 η=%.3f"
          % (ln_reff, eta), flush=True)

    # ---- H1 三点拟合 + ρ/overlap ----
    fits = {}
    rows = []
    for g_db in (-17.5, -18.0, -18.5):
        T1, Tc, Td, S = run_h1(g_db, 300, 555 + int(round((g_db + 20) * 10)))
        rc, rd, rs = np.sqrt(Tc), np.sqrt(Td), np.sqrt(S)
        rho_c = float(np.mean(rc > rs * 1.001))
        rho_d = float(np.mean(rd > rs * 1.001))
        ov_c = float(np.mean(np.clip(rc - rs, 0, None)))
        ov_d = float(np.mean(np.clip(rd - rs, 0, None)))
        rows.append((g_db, rc.mean(), rd.mean(), rc.std(), rd.std(),
                     rho_c, rho_d, ov_c, ov_d))
        print("H1 γ=%+.2f: E√T3c=%.2f E√T3cod=%.2f σ=%.2f/%.2f ρ=%.3f/%.3f "
              "E[(√T−√S)+]=%.2f/%.2f"
              % (g_db, rc.mean(), rd.mean(), rc.std(), rd.std(),
                 rho_c, rho_d, ov_c, ov_d), flush=True)
    x = np.array([10 ** (r[0] / 20.0) for r in rows])
    # 线性拟合 a*x+b
    def fitab(yv):
        A = np.vstack([x, np.ones_like(x)]).T
        a_, b_ = np.linalg.lstsq(A, yv, rcond=None)[0]
        return float(a_), float(b_)
    a_f, b_f = fitab(np.array([r[1] for r in rows]))
    a_c, b_c = fitab(np.array([r[2] for r in rows]))
    sigma_f = float(np.mean([r[3] for r in rows]))
    sigma_c = float(np.mean([r[4] for r in rows]))
    # frontier 解: a*10^{γ/20}+b−√thr−z90 σ = 0
    def frontier(a_, b_, thrsq, sig):
        ts = np.sqrt(thrsq)
        lo, hi = -30.0, -10.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if a_ * 10 ** (mid / 20.0) + b_ - ts - z90 * sig < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    f_full = frontier(a_f, b_f, thr[far][0], sigma_f)
    f_cod = frontier(a_c, b_c, thr[far][1], sigma_c)
    eps_pred = f_full - f_cod
    print("\n预测 frontier: T3c=%+.2f  T3cod=%+.2f -> recovery ε=%.2f dB "
          "(实测 rt_d: 0.21 dB)" % (f_full, f_cod, eps_pred), flush=True)
    marg = float(q12[0] + z90 * sigma_f - b_f)
    d_ov = float(np.mean([r[7] for r in rows]) - np.mean([r[8] for r in rows]))
    d_eff = d_thr - d_ov
    eps_closed = -20 * np.log10(max(1e-9, 1 - d_eff / marg))
    print("闭式 ε = −20log10(1−Δ_eff/marg*): Δ_overlap=%.2f Δ_eff=%.2f "
          "marg*=%.2f -> ε=%.2f dB" % (d_ov, d_eff, marg, eps_closed))
    print("检验: Δ_overlap/Δ_thr = %.2f (SF8 CR4/5 frontier 区)" % (d_ov / d_thr))

    # ---- 闭式实例化外推（η 塌缩近似，标注）----
    ratio = d_ov / d_thr
    print("\n== 实例化外推（η 线性塌缩上界；亚线性真实值更小）==")
    for tag, nominal_bits in (("CR4/8 (32 sym, sf_app=8 x (8,4))", 128),
                              ("SF12 CR4/5 (30 sym)", 72),
                              ("SF8 CR4/5 (30 sym, 实测锚)", 48)):
        ln_r = eta * nominal_bits
        dt = s_full * ln_r
        de = dt * (1 - ratio)
        eps = -20 * np.log10(max(1e-9, 1 - de / marg))
        print("  %-34s 名义 %3d bit -> ln(1/r_eff)=%5.2f Δ_thr=%5.2f "
              "ε≈%.2f dB" % (tag, nominal_bits, ln_r, dt, eps))
    rep = dict(thr_sqrt={str(k): v.tolist() for k, v in thr.items()},
               s_full=s_full, s_cod=s_cod, d_thr=d_thr,
               ln_reff=ln_reff, eta=eta, rows=[list(r) for r in rows],
               fit=dict(a_full=a_f, b_full=b_f, a_cod=a_c, b_cod=b_c,
                        sig_full=sigma_f, sig_cod=sigma_c),
               frontier=dict(T3c=f_full, T3cod=f_cod, eps_pred=eps_pred),
               closed=dict(d_overlap=d_ov, d_eff=d_eff, marg=marg,
                           eps=eps_closed, ratio=ratio),
               elapsed=time.time() - t0)
    with open('rt3b_h0lemma_results.json', 'w') as f:
        json.dump(rep, f, indent=1, default=float)
    print("saved rt3b_h0lemma_results.json (%.1f s)" % (time.time() - t0))


if __name__ == '__main__':
    main()
