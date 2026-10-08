# -*- coding: utf-8 -*-
"""audit_lib.py — 独立审计工具库（不依赖 scipy）。

统计原语全部自实现：卡方生存函数（正则化不完全伽马）、精确二项生存、
均匀分箱卡方 GOF、跨组齐性卡方 + Monte Carlo 精确 p 值。
不读写 checkpoint（只读），不改任何实验文件。
"""
import math

import numpy as np

PFA = 1e-2
EXP_RATE = math.log(100.0)          # 4.60517: q99 归一化后理论指数速率
THEO_Q50 = math.log(2.0) / EXP_RATE  # 0.150515
THEO_Q90 = math.log(10.0) / EXP_RATE # 0.500000
THEO_Q99 = 1.0


# ---------- 特殊函数 ----------

def _gser(a, x, itmax=500, eps=3e-9):
    """级数求 P(a,x)，x < a+1 时用。"""
    ap, s, d = a, 1.0 / a, 1.0 / a
    for _ in range(itmax):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * eps:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a, x, itmax=500, eps=3e-9):
    """连分式求 Q(a,x)，x >= a+1 时用。"""
    tiny = 1e-300
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, itmax):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc_upper(a, x):
    """Q(a,x) = 正则化上不完全伽马。"""
    if x < 0 or a <= 0:
        raise ValueError
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def chi2_sf(x, df):
    """卡方生存函数 P(X > x)。"""
    if x <= 0:
        return 1.0
    return gammainc_upper(df / 2.0, x / 2.0)


def binom_sf(k, n, p):
    """精确二项生存 P(X >= k)，n 大时用正态近似兜底。"""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if n > 10000:  # 罕见路径；这里 n ~ 数千，comb 可承受
        pass
    # log 空间累加避免大数问题
    logp, logq = math.log(p), math.log1p(-p)
    acc, sf = 0.0, 0.0
    for i in range(k, n + 1):
        logw = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + i * logp + (n - i) * logq)
        acc += math.exp(logw)
        if acc > 1e12:  # 防溢出，概率截断
            return 1.0
    sf = min(acc, 1.0)
    return sf


# ---------- 审计原语 ----------

def q99(x):
    return float(np.quantile(np.asarray(x, dtype=float), 1 - PFA))


def gof_exponential(norm_scores, nbins=20):
    """归一化分数 vs Exp(rate=4.605) 的等概率分箱卡方 GOF。

    返回 dict(chi2, df, p, bins)。注意：归一化用了同一样本的经验 99% 分位，
    相当于拟合了 1 个尺度参数，严格 df 应再减 1（此处不减，取保守上限）。
    """
    x = np.asarray(norm_scores, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    n = x.size
    edges = [-math.log1p(-k / nbins) / EXP_RATE for k in range(1, nbins)]
    idx = np.searchsorted(edges, x, side="right")
    obs = np.bincount(idx, minlength=nbins).astype(float)
    exp_ct = n / nbins
    chi2 = float(((obs - exp_ct) ** 2 / exp_ct).sum())
    df = nbins - 1
    return dict(chi2=chi2, df=df, p=chi2_sf(chi2, df), n=int(n),
                obs=obs.astype(int).tolist(), exp=exp_ct)


def homog_chisq(counts, totals):
    """跨组齐性：2xK 表（超阈/未超 x 组），条件于总超阈数的 MC 精确 p。

    counts: 各组超阈值数；totals: 各组总数。
    返回 dict(chi2, df, p asympt, p_mc, rates)。
    """
    K = len(counts)
    N = float(sum(totals))
    Ktot = float(sum(counts))
    p_pool = Ktot / N
    exp_k = [p_pool * t for t in totals]
    chi2 = 0.0
    for k, t, e in zip(counts, totals, exp_k):
        chi2 += (k - e) ** 2 / e + ((t - k) - (t - e)) ** 2 / (t - e)
    df = K - 1
    # MC：条件于 Ktot，按各组 n 占比多项分布
    rng = np.random.default_rng(20261003)
    nsim = 20000
    probs = np.array(totals, dtype=float) / N
    sims = rng.multinomial(int(Ktot), probs, size=nsim).astype(float)
    ek = np.array(exp_k, dtype=float)
    tt = np.array(totals, dtype=float)[None, :]
    kk = sims
    chi2_sim = ((kk - ek) ** 2 / ek).sum(1) \
        + (((tt - kk) - (tt - ek)) ** 2 / (tt - ek)).sum(1)
    p_mc = float((chi2_sim >= chi2 - 1e-12).mean())
    return dict(chi2=float(chi2), df=df, p_asym=chi2_sf(chi2, df),
                p_mc=p_mc, rates=[c / t for c, t in zip(counts, totals)],
                exp_k=exp_k)


def ols(x, y):
    """普通最小二乘，返回 (slope, intercept, r2)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom
    inter = (sy - slope * sx) / n
    yhat = slope * x + inter
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(inter), float(r2)


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    c = np.corrcoef(rx, ry)[0, 1]
    return float(c)
