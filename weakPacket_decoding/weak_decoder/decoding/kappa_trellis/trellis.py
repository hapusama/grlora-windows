# -*- coding: utf-8 -*-
"""κ 格跨符号格架：发射项、前向-后验（软读出 BCJR）与 Viterbi（硬读出 TREL）。

提取自 dera_battle_20260929/battle_runner.py 的 fb_post()/viterbi_path()
与 bcjr_marginal_20260928 的同源实现（数值逐位一致）。

发射项（prominence）：λ[k,d] = log(max_s m[s,d] − median_s m[s,d])，
即该 κ 假设下最强符号 bin 超出谱中位数的突出度。转移先验：|Δd|≤radius
均匀（对应 |Δκ| ≤ 0.25·radius bin/符号 的慢漂移，覆盖 SFO 残差）。
"""
from __future__ import annotations

import numpy as np


def emission_prominence(ms: list[np.ndarray]) -> np.ndarray:
    """(K, n_cols) 发射对数似然（∝）：突出度形态。"""
    return np.array([np.log(np.maximum(m.max(axis=0) - np.median(m, axis=0), 1e-30))
                     for m in ms])


def emission_max(ms: list[np.ndarray]) -> np.ndarray:
    """消融用：不扣中位数地板的原始 max 发射。"""
    return np.array([np.log(np.maximum(m.max(axis=0), 1e-30)) for m in ms])


def _lse(x: np.ndarray) -> float:
    m = x.max()
    return float(m + np.log(np.sum(np.exp(x - m))))


def forward_backward(lam: np.ndarray, radius: int = 1) -> np.ndarray:
    """前向-后验：返回 (K, n_cols) 逐符号平滑 κ 后验（行和为 1）。"""
    K, D = lam.shape
    alpha = np.zeros((K, D))
    alpha[0] = lam[0]
    beta = np.zeros((K, D))
    for k in range(1, K):
        for d in range(D):
            lo, hi = max(0, d - radius), min(D - 1, d + radius)
            alpha[k, d] = lam[k, d] + _lse(alpha[k - 1, lo:hi + 1])
            beta[k - 1, d] = _lse(lam[k, lo:hi + 1] + beta[k, lo:hi + 1])
    post = alpha + beta
    post = np.exp(post - post.max(axis=1, keepdims=True))
    return post / post.sum(axis=1, keepdims=True)


def per_symbol_posterior(lam: np.ndarray) -> np.ndarray:
    """消融用（-ind）：无跨符号耦合的逐符号 softmax。"""
    e = np.exp(lam - lam.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def viterbi(lam: np.ndarray, radius: int = 1) -> np.ndarray:
    """Viterbi 联合最优 κ 路径：返回 (K,) 每符号列号。"""
    K, D = lam.shape
    acc = lam[0].copy()
    back = np.zeros((K, D), dtype=int)
    for k in range(1, K):
        cand = np.stack([acc[np.clip(np.arange(D) + s, 0, D - 1)]
                         for s in range(-radius, radius + 1)])
        best = np.argmax(cand, axis=0)
        back[k] = best - radius
        acc = cand[best, np.arange(D)] + lam[k]
    path = np.zeros(K, dtype=int)
    path[-1] = int(np.argmax(acc))
    for k in range(K - 1, 0, -1):
        path[k - 1] = np.clip(path[k] + back[k, path[k]], 0, D - 1)
    return path
