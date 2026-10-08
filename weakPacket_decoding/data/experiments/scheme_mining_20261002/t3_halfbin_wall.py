# -*- coding: utf-8 -*-
"""任务三：半 bin 恒等式的逃逸口定量（信息墙的厚度）。

恒等式（κ=+0.5 vs κ=-0.5，STO 联合重参数化）：tone ν* = v+0.5 可被 (v, +0.5)
与 (v+1, -0.5) 完全相同地解释 —— 逐样点波形相同，任何接收机不可从波形区分。
唯一逃逸口：(a) 码本约束（v 或 v+1 只有一个是合法码字）；(b) 已知符号（协议）。

Part A（精确组合数学）：p = P(c+1 也是码字)（帧翻转歧义率），帧歧义 = p^M。
  Gray 映射下 v->v+1 = 单 bit 翻转，d>=2 的码 p=0（确定性击穿）。
  恒等映射下 p>0，N* = ln(0.5)/ln(p)。
Part B（波形级 MC）：SF8，真 κ=+0.5，联合 (κ,码字) ML 判 H0(κ=+0.5) vs H1(κ=-0.5)，
  M 扫描，两档 SNR（高/崖区）。对照组：无码（50% 硬墙）。
Part C（有码时 marg-vs-joint 方向，喂任务一）：κ=0.4、SF8、Hamming(8,4)/Gray，
  码约束下的逐符号边缘化(margC) vs 联合帧 ML(jmlC) 的门限差 —— 有码时间隙
  比无码时更大还是更小。
"""
import numpy as np
import time

rng = np.random.default_rng(20261002)
N = 256
T = np.arange(N)

# ---------- 码本 ----------
Gm = np.array([[1, 0, 0, 0, 1, 1, 1, 0],
               [0, 1, 0, 0, 1, 1, 0, 1],
               [0, 0, 1, 0, 1, 0, 1, 1],
               [0, 0, 0, 1, 0, 1, 1, 1]])
info = np.arange(16)
cb8 = (((info[:, None] >> np.arange(4)[::-1]) & 1) @ Gm) % 2     # [16,8]
cbi8 = [int(c) for c in (cb8 @ (1 << np.arange(7, -1, -1)))]
gray = lambda b: b ^ (b >> 1)
sym_ident = np.array(sorted(cbi8))         # 恒等映射（排序码本，索引一致）
sym_gray = np.array(sorted(gray(c) for c in cbi8))


def log_i0(x):
    x = np.abs(x)
    return np.where(x < 300.0,
                    np.log(np.i0(np.minimum(x, 300.0)) + 1e-300),
                    x - 0.5 * np.log(2 * np.pi * np.maximum(x, 300.0)))


print("== Part A: 翻转歧义率 p = P(c+1 也是码字), 帧歧义 = p^M ==")
for name, syms in (("identity", sym_ident), ("gray", sym_gray)):
    s = set(int(x) for x in syms)
    p = np.mean([((int(c) + 1) % N) in s for c in syms])
    if p > 0:
        nstar = np.log(0.5) / np.log(p)
        n95 = np.log(0.10) / np.log(p)
        print("  Hamming(8,4) %-8s: p = %d/16 = %.4f -> M*(amb<=0.5) = %.2f, "
              "M*(amb<=0.1) = %.2f" % (name, round(p * 16), p, nstar, n95))
    else:
        print("  Hamming(8,4) %-8s: p = 0  （v->v+1 = 单 bit 翻转，dmin=4 "
              "不可能落回码本 -> N*=1 确定性击穿）" % name)


# ---------- 波形级 ----------
def gen(n, M, kappa, snr_db, syms):
    ci = rng.integers(0, 16, size=(n, M))
    v = syms[ci]
    ph = rng.uniform(0, 2 * np.pi, size=(n, M))
    y = np.exp(2j * np.pi * (v[..., None] + kappa) * T / N + ph[..., None])
    sg = 10 ** (-snr_db / 10.0)
    y += np.sqrt(sg / 2) * (rng.standard_normal(y.shape)
                            + 1j * rng.standard_normal(y.shape))
    return y, ci, v


def F_of(y, kappa):
    tw = np.exp(-2j * np.pi * kappa * T / N)
    return np.fft.fft(y * tw[None, None, :], axis=-1)      # [n,M,N]


def part_B():
    print("== Part B: 波形级 κ=+0.5 vs -0.5 联合 (κ,码字) ML 判决 ==")
    ntr = 4000
    print("  %4s %8s | %18s | %18s" % ("M", "SNR",
          "P_correct (gray)", "P_correct (ident)"))
    for snr_db in (-4.0, -16.0):
        for M in (1, 2, 3, 4, 6, 8, 12):
            out = {}
            for name, syms in (("gray", sym_gray), ("ident", sym_ident)):
                y, ci, v = gen(ntr, M, 0.5, snr_db, syms)
                rho = 10 ** (snr_db / 10.0)
                Fp = F_of(y, +0.5)          # tone 网格: v+0.5
                Fm = F_of(y, -0.5)          # tone 网格: v-0.5 == (v+1)+0.5-1... 同一 F? 不:
                # F_of(y,+0.5)[.., v] 评估 ν=v+0.5; F_of(y,-0.5)[.., v'] 评估 ν=v'-0.5
                # H0 用码字符号 v 的 +0.5 档; H1 用码字符号 v' 的 -0.5 档。
                s = set(int(x) for x in syms)
                idx = np.array(sorted(s))
                lp0 = log_i0(2 * rho * np.abs(Fp[:, :, idx]))   # [n,M,|C|]
                lp1 = log_i0(2 * rho * np.abs(Fm[:, :, idx]))
                sc0 = lp0.max(axis=-1).sum(axis=-1)              # [n]
                sc1 = lp1.max(axis=-1).sum(axis=-1)
                tmask = np.isclose(sc0, sc1, rtol=1e-9, atol=0.0)
                ties = np.sum(tmask)
                pc = np.mean(np.where(tmask, 0.5, sc0 > sc1))
                out[name] = (pc, ties / ntr)
            print("  %4d %+8.1f | %.4f (tie %.3f) | %.4f (tie %.3f)"
                  % (M, snr_db, out['gray'][0], out['gray'][1],
                     out['ident'][0], out['ident'][1]))
        # 无码对照
        M = 8
        y, ci, v = gen(ntr, M, 0.5, snr_db, np.arange(N))
        rho = 10 ** (snr_db / 10.0)
        lp0 = log_i0(2 * rho * np.abs(F_of(y, +0.5)))
        lp1 = log_i0(2 * rho * np.abs(F_of(y, -0.5)))
        sc0 = lp0.max(axis=-1).sum(axis=-1)
        sc1 = lp1.max(axis=-1).sum(axis=-1)
        # F(ν=v+0.5) 与 F(ν'=v'-0.5) 在 v'=v+1 处逐点相等 -> sc0==sc1 恒成立
        print("  uncoded M=8: P_correct = %.4f, exact-tie rate = %.4f"
              % (np.mean(sc0 > sc1) + 0.5 * np.mean(sc0 == sc1),
                 np.mean(np.isclose(sc0, sc1))))


def part_C():
    print("== Part C: 有码 (Hamming(8,4)/Gray, κ=0.4) margC vs jmlC 门限 ==")
    KAP = 0.4
    grid = np.round(np.arange(-0.5, 0.5, 0.05), 2)
    idx = np.array(sorted(set(int(x) for x in sym_gray)))     # 16 码字符号
    M = 8

    def cer(kappa, snr_db, n, mode):
        tot = 0
        err = 0
        for s0 in range(0, n, 100):
            m = min(100, n - s0)
            y, ci, v = gen(m, M, KAP, snr_db, sym_gray)
            rho = 10 ** (snr_db / 10.0)
            if mode == 'margC':
                L = None
                for g in grid:
                    Fg = np.abs(F_of(y, g))[:, :, idx]
                    lg = log_i0(2 * rho * Fg)          # [m,M,16]
                    L = lg if L is None else np.logaddexp(L, lg)
                cd = np.argmax(L, axis=-1)
            else:
                scores = []
                for g in grid:
                    Fg = np.abs(F_of(y, g))[:, :, idx]
                    lg = log_i0(2 * rho * Fg)
                    scores.append(lg.max(axis=-1).sum(axis=-1))  # [m]
                gML = np.argmax(np.stack(scores), axis=0)        # [m]
                cd = np.empty((m, M), dtype=np.int64)
                for g in range(grid.size):
                    sel = np.where(gML == g)[0]
                    if len(sel) == 0:
                        continue
                    Fg = np.abs(F_of(y, grid[g]))[sel][:, :, idx]
                    cd[sel] = np.argmax(log_i0(2 * rho * Fg), axis=-1)
            err += int(np.sum(cd != ci))
            tot += m * M
        return err / tot

    def thr_of(mode, target=0.10):
        lo, hi = -26.0, -8.0
        # 粗扫
        for _ in range(5):
            mid = 0.5 * (lo + hi)
            if cer(KAP, mid, 150, mode) > target:
                lo = mid
            else:
                hi = mid
        # 细化
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            if cer(KAP, mid, 500, mode) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    tm = thr_of('margC')
    tj = thr_of('jmlC')
    print("  CER10%% threshold: margC = %+.2f dB, jmlC = %+.2f dB, gap = %.2f dB"
          % (tm, tj, tm - tj))


t0 = time.time()
part_B()
part_C()
print("elapsed %.1f s" % (time.time() - t0))
