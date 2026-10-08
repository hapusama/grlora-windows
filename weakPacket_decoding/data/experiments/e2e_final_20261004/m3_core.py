# -*- coding: utf-8 -*-
"""M3 核心库（2026-10-04）：OURS-E2E 端到端链组装（决战版）。

链规格（brief）：
  dep2 检测（系统门限，非 GT）→ ν̂0+payload 中心走动补偿频旋 → demod 两列
  （A=DeRa demod port / B=TREL-5 = wm 掩模 5 列格 trellis κ_sel，替代旧 γ
  直喂——M1 诊断 3：γ/π=κ_CFO+ε_t/4 与抽头核存在历元差，唯一稳定估计器
  = 谱移列格+trellis）→ 零走动反映射 + Δ0 CRC 仲裁（D2 终版保守）。
并列 OURS-E2E-cert：检测换 cert 机制级（GT 锚定模板，bookkeeping 上界），
解码链与 OURS-E2E 逐字相同。DeRa 全链 = paper_dera_detector top-5 候选 →
paper_dera_demod + δ∈{0,±1,±2} CRC fallback（front_runner 原样）。

相对 D2 battle C 的链差异（全部有据，报告 §1 声明）：
  D1. 检测器 dep→dep2（m2_core：前导相干锚+拆分确认，门限 11.46/11.93/
      12.07 dB@1e-3 per-P 池化，M2 §3 主表口径）；锚质量 p50 0.07-0.16 bin
      （dep-old 0.10-0.32）。
  D2. 盲对齐 = 内容平移批量技巧：Δ 网格 stack(seg_crop[Δ:]) 一次喂
      score_dep2 批量接口（窗在 hs_loc、内容平移 Δ ≡ 窗在 hs_loc+Δ，精确）。
  D3. 频旋 ν_rot = ν̂0 + (c_pay − c_e)·δ̂（c_pay = pre+12.25+(psym−1)/2 为
      payload 中心场 idx；c_e = 采集质心）—— 全部检测器自产量，无 GT。
      把 payload 走动扫频居中：δ=0.082 时 payload 对 ν̂0 残差 +0.9~+2.5 bin
      → 居中后 ±0.8 bin（整数部分残余由 Δ0 CRC 吸收，B 列 trellis 走格）。
  D4. γ-链整体移除（D2 三关 + M1 历元差）；κ 载体 = B 列 TREL-5 列格。
  D5. CRC 仲裁预算：我方 2 列 × Δ0∈{0,±1,±2,±3}（=14）vs DeRa top-5 ×
      δ∈{0,±1,±2}（=25）——两侧各用各的候选重试原语，错误 CRC 撞过
      概率 ~2⁻¹⁶/候选，可忽略。

铁律：噪声纯 AWGN 同一实现喂两链；注入（resample_sfo）在加噪前；GT 只
来自干净原生解 + 注入后干净模板（cert/评分侧）；运行时链内无 GT。
"""
import os
import sys

import numpy as np

_WD = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
for _p in (
    _WD,
    os.path.join(_WD, "data", "experiments", "keystone_battle_20261003"),
    os.path.join(_WD, "data", "experiments", "dep_anchor_20261004"),
    os.path.join(_WD, "data", "experiments", "dera_front_battle_20260930"),
    os.path.join(_WD, "data", "experiments", "e2e_final_20261004"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import d1_core as C                      # noqa: E402
import d2_core as D                      # noqa: E402
import m2_core as M2                     # noqa: E402
import front_runner as FR               # noqa: E402
from weak_decoder.decoding.kappa_trellis import KappaTrellisDemodulator  # noqa: E402
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator  # noqa: E402

SF, N, OS, NF = 10, 1024, 4, 4096
EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- 检测门限（M2 §3 per-P 池化精确分位 @FAR=1e-3，线性域）----
DEP2_THR = {8: 10 ** (11.46 / 10.0), 16: 10 ** (11.93 / 10.0),
            32: 10 ** (12.07 / 10.0)}
CERT_THR = 10.424901486084273          # D2 battle C（闭式 1e-2 + 2D 膨胀 1.11dB）

# ---- 盲对齐筛查格（接收机设计参数；S1/S2/战例探针定，报告声明）----
# P1/P1b：门事件 start 偏置 native 全 0（帧位与跳格对齐 ⇒ 偏置恒 512 倍
# 数）；带噪时链入噪声峰把组 start 拉早（战例实测 −2560）或 top-1 落到
# 远处伪事件 ⇒ top-3 事件 × ±3584/512 筛查格（hop 量化 ⇒ 命中真偏置时
# score 为满峰，次瓣 ≤0.4×）。P2：score(Δ) 真位尖峰 vs 48 倍数次瓣
# 0.2-0.43、远端 <0.2。ν̂0 旋转精确抵消 e/4 音移（anchor(Δ)=ν0+Δ/4 实测
# 线性）⇒ 对齐误差只以 −e/NF dB 能量损耗伤 demod。筛查用 confirm-lite
# （锚 bank 单点；胜点全统计量复核门限）。
SCAN_SCREEN = np.arange(-1024, 1025, 512)   # 每事件 5 偏移（紧链距分组下
                                             # start 偏置域 ±512 + 余量）

SCAN_HOP = NF // 8
SCAN_THRESH_DB = 4.0
GATE_EVENTS = 3


# ---------------------------------------------------------------- 门扫
def _robust_start(grp, span=None):
    """组内最密窗起点：在 0.6·前导长度窗内峰数最多的起点。

    真前导峰密集（窗内 ~pre/2 个 hop 峰），链入噪声峰稀疏孤悬头部——
    最密窗起点对头部离群免疫（战例 f15：min 起点 −2560，最密窗起点 0）。
    span=None 时退回 min（无 pre 信息场景）。"""
    pos = sorted(g[0] for g in grp)
    if span is None or len(pos) < 6:
        return int(pos[0])
    pos_arr = np.asarray(pos, dtype=np.int64)
    idx = np.searchsorted(pos_arr, pos_arr + int(span), side="right")
    cnt = idx - np.arange(len(pos_arr))
    return int(pos_arr[int(np.argmax(cnt))])


def gate_scan(seg, pre=None):
    """单窗门扫（DeRa scan 同款原语；链距 2NF 同 D2，起点=最密窗）。"""
    n = len(seg)
    span = None if pre is None else int(0.6 * pre * NF)
    peaks = []
    for s in range(0, n - NF + 1, SCAN_HOP):
        X = np.fft.fft(np.asarray(seg[s:s + NF]) * C.DOWN)
        row = np.abs(np.concatenate((X[NF - N // 2:], X[:N // 2]))) ** 2
        med = float(np.median(row))
        if med <= 0:
            continue
        k = int(np.argmax(row))
        r = 10 * np.log10(row[k] / med)
        if r > SCAN_THRESH_DB:
            peaks.append((s, k, r))
    events, used = [], [False] * len(peaks)
    for i, p in enumerate(peaks):
        if used[i]:
            continue
        grp, used[i] = [p], True
        for j in range(i + 1, len(peaks)):
            if used[j]:
                continue
            q = peaks[j]
            db = min(abs(q[1] - grp[-1][1]), N - abs(q[1] - grp[-1][1]))
            if db <= 4 and 0 < q[0] - grp[-1][0] <= 2 * NF:
                grp.append(q)
                used[j] = True
        if len(grp) >= 3:
            events.append(dict(start=_robust_start(grp),
                               power=sum(10 ** (g[2] / 10) for g in grp)))
    events.sort(key=lambda e: -e["power"])
    return events[:GATE_EVENTS]


# ---------------------------------------------------------------- 盲对齐
_KAP_AXIS = np.fft.fftfreq(M2.N_FINE)


def acquire_fast(segs, hs, pre):
    """m2_core.acquire 的 BLAS 化复刻（walk-bank einsum → 广播 matmul，
    complex64；与原版数值等价，仅舍入序不同——已对拍验证）。"""
    wins = C.field_windows(hs, pre)
    B = segs.shape[0]
    ka = min(M2.ACQ_K[pre], pre)
    idx_e = np.arange(ka, dtype=float)
    c_e = float(idx_e.mean())
    n_arr = np.arange(NF)

    Xs = np.empty((B, ka, 2 * N), dtype=np.complex128)
    for j in range(ka):
        s, r, l = wins[j]
        W = segs[:, s:s + l] * r[None, :l]
        X = np.fft.fft(W, M2.NFFT, axis=1)
        Xs[:, j] = np.concatenate((X[:, M2.NFFT - N:], X[:, :N]), axis=1)

    subs = C._far_subgrid([])
    TWf = np.exp(-2j * np.pi * (subs - N)[:, None] * n_arr[None, :]
                 / (2.0 * NF))
    sig2 = np.zeros(B)
    for j in range(ka):
        s, r, l = wins[j]
        W = segs[:, s:s + l] * r[None, :l]
        sig2 += np.sum(np.abs(W @ TWf.T) ** 2, axis=1) / subs.size
    sig2 /= ka

    Ekap = np.exp(-2j * np.pi * np.outer(_KAP_AXIS, idx_e - c_e)
                  ).astype(np.complex64)
    cols = np.arange(2 * N)
    best = np.full(B, -1.0)
    k_pre = np.zeros(B, dtype=np.int64)
    d_pre = np.zeros(B)
    for dg in M2.ACQ_DGRID:
        sh = np.rint(2.0 * (idx_e - c_e) * float(dg)).astype(int)
        Xsh = np.empty((B, ka, 2 * N), dtype=np.complex64)
        for j in range(ka):
            Xsh[:, j] = Xs[:, j][:, (cols + sh[j]) % (2 * N)]
        # 单 GEMM：T(q, b·n) = Ekap(q,k) @ Xshᵀ(k, b·n)（BLAS zgemm）
        Xt = np.ascontiguousarray(Xsh.transpose(1, 0, 2).reshape(ka, -1))
        T = np.abs(Ekap @ Xt).reshape(_KAP_AXIS.size, B, 2 * N)
        flat = T.transpose(1, 0, 2).reshape(B, -1)
        kf = np.argmax(flat, axis=1)
        v = flat[np.arange(B), kf]
        upd = v > best
        best = np.where(upd, v, best)
        k_pre = np.where(upd, (kf % (2 * N)).astype(np.int64), k_pre)
        d_pre = np.where(upd, float(dg), d_pre)

    k0 = (k_pre.astype(float) - N) / 2.0
    nu0h = k0.copy()
    khat = np.zeros(B)
    Wrows = [segs[:, wins[j][0]:wins[j][0] + wins[j][2]]
             * wins[j][1][None, :wins[j][2]] for j in range(ka)]
    for poff in M2.ACQ_P0_OFF:
        p0 = k0 + poff
        Xd = np.empty((B, ka), dtype=np.complex128)
        for j in range(ka):
            pj = p0 + (idx_e[j] - c_e) * d_pre
            tw = np.exp(-2j * np.pi * pj[:, None] * n_arr[None, :NF] / NF)
            Xd[:, j] = np.einsum("bl,bl->b", Wrows[j], tw)
        F = np.abs(np.fft.fft(Xd, M2.N_FINE, axis=1))
        q = np.argmax(F, axis=1)
        v = F[np.arange(B), q] ** 2
        upd = v > best
        best = np.where(upd, v, best)
        nu0h = np.where(upd, p0, nu0h)
        khat = np.where(upd, M2.kap_of_q(q), khat)
    return dict(nu0h=nu0h, khat=khat, acq_score=best / (ka * sig2),
                c_e=c_e, ka=ka, sig2_a=sig2)
def _scan_scores(seg, hs0, pre, offs, lite=False):
    """批量 dep2 分数 over 内容平移 offs（≡ 窗平移 −offs）。

    逐偏移自带 acquire（B 行批量 einsum）：每偏移独立全精度锚——事件
    start 被链入噪声峰拉早数千样本时参考行 acquire 会被场前内容污染
    （战例 f15@−20：锚错至 359 vs 真 −24.8），锚移位技巧失效，故必须
    逐偏移。lite=True：走查 bank 7→3 格 + 锚 bank 单点（排序用近似，
    ±0.1dB 级）；胜点由调用方以全统计量复核门限。返回 (score,bd,bk,diag)。"""
    base = hs0 - int((pre + 4.25) * NF)
    lo = max(base - NF, 0)
    hi = min(hs0 + 2 * NF, len(seg))
    hs_loc = hs0 - lo
    L = hi - lo
    w = max(int(np.max(np.abs(offs))), 0)
    buf = np.zeros((len(offs), L + w), dtype=np.complex128)
    for i, off in enumerate(offs):
        s0 = lo + off
        if s0 < 0:
            continue
        chunk = np.asarray(seg[s0:hi + off], dtype=np.complex128)
        buf[i, :len(chunk)] = chunk
    saved_dg, saved_aoff, saved_acq = M2.ACQ_DGRID, M2.ANCHOR_OFF, M2.acquire
    if lite:
        M2.ACQ_DGRID = np.array([-0.082, 0.0, 0.082])
        M2.ANCHOR_OFF = np.array([0.0])
    try:
        acq = acquire_fast(buf, hs_loc, pre)
        sc, bd, bk, ph = M2.confirm(buf, hs_loc, pre, acq["nu0h"])
    finally:
        M2.ACQ_DGRID, M2.ANCHOR_OFF = saved_dg, saved_aoff
    sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)  # 边缘零窗行
    diag = dict(nu0h=acq["nu0h"], c_e=acq["c_e"])
    return sc, bd, bk, diag


def dep2_blind_detect(seg, pre, thr=None):
    """OURS-E2E 检测：门扫 top-3 事件 × ±3584/512 lite 筛查 → 胜点全
    统计量过门限。返回 dict(score, hs, pay0, nu0h, dhat, khat, c_e)/None。"""
    thr = DEP2_THR[pre] if thr is None else thr
    best = None
    for ev in gate_scan(seg, pre)[:3]:
        hs0 = ev["start"] + int((pre + 4.25) * NF)
        if hs0 + 10 * NF > len(seg) or hs0 < int((pre + 5.25) * NF):
            continue
        sc, _bd, _bk, _dg = _scan_scores(seg, hs0, pre, SCAN_SCREEN, lite=True)
        i = int(np.argmax(sc))
        if best is None or sc[i] > best[1]:
            best = (int(hs0 + SCAN_SCREEN[i]), float(sc[i]))
    if best is None or not np.isfinite(best[1]) or best[1] < thr:
        return None
    hs_b = best[0]
    sc, bd, _bk, diag = _scan_scores(seg, hs_b, pre, np.array([0]))
    if not np.isfinite(sc[0]) or sc[0] < thr:
        return None
    return dict(score=float(sc[0]), hs=int(hs_b),
                pay0=int(hs_b) + 8 * NF,
                nu0h=float(diag["nu0h"][0]), dhat=float(bd[0]),
                khat=0.0, c_e=float(diag["c_e"]))


# ---------------------------------------------------------------- 反映射判据
def demap_judge(rows, ints, f):
    """逐符整数走动反映射 + Δ0 CRC 仲裁（D2 battle C 终版保守，逐字）。"""
    am = np.argmax(rows, axis=1)
    best_ints, best_score = None, None
    for cand in (ints, ints + 1):
        c = (am - cand) % N
        mode = np.bincount(c, minlength=N).argmax()
        dev = np.minimum((c - mode) % N, (mode - c) % N)
        score = float(np.sum(dev ** 2))
        if best_score is None or score < best_score:
            best_ints, best_score = cand, score
    ints = best_ints
    for d0 in (0, 1, -1, 2, -2, 3, -3):
        rows_dm = np.stack([np.roll(rows[k], -(ints[k] + d0))
                            for k in range(rows.shape[0])])
        if FR.judge_crc(rows_dm, 1, f["gt_hdr"], f["plen"], f["cr"]):
            hard = [(int(np.argmax(r)) - 1) % N for r in rows_dm]
            ser = int(sum(int(h != g) for h, g in zip(hard, f["gt"])))
            return True, ser, d0
    return False, -1, 0


# ---------------------------------------------------------------- 旋频
def nu_rot_of(nu0_ref, dhat, ref_idx, pre, psym):
    """payload 中心走动补偿旋频（检测器/模板自产量）。

    δ̂ 门 0.04（= M2 报告 δ̂ 噪声底：p50 偏置 0.017 的 2×）——|δ̂|<0.04 视为
    无漂移不补偿：δ≈0 帧的 ±0.08 伪峰补偿反引入 ~2.6 bin 旋频误差。"""
    c_pay = pre + 12.25 + (psym - 1) / 2.0
    dg = float(dhat) if abs(float(dhat)) >= 0.04 else 0.0
    return float(nu0_ref) + (c_pay - ref_idx) * dg


def _demod_two_columns(seg_p, pre, psym, kt, dd):
    rows_a = dd.demod_payload(seg_p, pre + 5, psym)[1]     # DeRa port 相干行
    rows_b = kt.demod_payload(seg_p, pre + 5, psym, readout="viterbi")
    return rows_a, rows_b


def ours_chain_decode(seg, det, f, kt, dd):
    """检出后解码（两列 + CRC 仲裁）。返回 dict。

    主仲裁（u，预注册链）= 零走动反映射 + Δ0∈{0,±1,±2,±3} CRC，A 列
    优先。u_walk（申报的探索列，非预注册链）= 主仲裁失败时以检测器 δ̂
    为中心的斜率梯 {±δc/2, ±δc, ±3δc/2} × (ints, ints+1) × Δ0 的整数
    走动反映射——δ=0.082 native 诊断：argmax−gt 呈 −1→0→+1 缓坡（净
    整数走动 ≈0.7×psym·δ），零走动吸收不了；斜率梯全为检测器自产量。"""
    pre, psym = f["pre"], f["psym"]
    pad = (pre + 5) * NF
    seg_p = np.concatenate((np.zeros(pad, dtype=np.complex128),
                            seg[det["pay0"]:]))
    n_rot = np.arange(len(seg_p))
    seg_p = seg_p * np.exp(-2j * np.pi * det["nu_rot"] * n_rot / NF)
    rows_a, rows_b = _demod_two_columns(seg_p, pre, psym, kt, dd)
    out = {}
    for tag, rows in (("a", rows_a), ("b", rows_b)):
        ok, ser, d0 = demap_judge(rows, np.zeros(psym, dtype=int), f)
        out[tag] = dict(ok=ok, ser=ser, d0=d0)
    # 仲裁顺序：A 列优先（repo 最优 demod），A 全败再 B（候选预算 §D5）
    if out["a"]["ok"]:
        out["u"] = dict(ok=True, ser=out["a"]["ser"], col="a")
    elif out["b"]["ok"]:
        out["u"] = dict(ok=True, ser=out["b"]["ser"], col="b")
    else:
        out["u"] = dict(ok=False, ser=out["a"]["ser"], col="a")
    # 注：δ=0.082 native 诊断（探针）——argmax−gt 呈 −1→0→+1 缓坡（净整
    # 数走动 ≈0.6×psym·δ，符号大部分正确、坡区 ±1 错）。零走动+Δ0 常数
    # 吸收不了；δ̂ 斜率梯反映射实测失败（rint 边界 vs 实际坡形差 ±3 符号，
    # 且窗时漂移抵消因子 ~0.6 使 δ̂ ≠ 净走动斜率）——按预注册纪律不做
    # 进一步链上调整，δ=0.082 如实报失效分解。
    return out
