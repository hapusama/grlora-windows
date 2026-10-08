# -*- coding: utf-8 -*-
"""M2b 探针 0（2026-10-04）：三件事，决定 dep3 设计。

① δ=0.082 采集墙真实原因：m2_battle 的 nu_err 用【未注入】模板当 GT
   （nu0 + c_e·delta_native），注入后真 GT 应为 tmq（注入帧模板）。分解
   「度量偏置」vs「真锚坏」。
② 确认场 θ 相干度实测（真实 keystone 帧集）：确认行在真锚+真 δ+κ 对齐后
   的组间相位结构（pre/sync/sfd 三组残相），测「无 θ 损失」与 ψ(δ) 分数
   bin 律可解释性——dep3b 的设计输入。
③ 双段采集可行性：6 前导+2 sync（已知偏移 +24/+32 折算）联合相干 vs
   纯 6 前导，采集成功率曲线（δ∈{0,0.082}×档）。

运行：python m2b_probe0.py → m2b_probe0.json（分钟级）
"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_core as C
import d1_battle as A
import d2_core as D
import m2_core as M

NF = C.NF
N = C.N
HERE = os.path.dirname(os.path.abspath(__file__))
SEED_CONST = 20261003
DELTAS = (0.0, 0.082)
LEVELS = (None, -28, -30, -32, -34, -36)


def add_noise(seg, S, N0, level, seed, salt, di):
    rng = np.random.default_rng((SEED_CONST * 7919
                                 + (int(level) + 100) * 131
                                 + seed * 17 + salt * 7919
                                 + di * 104729) % (2 ** 31))
    p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
    return seg + ((rng.standard_normal(len(seg))
                   + 1j * rng.standard_normal(len(seg)))
                  * np.sqrt(p_add / 2.0))


# ---------------------------------------------------------------- 双段采集
def acquire2(segs, hs, pre):
    """双段采集：K_a 前导 + 2 sync（已知偏移 O_SYNC，列折算）联合相干。

    与 m2_core.acquire 同型，E 行 = 前 K_a 前导 + sync 2 行；sync 行列
    位置按 (ν0 + o_sync) 折算——把 sync 行谱移 o_sync·2 列后并入联合搜索。
    返回 dict(nu0h, khat, acq_score, c_e, ka, d_pre)。c_e = 全部 K_a+2 行
    的 idx 质心（走动参考）。"""
    wins = C.field_windows(hs, pre)
    B = segs.shape[0]
    ka = min(M.ACQ_K[pre], pre)
    rows = list(range(ka)) + [pre, pre + 1]
    idx_e = np.array(rows, dtype=float)
    c_e = float(idx_e.mean())
    n_arr = np.arange(NF)

    Xs = np.empty((B, len(rows), 2 * N), dtype=np.complex128)
    for q, j in enumerate(rows):
        s, r, l = wins[j]
        W = segs[:, s:s + l] * r[None, :l]
        X = np.fft.fft(W, M.NFFT, axis=1)
        Xc = np.concatenate((X[:, M.NFFT - N:], X[:, :N]), axis=1)
        off = 0.0 if j < pre else M.O_SYNC[j - pre]
        sh = int(round(2.0 * off))
        Xs[:, q] = np.roll(Xc, -sh, axis=1) * np.exp(
            -2j * np.pi * (2.0 * off - sh) * n_arr[:1] / NF)  # 近似（列整数）
    # 注：off=24/32 均为整数 bin ⇒ 2·off 整数列，roll 精确，无分数残差。

    Ekap = np.exp(-2j * np.pi * np.outer(M._KAP_AXIS, idx_e - c_e))
    cols = np.arange(2 * N)
    best = np.full(B, -1.0)
    k_pre = np.zeros(B, dtype=np.int64)
    d_pre = np.zeros(B)
    for dg in M.ACQ_DGRID:
        sh = np.rint(2.0 * (idx_e - c_e) * float(dg)).astype(int)
        Xsh = np.empty_like(Xs)
        for q in range(len(rows)):
            Xsh[:, q] = Xs[:, q][:, (cols + sh[q]) % (2 * N)]
        T = np.abs(np.einsum("qk,bkn->bqn", Ekap, Xsh)) ** 2
        flat = T.reshape(B, -1)
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
             * wins[j][1][None, :wins[j][2]] for j in rows]
    for poff in M.ACQ_P0_OFF:
        p0 = k0 + poff
        Xd = np.empty((B, len(rows)), dtype=np.complex128)
        for q, j in enumerate(rows):
            oj = 0.0 if j < pre else M.O_SYNC[j - pre]
            pj = p0 + oj + (idx_e[q] - c_e) * d_pre
            tw = np.exp(-2j * np.pi * pj[:, None] * n_arr[None, :NF] / NF)
            Xd[:, q] = np.einsum("bl,bl->b", Wrows[q], tw)
        F = np.abs(np.fft.fft(Xd, M.N_FINE, axis=1))
        q_ = np.argmax(F, axis=1)
        v = F[np.arange(B), q_] ** 2
        upd = v > best
        best = np.where(upd, v, best)
        nu0h = np.where(upd, p0, nu0h)
        khat = np.where(upd, M.kap_of_q(q_), khat)
    return dict(nu0h=nu0h, khat=khat, acq_score=best / (len(rows) * 1.0),
                c_e=c_e, ka=len(rows), d_pre=d_pre)


# ---------------------------------------------------------------- θ 结构
def theta_structure():
    """干净注入帧的确认场相位结构（28 帧 × δ∈{0,0.082}）。

    对每帧：用 tmq（GT）把确认行对齐到真锚+真 δ+κ̂，取 DTFT 复值 z_j；
    减去公共相位与线性斜坡（κ-FFT 等价）后，看组内/组间残相分布：
      - 组内 std（sync 对、SFD 组）：ψ/o 模型误差上界
      - 组间角（θ_sync、θ_sfd 相对前导组）：跨帧分布（可预测性）
    同时记录 |Σ√b z|²/Σb²（无 θ 相干度）与逐行可预测量（ν0 frac、o_j）。
    """
    frames = A.build_frames()
    out = []
    for fi, f in enumerate(frames):
        pre = f["pre"]
        for dl in DELTAS:
            eps = D.eps_of_delta(dl)
            inj = D.resample_sfo(f["seg"],
                                 f["lead"] - (pre + 4.25) * M.NF,
                                 eps) if dl else f["seg"]
            tmq = D.clean_template_q(inj, f["lead"], pre)
            wins = C.field_windows(f["lead"], pre)
            L = M._confirm_layout(pre)
            rows, b = L["rows"], L["b"]
            c_e = L["c_e"]
            nu0, dl_t = tmq["nu0"], tmq["delta"]
            conj = tmq["conj_dn"]
            z = np.zeros(len(rows), dtype=np.complex128)
            for j, rw in enumerate(rows):
                sgn_j = tmq["sgn"][rw]
                pj = nu0 + tmq["o"][rw] + sgn_j * (L["idx"][j] - c_e) * dl_t
                z[j] = C.row_dtft(inj, wins[rw][0], wins[rw][1],
                                  wins[rw][2], pj)[0]
                if conj and sgn_j < 0:
                    z[j] = np.conj(z[j])
            # κ̂：前导组行相位的线性拟合斜率（确认行内前导行）
            pre_rows = [j for j, rw in enumerate(rows) if rw < pre]
            ph = np.unwrap(np.angle(z[pre_rows]))
            jj = L["idx"][pre_rows]
            sl = np.polyfit(jj, ph, 1)[0]
            zr = z * np.exp(1j * sl * (L["idx"] - np.mean(jj)))
            # 组残相（相对全组公共相位）
            res = dict(frame=fi, pre=pre, delta=dl, nu0=nu0,
                       nu0_frac=((nu0 + 0.5) % 1.0) - 0.5,
                       o_sfd=[float(tmq["o"][rw]) for rw in rows
                              if rw >= pre + 2],
                       o_sync=[float(tmq["o"][rw]) for rw in rows
                               if pre <= rw < pre + 2],
                       conj=conj, kappa_slope=float(sl))
            # 归一：zn = z/(NF·â)，â = 前导组行幅 RMS；相干度分母 (Σ√b)²
            ahat = float(np.sqrt(np.mean(np.abs(z[pre_rows]) ** 2)) / NF)
            zn = z / (NF * ahat)
            znr = zr / (NF * ahat)
            s2b = float(np.sum(np.sqrt(b)) ** 2)
            for tag, sel in (("ph_pre", pre_rows),
                             ("ph_sync", [j for j, rw in enumerate(rows)
                                          if pre <= rw < pre + 2]),
                             ("ph_sfd", [j for j, rw in enumerate(rows)
                                         if rw >= pre + 2])):
                zz = znr[sel] * np.sqrt(b[sel])
                res[tag] = float(np.angle(np.sum(zz)))     # 组相位
                res[tag + "_mag"] = float(np.abs(np.sum(zz))) \
                    / float(np.sum(np.sqrt(b[sel])))
            best = 0.0
            for kap in np.fft.fftfreq(M.N_FINE):
                w = np.exp(-2j * np.pi * kap * (L["idx"] - c_e))
                best = max(best, float(np.abs(np.sum(np.sqrt(b) * zn * w))))
            res["coh_notheta"] = best ** 2 / s2b
            th = np.zeros(len(rows))
            th[pre_rows] = res["ph_pre"]
            th[[j for j, rw in enumerate(rows) if pre <= rw < pre + 2]] = \
                res["ph_sync"]
            th[[j for j, rw in enumerate(rows) if rw >= pre + 2]] = \
                res["ph_sfd"]
            res["coh_theta"] = float(np.abs(np.sum(
                np.sqrt(b) * znr * np.exp(-1j * th))) ** 2) / s2b
            out.append(res)
    return out


def main():
    t0 = time.time()
    frames = A.build_frames()
    res = {}

    # ---- ①② 双段 vs 单段采集 + 真锚误差 ----
    if not os.environ.get("SKIP_ACQ"):
        diag = {}
        for di, dl in enumerate(DELTAS):
            for level in LEVELS:
                for seed in (0, 1, 2):
                    for fi, f in enumerate(frames):
                        pre = f["pre"]
                        eps = D.eps_of_delta(dl)
                        inj = D.resample_sfo(f["seg"],
                                             f["lead"] - (pre + 4.25) * M.NF,
                                             eps) if dl else f["seg"]
                        S, N0 = A.snr_parts(inj)
                        seg = inj if level is None else add_noise(
                            inj, S, N0, level, seed, f["hs"] % 4099, di)
                        seg = seg[None, :]
                        tmq = D.clean_template_q(inj, f["lead"], pre)
                        a1 = M.acquire(seg, f["lead"], pre)
                        a2 = acquire2(seg, f["lead"], pre)
                        e1 = a1["nu0h"][0] - (tmq["nu0"] + a1["c_e"]
                                              * tmq["delta"])
                        e2 = a2["nu0h"][0] - (tmq["nu0"] + a2["c_e"]
                                              * tmq["delta"])
                        k = "δ=%g lv=%s" % (dl, level)
                        r = diag.setdefault(k, dict(e1=[], e2=[], ok1=[],
                                                    ok2=[]))
                        r["e1"].append(e1)
                        r["e2"].append(e2)
                        r["ok1"].append(1.0 if abs(e1) < 0.5 else 0.0)
                        r["ok2"].append(1.0 if abs(e2) < 0.5 else 0.0)
        for k, r in diag.items():
            e1, e2 = np.abs(np.array(r["e1"])), np.abs(np.array(r["e2"]))
            diag[k] = dict(n=len(e1),
                           seg1=dict(ok=float(np.mean(r["ok1"])),
                                     p50=float(np.quantile(e1, .5)),
                                     p90=float(np.quantile(e1, .9))),
                           seg2=dict(ok=float(np.mean(r["ok2"])),
                                     p50=float(np.quantile(e2, .5)),
                                     p90=float(np.quantile(e2, .9))))
        res["acq_diag"] = diag

    # ---- ② θ 结构 ----
    ts = theta_structure()
    res["theta_rows"] = ts
    _gp(res, ts)
    json.dump(res, open(os.path.join(HERE, "m2b_probe0.json"), "w"),
              indent=1)
    print("acq_diag:")
    for k, v in res.get("acq_diag", {}).items():
        print(" %s seg1(ok=%.2f p50=%.3f p90=%.3f) seg2(ok=%.2f p50=%.3f"
              " p90=%.3f)" % (k, v["seg1"]["ok"], v["seg1"]["p50"],
                              v["seg1"]["p90"], v["seg2"]["ok"],
                              v["seg2"]["p50"], v["seg2"]["p90"]))
    print("group_phase:")
    for k, v in res.get("group_phase", {}).items():
        print(" ", k, {kk: (round(vv, 3) if isinstance(vv, float) else
                        [round(x, 1) for x in vv]) for kk, vv in v.items()})
    print("%.0fs -> m2b_probe0.json" % (time.time() - t0))


def _circ_stat(a_deg):
    a = np.radians(np.asarray(a_deg))
    m = np.angle(np.mean(np.exp(1j * a)))
    d = np.angle(np.exp(1j * (a - m)))
    return [float(np.degrees(m)), float(np.degrees(np.std(d)))]


def _gp(res, ts):
    for pre in (8, 16, 32):
        for dl in DELTAS:
            sel = [r for r in ts if r["pre"] == pre and r["delta"] == dl]
            if not sel:
                continue
            syn = np.degrees([np.angle(np.exp(1j * (r["ph_sync"]
                                                    - r["ph_pre"])))
                              for r in sel])
            sfd = np.degrees([np.angle(np.exp(1j * (r["ph_sfd"]
                                                    - r["ph_pre"])))
                              for r in sel])
            res.setdefault("group_phase", {})["P%d δ=%g" % (pre, dl)] = dict(
                n=len(sel),
                sync_vs_pre_deg=_circ_stat(syn),
                sfd_vs_pre_deg=_circ_stat(sfd),
                coh_notheta_med=float(np.median([r["coh_notheta"]
                                                 for r in sel])),
                coh_theta_med=float(np.median([r["coh_theta"]
                                               for r in sel])))


if __name__ == "__main__":
    main()
