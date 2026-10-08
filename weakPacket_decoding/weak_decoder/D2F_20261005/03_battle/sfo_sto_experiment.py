# -*- coding: utf-8 -*-
"""SFO vs STO 替代实验（2026-10-05）：低 SNR 下"单参数 STO 顶替 SFO"的
模型失配账与后果。用户命题："低 SNR 下不能把 SFO 简单用 STO 替代"。

设计（单一变量隔离：同一解码器 = DeRa port + Δ0∈{0,±1,±2,±3} CRC，
只换时间基补偿方式；全部参数在干净信号上估计/冻结 = 各臂最优形态，
实验B 哲学、机制级诊断，协议 §5A）：

  A_full   完整补偿 oracle：注入反转重采样（ε 精确）+ native τ̂₀ 时延
  B_sto    最强单 STO：GT 搜出的全局最优常数时延（参考点自由——任何
           STO 方法的上界）
  C_sto_pre 前导参考 STO：τ̂₀(native) 施加于前导参考，payload 残差
           自由漂移（真实接收机"前导估一次 τ、包内不管"的形态）
  D_ladder 斜率梯（ε 一阶近似，GT 斜率 0.02 预旋）= D2F 现行机制
  （参照行：m3p2 实战链 PER，join 自 checkpoint）

场景：δ=0.02 注入 × SNR{native,−18,−20,−22,−24} × 2 种子 × 28 帧。
指标：PER / SER / **逐符号位置错误图**（残差集中度）/ 机制层逐符号
统计量损失（clean，ref_template）。

预注册预测：
  P1 native：全臂 SER≈0（约定/符号自检）；
  P2 机制层（clean）：B 损失中位 −1~−2dB、尾部 −4~−7dB、随 k 线性增长；
     A 平坦（贴界）；
  P3 判决层：−18 各臂接近（FEC 原谅尾部损伤）；−22/−24 差距爆炸，
     C 最差（尾部 bin 翻转，Δ0 常数吸收不了斜率）> B > D ≈ A；
  P4 C 的错误集中在 payload 尾部符号（位置图）。
→ sfo_sto_results.jsonl（本包 04_results，新实验无 join 依赖）
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\weak_decoder\D2F_20261005")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\weak_decoder\D2F_20261005\01_core")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\dera_front_battle_20260930")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\e2e_final_20261004")

import d2_core as D2
import m3_core as M3
import m3p_core as M3P
import front_runner as FR
import ref_template as RT
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator

NF, N, OSF = 4096, 1024, 4
DD = DeRaDemodulator(10, 4)
SEED_CONST = 20261005
DELTA = 0.02
LEVELS = (None, -18, -20, -22, -24)
SEEDS = (0, 1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "04_results", "sfo_sto_results.jsonl")


def frac_delay(seg, tau):
    """全段分数时延（带限，频域相位坡；τ 采样，正=补偿提前）。"""
    if abs(tau) < 1e-9:
        return seg
    f = np.fft.fftfreq(len(seg))
    return np.fft.ifft(np.fft.fft(seg) * np.exp(-2j * np.pi * f * tau))


def run_arm(arm, seg_inj, seg0, f, pre, psym, nu0, tau0_nat, tau0_best,
            eps_undo, eps, origin, m0, lead, slope=DELTA):
    pad = (pre + 5) * NF
    if arm == "A_full":
        seg_c = D2.resample_sfo(seg_inj, origin, eps_undo)
        seg_c = frac_delay(seg_c, tau0_nat)
    elif arm == "B_sto":
        seg_c = frac_delay(seg_inj, tau0_best)
    elif arm == "C_sto_pre":
        seg_c = frac_delay(seg_inj, tau0_nat)
    else:                                   # D_ladder（= m3p2 真实机制：
        seg_c = seg_inj                     # ν_rot 居中 + 梯，逐字镜像）
    if arm == "D_ladder":
        nu_arm = M3.nu_rot_of(nu0, slope, (pre - 1) / 2.0, pre, psym)
    else:
        nu_arm = nu0
    seg_p = np.concatenate((np.zeros(pad, dtype=np.complex128),
                            seg_c[m0:]))
    base = seg_p * np.exp(-2j * np.pi * nu_arm * np.arange(len(seg_p)) / NF)
    if arm == "D_ladder":
        n_win = np.arange(NF)
        last = None
        for sl in (0.0, 0.01, 0.02, 0.03):   # 真实 m3p2 机制：梯 CRC 早退
            sp = base
            if sl:
                sp = base.copy()
                for s in range(psym):
                    w0 = (pre + 5 + s) * NF
                    sp[w0:w0 + NF] *= np.exp(
                        -2j * np.pi * sl * (s - (psym - 1) / 2.0)
                        * n_win / NF)
            rows = DD.demod_payload(sp, pre + 5, psym)[1]
            ok, ser, d0 = M3.demap_judge(rows, np.zeros(psym, dtype=int), f)
            if ok:
                am = np.argmax(rows, axis=1)
                err = ((am - d0 - 1) % N) != np.asarray(f["gt"])
                return dict(ok=True, ser=int(err.sum()),
                            err=err.astype(int).tolist(), slope=sl)
            last = (rows, d0)
        rows, d0 = last
        am = np.argmax(rows, axis=1)
        err = ((am - d0 - 1) % N) != np.asarray(f["gt"])
        return dict(ok=False, ser=int(err.sum()),
                    err=err.astype(int).tolist(), slope=None)
    rows = DD.demod_payload(base, pre + 5, psym)[1]
    ok, ser, d0 = M3.demap_judge(rows, np.zeros(psym, dtype=int), f)
    am = np.argmax(rows, axis=1)
    err = ((am - d0 - 1) % N) != np.asarray(f["gt"])
    return dict(ok=bool(ok), ser=int(err.sum()), err=err.astype(int).tolist())


def main():
    t0 = time.time()
    frames = FR.build_frames()
    out_fh = open(OUT, "a", encoding="utf-8")
    n_units = 0
    for fi, f in enumerate(frames):
        pre, psym = f["pre"], f["psym"]
        lead = pre + 6
        tail = (8 + psym + 2) * NF + 64
        seg0 = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + tail],
                          dtype=np.complex128)
        origin = (lead - (pre + 4.25)) * NF
        m0 = lead * NF + 8 * NF
        eps = D2.eps_of_delta(DELTA)
        eps_undo = -eps / (1.0 + eps)
        seg_inj = D2.resample_sfo(seg0, origin, eps)
        # ν̂₀ 基准 = dep4 连续锚（干净注入信号上；与 demod 端约定一致——
        # clean_template_q 的 ν₀ 约定差 ~0.146 bin，分数 bin 边界即生死线，
        # 本轮逐位对拍实锤）
        nu0 = float(M3P.dep4_blind_detect(seg_inj, pre)["nu0h"])
        gt1 = np.asarray(f["gt"]) + 1
        tau0_nat = RT.est_tau0(seg0, m0, gt1, nu0, 0.0, 0.0, None)
        # B 臂最优常数时延：解析中心化（GT ε 残差斜坡的均值置零）——
        # 不用相关搜索（啁啾时延目标函数 Fresnel 多瓣，全局最大落旁瓣，
        # 本轮校准诊断实测 τ̂=−0.9 旁瓣陷阱）
        r_mean = float(np.mean([m0 + k * NF - origin
                                for k in range(psym)])) * eps
        tau0_best = tau0_nat + r_mean
        # 机制层（clean）：A vs B 逐符号统计量
        ZA, MA = RT.z_full(seg_inj, m0, gt1, nu0, eps, origin,
                           tau0=tau0_nat)
        ZB, MB = RT.z_full(seg_inj, m0, gt1, nu0, 0.0, origin,
                           tau0=tau0_best)
        S, n0 = FR.snr_parts(seg_inj[:lead * NF + 8 * NF])
        mech = dict(
            snrA=(10 * np.log10(np.abs(ZA) ** 2 / (n0 * MA))).round(2).tolist(),
            snrB=(10 * np.log10(np.abs(ZB) ** 2 / (n0 * MB))).round(2).tolist(),
            tau0_nat=float(tau0_nat), tau0_best=float(tau0_best))
        for lv in LEVELS:
            for sd in SEEDS:
                if lv is None:
                    seg = seg_inj
                else:
                    rng = np.random.default_rng(
                        (SEED_CONST * 7919 + (int(lv) + 100) * 131
                         + sd * 101 + fi * 7919 + 3 * 104729) % (2 ** 31))
                    p_add = max(S / 10 ** (lv / 10.0) - n0, 1e-30)
                    seg = seg_inj + (
                        (rng.standard_normal(len(seg_inj))
                         + 1j * rng.standard_normal(len(seg_inj)))
                        * np.sqrt(p_add / 2.0))
                rec = dict(frame=fi, pre=pre, level=lv, seed=sd,
                           delta=DELTA)
                for arm in ("A_full", "B_sto", "C_sto_pre", "D_ladder"):
                    r = run_arm(arm, seg, seg0, f, pre, psym, nu0,
                                tau0_nat, tau0_best, eps_undo, eps,
                                origin, m0, lead)
                    rec[arm] = dict(ok=r["ok"], ser=r["ser"])
                    if lv == -22 or lv is None:
                        rec[arm + "_err"] = r["err"]
                if lv is None and fi < 3:
                    rec["mech"] = mech
                out_fh.write(json.dumps(rec) + "\n")
                n_units += 1
        out_fh.flush()
    out_fh.close()
    print("units:", n_units, "%.0fs" % (time.time() - t0))

    # ---- 汇总 ----
    import collections
    agg = collections.defaultdict(
        lambda: collections.defaultdict(lambda: dict(n=0, per=0, ser=0)))
    poserr = collections.defaultdict(
        lambda: collections.zeros if False else None)
    pos = collections.defaultdict(lambda: collections.Counter())
    for line in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        for arm in ("A_full", "B_sto", "C_sto_pre", "D_ladder"):
            a = agg[r["level"]][arm]
            a["n"] += 1
            a["per"] += int(not r[arm]["ok"])
            a["ser"] += r[arm]["ser"]
            key = (r["level"], arm)
            e = r.get(arm + "_err")
            if e:
                for k, v in enumerate(e):
                    pos[key][k] += v
    print("δ=0.02  PER / SER（n=56/档，psym≈35-40）")
    print("SNR     | A_full      | B_sto       | C_sto_pre   | D_ladder")
    for lv in sorted(agg, key=lambda x: (x is not None, x)):
        row = []
        for arm in ("A_full", "B_sto", "C_sto_pre", "D_ladder"):
            a = agg[lv][arm]
            tot = a["n"] * (35 + 40) // 2
            row.append("PER %.3f SER %.3f" % (a["per"] / a["n"],
                                              a["ser"] / max(tot, 1)))
        print("%-7s | %s | %s | %s | %s" % (lv, *row))
    print("\n−22 档逐符号错误位置（bin=10 符号）：")
    for arm in ("A_full", "B_sto", "C_sto_pre", "D_ladder"):
        c = pos[(-22, arm)]
        if c:
            hist = [sum(c.get(k, 0) for k in range(b, b + 10))
                    for b in range(0, 40, 10)]
            print("  %-10s %s" % (arm, hist))


if __name__ == "__main__":
    main()
