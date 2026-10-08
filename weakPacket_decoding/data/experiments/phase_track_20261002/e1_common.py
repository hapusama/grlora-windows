# -*- coding: utf-8 -*-
"""E1 相位轨迹物理验证（2026-10-02）——共用库。

遵循 EXPERIMENT_PROTOCOL.md：
  - 数据集规则（§1）：header_valid=1、payload_len>0、干净信号 GT 可解且
    header 校验和 + payload CRC16 通过（decode_explicit_frame_symbols）；
  - SNR 口径（§2）：整包加噪 SNR，S = 段功率 − 带外噪声底（|f|∈(70,240) kHz），
    N0 = 带外噪声底，P_add = S/10^(SNR/10) − N0；噪声 = (randn+j·randn)·sqrt(P_add/2)，
    种子由 (实验日, 档位, 种子号, 帧号) 确定性派生；
  - 噪声有且仅有 AWGN，同一实现口径；native（不加噪）档必须包含。

物理域约定（本实验自定义，全部经 OTA 数据实测核实，见 e1_smoke/probe）：
  - 对齐：整 bin+分数 CFO × 全段 + STO 亚 chip 分数时延（= battle seg_aligned 同式）；
  - 提取：MASK 带限（= battle wm 同式）→ OS 抽取 → 乘抽取域 downchirp 参考
    ref_dn = conj(build_upchirp(sf,0,1)) → N 点 FFT。该参考下实测音位置：
    payload 非 LDRO 音在 v、LDRO 音在 4v、header 音在 4·gt_hdr（gt_hdr 为 //4
    域值）、前导音在 −1、sync word 0x34 两符号音在 {23, 31}。
  - 帧结构（实测核实）：前导 P @ hs−(P+4.25−k)NF + sync#2(23/31) 2 upchirp
    + SFD 2.25 downchirp（止于 hs）+ header 8 @ hs+j·NF + payload；
    header/payload 网格与 battle 完全一致。
  - 两级测量（measure_frame）：第一级逐符 argmax/抛物线/PCM γ/解斜产物 dr；
    第二级 mod-1 圆统计跟踪帧音轨迹 ν̂_i = c_i + κ₀ + εν·i（对整数 argmax 滑步
    免疫），θ_i = 跟踪频率处 DTFT 相位（接收机同款操作定义）。
"""
import sys
import csv
import os
import json
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols
from weak_decoder.chirp import build_upchirp

ROOT = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\phase_track_20261002"
HF = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
      r"\header_first")
USRP = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ"
DAY = 20261002                      # 实验日（种子派生）
FS = 500000.0
SNR_LEVELS = [-16, -18, -20, -22, -24]
N_SEEDS = 20

SF10_SOURCES = [
    ("0_0_0_10_14_%d" % p, os.path.join(USRP, "0_0_0_10_14_%d.bin" % p),
     os.path.join(HF, "0_0_0_10_14_%d_header_first_frames.csv" % p), 10, p, 0)
    for p in (8, 16, 32)]
SF11_SOURCES = [
    ("1_0_%d_11_2_16" % p, os.path.join(USRP, "lab1_sf11_TP2", "1_0_%d_11_2_16.bin" % p),
     os.path.join(HF, "1_0_%d_11_2_16_header_first_frames.csv" % p), 11, 16, 1)
    for p in range(8, 17)] + \
    [("1_1_%d_11_2_16" % p, os.path.join(USRP, "lab1_sf11_TP2", "1_1_%d_11_2_16.bin" % p),
      os.path.join(HF, "1_1_%d_11_2_16_header_first_frames.csv" % p), 11, 16, 1)
     for p in range(0, 8)]


def fv(x):
    """CSV 数值字段安全解析（branch_* 多值取首值 = 有效支路）。"""
    s = (x or "0").strip()
    if not s:
        return 0.0
    return float(s.split("|")[0])


def wrap(x):
    """卷绕到 (−π, π]。"""
    return np.angle(np.exp(1j * np.asarray(x)))


def frac_delay(x, samples):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x))
    return np.fft.ifft(X * np.exp(-2j * np.pi * f * samples))


class DS:
    """一个数据集（SF/前导/LDRO 相同的一组 capture）的静态参数。"""

    def __init__(self, sf):
        self.sf = int(sf)
        self.n = 1 << self.sf
        self.os = 4
        self.nf = self.n * self.os
        self.mask = np.abs(np.fft.fftfreq(self.nf)) <= 0.125 + 40.0 / self.nf
        self.ref_up = build_upchirp(self.sf, 0, 1).astype(np.complex128)      # N 点抽取域 upchirp id=0
        self.ref_dn = np.conj(self.ref_up)                                     # downchirp id=0（主参考）
        self.dnchirp_os = np.conj(build_upchirp(self.sf, 0, self.os)).astype(np.complex128)  # NF 域 downchirp

    def ref_for(self, kind):
        """upchirp 类符号窗（pre/sync/hdr/pay）用 downchirp 参考；SFD downchirp 窗用 upchirp 参考。"""
        return self.ref_dn if kind != "sfd" else self.ref_up


def make_slots(P, psym):
    """返回 [(kind, k, off_nf, use_in_fit)]；off_nf 为 seg 内窗口起点（NF 单位，可分数）。

    帧结构（OTA 实测核实，见模块 docstring）：前导 P @ hs−(P+4.25−k)NF，
    sync#1(id 23) @ [hs−4.25,hs−3.25)NF，sync#2(id 31) @ [hs−3.25,hs−2.25)，
    sfd 2.25 downchirp（两整窗 + 1/4 窗）止于 hs，header 8 @ hs+jNF，
    payload @ hs+(8+k)NF。seg 原点 = hs−(P+5)NF。
    """
    slots = [("pre", k, 1.0 + k - 0.25, True) for k in range(P)]
    slots += [("sync", 0, P + 0.75, True), ("sync", 1, P + 1.75, True)]
    slots += [("sfd", j, P + 2.75 + j, False) for j in range(2)]
    slots += [("sfdq", 0, P + 4.75, False)]                    # 1/4 downchirp，仅记录
    slots += [("hdr", j, P + 5.0 + j, True) for j in range(8)]
    slots += [("pay", k, P + 13.0 + k, True) for k in range(psym)]
    return slots


C_PRE, C_SYNC = -1, (23, 31)     # 前导/sync 音位置（OTA 实测；整数部分，分数由跟踪吸收）


def c_nominals(P, psym, ldro, gt_hdr, gt_pay):
    """fit 符号的名义音 bin c_i（= 实测音位置整数约定）：pre −1、sync {23,31}、
    header 4·v_hdr、payload 非 LDRO v / LDRO 4v。与 make_slots 的 fit 序列对齐。"""
    cs = [C_PRE] * P + list(C_SYNC)
    cs += [4 * int(v) for v in gt_hdr]
    cs += [(4 * int(v)) if ldro else int(v) for v in gt_pay]
    return cs


def measure_frame(seg, P, psym, ldro, gt_hdr, gt_pay, ds, fine=False):
    """整帧两级测量（native 与注噪共用）：逐符原始测量 → 逐类整数 c 校正 →
    mod-1 帧音跟踪 → 跟踪频率处 DTFT 相位。返回 (syms, track0, c_list)。

    syms 字段：kind,k,c,b,kappa,dnu_raw,nu,theta,theta_pk,amp,snr_eff,
    gamma_pcm,z_pcm[,y_v1v2,z_v1v2][,nu_fine,theta_fine,nu_h1,th_h1,nu_h2,th_h2]。
    """
    n = ds.n
    slots = make_slots(P, psym)
    cs_nom = c_nominals(P, psym, ldro, gt_hdr, gt_pay)
    it = iter(cs_nom)
    recs = []
    for kind, k, off, use in slots:
        c0 = next(it) if use else None
        m = extract_raw(seg, off, ds, ds.ref_for(kind), c_nom=(c0 if use else 0))
        recs.append((kind, k, off, use, m, c0))
    # 逐类整数 c 校正（强符号 argmax 众数；mod N → 有符号）
    def cls_mode(kind_sel, nominal):
        bb = [rec[4]["b"] for rec in recs
              if rec[3] and rec[0] in kind_sel and rec[4]
              and rec[4]["snr_eff"] > 30.0]
        if len(bb) < 2:
            return 0
        d = np.array([(b - nominal) % n for b in bb])
        cnts = np.bincount(d, minlength=n)
        mode = int(cnts.argmax())
        if mode > n // 2:
            mode -= n
        return mode if cnts.max() >= max(2, len(bb) - 1) else 0
    g_pre = cls_mode(("pre",), C_PRE)
    g_hp = cls_mode(("hdr", "pay"), 0)
    g_syn = cls_mode(("sync",), 0)      # sync 两符各自 b 与 23/31 的整体偏移核查
    c_list = []
    for kind, k, off, use, m, c0 in recs:
        if not use:
            c_list.append(None)
        elif kind == "pre":
            c_list.append(c0 + g_pre)
        elif kind == "sync":
            c_list.append(c0 + g_syn)
        else:
            c_list.append(c0 + g_hp)
    fit = [j for j, rec in enumerate(recs) if rec[3] and rec[4]]
    dnu = []
    for j in fit:
        m = recs[j][4]
        dnu.append(float((m["b"] + m["kappa"] - c_list[j] + n / 2.0) % n - n / 2.0))
    k0, enu = track_tone(dnu, np.arange(len(fit), dtype=float),
                         [recs[j][4]["snr_eff"] for j in fit])
    syms = []
    for jj, j in enumerate(fit):
        kind, k, off, use, m, _ = recs[j]
        nu = c_list[j] + k0 + enu * jj
        th = theta_at(m["dr"], nu, n)
        rec = dict(kind=kind, k=int(k), c=int(c_list[j]), b=m["b"], kappa=m["kappa"],
                   dnu_raw=m["dnu"], nu=nu, theta=th, theta_pk=m["theta_pk"],
                   amp=m["amp"], snr_eff=m["snr_eff"],
                   gamma_pcm=m["gamma_pcm"], z_pcm=m["z_pcm"])
        g = v1v2_gamma(seg, off, ds, m["b"])
        if g is not None:
            rec["y_v1v2"], rec["z_v1v2"] = g[0], g[1]
        if fine and (kind in ("pre", "sync")
                     or (kind == "pay" and k in (0, psym // 2, psym - 1))):
            (nf_, tf_, af_), h1, h2 = fine_scan_dr(m["dr"], nu, n)
            rec.update(nu_fine=nf_, theta_fine=tf_,
                       nu_h1=h1[0], th_h1=h1[1], nu_h2=h2[0], th_h2=h2[1])
        syms.append(rec)
    return syms, dict(kappa0=k0, enu=enu, g_pre=g_pre, g_syn=g_syn, g_hp=g_hp), c_list


def align_seg(iq, r, ds, P, psym, lead=5):
    """全先验对齐段（= battle seg_aligned 同式，段向前扩 lead 符号以覆盖全前导）。

    返回 (seg complex64, i0, battle_off)；battle 段（SNR 标定口径）= seg[battle_off*NF:]。
    """
    hs = int(r["header_start_sample"])
    nf = ds.nf
    i0 = hs - (P + lead) * nf
    i1 = hs + (8 + psym + 1) * nf
    if i0 < 0:
        return None, i0, 0
    seg = np.asarray(iq[i0:i1], dtype=np.complex128)
    n = np.arange(len(seg))
    f = int(r["source_grlora_cfo_int"]) + fv(r.get("source_grlora_cfo_frac"))
    seg = seg * np.exp(-2j * np.pi * f * n / nf)
    seg = frac_delay(seg, -fv(r.get("source_grlora_payload_sto_frac")) * ds.os)
    battle_off = (P + lead - (8 if ds.sf == 10 else 16))
    return seg.astype(np.complex64), i0, battle_off


def snr_parts(seg, nf):
    """协议 §2：S = 段功率 − 带外噪声底，N0 = 带外噪声底（Parseval 单位）。"""
    X = np.fft.fftshift(np.fft.fft(seg))
    p = np.abs(X) ** 2 / len(seg)
    f = np.fft.fftshift(np.fft.fftfreq(len(seg))) * FS
    ob = (np.abs(f) > 70000) & (np.abs(f) < 240000)
    n0 = float(np.mean(p[ob]))
    total = float(np.mean(np.abs(seg) ** 2))
    return max(total - n0, 1e-30), n0


def unit_rng(snr, seed, gid):
    """协议 §2 种子派生：(实验日, 档位, 种子号, 帧号)。"""
    lv = 0 if snr is None else int(snr)
    return np.random.default_rng((DAY * 7919 + (lv + 100) * 131 + seed * 17 + gid * 7919) % (2 ** 31))


def inject_noise(seg, snr, seed, gid, S, N0):
    """整包 AWGN（协议铁律 1 / §2）：返回带噪 seg（complex64）。"""
    if snr is None:
        return seg
    rng = unit_rng(snr, seed, gid)
    p_add = max(S / 10 ** (snr / 10.0) - N0, 1e-30)
    nz = (rng.standard_normal(len(seg)) + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    return (seg.astype(np.complex128) + nz).astype(np.complex64)


def extract_raw(seg, off_nf, ds, ref, c_nom):
    """单符号原始测量（第一级）：argmax b、抛物线 κ、dnu_raw、幅度/峰SNR、PCM γ、
    解斜产物 dr（复数128，供第二级跟踪后测 θ）。

    dnu_raw = ((b+κ̂) − c_nom) 卷绕到 (−N/2, N/2]；整数 argmax 滑步以 mod-1 群聚类
    在第二级吸收（圆统计对整数滑步免疫）。
    """
    nf, n, os = ds.nf, ds.n, ds.os
    off = int(round(off_nf * nf))
    w = seg[off:off + nf]
    if w.size < nf:
        return None
    W = np.fft.fft(w.astype(np.complex128)) * ds.mask
    d = np.fft.ifft(W)
    s0 = d[::os]
    dr = s0 * ref
    X = np.fft.fft(dr)
    p2 = np.abs(X) ** 2
    b = int(np.argmax(p2))
    l, m, r = abs(X[(b - 1) % n]), abs(X[b]), abs(X[(b + 1) % n])
    kq = float(np.clip(0.5 * (l - r) / (l - 2 * m + r + 1e-30), -0.5, 0.5))
    dnu = float((b + kq - c_nom + n / 2.0) % n - n / 2.0)
    e = np.exp(-1j * np.pi / n)
    z1 = X[(b - 1) % n] - e * X[b]
    z0 = X[b] - e * X[(b + 1) % n]
    zp = z1 if abs(z1) >= abs(z0) else z0
    floor = float(np.median(p2))
    return dict(b=b, kappa=kq, dnu=dnu, amp=float(abs(X[b])), floor=floor,
                snr_eff=float(p2[b] / max(floor, 1e-30)),
                gamma_pcm=float(np.angle(zp)), z_pcm=float(abs(zp)),
                theta_pk=float(np.angle(X[b])), dr=dr)


def track_tone(dnu, idx, snr_eff, span=0.02, npts=161):
    """第二级帧音跟踪（mod-1 圆统计，对整数 argmax 滑步免疫）：

    ν̂_i = c_i + κ₀ + εν·i；κ₀ = 加权圆均值（权重 ∝ snr_eff），
    εν = wrapped 线性斜率（|εν|≤span 网格搜索，再抛物线细化）。
    """
    dnu = np.asarray(dnu, dtype=float)
    idx = np.asarray(idx, dtype=float)
    w = np.asarray(snr_eff, dtype=float)
    w = w / (np.max(w) + 1e-30)
    ph = np.exp(2j * np.pi * dnu)
    g = np.linspace(-span, span, npts)
    sc = np.abs((w * ph) @ np.exp(-2j * np.pi * np.outer(idx, g)))
    j = int(np.argmax(sc))
    if 0 < j < npts - 1:
        y0, y1, y2 = sc[j - 1], sc[j], sc[j + 1]
        dmax = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30)
        enu = float(g[j] + np.clip(dmax, -0.5, 0.5) * (g[1] - g[0]))
    else:
        enu = float(g[j])
    z = float(np.angle(np.sum(w * ph * np.exp(-2j * np.pi * enu * idx))) / (2 * np.pi))
    k0 = float((z + 0.5) % 1.0 - 0.5)
    return k0, enu


def theta_at(dr, nu, n):
    """跟踪频率处的 DTFT 相位（= 音起始相位，跟踪误差 π·Δν·(N−1)/N 计入残差）。"""
    return float(np.angle(np.dot(dr, np.exp(-2j * np.pi * nu * np.arange(len(dr)) / n))))


def fine_scan_dr(dr, c_ref, n, half_span=0.9, npts=181, halves=True):
    """精细 DTFT 扫描（在 dr 上）：主峰 + 可选前后半窗。"""
    out = []
    segs = [dr] + ([dr[:n // 2], dr[n // 2:]] if halves else [])
    for x in segs:
        fr = c_ref + np.linspace(-half_span, half_span, npts)
        V = np.exp(-2j * np.pi * np.outer(fr, np.arange(len(x))) / n) @ x
        j = int(np.argmax(np.abs(V)))
        out.append((float(fr[j]), float(np.angle(V[j])), float(np.abs(V[j]))))
    return out


def pcm_gamma_X(X, b, n):
    """跨踞对 PCM 读出：γ = arg(X[j] − e^{−jπ/N}·X[j+1])，j ∈ {b−1, b} 取 |Z| 大者。"""
    j1 = (b - 1) % n
    e = np.exp(-1j * np.pi / n)
    z1 = X[j1] - e * X[b]
    z0 = X[b] - e * X[(b + 1) % n]
    z = z1 if abs(z1) >= abs(z0) else z0
    return float(np.angle(z)), float(abs(z))


def v1v2_gamma(seg, off_s, ds, m_hat):
    """γ-链 V1/V2 读出（weak_decoder.baselines.dera._wrap_split_matrices 的逐候选闭式）：

    d_m[n] = w[n]·downchirp[(n+m·OS) mod NF]；split=(N−m)·OS；V1=Σ前段, V2=Σ后段；
    y = angle(V2·conj(V1))/π（= πκ 差分相位，π 周期模糊 → mod-1 卷绕域）。
    """
    nf, n, os = ds.nf, ds.n, ds.os
    off = int(round(off_s))
    w = seg[off:off + nf].astype(np.complex128)
    if w.size < nf:
        return None
    m = int(m_hat) % n
    d = w * np.roll(ds.dnchirp_os, -m * os)
    ts = nf - m * os
    v1 = float(np.sum(d[:ts].real)) + 1j * float(np.sum(d[:ts].imag))
    v2 = float(np.sum(d[ts:].real)) + 1j * float(np.sum(d[ts:].imag))
    z = v2 * np.conj(v1)
    return float(np.angle(z)) / np.pi, float(abs(z))


def freeze_gt(iq, r, ds, psym, ldro):
    """协议 §1 GT 冻结：CSV 先验在干净信号上解全帧符号，CRC 必须通过。"""
    res = demod_symbol_sequence(
        samples=np.asarray(iq, dtype=np.complex64),
        header_start_sample=int(r["header_start_sample"]), sf=ds.sf, os_factor=ds.os,
        cfo_int=int(r["source_grlora_cfo_int"]), cfo_frac=fv(r.get("source_grlora_cfo_frac")),
        sfo_hat=fv(r.get("source_grlora_sfo_hat")),
        sfo_cum_initial=fv(r.get("source_grlora_branch_sfo_cum_initial")),
        header_count=8, payload_count=psym, payload_ldro=bool(ldro))
    gt_hdr = [x.symbol_value for x in res[:8]]
    gt = [x.symbol_value for x in res[8:]]
    dec = decode_explicit_frame_symbols(gt_hdr, gt, sf=ds.sf, bw=125000.0,
                                        ldro_mode=1 if ldro else 2)
    if not (dec.header.header_valid and dec.payload.crc_valid):
        raise ValueError("GT CRC 未过")
    return gt_hdr, gt


def c_of(v, ldro, is_header):
    """值 → 名义音 bin（TX 循环移位）。header 恒全分辨率；LDRO payload 音在 4v+1。"""
    if is_header or not ldro:
        return int(v) + 1
    return 4 * int(v) + 1
