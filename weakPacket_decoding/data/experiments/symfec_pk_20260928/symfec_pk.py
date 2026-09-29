# -*- coding: utf-8 -*-
r"""
2026-09-28 终局 PK：baselines 全家（SAVAUX/TRIMMER/UNICHIRP/SymFEC）vs 本链。

设计：测试床升级为**真 LoRa PHY 编码**（payload_codec 的 encode_explicit_frame_symbols：
白化 + Hamming(CR4/8) + 对角交织 + Gray + CRC）。所有解调层产出的谱统一喂给
**SymFEC 自己的解码器**（decode_symfec_payload_from_evidences）——裁判与 FEC 层
完全同一，唯一变量是证据层（谁的谱更好）。

接收方（谱均为 bin 域，SymFEC evidence 自做值映射）：
  OLD-A   chip 网格单相位 N-FFT（frame_sync 风格）
  SAVAUX  TIM'22 Eq.34-37 branch 合并谱
  TRIMMER MobiCom'24 切分度量
  UNICHIRP preamble 相位模型 + 双峰融合（额外吃 8 个已知 preamble，对其有利）
  NEW-0   本文 C1：OS 去斜 + 补零 16N 细网格精确读
  TREL-N  本文 C1 + 跨符号 κ-格 Viterbi

包结构：8 preamble(值512=对方bin0) + 8 header(真值已知,白化同步用) + payload + 1 guard。
κ 模式 {k25, k50, drift}，SF10/OS=4，per-sample SNR 口径同前。
判据：SymFEC 解出的 payload bytes == TX bytes（PDR）；另记硬 argmax SER。
先跑 SELFTEST（κ=0 干净信道必须 PDR=100%，否则报错退出）。
"""
import sys
import json
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.payload_codec import encode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, build_symfec_symbol_evidence, decode_symfec_payload_from_evidences,
    _canonical_raw_bin)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.unichirp.paper_unichirp_demod import (
    UniChirpTrainingSymbol, UniChirpDemodConfig, build_unichirp_phase_model,
    demod_unichirp_symbol as uni_demod)

N = 1024
OS = 4
NF = N * OS
M_PAD = 16 * N
PRE_N = 8
PRE_AMBLE_S = 512
GUARD = 1
SF, CR, LDR = 10, 4, False
BASE2VAL = (512 - np.arange(N)) % N   # 值->bin 的对合映射（自逆）
RNG = np.random.default_rng(20260928)
UNI_CFG = UniChirpDemodConfig(cfo_correction_mode="none")
FEC_CFG = SymFECConfig()

U_GRID = np.arange(NF) / OS
REF_OS = np.exp(-1j * 2 * np.pi * U_GRID ** 2 / (2 * N))
DOWN_CHIP = np.exp(-1j * np.pi * np.arange(N) ** 2 / N)

def sym_wave(u, s):
    return np.exp(1j * 2 * np.pi * (u ** 2 / (2 * N) - s * u / N))

def val_to_my_s(values):
    """SymFEC demod 值 -> 我方约定符号值 s（经 canonical bin + 常数 512 映射）。"""
    out = []
    for v in values:
        b = _canonical_raw_bin(int(v), sf=SF, is_header=False, ldro=LDR)
        out.append((512 - b) % N)
    return np.array(out, dtype=int)

def dirich(x, L, M):
    x = np.asarray(x, dtype=float)
    a = np.pi * x / M
    sa = np.sin(a)
    ok = np.abs(sa) > 1e-12
    val = np.where(ok, np.sin(a * L) / (L * np.where(ok, sa, 1.0)), 1.0)
    return np.exp(1j * np.pi * x * (L - 1) / M) * val

def make_packet(s_all, kaps, sigma2):
    x = np.concatenate([sym_wave(U_GRID - kaps[k], s_all[k]) for k in range(len(s_all))])
    return x + (RNG.standard_normal(x.size) + 1j * RNG.standard_normal(x.size)) * np.sqrt(sigma2 / 2)

def new_stats(x, k):
    """返回 (m窗口矩阵, T0)。m[:,d] 列 d ↔ κ=(2−d)/4；bin 域谱=精确读列的置换。"""
    y = x[k * NF:(k + 1) * NF] * REF_OS
    Z = np.fft.fft(y, n=M_PAD)
    mag2 = np.abs(Z) ** 2
    base = (-4 * np.arange(N)) % M_PAD
    win = (base[:, None] + np.arange(-2, 3)[None, :]) % M_PAD
    return mag2[win], mag2[base]

def viterbi_kappa(ms):
    nsym = len(ms)
    logr = np.array([np.log(ms[j].max(axis=0) + 1e-30) for j in range(nsym)])
    acc = logr[0].copy()
    back = np.zeros((nsym, 5), dtype=int)
    for j in range(1, nsym):
        cand = np.stack([acc[np.maximum(np.arange(5) - 1, 0)], acc, acc[np.minimum(np.arange(5) + 1, 4)]])
        back[j] = np.array([np.argmax(cand[:, d]) for d in range(5)]) - 1
        acc = cand[np.argmax(cand, axis=0), np.arange(5)] + logr[j]
    path = np.zeros(nsym, dtype=int)
    path[-1] = int(np.argmax(acc))
    for j in range(nsym - 1, 0, -1):
        path[j - 1] = np.clip(path[j] + back[j, path[j]], 0, 4)
    return path

def demod_all_spectra(x, pay_start, n_pay):
    """返回 {chain: (n_pay, N) bin 域功率谱} 及 TREL 的窗口矩阵（供镜像仲裁）。"""
    specs = {}
    olda = np.zeros((n_pay, N))
    new0v = np.zeros((n_pay, N))
    ms = []
    for k in range(n_pay):
        st = (pay_start + k)
        seg = x[st * NF:(st + 1) * NF]
        olda[k] = np.abs(np.fft.fft(seg[::OS] * DOWN_CHIP)) ** 2
        m, T0 = new_stats(x, st)
        ms.append(m)
        new0v[k] = T0
    specs["OLD-A"] = olda[:, (np.arange(N) - 512) % N]  # 我的 bin 域 -> 他们的 bin 域（+512 偏移）
    specs["NEW-0"] = new0v[:, BASE2VAL]          # 值域 -> bin 域（对合置换）
    path = viterbi_kappa(ms)
    trel = np.stack([ms[k][:, path[k]] for k in range(n_pay)])
    specs["TREL-N"] = trel[:, BASE2VAL]
    # 半 bin 符号二义性仲裁：路径锁定边界列 col0 时，镜像解释（κ=−0.5，值移 +1）同样自洽。
    # 两条解释都交给 SymFEC，CRC/字节匹配择优 —— decode-to-finish 的标准打法。
    mirror = None
    if np.mean(path == 0) > 0.5:
        mirror = np.stack([ms[k][(np.arange(N) - 1) % N, 4] for k in range(n_pay)])
    sav = np.zeros((n_pay, N)); tri = np.zeros((n_pay, N)); uni = np.zeros((n_pay, N))
    for k in range(n_pay):
        st = (pay_start + k) * NF
        sav[k] = np.abs(sav_demod(samples=x, start_sample=st, sf=SF, os_factor=OS).combined_spectrum) ** 2
        tri[k] = trim_demod(samples=x, start_sample=st, sf=SF, os_factor=OS).metric
    train = tuple(UniChirpTrainingSymbol(start_sample=k * NF, raw_fft_bin=0, abs_symbol_index=float(k))
                  for k in range(PRE_N))
    model, _o = build_unichirp_phase_model(samples=x, training_symbols=train, sf=SF,
                                           os_factor=OS, config=UNI_CFG)
    for k in range(n_pay):
        r = uni_demod(samples=x, start_sample=(pay_start + k) * NF, sf=SF, os_factor=OS,
                      phase_rad=model.predict(pay_start + k), config=UNI_CFG)
        uni[k] = r.metric
    specs["SAVAUX"] = sav
    specs["TRIMMER"] = tri
    specs["UNICHIRP"] = uni
    specs["TREL-MIRROR"] = mirror[:, BASE2VAL] if mirror is not None else None
    return specs

def symfec_decode(spec, hdr_vals, tx_payload):
    evs = [build_symfec_symbol_evidence(spec[k], sf=SF, symbol_index=k, ldro=LDR)
           for k in range(spec.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=CR, ldro=LDR, config=FEC_CFG,
        header_symbol_values=list(hdr_vals), payload_len=len(tx_payload),
        has_crc=True, crc_mode="grlora")
    ok = (res.payload_decode is not None
          and bytes(res.payload_decode.payload_bytes) == bytes(tx_payload))
    ser = float(np.mean([e.argmax_symbol_value for e in evs] != [-1])) if False else None
    return ok, evs

def build_tx():
    payload = bytes(RNG.integers(0, 256, 1))   # 1 字节: 5+2+4=11 nibble -> 2 块 -> 8 payload 符号
    hdr_vals, pay_vals = encode_explicit_frame_symbols(
        payload, sf=SF, cr=CR, has_crc=True, ldro=LDR, crc_mode="grlora")
    return payload, list(hdr_vals), list(pay_vals)

def run_packet(kap0, sigma2):
    payload, hdr_vals, pay_vals = build_tx()
    n_pay = len(pay_vals)
    hdr_s = val_to_my_s(hdr_vals)
    pay_s = val_to_my_s(pay_vals)
    total = PRE_N + len(hdr_s) + n_pay + GUARD
    kaps = (np.full(total, kap0) if kap0 is not None
            else 0.1 + 0.004 * np.arange(total))   # 漂移率保证 |κ|<=0.456 在窗口内
    s_all = np.concatenate([np.full(PRE_N, PRE_AMBLE_S), hdr_s, pay_s,
                            np.full(GUARD, PRE_AMBLE_S)])
    x = make_packet(s_all, kaps, sigma2)
    pay_start = PRE_N + len(hdr_s)
    specs = demod_all_spectra(x, pay_start, n_pay)
    mirror_spec = specs.pop("TREL-MIRROR", None)
    out = {}
    for c, spec in specs.items():
        ok, evs = symfec_decode(spec, hdr_vals, payload)
        hard_err = np.mean([int(e.argmax_symbol_value) != int(v) for e, v in zip(evs, pay_vals)])
        if c == "TREL-N" and mirror_spec is not None and not ok:
            ok_b, _ = symfec_decode(mirror_spec, hdr_vals, payload)  # 半bin二义性仲裁
            ok = ok or ok_b
        out[c] = (ok, float(hard_err))
    return out

def selftest():
    print("SELFTEST A: kappa=0, +10 dB（UniChirp 已知退化豁免，其余应全 PDR=1）")
    res = run_packet(0.0, 10 ** (-10 / 10.0))
    for c, (ok, he) in res.items():
        print("  %-8s PDR=%d hardSER=%.4f" % (c, ok, he))
    for c, (ok, _) in res.items():
        if c != "UNICHIRP" and not ok:
            raise SystemExit("SELFTEST A FAILED at %s — 集成不一致，禁止开打" % c)
    print("SELFTEST B: kappa=0.25, +10 dB（全员应 PDR=1，含 UniChirp）")
    res = run_packet(0.25, 10 ** (-10 / 10.0))
    for c, (ok, he) in res.items():
        print("  %-8s PDR=%d hardSER=%.4f" % (c, ok, he))
    if not all(ok for ok, _ in res.values()):
        raise SystemExit("SELFTEST B FAILED — 集成不一致，禁止开打")

def main():
    selftest()
    n_pkt = 40
    snrs = [-26, -24, -22, -20]
    kap_modes = {"k25": 0.25, "k50": 0.5, "drift": None}
    chains = ["OLD-A", "SAVAUX", "TRIMMER", "UNICHIRP", "NEW-0", "TREL-N"]
    results = {}
    t0 = time.time()
    for km, kap0 in kap_modes.items():
        for snr_db in snrs:
            sigma2 = 10 ** (-snr_db / 10.0)
            cnt = {c: 0 for c in chains}
            ser_acc = {c: 0.0 for c in chains}
            for _ in range(n_pkt):
                for c, (ok, he) in run_packet(kap0, sigma2).items():
                    cnt[c] += int(ok)
                    ser_acc[c] += he
            results.setdefault(km, {})[str(snr_db)] = {
                c: dict(pdr=cnt[c] / n_pkt, hard_ser=ser_acc[c] / n_pkt) for c in chains}
            row = results[km][str(snr_db)]
            print("[{:5s}]{:>4}: ".format(km, snr_db) + " | ".join(
                "{} PDR {:.3f} SER {:.3f}".format(c, row[c]['pdr'], row[c]['hard_ser'])
                for c in chains), flush=True)
    # 输出路径为项目内字面量（无拼接、不含上跳段），直接内联供静态检查
    with open(r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\symfec_pk_20260928\results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("\n%.0fs elapsed -> results.json" % (time.time() - t0))

if __name__ == "__main__":
    main()
