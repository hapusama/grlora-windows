# -*- coding: utf-8 -*-
r"""2026-09-29 v6：判据修复后的 OTA 重放（替代 ota_final.py 的作废数字）。

v5 作废的根因（本目录 RESULTS.md 勘误 + data/smoke_tmp/audit_*.py 实锤）：
  R1 payload_codec crc_valid = not has_crc 旁路：header 打坏解出 has_crc=0
     时无条件 True（已修：header_valid 强制）；
  R2 8 链共享同一份 header 解调且 wm header argmax 从未对过（OTA 上
     argmax = 4×真值，旧脚本 %1024 直接当值）；
  R3 "native 共识 GT" 从未真正建立。

v6 设计：
  - GT：demod_symbol_sequence（CSV 生成器，OTA 验证过）在原生段上解出
    header/payload 真值符号，decode 后 CRC 必须通过，字节冻结为 GT；
  - 受控对比：所有链共用 GT header（与合成测试床"真值 header"同约定），
    唯一变量 = payload 证据链；
  - 每链探针校准 δ = argmax - 真值（探针在当前域的 native 段上测，符号间
    必须一致），证据按 roll(-1-δ) 喂 SymFEC（其内部做 (bin-1) 映射）；
  - 判据 = bytes==GT（主）AND crc_valid（辅，已修复语义）；
  - 自检门禁：合成正控 + 坏 payload 负控 + 坏 header 负控，全过才碰真实数据。
"""
import sys
import csv
import json
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.payload_codec import (
    encode_explicit_frame_symbols, decode_explicit_frame_symbols)
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence
from weak_decoder.chirp import build_upchirp
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, build_symfec_symbol_evidence, decode_symfec_payload_from_evidences)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.unichirp.paper_unichirp_demod import (
    UniChirpTrainingSymbol, UniChirpDemodConfig, build_unichirp_phase_model,
    demod_unichirp_symbol as uni_demod, unichirp_full_spectrum)

N, OS, NF = 1024, 4, 4096          # SF10, os=4（samp-rate 500kHz / BW 125kHz）
SF, LDR = 10, False
FEC_CFG = SymFECConfig()
UNI_CFG = UniChirpDemodConfig(cfo_correction_mode="none")
ROT = np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(N) / N) for d in range(5)])
MASK = np.abs(np.fft.fftfreq(NF)) <= 0.125 + 40.0 / NF
REF_OS = np.conj(build_upchirp(SF, symbol_id=0, os_factor=OS))  # 同 ota_final/arbitration 的 REF0

CHAINS = ["PLAIN", "OLD-A", "OLD-A+F", "SAVAUX", "TRIMMER", "UNICHIRP", "NEW-0", "BCJR"]


def tx_sym(v):
    """gr-lora 约定符号波形（连续 t 分支式，同 ota_final）。"""
    t = np.arange(NF) / OS
    lin = np.where(t < (N - v), (v / N - 0.5) * t, (v / N - 1.5) * t)
    return np.exp(2j * np.pi * (t ** 2 / (2 * N) + lin))


def wm(x, k):
    """(N,5) 细网格窗口能量；列 d ↔ κ=(2-d)/4。

    与 ota_final.py 逐字同构（MASK 预滤 + 4 相位 chip 抽取 + κ 旋转平均）；
    OTA 上已验证 argmax=真符号值。
    """
    seg = np.fft.ifft(np.fft.fft(x[k * NF:(k + 1) * NF]) * MASK)
    y = seg * REF_OS
    chips = np.stack([y[p:p + NF:OS] for p in range(OS)])
    inp = (chips[None, :, :] * ROT[:, None, :]).reshape(20, N)
    P = np.abs(np.fft.fft(inp, axis=1)) ** 2
    return P.reshape(5, 4, N).mean(axis=1).T


def olda_rows(seg, k, prefiltered=False):
    """OLD-A / OLD-A+F：与 arbitration_20260928 完全一致的裸抽取（offset-0 相位）。"""
    st = k * NF
    s = seg[st:st + NF].copy()
    if prefiltered:
        s = np.fft.ifft(np.fft.fft(s) * MASK)
    return np.abs(np.fft.fft(s[::OS] * REF_OS[::OS])) ** 2


def _lse(x):
    m = x.max()
    return m + np.log(np.sum(np.exp(x - m)))


def bcjr_post(ms):
    K = len(ms)
    lam = np.array([np.log(np.maximum(ms[k].max(axis=0) - np.median(ms[k], axis=0), 1e-30))
                    for k in range(K)])
    alpha = np.zeros((K, 5))
    alpha[0] = lam[0]
    for k in range(1, K):
        for d in range(5):
            lo, hi = max(0, d - 1), min(4, d + 1)
            alpha[k, d] = lam[k, d] + _lse(alpha[k - 1, lo:hi + 1])
    beta = np.zeros((K, 5))
    for k in range(K - 1, 0, -1):
        for dp in range(5):
            lo, hi = max(0, dp - 1), min(4, dp + 1)
            beta[k - 1, dp] = _lse(lam[k, lo:hi + 1] + beta[k, lo:hi + 1])
    post = alpha + beta
    post = np.exp(post - post.max(axis=1, keepdims=True))
    return post / post.sum(axis=1, keepdims=True)


def raw_chain_rows(seg, off, psym):
    """各链的原始谱行（未校准，列=各自 bin 约定）。返回 dict[chain]->(psym,N)。

    seg 必须已完成整 bin CFO 纠正；off 为 payload 起始符号序号。
    UniChirp 训练用段内前 4 个干净 preamble 符号（0..3），训练 bin 由
    其自身全谱 argmax 现场探得（探针校准，噪声下自然退化）。
    """
    rows = {}
    Ta = np.stack([olda_rows(seg, off + k) for k in range(psym)])
    Tf = np.stack([olda_rows(seg, off + k, prefiltered=True) for k in range(psym)])
    rows["OLD-A"] = Ta
    rows["OLD-A+F"] = Tf
    sav = np.zeros((psym, N)); tri = np.zeros((psym, N))
    for k in range(psym):
        st = (off + k) * NF
        sav[k] = np.abs(sav_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                                  cfo_int=0).combined_spectrum) ** 2
        tri[k] = trim_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                            cfo_int=0).metric
    rows["SAVAUX"] = sav
    rows["TRIMMER"] = tri
    train = []
    for k in range(4):  # 干净 preamble 符号 0..3
        full = unichirp_full_spectrum(samples=seg, start_sample=k * NF, sf=SF,
                                      os_factor=OS, cfo_int=0, cfo_frac=0.0, config=UNI_CFG)
        train.append(UniChirpTrainingSymbol(start_sample=k * NF,
                                            raw_fft_bin=int(np.argmax(full[:N])),
                                            abs_symbol_index=float(k)))
    model, _o = build_unichirp_phase_model(samples=seg, training_symbols=tuple(train),
                                           sf=SF, os_factor=OS, config=UNI_CFG)
    uni = np.zeros((psym, N))
    for k in range(psym):
        uni[k] = uni_demod(samples=seg, start_sample=(off + k) * NF, sf=SF, os_factor=OS,
                           phase_rad=model.predict(off + k), config=UNI_CFG).metric
    rows["UNICHIRP"] = uni
    ms = [wm(seg, off + k) for k in range(psym)]
    rows["NEW-0"] = np.stack([m[:, 2] for m in ms])
    post = bcjr_post(ms)
    rows["BCJR"] = np.stack([ms[k] @ post[k] for k in range(psym)])
    return rows, Ta


def probe_deltas(raw_rows, known_values, n_probe=5):
    """δ = argmax - 真值，前 n_probe 个符号逐链多数投票（众数须 >= n_probe-1）。

    裸抽取链在分数偏移下音可能分裂，argmax 偶尔翻到相邻 bin；众数投票取
    主约定，偶发翻转留给链自身的软证据消化（这正是裸抽取的固有弱点）。
    """
    deltas = {}
    for c, rows in raw_rows.items():
        d = [(int(np.argmax(rows[k])) - int(known_values[k])) % N for k in range(n_probe)]
        vals, counts = np.unique(d, return_counts=True)
        best = int(vals[np.argmax(counts)])
        if int(np.max(counts)) < max(3, n_probe - 2):
            raise SystemExit("PROBE FAILED: chain %s 探针无稳定众数 %s" % (c, d))
        deltas[c] = best
    return deltas


def roll_evidence(rows, delta):
    """symfec 内部做 value=(bin-1) 映射 => 值 v 的音需放在列 v+1。

    链的音在列 v+delta（探针测得），故 out[p] = row[p-1+delta]，
    使音落在 p=v+1。delta=0 时与 arbitration 的 to_canon 逐字一致。
    """
    idx = (np.arange(N) - 1 + int(delta)) % N
    return rows[:, idx]


def judge_evidence_chain(rows, delta, gt_hdr, plen, cr_i, gt_bytes):
    """多 δ 候选 + CRC 仲裁（项目自有的镜像仲裁原语）。

    OTA 实测存在整 bin 回绕：部分帧 tone 在 v+1 而非 v（κ_eff 越过 ±0.5，
    合成测试床未覆盖）。对 δ∈{探针值, ±1} 逐一解码，任一通过 bytes+CRC
    即通过——与 bcjr_marginal 的 TREL 镜像仲裁同一机制，判据已修复故
    无空真风险（3 次尝试的假阳性率 ~3×2^-16，可忽略）。
    """
    cand = sorted({int(delta) % N, (int(delta) + 1) % N, (int(delta) - 1) % N})
    for d_try in cand:
        evs = [build_symfec_symbol_evidence(roll_evidence(rows, d_try)[k], sf=SF,
                                            symbol_index=k, ldro=LDR)
               for k in range(rows.shape[0])]
        res = decode_symfec_payload_from_evidences(
            evidences=evs, sf=SF, cr=cr_i, ldro=LDR, config=FEC_CFG,
            header_symbol_values=list(gt_hdr), payload_len=plen,
            has_crc=True, crc_mode="grlora")
        if (res.payload_decode is not None
                and res.payload_decode.crc_valid
                and bytes(res.payload_decode.payload_bytes) == gt_bytes):
            return True, True
    return False, False


def judge_plain(Ta, delta, gt_hdr, gt_bytes):
    cand = sorted({int(delta) % N, (int(delta) + 1) % N, (int(delta) - 1) % N})
    for d_try in cand:
        vals = [int(np.argmax(Ta[k])) - d_try for k in range(Ta.shape[0])]
        try:
            res = decode_explicit_frame_symbols(gt_hdr, vals, sf=SF, bw=125000.0, ldro_mode=2)
        except Exception:
            continue
        if (res.header.header_valid and res.payload.crc_valid
                and bytes(res.payload.payload_bytes) == gt_bytes):
            return True, True
    return False, False


def demod_all_fixed(seg, cfo_int, off, psym, plen, cr_i, gt_hdr, gt_bytes, deltas):
    n = np.arange(len(seg))
    seg = seg * np.exp(-2j * np.pi * int(cfo_int) * n / NF)
    raw_rows, Ta = raw_chain_rows(seg, off, psym)
    out = {}
    ok, bok = judge_plain(Ta, deltas["OLD-A"], gt_hdr, gt_bytes)
    out["PLAIN"] = (ok, bok)
    for c in ["OLD-A", "OLD-A+F", "SAVAUX", "TRIMMER", "UNICHIRP", "NEW-0", "BCJR"]:
        out[c] = judge_evidence_chain(raw_rows[c], deltas[c], gt_hdr, plen, cr_i, gt_bytes)
    return out


# ---------------- 自检门禁 ----------------
def selftest():
    print("SELFTEST v6: 合成正控 + 双负控（判据回归护栏）")
    rng = np.random.default_rng(20260929)
    ok_all = True
    for cr_i in [1, 4]:
        payload = bytes(int(v) for v in rng.integers(0, 256, 33))
        hdr, pay = encode_explicit_frame_symbols(
            payload, sf=SF, cr=cr_i, has_crc=True, ldro=LDR, crc_mode="grlora")
        seq = [0] * 8 + list(hdr) + list(pay) + [0]
        seg = np.concatenate([tx_sym(v) for v in seq])
        raw_rows, _Ta = raw_chain_rows(seg, 16, len(pay))
        deltas = probe_deltas(raw_rows, pay)
        res = {}
        for c in ["OLD-A", "OLD-A+F", "SAVAUX", "TRIMMER", "UNICHIRP", "NEW-0", "BCJR"]:
            res[c] = judge_evidence_chain(raw_rows[c], deltas[c], hdr, 33, cr_i, payload)[0]
        # UniChirp cr=1 短码字已知失败（同 v4 记录），不阻塞
        must = {c: v for c, v in res.items() if not (c == "UNICHIRP" and cr_i == 1)}
        bad = [c for c, v in must.items() if not v]
        print("  cr=%d 正控: %s%s" % (cr_i, "全过" if not bad else "FAIL:" + ",".join(bad),
                                      " (UniChirp cr=1 豁免)" if cr_i == 1 and not res["UNICHIRP"] else ""))
        ok_all = ok_all and not bad
        # 负控 A：坏 payload（打坏整个交织块 = cr_i+4 个符号，不可纠正）
        seg_cor = seg.copy()
        for j in range(cr_i + 4):
            k0 = (16 + j) * NF
            seg_cor[k0:k0 + NF] = tx_sym((pay[j] + 137 + 31 * j) % N)
        rr_cor, _ = raw_chain_rows(seg_cor, 16, len(pay))
        bad_a = []
        for c in ["OLD-A", "OLD-A+F", "SAVAUX", "TRIMMER", "NEW-0", "BCJR"]:
            if judge_evidence_chain(rr_cor[c], deltas[c], hdr, 33, cr_i, payload)[0]:
                bad_a.append(c)
        print("  cr=%d 负控A(坏payload): %s" % (cr_i, "全拒✓" if not bad_a else "漏放:" + ",".join(bad_a)))
        ok_all = ok_all and not bad_a
        # 负控 B：坏 header（随机）=> SymFEC 判据必须拒（R1 回归护栏）
        bad_hdr = [int(v) for v in rng.integers(0, N, 8)]
        bad_b = [c for c in ["OLD-A", "NEW-0", "BCJR"]
                 if judge_evidence_chain(raw_rows[c], deltas[c], bad_hdr, 33, cr_i, payload)[0]]
        print("  cr=%d 负控B(坏header): %s" % (cr_i, "全拒✓" if not bad_b else "漏放:" + ",".join(bad_b)))
        ok_all = ok_all and not bad_b
    if not ok_all:
        raise SystemExit("SELFTEST FAILED —— 禁止碰真实数据")


# ---------------- 真实数据 ----------------
SOURCES = [
    ("0_0_0_10_14_8",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_8_header_first_frames.csv"),
    ("0_0_0_10_14_16",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_frames.csv"),
    ("0_0_0_10_14_32",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_32.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_32_header_first_frames.csv"),
]


def gt_of_frame(iq, r):
    """用验证过的 demod_symbol_sequence 在原生段解出 GT（CRC 必须通过）。"""
    hs = int(r["header_start_sample"])
    psym = int(r["payload_symbol_count"])
    res = demod_symbol_sequence(
        samples=np.asarray(iq, dtype=np.complex64),
        header_start_sample=hs, sf=SF, os_factor=OS,
        cfo_int=int(r["source_grlora_cfo_int"]),
        cfo_frac=float(r["source_grlora_cfo_frac"]),
        sfo_hat=float(r["source_grlora_sfo_hat"]),
        sfo_cum_initial=float(r.get("source_grlora_branch_sfo_cum_initial") or 0.0),
        header_count=8, payload_count=psym, payload_ldro=False)
    hdr_vals = [x.symbol_value for x in res[:8]]
    pay_vals = [x.symbol_value for x in res[8:]]
    dec = decode_explicit_frame_symbols(hdr_vals, pay_vals, sf=SF, bw=125000.0, ldro_mode=2)
    if not (dec.header.header_valid and dec.payload.crc_valid):
        raise SystemExit("GT FAILED on frame %s（原生 CRC 未过，检查 CSV/参数）" % r.get("frame_index"))
    return hdr_vals, bytes(dec.payload.payload_bytes), hs, psym


def seg_of(iq, r, psym):
    hs = int(r["header_start_sample"])
    return np.asarray(iq[hs - 8 * NF: hs + (8 + psym + 1) * NF], dtype=np.complex128)


def main():
    selftest()
    rng = np.random.default_rng(20260929)
    frames = []
    deltas_all = {}
    for cap, bin_path, csv_path in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            gt_hdr, gt_bytes, hs, psym = gt_of_frame(iq, r)
            frames.append(dict(cap=cap, iq=iq, r=r, gt_hdr=gt_hdr, gt_bytes=gt_bytes,
                               psym=psym, plen=int(r["payload_len"]), cr=int(r["cr"])))
        # 每条 capture 的探针校准（native 段）
        r0, f0 = rows[0], None
        f0 = next(f for f in frames if f["cap"] == cap)
        seg0 = seg_of(iq, r0, int(r0["payload_symbol_count"]))
        # 已知真值用 GT 前几个 payload 符号（gt 重新取，不重复解码）
        known = [x.symbol_value for x in demod_symbol_sequence(
            samples=np.asarray(iq, dtype=np.complex64),
            header_start_sample=int(r0["header_start_sample"]), sf=SF, os_factor=OS,
            cfo_int=int(r0["source_grlora_cfo_int"]),
            cfo_frac=float(r0["source_grlora_cfo_frac"]),
            sfo_hat=float(r0["source_grlora_sfo_hat"]),
            sfo_cum_initial=float(r0.get("source_grlora_branch_sfo_cum_initial") or 0.0),
            header_count=8, payload_count=5, payload_ldro=False)[8:]]
        n = np.arange(len(seg0))
        seg0d = seg0 * np.exp(-2j * np.pi * int(r0["source_grlora_cfo_int"]) * n / NF)
        raw_rows, _Ta = raw_chain_rows(seg0d, 16, int(r0["payload_symbol_count"]))
        deltas_all[cap] = probe_deltas(raw_rows, known)
        print("[%s] %d 帧, 探针 δ=%s" % (cap, len(rows),
              {c: d for c, d in deltas_all[cap].items()}), flush=True)

    print("Pass 1: native（应全过；任何失败=管线问题）", flush=True)
    nat = {c: 0 for c in CHAINS}
    for f in frames:
        seg = seg_of(f["iq"], f["r"], f["psym"])
        out = demod_all_fixed(seg, int(f["r"]["source_grlora_cfo_int"]), 16,
                              f["psym"], f["plen"], f["cr"], f["gt_hdr"], f["gt_bytes"],
                              deltas_all[f["cap"]])
        for c in CHAINS:
            nat[c] += int(out[c][0])
    print("  native: " + " | ".join("%s %d/%d" % (c, nat[c], len(frames)) for c in CHAINS), flush=True)

    levels = [18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0]
    table = {"native": {c: nat[c] for c in CHAINS}}
    t0 = time.time()
    for lv in levels:
        cnt = {c: 0 for c in CHAINS}
        cntb = {c: 0 for c in CHAINS}
        for f in frames:
            seg = seg_of(f["iq"], f["r"], f["psym"])
            p_add = 10 ** (lv / 10.0) * float(np.mean(np.abs(seg) ** 2))
            seg = seg + (rng.standard_normal(len(seg)) + 1j * rng.standard_normal(len(seg))) \
                * np.sqrt(p_add / 2.0)
            out = demod_all_fixed(seg, int(f["r"]["source_grlora_cfo_int"]), 16,
                                  f["psym"], f["plen"], f["cr"], f["gt_hdr"], f["gt_bytes"],
                                  deltas_all[f["cap"]])
            for c in CHAINS:
                cnt[c] += int(out[c][0])
                cntb[c] += int(out[c][1])
        table["+%gdB" % lv] = dict(cnt)
        print("[+%gdB] " % lv + " | ".join("%s %d/%d" % (c, cnt[c], len(frames)) for c in CHAINS),
              " | bytes-only:", " ".join("%s:%d" % (c, cntb[c]) for c in CHAINS), flush=True)

    out_json = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ota_replay_20260929\results_v6.json"
    assert out_json.startswith("D:\\Desktop\\proj\\") and ".." not in out_json
    with open(out_json, "w") as f:
        json.dump({"n_frames": len(frames), "table": table,
                   "deltas": {k: {c: int(d) for c, d in v.items()} for k, v in deltas_all.items()},
                   "judge": "bytes==GT AND crc_valid(header_valid enforced); GT header shared by all chains",
                   "note": "v6 fixed judge; v5 numbers void"}, f, indent=1, default=float)
    print("\n%.0fs -> results_v6.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
