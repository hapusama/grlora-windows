# -*- coding: utf-8 -*-
r"""SUR-Lite：轻量完整弱包同步解码链（在线版，2026-10-05）。

设计（用户指令 2026-10-05：重心放 decoding，不纠结 CFAR/重检测）：
  前端（轻量）= 帧位置 + 前导参数（实验B 型先验冻结，同 G1）
  候选调度    = net 等价类静态序（创新点 2：能量层结构；不依赖该帧
                解码结果的固定序，公平）
  解码引擎    = DERA port v2 两段相干合并（最强引擎，DeRa 的肩膀）
  仲裁        = CRC16 早退（判据已过负控）
四臂一次跑出（同一次在线循环内按停机位统计）：
  A0      = 第 1 候选（锚点，= DeRa Stages 1-2）
  STAGE3  = 锚点 + |net| 序前 5 候选（DeRa Stage 3 代理，对其有利）
  SUR13   = 序前 13（本链 v0 默认预算）
  SUR65   = 全 65（上界）

协议合规：实验B 型（先验给定，隔离检测——检测损失由产线/另线报告）；
AWGN 整包口径；GT/δ 冻结；噪声种子公式与 g1_dera_grid 完全一致 ⇒
−24 档结果应与 checkpoint_dera_m24.jsonl 逐位吻合（自校验）。
native 冒烟：A0 CRC 应 ≈100%（DERA native SER .002）。
"""
import sys
import json
import os
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dts_residual_20261004")
import dts_runner as D

SF, N, OS, NF = D.SF, D.N, D.OS, D.NF

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\g1_rescue_map_20261004"
NUS = [round(-0.30 + 0.05 * i, 2) for i in range(13)]
TAUS = [-0.5, -0.25, 0.0, 0.25, 0.5]
SEEDS = [0, 1, 2]
LEVELS = [None, -20, -22, -24]

# net 等价类静态候选序（只依赖格点集合，数据无关）
_CLS = {}
for _nu in NUS:
    for _tau in TAUS:
        _CLS.setdefault(round(_nu - _tau, 2), []).append((_nu, _tau))
_REPS = {e: min(v, key=lambda p: abs(p[0]) + abs(p[1])) for e, v in _CLS.items()}
ORDER = [(0.0, 0.0)]
for _e in sorted(_CLS, key=lambda x: (abs(x), abs(_REPS[x][0]))):
    ORDER.append(_REPS[_e])
    ORDER += sorted([p for p in _CLS[_e] if p != _REPS[_e]],
                    key=lambda p: abs(p[0]) + abs(p[1]))
K_A0, K_S3, K_SUR = 1, 6, 13      # A0 / 锚点+5 / 默认预算


def decode_at(f, seg, nu, tau):
    s = D.apply_residual(seg, "nu", nu)
    if tau != 0.0:
        s = D.apply_residual(s, "tau", tau)
    _s1, cohd = D.DERA.demod_payload(s, 16, f["psym"])
    cc = f["delta"]
    hard = [(int(np.argmax(cohd[k])) - cc) % N for k in range(f["psym"])]
    ser = sum(int(h != g) for h, g in zip(hard, f["gt"])) / f["psym"]
    return int(D.judge_crc_fast(cohd, cc, f["gt_hdr"], f["plen"], f["cr"])), ser


def run_unit(u):
    lv, seed, fi = u
    f = D.G["frames"][fi]
    seg = f["seg"]
    if lv is not None:
        rng = np.random.default_rng((20261004 * 7919 + (int(lv) + 100) * 131
                                     + seed * 17 + fi) % (2 ** 31))
        p_add = max(f["S"] / 10 ** (lv / 10.0) - f["N0"], 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    t0 = time.perf_counter()
    trials = 0
    stop_at = None      # 第一个 CRC 通过的候选序号（1-based；None=全败）
    ser_at = {}
    for p in ORDER:
        crc, ser = decode_at(f, seg, p[0], p[1])
        trials += 1
        ser_at[p] = ser
        if crc:
            stop_at = trials
            break
    wall = time.perf_counter() - t0
    return {"level": lv, "seed": seed, "frame": fi,
            "stop_at": stop_at, "trials": trials, "wall": round(wall, 3),
            "budget_used": trials, "ser0": ser_at.get((0.0, 0.0)),
            "ser_stop": ser_at.get(ORDER[(stop_at or trials) - 1], None)}


def main():
    t0 = time.time()
    ckpt = os.path.join(EXP_DIR, "sur_lite_online.jsonl")
    done = set()
    if os.path.exists(ckpt):
        for line in open(ckpt, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["level"], r["seed"], r["frame"]))
            except Exception:
                pass
    units = [(lv, sd, fi) for lv in LEVELS for sd in SEEDS for fi in range(28)
             if (lv, sd, fi) not in done]
    print("待跑 %d 单元（每单元在线解码≤65 候选，CRC 早退）" % len(units), flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, os.cpu_count() - 1),
                  initializer=D.init_worker) as pool:
        with open(ckpt, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units, chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 100 == 0:
                    print("  %d/%d (%.0fs)" % (i + 1, len(units), time.time() - t0),
                          flush=True)

    # 汇总四臂战表
    recs = [json.loads(l) for l in open(ckpt, encoding="utf-8")]
    print("\n[SUR-Lite 在线战表]（每档 %d 单元；恢复=stop_at≤预算）" %
          (len(recs) // len(LEVELS)))
    print("  %-7s %-8s %-9s %-9s %-9s | %-8s %-7s" %
          ("档", "A0", "STAGE3", "SUR13", "SUR65", "E[试验]", "单包耗时"))
    for lv in LEVELS:
        sel = [r for r in recs if r["level"] == lv]
        n = len(sel)
        if not n:
            continue
        a0 = 100 * sum(1 for r in sel if r["stop_at"] and r["stop_at"] <= K_A0) / n
        s3 = 100 * sum(1 for r in sel if r["stop_at"] and r["stop_at"] <= K_S3) / n
        s13 = 100 * sum(1 for r in sel if r["stop_at"] and r["stop_at"] <= K_SUR) / n
        s65 = 100 * sum(1 for r in sel if r["stop_at"]) / n
        et = float(np.mean([r["trials"] for r in sel]))
        wt = float(np.mean([r["wall"] for r in sel]))
        print("  %-7s %-8.1f %-9.1f %-9.1f %-9.1f | %-8.1f %.2fs" %
              ("native" if lv is None else lv, a0, s3, s13, s65, et, wt))
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()
