# -*- coding: utf-8 -*-
"""D1 Battle A：DeRa 主场（无漂移）——OURS 12.25 全场 keystone CFAR vs
DeRa 检测 port（v2 Eq.47 统计量），同 FAR 精确分位数 ROC。

协议（EXPERIMENT_PROTOCOL v1.0）：
  - AWGN 同一实现喂两臂（同一 (帧,档,种子) 噪声）；整包 SNR 口径
    （带外底标定逐包 P_add）；native 档含；
  - GT 锚定窗（实验B 哲学，ff_runner 先例）：模板在干净信号逐帧冻结
    （ν0/o_j/θ/conj/δ̂），两臂同一份；DeRa 臂自带带噪 argmax 列搜索
    （其统计量设计原样，仅 σ̂² 用冻结列远子格[稳定化方向，声明]）；
  - H0：帧间纯噪声窗（避所有帧 ±2(P+8)NF）+ 同款加噪；模板取对应帧
    （cert 统计量 = 固定线性变换 ⇒ H0 分布参数无关，池化合法）；
  - 种子派生常数 20261003（ff_runner 同构）。
单位：H1 = 28 帧 × {native,−16..−24} × 20 种子；H0 = 2240 窗/档
（28 帧 × 20 种子 × 4 窗），池化 11200（cert 自归一 ⇒ H0 参数无关，
池化合法；U4 合成 1e5 MC 验证闭式）。JSONL 断点续跑（作业级标记）。
"""
import csv
import json
import multiprocessing as mp
import os
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # 小矩阵多线程 BLAS 在本机抖动（37x 0.1s vs 17s）

import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C

SF, N, NF = 10, 1024, 4096
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "d1_battle_checkpoint.jsonl")
SEED_CONST = 20261003
LEVELS = [None] + [-v for v in (16, 18, 20, 22, 24, 26, 28, 30,
                                32, 34, 36, 38)]
N_SEEDS = 20
N_H0_CENTERS = 4
HF = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
      r"\header_first")
USRP = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ"
SOURCES = [(p, "%s/0_0_0_10_14_%d.bin" % (USRP, p),
            "%s/0_0_0_10_14_%d_header_first_frames.csv" % (HF, p))
           for p in (8, 16, 32)]

G = {}


def snr_parts(seg):
    X = np.fft.fftshift(np.fft.fft(seg))
    p = np.abs(X) ** 2 / len(seg)
    f = np.fft.fftshift(np.fft.fftfreq(len(seg))) * 500000.0
    ob = (np.abs(f) > 70000) & (np.abs(f) < 240000)
    n0 = float(np.mean(p[ob]))
    total = float(np.mean(np.abs(seg) ** 2))
    return max(total - n0, 1e-30), n0


def build_frames():
    frames = []
    for pre, bin_path, csv_path in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1"
                and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            hs = int(r["header_start_sample"])
            lead = (pre + 6) * NF
            seg = np.asarray(iq[hs - lead:hs + 8 * NF], dtype=np.complex128)
            tm = C.clean_template(seg, lead, pre)
            frames.append(dict(pre=pre, hs=hs, iq=iq, lead=lead,
                               seg=seg, tm=tm, snr=snr_parts(seg)))
    by_cap = {}
    for i, f in enumerate(frames):
        by_cap.setdefault(f["pre"], []).append(f["hs"])
    for f in frames:
        f["all_hs"] = by_cap[f["pre"]]
    return frames


def add_noise(seg, S, N0, level, seed, salt):
    if level is None:
        return seg
    rng = np.random.default_rng((SEED_CONST * 7919
                                 + (int(level) + 100) * 131
                                 + seed * 17 + salt * 7919) % (2 ** 31))
    p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
    return seg + ((rng.standard_normal(len(seg))
                   + 1j * rng.standard_normal(len(seg)))
                  * np.sqrt(p_add / 2.0))


def h0_centers(f, n, level, seed):
    rng = np.random.default_rng((SEED_CONST * 31 + (int(level) + 100) * 7
                                 + seed * 13 + f["hs"] % 9973) % (2 ** 31))
    guard = 100 * NF   # 帧总前向延伸（payload ~45NF）+裕量，防 H0 窗落帧内
    span = (f["pre"] + 8) * NF
    lo, hi = span, len(f["iq"]) - span
    out, tries = [], 0
    while len(out) < n and tries < 50000:
        tries += 1
        c = int(rng.integers(lo, hi))
        if all(abs(c - h) > guard for h in f["all_hs"]):
            out.append(c)
    return out


def run_h1_unit(u):
    level, seed = u
    frames = G["frames"]
    out = []
    for fi, f in enumerate(frames):
        seg = add_noise(f["seg"], f["snr"][0], f["snr"][1], level, seed,
                        f["hs"] % 4099)
        sc, kap, _ = C.score_cert_batch(seg[None, :], f["lead"], f["pre"],
                                        f["tm"])
        sd, kd, _ = C.score_dera_batch(seg[None, :], f["lead"], f["pre"],
                                       f["tm"])
        out.append(dict(kind="h1", level=level, seed=seed, frame=fi,
                        cert=float(sc[0]), dera=float(sd[0]),
                        kappa=float(kap[0]), kappa_d=float(kd[0])))
    return out


def run_h0_unit(u):
    """H0 = 纯 AWGN 窗（协议 §0：噪声有且仅有 AWGN）。

    发现声明：capture 帧间段含强非噪声突发（连续 RSSI 发射变体，窗功率
    可达帧功率 86×，见战报），纯噪声 H0 = 合成复高斯，底噪 = 该帧带外底
    N0，加噪同款到档位 ⇒ H1 噪声密度与 H0 完全一致。"""
    level, seed = u
    frames = G["frames"]
    out = []
    for fi, f in enumerate(frames):
        L = f["lead"] + 8 * NF
        rng0 = np.random.default_rng((SEED_CONST * 131
                                      + (int(level) + 100) * 17 + seed * 31
                                      + f["hs"] % 7919) % (2 ** 31))
        n0 = f["snr"][1]
        p_add = max(f["snr"][0] / 10 ** (level / 10.0) - n0, 1e-30)
        sig2_tot = n0 + p_add
        ws = []
        for k in range(N_H0_CENTERS):
            ws.append((rng0.standard_normal(L) + 1j * rng0.standard_normal(L))
                      * np.sqrt(sig2_tot / 2.0))
        sub = np.stack(ws)
        sc, _, _ = C.score_cert_batch(sub, f["lead"], f["pre"], f["tm"])
        sd, _, _ = C.score_dera_batch(sub, f["lead"], f["pre"], f["tm"])
        for j in range(sub.shape[0]):
            out.append(dict(kind="h0", level=level, seed=seed, frame=fi,
                            cert=float(sc[j]), dera=float(sd[j])))
        del ws, sub
    return out


def init_worker():
    G["frames"] = build_frames()


def main():
    t0 = time.time()
    print("冻结帧集/模板…", flush=True)
    frames = build_frames()
    print("帧数 %d（%.0fs）" % (len(frames), time.time() - t0), flush=True)
    done_jobs = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("kind") in ("h1done", "h0done"):
                    done_jobs.add((r["kind"], r["level"], r["seed"]))
            except Exception:
                pass
        print("断点恢复：%d 完成作业" % len(done_jobs), flush=True)
    h1_jobs = [(lv, sd) for lv in LEVELS for sd in
               (range(1) if lv is None else range(N_SEEDS))]
    h0_jobs = [(lv, sd) for lv in LEVELS[1:] for sd in range(N_SEEDS)]
    todo_h1 = [u for u in h1_jobs if ("h1done", u[0], u[1]) not in done_jobs]
    todo_h0 = [u for u in h0_jobs if ("h0done", u[0], u[1]) not in done_jobs]
    print("待跑 H1 作业 %d / H0 作业 %d" % (len(todo_h1), len(todo_h0)),
          flush=True)

    ctx = mp.get_context("spawn")
    with open(CKPT, "a", encoding="utf-8") as fh:
        with ctx.Pool(processes=6, initializer=init_worker) as pool:
            for tag, jobs, fn in (("h1", todo_h1, run_h1_unit),
                                  ("h0", todo_h0, run_h0_unit)):
                for i, recs in enumerate(pool.imap_unordered(fn, jobs,
                                                             chunksize=1)):
                    for r in recs:
                        fh.write(json.dumps(r) + "\n")
                    fh.write(json.dumps(dict(kind=tag + "done",
                                             level=jobs[i][0],
                                             seed=jobs[i][1])) + "\n")
                    fh.flush()
                    if (i + 1) % 10 == 0:
                        print("%s %d/%d (%.0fs)" % (tag, i + 1, len(jobs),
                                                    time.time() - t0),
                              flush=True)
    print("完成 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
