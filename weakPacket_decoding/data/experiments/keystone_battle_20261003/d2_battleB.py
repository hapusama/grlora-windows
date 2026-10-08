# -*- coding: utf-8 -*-
"""D2 Battle B：漂移域（induced-SFO）检测战——三臂 × δ × SNR。

臂：① DeRa（port Eq.47 统计量，恒定列假设=物理律#5 受害者）
    ② OURS cert-q（机制级：GT 锚定二次模板，冻结固定线性变换，闭式 CFAR）
    ③ OURS dep（可部署：非相干 keystone 粗捕获锚 + δ 网格 + κ-FFT，无 θ，
       门限 MC——论文系统节的数字）
δ 档：{0(控制), 0.005, 0.01, 0.02, 0.04, 0.082} bin/符（ε=δ/N 重采样注入）
SNR 档：−26..−38（2dB 步，7 档）；H0：{−28,−34,−38}×4 窗/帧；20 种子。
协议：E2 已获用户批准（2026-10-03 指令）；其余铁律与 d1_battle 一致——
同一 (帧,δ,档,种子) 加噪喂三臂；H0=纯合成 AWGN（带外底标定，§0）；
GT 模板在【注入后干净信号】上逐 (帧,δ) 冻结（GT 锚定，实验B 哲学）；
种子常数 20261003（派生式与 d1 同构 +δ 因子）。JSONL 断点续跑。
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C
import d1_battle as A
import d2_core as D

NF = 4096
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "d2_battleB_checkpoint.jsonl")
SEED_CONST = 20261003
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.082)
LEVELS = [-26, -28, -30, -32, -34, -36, -38]
H0_LEVELS = [-28, -34, -38]
N_SEEDS = 20
N_H0_CENTERS = 4

G = {}


def field_origin(lead, pre):
    return lead - (pre + 4.25) * NF


def inj_cache(fi):
    """(帧, δ) 注入段 + cert-q 模板 + SNR 口径（FIFO 40 项）。"""
    frames = G["frames"]
    key = (fi, G["di"])
    cch = G.setdefault("cache", {})
    if key not in cch:
        f = frames[fi]
        pre = f["pre"]
        eps = D.eps_of_delta(DELTAS[G["di"]])
        inj = D.resample_sfo(f["seg"], field_origin(f["lead"], pre), eps)
        tm = D.clean_template_q(inj, f["lead"], pre)
        cch[key] = (inj, tm, A.snr_parts(inj))
        if len(cch) > 40:
            for k in list(cch)[:len(cch) - 40]:
                del cch[k]
    return cch[key]


def add_noise_inj(inj, S, N0, level, seed, salt, di):
    rng = np.random.default_rng((SEED_CONST * 7919
                                 + (int(level) + 100) * 131
                                 + seed * 17 + salt * 7919
                                 + di * 104729) % (2 ** 31))
    p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
    return inj + ((rng.standard_normal(len(inj))
                   + 1j * rng.standard_normal(len(inj)))
                  * np.sqrt(p_add / 2.0))


def run_h1_unit(u):
    di, level, seed = u
    G["di"] = di
    frames = G["frames"]
    out = []
    for fi, f in enumerate(frames):
        inj, tm, (S, N0) = inj_cache(fi)
        seg = add_noise_inj(inj, S, N0, level, seed, f["hs"] % 4099, di)
        sc, kap, _ = C.score_cert_batch(seg[None, :], f["lead"], f["pre"], tm)
        sd, kd, _ = C.score_dera_batch(seg[None, :], f["lead"], f["pre"], tm)
        sp, dhat, khp, _ = D.score_dep_batch(seg[None, :], f["lead"], f["pre"],
                                             tm)
        out.append(dict(kind="h1", di=di, delta=DELTAS[di], level=level,
                        seed=seed, frame=fi, pre=f["pre"],
                        cert=float(sc[0]), dera=float(sd[0]),
                        dep=float(sp[0]), dhat=float(dhat[0]),
                        khat=float(khp[0])))
    return out


def run_h0_unit(u):
    di, level, seed = u
    G["di"] = di
    frames = G["frames"]
    out = []
    for fi, f in enumerate(frames):
        inj, tm, (S, N0) = inj_cache(fi)   # 模板与 SNR 口径；噪声纯合成
        L = len(inj)
        rng0 = np.random.default_rng((SEED_CONST * 131
                                      + (int(level) + 100) * 17 + seed * 31
                                      + f["hs"] % 7919
                                      + di * 104729) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        sig2_tot = N0 + p_add
        ws = []
        for k in range(N_H0_CENTERS):
            ws.append((rng0.standard_normal(L) + 1j * rng0.standard_normal(L))
                      * np.sqrt(sig2_tot / 2.0))
        sub = np.stack(ws)
        sc, _, _ = C.score_cert_batch(sub, f["lead"], f["pre"], tm)
        sd, _, _ = C.score_dera_batch(sub, f["lead"], f["pre"], tm)
        sp, _, _, _ = D.score_dep_batch(sub, f["lead"], f["pre"], tm)
        for j in range(sub.shape[0]):
            out.append(dict(kind="h0", di=di, delta=DELTAS[di], level=level,
                            seed=seed, frame=fi, pre=f["pre"],
                            cert=float(sc[j]), dera=float(sd[j]),
                            dep=float(sp[j])))
        del ws, sub
    return out


def init_worker():
    G["frames"] = A.build_frames()


def main():
    t0 = time.time()
    print("Battle B：δ 档 %s｜H1 档 %s｜种子 %d"
          % (list(DELTAS), LEVELS, N_SEEDS), flush=True)
    done_jobs = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("kind") in ("h1done", "h0done"):
                    done_jobs.add((r["kind"], r["di"], r["level"], r["seed"]))
            except Exception:
                pass
        print("断点恢复：%d 完成作业" % len(done_jobs), flush=True)
    h1_jobs = [(di, lv, sd) for di in range(len(DELTAS))
               for lv in LEVELS for sd in range(N_SEEDS)]
    h0_jobs = [(di, lv, sd) for di in range(len(DELTAS))
               for lv in H0_LEVELS for sd in range(N_SEEDS)]
    todo_h1 = [u for u in h1_jobs if ("h1done",) + u not in done_jobs]
    todo_h0 = [u for u in h0_jobs if ("h0done",) + u not in done_jobs]
    print("待跑 H1 %d / H0 %d 作业" % (len(todo_h1), len(todo_h0)), flush=True)

    ctx = mp.get_context("spawn")
    n_written = 0
    with open(CKPT, "a", encoding="utf-8") as fh:
        with ctx.Pool(processes=6, initializer=init_worker) as pool:
            for tag, jobs, fn in (("h1", todo_h1, run_h1_unit),
                                  ("h0", todo_h0, run_h0_unit)):
                for i, recs in enumerate(pool.imap_unordered(fn, jobs,
                                                             chunksize=1)):
                    for r in recs:
                        fh.write(json.dumps(r) + "\n")
                    fh.write(json.dumps(dict(kind=tag + "done", di=jobs[i][0],
                                             level=jobs[i][1],
                                             seed=jobs[i][2])) + "\n")
                    fh.flush()
                    n_written += 1
                    if n_written % 30 == 0:
                        print("%s %d/%d (%.0fs)" % (tag, i + 1, len(jobs),
                                                    time.time() - t0),
                              flush=True)
    print("完成 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
