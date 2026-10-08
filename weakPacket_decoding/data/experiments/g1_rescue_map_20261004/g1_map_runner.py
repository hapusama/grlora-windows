# -*- coding: utf-8 -*-
r"""G1 同步救援地图（关口1+2 最小版，2026-10-04）。

问题（用户路线 2026-10-04）：弱信号下前导同步参数的不确定性中，哪些真正
影响 payload 解码？固定解码器，对同一批包做受控同步参数干预：
  - A 臂 = 前导估计参数（冻结先验，= 地图 (ν,τ)=(0,0) 点）
  - B 臂 = 有限邻域搜索（地图上 CRC 通过点；救回 = A fail 而 B 有 pass）
  - C 臂 = 已知内容拟合（地图上 GT-SER 最小点 = 同步信息完美的解码潜力上界；
    离线诊断参照，非算法性能上界，报数必须带此声明）
一次扫描出三臂。同时输出逐符号错误轨迹与 top1-top2 margin（关口2 的
"为什么"证据：错误位置/分数差/漂移趋势）。

网格：ν_frac ∈ ±0.30 bin（步 0.05，13 点）× τ ∈ ±0.5 chip（步 0.25，5 点）。
噪声：−20 dB 工作区 × 3 种子（关口3 最小版=跨噪声稳定性）。
铁律：AWGN 整包口径；残差后加噪；所有臂同一实现；GT/δ 冻结（实验B 规则）；
CRC 误接受风险由 28 帧 × 逐符号轨迹交叉核对（RESULTS 报数时说明）。
解码器固定 = TREL-5（kappa_trellis 语义，与 dts_runner 同实现）；
PLAIN（argmax）同行计算作对照（≈等预算朴素基线的地图）。
"""
import sys
import json
import os
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dts_residual_20261004")
import dts_runner as D  # 复用：build_frames / apply_residual / wm / viterbi_path / fast_evidence / judge_crc_fast

SF, N, OS, NF = D.SF, D.N, D.OS, D.NF

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\g1_rescue_map_20261004"
CKPT = os.path.join(EXP_DIR, os.environ.get("G1_CKPT", "checkpoint.jsonl"))

LEVEL = int(os.environ.get("G1_LEVEL", "-20"))
SEEDS = [0, 1, 2]
NUS = [round(-0.30 + 0.05 * i, 2) for i in range(13)]   # ±0.30 bin
TAUS = [-0.5, -0.25, 0.0, 0.25, 0.5]                    # ±0.5 chip（=±0.5 bin 等效）


def trel_plain_rows(seg, psym):
    """固定解码器：TREL-5（κ 格 Viterbi）+ PLAIN（argmax 对照）。返回行与 margin。"""
    ms = [D.wm(seg, 16 + k) for k in range(psym)]
    lam = np.array([np.log(np.maximum(m.max(axis=0) - np.median(m, axis=0), 1e-30))
                    for m in ms])
    path = D.viterbi_path(lam)
    trel = np.stack([ms[k][:, path[k]] for k in range(psym)])
    olda = np.stack([D.olda_rows(seg, 16 + k) for k in range(psym)])
    # margin：TREL 支路内 top1-top2（dB），逐符号
    margins = []
    for row in trel:
        s = np.sort(row)[::-1]
        margins.append(float(10.0 * np.log10(max(s[0], 1e-30) / max(s[1], 1e-30))))
    return trel, olda, margins


def run_unit(u):
    seed, fi, nu, tau = u
    f = D.G["frames"][fi]
    seg = D.apply_residual(f["seg"], "nu", nu)          # ν 先（线性，顺序无碍）
    if tau != 0.0:
        seg = D.apply_residual(seg, "tau", tau)
    if LEVEL is not None:
        rng = np.random.default_rng((20261004 * 7919 + (LEVEL + 100) * 131
                                     + seed * 17 + fi) % (2 ** 31))
        p_add = max(f["S"] / 10 ** (LEVEL / 10.0) - f["N0"], 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    trel, olda, margins = trel_plain_rows(seg, f["psym"])
    out = {}
    for name, rc in (("trel", trel), ("plain", olda)):
        cc = f["delta"]
        hard = [(int(np.argmax(rc[k])) - cc) % N for k in range(f["psym"])]
        err = [int(h != g) for h, g in zip(hard, f["gt"])]
        if name == "plain":
            try:
                dec = D.decode_explicit_frame_symbols(f["gt_hdr"], hard, sf=SF,
                                                      bw=125000.0, ldro_mode=2)
                crc = bool(dec.header.header_valid and dec.payload.crc_valid)
            except Exception:
                crc = False
        else:
            crc = D.judge_crc_fast(rc, cc, f["gt_hdr"], f["plen"], f["cr"])
        out[name] = {"ser": sum(err) / f["psym"], "crc": int(crc),
                     "err_mask": err, "first_err": (err.index(1) if 1 in err else -1)}
    return {"frame": fi, "seed": seed, "nu": nu, "tau": tau,
            "psym": f["psym"], "margins": [round(m, 2) for m in margins],
            "chains": out}


def main():
    t0 = time.time()
    os.makedirs(EXP_DIR, exist_ok=True)
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["seed"], r["frame"], r["nu"], r["tau"]))
            except Exception:
                pass
        print("断点恢复：已有 %d 单元" % len(done), flush=True)

    units = [(sd, fi, nu, tau) for sd in SEEDS for fi in range(28)
             for nu in NUS for tau in TAUS
             if (sd, fi, nu, tau) not in done]
    print("待跑 %d 单元（%d workers）" % (len(units), os.cpu_count()), flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, os.cpu_count() - 1),
                  initializer=D.init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units, chunksize=4)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 500 == 0:
                    print("  %d/%d (%.0fs)" % (i + 1, len(units), time.time() - t0),
                          flush=True)

    # 汇总：A/B/C 三臂 + 方向性（净偏移 ν+τ 直方图，方向由数据说话）
    maps = {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        maps.setdefault((r["seed"], r["frame"]), {})[(r["nu"], r["tau"])] = r
    n_pk = len(maps)
    a_fail = b_rescue = c_pot = 0
    net_hist = {}
    for (sd, fi), g in sorted(maps.items()):
        a = g.get((0.0, 0.0))
        if a is None:
            continue
        a_crc = a["chains"]["trel"]["crc"]
        b_pts = [k for k, v in g.items() if v["chains"]["trel"]["crc"]]
        c_key = min(g, key=lambda k: g[k]["chains"]["trel"]["ser"])
        if not a_crc:
            a_fail += 1
            if b_pts:
                b_rescue += 1
            if g[c_key]["chains"]["trel"]["ser"] < 0.5:
                c_pot += 1
        for (nu, tau) in b_pts:
            net = round(nu - tau, 2)   # 快验实测：τ(chip) 等效频移与 ν 反向，(ν,τ)=(.2,.2) 互消后成功
            net_hist[net] = net_hist.get(net, 0) + 1
    print("\n[G1 三臂汇总] 单元(seed×frame)=%d  @%ddB" % (n_pk, LEVEL))
    print("  A臂(前导参数) CRC fail: %d (%.1f%%)" % (a_fail, 100 * a_fail / n_pk))
    print("  B臂(邻域搜索) 救回: %d / %d fail (%.1f%%)" %
          (b_rescue, a_fail, 100 * b_rescue / max(a_fail, 1)))
    print("  C臂(GT拟合 SER<0.5) 潜力: %d / %d fail (%.1f%%)" %
          (c_pot, a_fail, 100 * c_pot / max(a_fail, 1)))
    print("  成功点净偏移 ν−τ 直方图: %s" %
          dict(sorted(net_hist.items())))
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()
