# -*- coding: utf-8 -*-
"""M4（2026-10-04 深夜）：turbo 相位列 × DeRa demod Pass1 —— 增量测量。

问题：M3 链（A=DeRa demod 优先/B=TREL/CRC 仲裁）无任何相位利用；
e3 已证 turbo 相位在 A 基线上 PER 捕获 55~61%（71:0 零败北）。本实验
量：**Pass1 换成更强的 DeRa demod 行后，turbo 在其残差帧上还能收回
多少 PER**——该数字直接决定 M3+ 终判形态（是否加 D 列）。

口径：实验B（先验给定，e1 对齐段/注噪/测量与 e2/e3 逐字节一致，
种子派生同式 ⇒ 与 e3 的 D 臂结果可配对）；协议 v1.1 §5A：诊断级。
臂：
  A      = e2 非相干基线行（回归锚）
  DERA   = DeRa port stage1 相干行（per-frame 干净 δ 冻结，同协议）
  D|A    = turbo 相位（Pass1=A 行，= e3 臂 D 复算）
  D|DERA = turbo 相位（Pass1=DERA 行，本轮主问题）
  CHAIN  = DERA → CRC 过即用；不过 → D|DERA → CRC（M3+ 形态）
→ m4_units.jsonl + stdout 汇总
"""
import json
import os
import sys
import time

import numpy as np
from multiprocessing import Pool

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C
import e2_arm as E2
import e3_common as D3
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator

HERE = os.path.dirname(os.path.abspath(__file__))
OUTU = os.path.join(HERE, "m4_units.jsonl")
IN = os.path.join(HERE, "e1_native.jsonl")
SNRS = (None, -22, -24, -26)
N = 1024
NF = 4096

_W = {}


def dera_rows(port, seg, fr, dsel):
    """DeRa port coherent 行（demod_payload，窗口 = e1 payload 窗整数重合），
    per-frame 干净 δ 冻结 roll，log 域归一（同 arm_A_twin 行口径）。"""
    P, psym = fr["P"], fr["psym"]
    need = int(round((P + 13 + psym) * NF)) + NF
    if len(seg) < need:
        return None
    _s1, coh = port.demod_payload(seg, P + 13, psym)
    rows = np.log(coh + 1e-30)
    rows -= rows.max(axis=1, keepdims=True)
    if dsel:
        rows = np.roll(rows, -dsel, axis=1)
    return rows


def calib_dsel(port, seg0, fr):
    """干净信号上冻结 DERA 行的常数 δ（payload argmax−gt 众数）。"""
    rows = dera_rows(port, seg0, fr, 0)
    if rows is None:
        return 0
    gt = np.asarray(fr["gt"])
    d = (np.argmax(rows, 1) - gt) % N
    d[d > N // 2] -= N
    return int(np.bincount(d + N // 2).argmax()) - N // 2


def run_unit(u):
    gid, snr, seed = u["gid"], u["snr"], u["seed"]
    fr, seg0, ds = _W["frames"][gid], _W["segs"][gid], _W["ds"][gid]
    port = _W["port"]
    seg = (C.inject_noise(seg0, snr, seed, gid, fr["S"], fr["N0"])
           if snr is not None else seg0)
    mt = E2.unit_measure(seg, fr, ds)
    if mt is None:
        return dict(u, skip="extract_fail")
    gt = np.asarray(fr["gt"])
    ntot = fr["P"] + 2 + fr["psym"]
    dummy_g = dict(phi_hat=np.zeros(0), tau_hat=99.0)   # 诊断门永不过
    rowsA, _ = E2.arm_A_twin(mt)
    rowsDERA = dera_rows(port, seg, fr, _W["dsel"][gid])
    out = dict(gid=gid, snr=snr, seed=seed)

    def ev(rows):
        return dict(sym_err=int((np.argmax(rows, 1) != gt).sum()),
                    crc_ok=bool(E2.judge(rows, fr["gt_hdr"], fr["plen"],
                                         fr["cr"])))

    out["A"] = ev(rowsA)
    if rowsDERA is None:
        out["skip"] = "dera_window"
        return out
    out["DERA"] = ev(rowsDERA)
    rowsDA, _ = D3.arm_D_turbo(mt, fr, dummy_g, rowsA)
    out["D_A"] = ev(rowsDA)
    rowsDD, _ = D3.arm_D_turbo(mt, fr, dummy_g, rowsDERA)
    out["D_DERA"] = ev(rowsDD)
    out["CHAIN"] = dict(crc_ok=bool(out["DERA"]["crc_ok"]
                                    or out["D_DERA"]["crc_ok"]))
    return out


def init_worker(gids):
    E2.build_static()
    _W["port"] = DeRaDemodulator(sf=10, os_factor=4)
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    keep = [fr for fr in frames if fr["gid"] in gids]
    caps = sorted(set(fr["cap"] for fr in keep))
    _W["frames"] = {fr["gid"]: fr for fr in keep}
    _W["segs"] = {}
    _W["ds"] = {}
    _W["dsel"] = {}
    for cap in caps:
        src = next(s for s in C.SF10_SOURCES + C.SF11_SOURCES if s[0] == cap)
        ds = C.DS(src[3])
        iq = np.memmap(src[1], dtype=np.complex64, mode="r")
        for fr in (f for f in keep if f["cap"] == cap):
            r = dict(header_start_sample=fr["hs"],
                     source_grlora_cfo_int=fr["cfo_int"],
                     source_grlora_cfo_frac=str(fr["cfo_frac"]),
                     source_grlora_payload_sto_frac=str(fr["sto_frac"]))
            seg, _i0, _bo = C.align_seg(iq, r, ds, fr["P"], fr["psym"])
            _W["segs"][fr["gid"]] = seg
            _W["ds"][fr["gid"]] = ds
        del iq
    for gid, fr in _W["frames"].items():
        _W["dsel"][gid] = calib_dsel(_W["port"], _W["segs"][gid], fr)


def main():
    n_seed_cap = int(sys.argv[1]) if len(sys.argv) >= 2 else 5
    t0 = time.time()
    E2.build_static()
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    frames = [f for f in frames if f["sf"] == 10]
    by_cap = {}
    for fr in frames:
        by_cap.setdefault(fr["cap"], []).append(fr["gid"])
    units = [dict(gid=g, snr=s, seed=sd)
             for gids in by_cap.values() for g in gids
             for s in SNRS for sd in range(min(n_seed_cap, C.N_SEEDS))]
    done = set()
    if os.path.exists(OUTU):
        for l in open(OUTU, encoding="utf-8"):
            try:
                r = json.loads(l)
                done.add((r["gid"], r["snr"], r["seed"]))
            except Exception:
                pass
        print("断点恢复：已完成 %d 单元" % len(done), flush=True)
    todo = [u for u in units if (u["gid"], u["snr"], u["seed"]) not in done]
    print("待跑 %d / %d 单元" % (len(todo), len(units)), flush=True)
    ctx = Pool(len(by_cap), initializer=init_worker,
               initargs=(list({g for gids in by_cap.values() for g in gids}),))
    n = 0
    with open(OUTU, "a", encoding="utf-8") as fh:
        try:
            for r in ctx.imap_unordered(run_unit, todo, chunksize=1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                n += 1
                if n % 60 == 0:
                    print("%d/%d (%.0fs)" % (n, len(todo), time.time() - t0),
                          flush=True)
        finally:
            ctx.close()
            ctx.join()
    # ---- 汇总 ----
    agg = {}
    for l in open(OUTU, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("skip"):
            continue
        k = "native" if r["snr"] is None else int(r["snr"])
        a = agg.setdefault(k, dict(n=0, dera_fail=0, chain_fail=0,
                                   a_fail=0, da_fail=0, dd_fail=0,
                                   rec=0, hurt=0))
        a["n"] += 1
        a["dera_fail"] += int(not r["DERA"]["crc_ok"])
        a["chain_fail"] += int(not r["CHAIN"]["crc_ok"])
        a["a_fail"] += int(not r["A"]["crc_ok"])
        a["da_fail"] += int(not r["D_A"]["crc_ok"])
        a["dd_fail"] += int(not r["D_DERA"]["crc_ok"])
        if not r["DERA"]["crc_ok"]:
            a["rec"] += int(r["D_DERA"]["crc_ok"])
            a["hurt"] += int(r["A"]["crc_ok"] and not r["D_DERA"]["crc_ok"])
    print("== 汇总（n=%d 种子）==" % n_seed_cap)
    print("snr      n   |  PER:  A   DERA   D|A   D|D  CHAIN | DERA失败帧上D|D回收")
    for k in sorted(agg, key=lambda x: (x != "native", x)):
        a = agg[k]
        print("%-7s %4d |      %.3f    %.3f     %.3f    %.3f   %.3f  |  %d/%d"
              % (k, a["n"], a["a_fail"] / a["n"], a["dera_fail"] / a["n"],
                 a["da_fail"] / a["n"], a["dd_fail"] / a["n"],
                 a["chain_fail"] / a["n"], a["rec"], a["dera_fail"]))
    print("用时 %.0fs → %s" % (time.time() - t0, OUTU))


if __name__ == "__main__":
    main()
