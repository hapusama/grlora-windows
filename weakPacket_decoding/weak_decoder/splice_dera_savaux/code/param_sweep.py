# -*- coding: utf-8 -*-
"""网格参数平台扫描（红队二轮：PEN 扫平只卸了一个旋钮）。

扫细步长 × 副本间距（5 配置），档 {−22,−24} × seeds {0,1} × 28 帧。
对照基线 = step 0.125 / spacing 0.75（v4-b 配置）。判据：PER/SER 在
参数面上平坦（平台非尖峰）。并行 4 workers。
"""
import sys
import numpy as np
import importlib.util as ilu

sr_spec = ilu.spec_from_file_location(
    "sr", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
          r"\splice_dera_savaux\code\splice_runner.py")
sr = ilu.module_from_spec(sr_spec)
sr_spec.loader.exec_module(sr)

NF, N, OS, SF_ = sr.NF, sr.N, sr.OS, sr.SF
G = {}

CONFIGS = [
    ("base  s=.125 c=.75", 0.125, 0.75),
    ("fine  s=.0625 c=.75", 0.0625, 0.75),
    ("coarse s=.25 c=.75", 0.25, 0.75),
    ("dense s=.125 c=.5", 0.125, 0.5),
    ("sparse s=.125 c=1.0", 0.125, 1.0),
]


def make_grids(step, spacing):
    span = 0.375
    fine = np.arange(-span, span + step / 2, step)
    n_c = int(np.floor(2 * (1.875 - span) / spacing)) + 1
    centers = np.linspace(-(1.875 - span) - spacing * (n_c - 1) / 2 + 0,
                          (1.875 - span) + spacing * (n_c - 1) / 2, n_c)
    centers = np.round(centers / (spacing / 2)) * (spacing / 2)
    centers = np.unique(centers)
    return centers, fine


def savt2_gen(seg_a, pay0, psym, centers, fine, pen=0.2):
    abs_d = np.concatenate([c + fine for c in centers])
    n_d = len(abs_d)
    E = np.empty((psym, n_d))
    rows_all = np.empty((psym, n_d, N))
    for j, d in enumerate(abs_d):
        seg_d = seg_a if d == 0.0 else sr.frac_delay(seg_a, float(d))
        for i in range(psym):
            res = sr.sav_demod(samples=seg_d, start_sample=(pay0 + i) * NF,
                               sf=SF_, os_factor=OS, cfo_int=0)
            p = np.abs(res.combined_spectrum.astype(np.complex128)) ** 2
            rows_all[i, j] = p
            E[i, j] = np.log(np.max(p) + 1e-30)
    outs = []
    n_f = len(fine)
    n_c = len(centers)
    for k in range(n_c):
        sl = slice(k * n_f, (k + 1) * n_f)
        Ek = E[:, sl]
        score = Ek[0].copy()
        back = np.zeros((psym, n_f), dtype=int)
        for i in range(1, psym):
            ns = np.full(n_f, -1e30)
            bk = np.zeros(n_f, dtype=int)
            for s in range(n_f):
                best, arg = -1e30, s
                for s0 in (s - 1, s, s + 1):
                    if 0 <= s0 < n_f:
                        vv = score[s0] - pen * abs(s - s0)
                        if vv > best:
                            best, arg = vv, s0
                ns[s], bk[s] = best, arg
            score = ns + Ek[i]
            back[i] = bk
        path = np.zeros(psym, dtype=int)
        path[-1] = int(np.argmax(score))
        for i in range(psym - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        outs.append((float(score[path[-1]]),
                     rows_all[:, sl][np.arange(psym), path]))
    outs.sort(key=lambda t: -t[0])
    return [o[1] for o in outs]


def init_worker():
    sr.init_worker()


def run_unit(u):
    lv, sd, fi = u
    f = sr.G["frames"][fi]
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    S, N0 = sr.G["snr"][fi]
    rng = np.random.default_rng((20260930 * 7919 + (lv + 100) * 131
                                 + sd * 17 + fi * 7919) % (2 ** 31))
    p_add = max(S / 10 ** (lv / 10.0) - N0, 1e-30)
    seg = seg + (rng.standard_normal(len(seg))
                 + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
    if not cands:
        return None
    out = {}
    for name, step, spacing in CONFIGS:
        centers, fine = make_grids(step, spacing)
        outs = savt2_gen(seg_as[0], pay0s[0], f["psym"], centers, fine)
        _d, ok, rows = sr.decode_chain(None, f, top5=outs)
        hard = [(int(np.argmax(rows[k])) - _d) % N for k in range(f["psym"])]
        err = sum(int(h != g) for h, g in zip(hard, f["gt"]))
        out[name] = (err, int(not ok))
    return out


def main():
    units = [(lv, sd, fi) for lv in (-22, -24) for sd in (0, 1)
             for fi in range(28)]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    agg = {}
    n_by = {}
    with ctx.Pool(processes=4, initializer=init_worker) as pool:
        for res in pool.imap_unordered(run_unit, units, chunksize=1):
            if res is None:
                continue
            for name, (err, fail) in res.items():
                a = agg.setdefault(name, [0, 0])
                a[0] += err
                a[1] += fail
                n_by[name] = n_by.get(name, 0) + 1
    n = max(n_by.values())
    den = n * 35
    print("平台扫描（−22/−24 合并，n=%d 单元）：" % n)
    for name, _s, _c in CONFIGS:
        err, fail = agg[name]
        print("  %-20s SER=%.4f  PER=%.3f" % (name, err / den, fail / n))


if __name__ == "__main__":
    main()
