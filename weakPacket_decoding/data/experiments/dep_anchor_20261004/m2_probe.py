# -*- coding: utf-8 -*-
"""M2 探针：①锚质量对照（dep2 相干锚 vs dep-old 功率质心锚 vs cert GT）
native+各 SNR+漂移；②o_sfd 域；③δ̂ 伪峰分布（native+漂移）；
④确定性抽查（cert/dera 与 d2_battleB 老 checkpoint 逐位一致）。
→ m2_probe.json
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_core as C
import d1_battle as A
import d2_core as D
import m2_core as M

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_CONST = 20261003
DELTAS = (0.0, 0.02, 0.082)
LEVELS = (None, -26, -30, -34, -36)


def add_noise(seg, S, N0, level, seed, salt, di):
    rng = np.random.default_rng((SEED_CONST * 7919
                                 + (int(level) + 100) * 131
                                 + seed * 17 + salt * 7919
                                 + di * 104729) % (2 ** 31))
    p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
    return seg + ((rng.standard_normal(len(seg))
                   + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0))


def main():
    frames = A.build_frames()
    res = {}

    # ---- o_sfd 域（GT 模板 28 帧）----
    alls = []
    for f in frames:
        tm = f["tm"]
        alls += [tm["o"][j] for j in range(f["pre"] + 2, f["pre"] + 5)]
    res["o_sfd_range"] = dict(lo=float(min(alls)), hi=float(max(alls)),
                              n=len(alls))

    # ---- 锚质量 + δ̂（dep2 vs dep-old vs cert）----
    anch = {}
    for di, dl in enumerate(DELTAS):
        for level in LEVELS:
            for seed in (0, 1, 2):
                for fi, f in enumerate(frames):
                    pre = f["pre"]
                    eps = D.eps_of_delta(dl)
                    inj = D.resample_sfo(f["seg"],
                                         f["lead"] - (pre + 4.25) * M.NF,
                                         eps) if dl != 0.0 else f["seg"]
                    S, N0 = A.snr_parts(inj)
                    seg = inj if level is None else add_noise(
                        inj, S, N0, level, seed, f["hs"] % 4099, di)
                    seg = seg[None, :]
                    tmq = D.clean_template_q(inj, f["lead"], pre)
                    acq = M.acquire(seg, f["lead"], pre)
                    c_e = acq["c_e"]
                    nu_gt = tmq["nu0"] + c_e * tmq["delta"]
                    e2 = abs(acq["nu0h"][0] - nu_gt)
                    sc, dh, kh, ph, dg = M.score_dep2(seg, f["lead"], pre,
                                                      ret_diag=True)
                    # dep-old 锚（d2 功率质心）
                    _, _, _, _, anc = D.score_dep_batch(
                        seg, f["lead"], pre, tmq, ret_anchor=True)
                    eo = abs(anc["nu0h"][0]
                             - (tmq["nu0"] + D._centroids(pre)[0]
                                * tmq["delta"]))
                    k_ = "δ=%g lv=%s" % (dl, level)
                    r = anch.setdefault(k_, dict(nu2=[], nuo=[], dhat=[],
                                                 acq=[]))
                    r["nu2"].append(e2)
                    r["nuo"].append(eo)
                    r["dhat"].append(dh[0] - dl)
                    r["acq"].append(1.0 if e2 < 0.5 else 0.0)
    for k, r in anch.items():
        nu2, nuo = np.array(r["nu2"]), np.array(r["nuo"])
        dh = np.array(r["dhat"])
        anch[k] = dict(
            n=len(nu2),
            dep2_nu_p50=float(np.quantile(nu2, 0.5)),
            dep2_nu_p90=float(np.quantile(nu2, 0.9)),
            dep2_nu_mean=float(nu2.mean()),
            depold_nu_p50=float(np.quantile(nuo, 0.5)),
            depold_nu_p90=float(np.quantile(nuo, 0.9)),
            depold_nu_mean=float(nuo.mean()),
            acq_ok=float(np.mean(r["acq"])),
            dhat_bias=float(np.median(dh)),
            dhat_mad=float(np.median(np.abs(dh - np.median(dh)))),
            dhat_p90abs=float(np.quantile(np.abs(dh), 0.9)),
            dhat_frac_gt008=float(np.mean(np.abs(dh) > 0.08)))
    res["anchor_quality"] = anch

    # ---- 确定性抽查：cert/dera 3 单元 vs 老 checkpoint ----
    old = {}
    for line in open(os.path.join(HERE, "..", "keystone_battle_20261003",
                                  "d2_battleB_checkpoint.jsonl"),
                     encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") == "h1" and (r["di"], r["level"], r["seed"]) \
                in ((0, -30, 0), (3, -34, 1), (5, -26, 2)):
            old[(r["di"], r["level"], r["seed"], r["frame"])] = r
    det = []
    for (di, level, seed) in ((0, -30, 0), (3, -34, 1), (5, -26, 2)):
        for fi, f in enumerate(frames):
            key = (di, level, seed, fi)
            if key not in old:
                continue
            dl = (0.0, 0.005, 0.01, 0.02, 0.04, 0.082)[di]
            pre = f["pre"]
            inj = D.resample_sfo(f["seg"],
                                 f["lead"] - (pre + 4.25) * M.NF,
                                 D.eps_of_delta(dl)) if dl else f["seg"]
            S, N0 = A.snr_parts(inj)
            seg = add_noise(inj, S, N0, level, seed, f["hs"] % 4099, di)
            tmq = D.clean_template_q(inj, f["lead"], pre)
            sc, _, _ = C.score_cert_batch(seg[None, :], f["lead"], pre, tmq)
            sd, _, _ = C.score_dera_batch(seg[None, :], f["lead"], pre, tmq)
            det.append(dict(key=list(key),
                            cert_rel=float(sc[0] / old[key]["cert"]),
                            dera_rel=float(sd[0] / old[key]["dera"])))
    res["determinism"] = det
    res["determinism_max_rel_dev"] = float(max(
        max(abs(d["cert_rel"] - 1), abs(d["dera_rel"] - 1)) for d in det))

    json.dump(res, open(os.path.join(HERE, "m2_probe.json"), "w"), indent=1)
    print(json.dumps(res["determinism_max_rel_dev"]))
    for k, v in anch.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                  for kk, vv in v.items()})
    print("→ m2_probe.json")


if __name__ == "__main__":
    main()
