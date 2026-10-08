# -*- coding: utf-8 -*-
"""探针：坏帧定位——argmax 锚定拟合 vs GT 锚定拟合 vs CSV。"""
import importlib.util
import csv
import sys

import numpy as np

BATTLE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
          r"\data\experiments\dera_battle_20260929")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


br = _load("battle_runner", BATTLE + r"\battle_runner.py")

from weak_decoder.decoding.gamma_link import GammaLinkDemodulator

GAM = GammaLinkDemodulator(br.SF, br.OS)
N = br.N
frames = br.build_frames()
cfo_list = []
for cap, bp, cp in br.SOURCES:
    for r in csv.DictReader(open(cp, encoding="utf-8")):
        if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0:
            cfo_list.append(int(r["source_grlora_cfo_int"])
                            + float(r["source_grlora_cfo_frac"].split("|")[0]))
print("idx  SER    kappa0(argmax)  kappa(GT截距)  CSVfrac   drift")
for i, f in enumerate(frames):
    rows = GAM.demod_payload(f["seg"], 16, f["psym"])
    d = int(f["delta"])
    hard = [(int(np.argmax(rows[k])) - d) % N for k in range(f["psym"])]
    e = sum(int(h != g) for h, g in zip(hard, f["gt"]))
    fit = GAM.last_fit
    res = []
    for k, g in enumerate(f["gt"]):
        s = (16 + k) * GAM.nf
        w = np.asarray(f["seg"][s:s + GAM.nf], dtype=np.complex64)
        res.append(np.angle((GAM._tail @ w)[g]
                            * np.conj((GAM._front @ w)[g])) / np.pi)
    res = np.array(res)
    med = float(np.median(res))
    res = ((res - med + 0.5) % 1.0) - 0.5 + med
    sl, ic = np.polyfit(np.arange(len(res), dtype=float), res, 1)
    if e > 2 or i < 3:
        print("%2d  %3d/%d  %+9.4f  %+9.4f  %+8.4f  %+.5f"
              % (i, e, f["psym"], fit["kappa0"], ic, cfo_list[i] % 1.0,
                 fit["drift"]))
