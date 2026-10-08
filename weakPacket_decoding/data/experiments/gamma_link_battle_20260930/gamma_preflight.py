# -*- coding: utf-8 -*-
"""γ-链 v3 preflight：native 门禁（≤2/980 错）+ Q1 诊断（δ̂ 分布、κ̂ 精度、ρ）。

Q1 判定（神招辩论 E1）：|δ̂|<0.01 bins/符 的帧占 ≥90% → 漂移支柱移出
默认帧叙事（默认帧上 γ-链 ≈ DeRa + 量化项，+0.2dB 级）。
用法：D:\mysoft2\miniconda3\envs\gr-lora\python.exe gamma_preflight.py
"""
import importlib.util
import sys

import numpy as np

HERE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
        r"\data\experiments\gamma_link_battle_20260930")
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

print("building frames ...", flush=True)
frames = br.build_frames()
print("%d frames" % len(frames), flush=True)

# CSV cfo（对齐前）→ 对齐后残余 κ ≈ frac(cfo)
import csv
cfo_list = []
for cap, bin_path, csv_path in br.SOURCES:
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0:
            cfo_list.append(int(r["source_grlora_cfo_int"])
                            + float(r["source_grlora_cfo_frac"].split("|")[0]))

tot = err = 0
drifts, k_err, rhos, gt_res = [], [], [], []
for i, f in enumerate(frames):
    rows = GAM.demod_payload(f["seg"], 16, f["psym"])
    d = int(f["delta"])
    hard = [(int(np.argmax(rows[k])) - d) % N for k in range(f["psym"])]
    err += sum(int(h != g) for h, g in zip(hard, f["gt"]))
    tot += f["psym"]
    fit = GAM.last_fit
    drifts.append(fit["drift"])
    k_err.append(((fit["kappa0"] - (cfo_list[i] % 1.0) + 0.5) % 1.0) - 0.5)
    rhos.append(fit["rho_mean"])
    # GT 锚定 γ（Q1 原设计：恒等式在真候选上的逐符轨迹）
    F = GAM._front
    T = GAM._tail
    res = []
    for k, g in enumerate(f["gt"]):
        s = (16 + k) * GAM.nf
        w = np.asarray(f["seg"][s:s + GAM.nf], dtype=np.complex64)
        zgt = (T @ w)[g] * np.conj((F @ w)[g])
        res.append(np.angle(zgt) / np.pi)
    res = np.array(res)
    med = float(np.median(res))
    res_w = ((res - med + 0.5) % 1.0) - 0.5 + med
    t_ax = np.arange(len(res_w), dtype=float)
    sl, ic = np.polyfit(t_ax, res_w, 1)
    gt_res.append(float(np.std(res_w - (sl * t_ax + ic))))
    if i < 4:
        print("  frame %2d: κ̂0=%+.4f δ̂=%+.5f ρ̄=%.3f GTγ残差σ=%.4f  SER %d/%d"
              % (i, fit["kappa0"], fit["drift"], fit["rho_mean"],
                 gt_res[-1],
                 sum(int(h != g) for h, g in zip(hard, f["gt"])), f["psym"]))

drifts_a = np.abs(np.array(drifts))
print("\nNATIVE：GAMMA %d/%d = %.4f（门禁 ≤0.002）" % (err, tot, err / tot))
print("Q1 诊断：|δ̂| 中位 %.5f，90 分位 %.5f，最大 %.5f bins/符"
      % (np.median(drifts_a), np.percentile(drifts_a, 90), drifts_a.max()))
print("  |δ̂|<0.01 帧占比 %.1f%%（E1 门：≥90%% → 漂移支柱移出默认帧叙事）"
      % (100.0 * np.mean(drifts_a < 0.01)))
print("κ̂0 vs CSV frac CFO：|err| 中位 %.4f max %.4f"
      % (np.median(np.abs(k_err)), np.max(np.abs(k_err))))
print("GT 锚定 γ 轨迹线性残差 σ：中位 %.4f max %.4f bins（恒等式+漂移模型在真实数据上的误差）"
      % (np.median(gt_res), np.max(gt_res)))
print("ρ̄（自标定工作点）: 中位 %.3f min %.3f（s→0=DeRa, s→∞=Trimmer）"
      % (np.median(rhos), np.min(rhos)))
