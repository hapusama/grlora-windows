# -*- coding: utf-8 -*-
"""DeRa port 修复探针 1：实测 trimmer 几何的 argmax 映射与 F/T 相位结构。

问题：
  P1: 候选 k 几何下，真符号 c 的 metric 峰在哪（k=c? c+1? N-c?）
  P2: 真候选处 front/tail 投影的相对相位 gamma(c, kappa) 是什么结构
      —— kappa 的贡献是公共相位（Stage-2 ML 可吸收）还是依赖 c（需闭式）
"""
import sys
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.chirp import build_upchirp
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    build_loratrimmer_matrices, loratrimmer_metric_from_symbol)

SF, N, OS = 10, 1024, 4
NF = N * OS
mat = build_loratrimmer_matrices(SF, OS)


def probe(c, kappa):
    sig = build_upchirp(SF, c, OS).astype(np.complex128)
    n = np.arange(NF)
    sig = sig * np.exp(2j * np.pi * kappa * n / NF)
    metric, front, tail, _m = loratrimmer_metric_from_symbol(
        sig, SF, OS, return_projections=True)
    am = int(np.argmax(metric))
    gamma = float(np.angle(tail[am] * np.conj(front[am])))
    return am, gamma


print("== P1: argmax mapping (c -> argmax) at kappa=0 ==")
for c in [0, 1, 2, 7, 100, 300, 511, 512, 513, 700, 1022, 1023]:
    am, _ = probe(c, 0.0)
    print("c=%4d -> argmax=%4d   (c+1=%4d, N-1-c=%4d)" % (c, am, (c + 1) % N, (N - 1 - c) % N))

print()
print("== P2: gamma(c, kappa) at the TRUE-symbol argmax ==")
cs = [0, 1, 100, 300, 511, 512, 513, 700, 1022, 1023]
kappas = [0.0, 0.1, 0.25, 0.5]
table = {}
for c in cs:
    row = []
    for kap in kappas:
        am, gamma = probe(c, kap)
        row.append((am, gamma))
    table[c] = row
    print("c=%4d: " % c + "  ".join("kap=%.2f am=%4d gamma=%+.3f" % (k, a, g)
                                    for (k, (a, g)) in zip(kappas, row)))

print()
print("== P3: kappa-induced phase shift delta_gamma(c) = gamma(c,kap)-gamma(c,0) ==")
for c in cs:
    base = table[c][0][1]
    print("c=%4d: " % c + "  ".join("%+.3f" % (g - base) for (_a, g) in table[c][1:]))
