# -*- coding: utf-8 -*-
"""DTS-1 前置快验：帧0，native，ε∈{0, 0.3}，串行。通过标准：
ε=0 TREL SER=0（实验B 一致性）；ε=0.3 各链 SER 显著>0（注入有效）。"""
import sys
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dts_residual_20261004")
import dts_runner as D

D.init_worker()
for eps in (0.0, 0.3):
    r = D.run_unit(("nu", eps, None, 0, 0))
    print("nu ε=%.2f native frame0:" % eps,
          {c: "%d/%d crc_fail=%d" % (v["sym_err"], r["sym_tot"], v["crc_fail"])
           for c, v in r["chains"].items()})
