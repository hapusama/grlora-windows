# -*- coding: utf-8 -*-
"""G1 前置快验：frame0 native 全网格（65 点），验证
1) (ν,τ)=(0,0) TREL SER=0（与实验B 一致）
2) 净偏移方向：比较 (0.2,0)/(0,0.2)/(0.2,-0.2)/(0.2,0.2) 的 SER
   ——若 ν(chip)≡ν(bin) 同向，则 (0.2,-0.2) 好、(0.2,0.2) 最差
3) 逐符号轨迹字段完整"""
import sys
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dts_residual_20261004")
import g1_map_runner as G1
import dts_runner as D

D.init_worker()
G1.LEVEL = None  # native 快验
print("== (0,0) 一致性 ==")
r = G1.run_unit((0, 0, 0.0, 0.0))
print("trel ser=%.3f crc=%d | plain ser=%.3f" %
      (r["chains"]["trel"]["ser"], r["chains"]["trel"]["crc"], r["chains"]["plain"]["ser"]))

print("\n== 方向性探针（native, TREL SER）==")
for nu, tau in [(0.2, 0.0), (0.0, 0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)]:
    r = G1.run_unit((0, 0, nu, tau))
    print("nu=%+.2f tau=%+.2f (net=%+.2f): trel ser=%.3f crc=%d plain ser=%.3f" %
          (nu, tau, nu + tau, r["chains"]["trel"]["ser"], r["chains"]["trel"]["crc"],
           r["chains"]["plain"]["ser"]))
