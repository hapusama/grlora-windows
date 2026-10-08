# 方法 — DeRa 前端 × Savaux 内核的拼接、失效机制与 fd 修复

## 1. 系统矩阵（7 链）

| 链 | 前端 | 解调 | 角色 |
|---|---|---|---|
| DERA×SAVAUX | DeRa 检测 port（top-5 候选） | Savaux Eq.34-37 相干分支合并 | **主角**：原样拼接 |
| DERAfd×SAVAUX | DeRa + 前导能量分数时延精化 | 同上 | **修复臂**（混血+，须标注非 DeRa 原生） |
| DERA×TRIMMER | DeRa | LoRaTrimmer | front battle 参照（逐位复现锚点） |
| DERA×TREL-5 | DeRa | 我方 κ 格 Viterbi | 参照 |
| DERA×DERA | DeRa | DeRa 解码器 v2 port | **全 DeRa** = 当前最强系统 |
| OURS×SAVAUX | 我方产线前端 | Savaux | 前端对照 |
| PRIOR×SAVAUX | 干净 CSV 全先验（oracle 标签） | Savaux | 诊断天花板（非参赛链，§5A-4 分报） |

所有链共享：同一段整包加噪信号、同一 δ∈{0,±1,±2} CRC 仲裁、对称 top-5
候选重试（DeRa Stage-3 原语）、同一判据（symfec 证据 + payload CRC16）。

## 2. 失效机制：亚样本定时 → bin 边界混跳

DeRa 前端交付的 f̂ 精度很高（|误差|中位 ≤0.004 bin，hs 误差 0 样本），
但**不施加分数时延补偿**（−sto_frac·OS，真实帧上 ±1.9 样本内分布）。
Savaux 的相干合并没有吸收该项的自由度，后果链条：

1. 残余时延使合并谱的单音峰落在两个 FFT bin 之间（bin 边界）；
2. 噪声扰动下 argmax 在 bin k 与 k±1 之间**逐符号随机跳变**——行谱 diff
   呈 {0, −1} 混合（例：f00 {0:14, 1023:21}），**任何单一 δ 卷积都修不回**；
3. 逐符号错误直接进 SER/PER——native 无噪即劣化（10/28 帧 SER .2~.5）。

对照：κ 格解调器（TREL/DERA 解码器）原生容忍亚 bin 残差（native 全对），
LoRaTrimmer 部分吸收（native .056），Savaux 最伤（native .123）。
**病不在解调器强度，在"相干合并器是否原生容忍亚 bin 残差"。**

时延响应呈**双平台**（f00 细扫，步 0.125 样本）：d ≤ −0.625 全对；
d ≥ +0.375 恒定 −1（δ 可修）；仅中间 ~0.6-0.9 样本宽的混跳区致命。
对抗测试：干净帧加同样时延不变坏 ⇒ 修复的本质是"避开混跳区"，
不是"加了时延就好"。

## 3. fd 修复（可部署，无 GT）

```
d* = argmax_{d ∈ [−2,+2], 步 0.25}  Σ_{k=前导4符号}  max_bin( dechirp 峰能量(seg_d) )
seg_fd = frac_delay(seg_a, d*)       # 全段一次分数时延
```

- 能量目标在双平台上平坦——argmax 落在任一平台即净（不必选对平台，
  两平台都 δ 可修）；
- 17 点网格全部落在平台上：7/7 坏帧 raw SER 0.000；
- native .000/.000；−20 PER .024 追平全 DeRa；−22 .107 打平；−24 .583
  vs .619（3/84 包，噪声量级，**不称反超**）；−17（1/84 包）与 −26
  （SER .655 vs .521）仍小输。

已知弱点：能量目标平坦 ⇒ 偶尔落在平台边缘（−17 的 1/84 包代价）；
深端前导能量本身被噪声淹没 ⇒ 网格失效。→ 下一代判据候选：交叉项
一致性统计量（见 NOTES §4）。

## 4. 协议遵从（EXPERIMENT_PROTOCOL v1.1）

- 实验A型：整包加噪（含前导）、无先验、同步在带噪信号上重做；
- 纯 AWGN 同一实现喂所有链；整包 SNR 口径（带外噪底法）；
- GT 只来自干净原生解（CSV 先验，帧级冻结）；
- 噪声种子派生常数 20260930 与 dera_front_battle_20260930 同源 →
  参照链六档逐位复现（harness 完整性 100%）；
- native 冒烟 + 判据负控（单符号损坏必拒；header 子项声明豁免——
  判据层不重验 header，与 exp2/battle/front battle 同款）；
- oracle 臂（PRIOR）分表分报，不作参赛链。

## 5. Savaux 方法与其边界（论文定位）

Savaux（TechRxiv 16657063 = IJMNDI'22，TIM'22 为期刊版，port 编号
(34)-(37) 对应期刊版）：按过采样相位拆 branch → 逐 branch 专用 DFT
（Heaviside wrap 相位项）→ Eq.37 相干合并 → argmax。

论文模型 Eq.(2)：`r[n] = h·s[n] + w[n]`——**零 CFO/STO/SFO**，明文
"We assume a synchronized reception"，sync mismatch 列为 future work；
3dB 声明 = OSR2 vs OSR1 过采样相干合并增益（仿真 SF7/8/9，无 OTA）。
**本研究实测的正是该论文未做的非理想接收半场**——fd 修复臂相当于
替它补上第一块实验。
