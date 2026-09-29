# enhanced_demod_chain_awgn_20260928 — RESULTS

日期：2026-09-28（晚）。脚本：`enhanced_chain_test.py`（numpy-only），
数据：`results.json`，图：`pdr_curves.png`。前置：`../oversampling_accounting_20260928/`。

## 目的

用已验证的泄漏性质组装增强解调链，在同一物理信号 + 同一噪声 + 同一 FEC 下
与旧链正面对比，可控 AWGN（κ 与 SNR 手动旋钮）。

## 链定义

| 链 | 解调层 | 用到的性质 |
|---|---|---|
| OLD-A | chip 网格单相位抽取 + N-FFT argmax（frame_sync 风格） | — |
| OLD-B | 4 相位抽取 + 功率谱平均（朴素"副本叠加"） | P1'（非相干分集） |
| NEW-1 | OS 域去斜 + 补零 16N 细网格 FFT + ±2 细bin 窗 max | P1 相位对齐 + P3 免估计 κ-GLRT |
| NEW-2 | 同上 + 幅值比 κ̂ + Dirichlet 相干重组 | P1 + P2 + P3 全家 |
| ORACLE | 真 κ 相干 ML | 上界 |

FEC stand-in（明确非完整 LoRa FEC，仅公平对比）：40 payload bit →
10×Hamming(8,4)+总校验 → 对角交织到 8 个 SF10 符号 → Gray；
解调 T(s) 全谱 → bit LLR → Chase-16 MAP（软）/ 最近码字（硬）。
噪声模型：os 域 iid（未滤波宽带），口径同前置实验——**若噪声预滤波到带内，
NEW 相对 OLD-A 的增益会缩到量化税量级（~2 dB），此条件必须写进论文。**

## 主要读数（soft-FEC PDR50，相对 OLD-A）

| κ 模式 | OLD-A | OLD-B | NEW-1 | NEW-2 | ORACLE | NEW-2 − OLD-A |
|---|---|---|---|---|---|---|
| 0 | ≈−15.5 | ≈−17 | ≈−21 | ≈−20.5 | ≈−21.5 | **+5~5.5 dB** |
| 0.25 | ≈−15 | ≈−17.5 | ≈−20 | ≈−20.2 | ≈−21.5 | **+5 dB** |
| drift(+0.02/sym) | ≈−15.5 | ≈−18 | ≈−18.5 | ≈−19 | ≈−19.5 | **+3.5~4.5 dB** |
| 0.5 | （地板 0.53 SER） | PDR50≈−18 | 平局假象 | 0.5 随机 | ≈−20 | 见下 |

符号级：κ=0 时 NEW 链在 −22 dB 达 SER=0，OLD-A 需 −16（6 dB）。

## 三个诚实的发现

1. **半 bin 不可分命题被现场验证（并抓到一个平局裁决假象）**：κ=0.5 时音
   落在补零网格的整数点上，恰好是 s 与 s+1 假设窗口的**公共边界**——两窗
   max 逐 bit 相等，argmax 按先返回小索引破平 → κ=+0.5 单侧仿真里 NEW-1
   SER=0 是**裁决假象**（κ=−0.5 时会 100% 错；κ 对称分布时 50%）；
   NEW-2 相邻权不一致 → 0.5 随机；只有 ORACLE（知 κ 旋转）满分。
   **细网格不解决半 bin 歧义，κ 信息才解决**——命题的仿真版证明。
2. **软 FEC 增益实测 2~3 dB**：NEW-2 κ=0 −20 dB：sP 0.408 vs hP 0.092。
   且 κ=0.5 处 soft 把 0.5-SER 的 NEW-2 救回 PDR 0.27（邻 bin=Gray 1bit
   错，FEC 可纠）——软证据对歧义场景的兜底价值。
3. **NEW-2 与 ORACLE 剩 1~2 dB 差距 = κ̂ 质量**：补零网格上 Dirichlet 主瓣
   宽（零点在 ±4 细bin），相邻两点幅值比几乎不变（0.9~1），δ̂=m2/(m1+m2)
   系统性偏 ~0.5。修正方向：5 点窗内做核形状拟合（或回到主 bin 网格用
   反相比值估 κ 再旋转）。这是下一轮唯一的算法缺口。

## 脚本勘误记录（复现注意）

- 4N 点 FFT 不给细网格（分辨率=Δf 不变），必须补零到 16N；
- NEW/ORACLE 的 T 按符号值索引，OLD 按 bin 索引（−值），映射勿混；
- FEC 编码侧 payload 行=nibble，`reshape(-1)` 会把 bit 当 nibble 查表；
- numpy `where` 两分支先求值，索引需先 clip。

## 结论

**发现的三条性质（相位对齐合并、反相指纹、免估计 κ-GLRT）组装成链后，
在未滤波宽带噪声 AWGN 下对 frame_sync 风格旧链实测 +5 dB（κ=0/0.25）、
+3.5~4.5 dB（SFO 漂移），距真 κ oracle 仅 1~2 dB，缺口已定位为 κ̂ 估计。**
下一步：①κ̂ 5 点核拟合（关掉与 ORACLE 的差距）；②把 NEW-2 的 T(s) 接进
weak_decoder 的 soft-Hamming/CRC 仲裁真链；③在 OTA capture 上重放。
