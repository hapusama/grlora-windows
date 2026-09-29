# PC-Ridge：面向 Savaux 过采样接收机的前导相干列表同步设计

**日期：** 2026-09-14
**定位：** Savaux 过采样解调解决了"多采样偏移副本叠加"的解调端；本设计解决它没解决的
同步端——把 `doc/初始状态估计.md` 已有的相干估计数学，改造成**低 SNR 下的候选列表
生成器**，接入现有 ambiguity-ridge + CRC 仲裁主线。
**一句话：** 高 SNR 相干估计器给唯一解没问题；低 SNR 的正确形态是"相干谱的多个局部
极大 + 软证据排序 + 解码器仲裁"，而不是把 argmax 的门开得更松。

---

## 1. 要解决的缺口（用 2026-09-14 Step-1 数据量化）

同一 560-trial 协议下（8 OTA × 10 seeds × 7 Es/N0）：

| Es/N0 | 仲裁 PDR（K4+soft） | soft-oracle（干净同步） | 同步缺口 |
|---:|---:|---:|---:|
| 11 dB | 18.8% | 36.2% | **17.4 pts** |
| 12 dB | 65.0% | 78.8% | **13.8 pts** |
| 13 dB | 90.0% | 98.8% | 8.8 pts |
| 10 dB | 0% | 1.2% | demod 已死，同步无意义 |

同步估计可用率（官方 run 的 estimate_available_rate）：10 dB 仅 53.8%，11 dB 80%。

**关键实验空白：** chip/virtual_simo 相干检测器只在 13 dB 评估过
（`virtual_phase_sfd_sync_ota_awgn_20260826_8pkt_13db`：strict 0.375 = 0.375 持平，
但当时 estimate_available 已是 100%）——**统计量不是瓶颈，门才是**。而在可用性真正
坍塌的 10–12 dB 区，相干积累从未被评估。这不是"已证伪"，是"没打中战场"。

## 2. 三条硬约束（来自自家负结果，设计不得违反）

1. **分支不是分集。** OSR 各 branch 是同一波形的确定性投影，噪声相关，无 R 折独立
   增益（2026-07 handoff §结论；virtual_simo docstring 自己也声明了）。任何声称
   "8 分支 = 8 天线"的设计直接枪毙。分支只允许两个角色：Savaux 匹配滤波前端
   （已有），和分数域观测的辅助孔径（需生死线验证）。
2. **门是杀手不是统计量。** 15 dB 审计：estimate_available 100% 而 strict PDR 75%。
   新设计里一切"验证"都是排序分数，唯一裁决权在 payload CRC。
3. **高 SNR 退化检验。** 理想 AWGN、同步正确时，每个新部件必须退化为标准
   dechirp-FFT / 现有 fractional refinement，否则模型有私货（handoff 审查问题 3）。

## 3. 设计：五个阶段

### Stage A —— 相干谱（已有数学，照 `初始状态估计.md` 实现）

参数 θ = (τ₀ chip 级初始 STO, β = CFO·T_s bin 单位, ζ 常值 SFO 项)。对 L 个前导
upchirp，第 s 个 chirp 按 τ₀+β+ζ·s 补偿后做 FFT，目标函数是所有 chirp 在预测 bin
上的**相干叠加能量**：

$$S(\tau_0,\beta) = \Big| \sum_{s=0}^{L_s-1} X_s\big(b(\tau_0,\beta,s)\big)\cdot e^{-j\phi(\tau_0,\beta,s)} \Big|^2$$

两级搜索（粗：整 bin 网格 + 1/4 chip τ₀；细：局部 ±1 bin ±1/2 chip）。
注意 doc 已警告的**重复补偿**问题：τ₀ 只能出现一次（裁窗或频偏补偿二选一）。

**与前导长窗 FFT 的关系：** virtual_simo 的长窗 single-steering 检测是 S 在 β=0
漂移假设下的特例；带 ζ 的分段补偿是它的推广。Stage A 输出的不是 argmax，而是
**保留整个 S 的局部极大集合**。

### Stage B —— 候选 = 相干谱的模态（本设计的核心改动）

低 SNR 下 S 的真值峰与噪声峰不可分，且 (τ₀, β) 沿 chirp 时频对耦方向模糊——
这正是 ridge 关系 `Δpayload_samples ≈ OSR·ΔCFO_bins` 的二维推广。做法：

- 取 S 的 Top-M 局部极大（M≈8~16），每个是一个 (τ₀, β, ζ) 候选；
- 沿模糊线方向若多个极大共线，合并为一条线上的有序候选（复用
  `build_ambiguity_ridge_sync_list` 的耦合结构与 fractional refinement 接口）；
- **不做唯一判决**。candidate 0 = S 最大模态（替代现在的 noisy FrameSync 坐标）。

### Stage C —— 已知符号软证据（排序，不裁决）

- **sync word 两个符号值已知**：候选参数下它们的预测 bin 可相干积分，作为净 ID
  证据打分——替代现在的 netID 硬门；
- **SFD 上下 chirp 配对**：upchirp 积累 bin 与 downchirp bin 的互补关系
  （b_u ≈ Δτ+Δf，b_d ≈ −Δτ+Δf）解耦 (τ₀, β)；SFD 弱（2.25 符号）时检测不到就
  不用，**不作为门**——退化为 Stage B 的模糊线候选；
- 所有分数线性加权进候选排序，权重在 8 包上调，扩数据后复核（防过拟合声明）。

### Stage D —— 时间孔径分数精测（相干积累的第二用途）

对每个候选，用 L 个前导符号在预测 bin 上的**复数序列相位拟合**精测分数 CFO
（DFT zoom 到 ±0.5 bin，分辨率 ~1/L_s bin ≈ 0.06 bin），τ₀ 分数部分交给现有的
fractional STO/SFO refinement（Savaux 匹配滤波已隐式处理分数延迟）。ζ 由前导段
内 bin 漂移初估，payload 段沿用现有 sfo 递推。

**分支孔径精测（可选、带生死线）：** 分支间相位递进含分数 STO 信息，但按约束 1
它是投影受限观测——若与现 fractional refinement 相比无独立增益（同 8 包协议
A/B），直接删除，不进主线。

### Stage E —— 仲裁（现有设施，零改动）

每个候选 → `decode_soft_hamming_sync_candidate` / upstream soft → header CR 合法性
→ payload CRC 仲裁，首个 CRC 成功当选，expected bytes 只做事后审计。

## 4. 过一遍自家审查问题

| 审查问题 | 回答 |
|---|---|
| 利用的物理现象 | L 个前导 upchirp dechirp 后是同一单音的多份时间副本；chirp 时频对耦使 (τ₀,β) 模糊呈线结构 |
| 改变什么 | 同步候选的**可用性**（10–12 dB 坍塌区）与**排序质量**；不改解调、不造新信息 |
| 高 SNR 退化 | S 最大模态 → 标准 preamble bin 估计；D 相位拟合 → 现有 fractional refinement 同解 |
| 与分支分集的距离 | 分支只做前端/辅助孔径，无独立性主张 |

## 5. 复杂度预算

- Stage A 粗搜：L 次 FFT/网格点 ≈ 现有 scan_preamble_windows 量级；两级搜索已把
  2D 网格压到 O(N·(N/4)) 的一次性代价，离线 Python 可接受（现在 noisy FrameSync
  ~13 s/trial 是锚点，Stage A 目标 <5 s/trial）；
- Stage B–D 每候选代价与现有 ridge refinement 同阶（ms 级）；
- M=16 候选 × 平均 <2 次解码尝试（今日实测 1.7），仲裁成本不变。

## 6. 失败模式与对策

| 失败模式 | 对策 |
|---|---|
| 分数 CFO 超相干窗（>1/(L·T_s)），积累退化为非相干 | 分段（4 符号）相干 + 段间非相干合并；β 网格加密 |
| 多包/碰撞下积累旁瓣成伪模态 | M 上限 + sync word 软证据降权 + CRC 兜底（本来就是仲裁制） |
| 8 包上排序权重过拟合 | 权重只调排序不裁决；扩数据（≥100 包，H7 板 SOP）后必须复核 |
| 窄带干扰聚成强伪峰 | 本版不处理（AWGN 协议优先）；留 AliasTrim 历史思路做扩展 |

## 7. 诚实收益预算与生死线

- **上限**：11 dB 17.4 pts / 12 dB 13.8 pts（oracle 差距）；现实目标
  **11 dB +8~12 pts（18.8% → 27~31%），PDR50 从 11.676 dB 推到 ~11.4 dB**；
- **生死线**：同 560-trial 协议，仲裁 PDR@11 dB 提升 ≥5 pts → 进主线；
  <2 pts → 相干积累对本协议无增量，降级为工程优化（候选 0 质量改进），不再当
  论文主张；
- 附带收益（不构成主张）：estimate_available@10–11 dB 提升、candidate 0 命中率、
  平均解码尝试数下降。

## 8. 实现与评估计划

1. 新模块 `weak_decoder/os_lora/system/preamble_coherent_candidates.py`：
   输入整包 IQ → 输出与 `AmbiguityRidgeCandidate` 同构的候选列表 + 排序分数；
   FFT 核复用 `paper_oversampled_spectrum` / `scan_preamble_windows`；
2. 消融入口：`evaluate_decoder_aware_crc_pdr_ota.py` 加
   `--candidate-source noisy|preamble_coherent`（仿照今日 --ridge-upstream-soft 的
   接法）；
3. 阶段指标（先于端到端）：availability@10/11/12 dB、hit@K（正确坐标在列表内的
   比例与排名）、candidate-0 命中率；
4. 单元测试：clean 包 rank-1 唯一命中；合成 β 扫描恢复精度 ≤0.1 bin；高分段
   相干窗退化行为；
5. 通过后再打端到端 560-trial 与生死线。

## 9. 相关工作定位（novelty 边界，诚实声明）

- **相干前导积累作为原语不新**：Ameloot et al.（IoT-J 2021，极低 SNR 同步）是
  最近邻经典对手，§11 Step 4 必须正面对比——差异点在：我们的积累统计量建立在
  Savaux 分支合并周期图上（同步与解调共用同一副本前端）、输出列表而非唯一解、
  且以 CRC 仲裁收尾（decode-to-finish 架构）；
- **XCopy（MobiCom'23）**：重传副本相干合并，TX 侧时间冗余——与本设定（单包、
  单节点、无重传）正交，related work 引用，不做 dB 对比（§9.3 排除）；
- **MALoRa（INFOCOM'22）**：多天线相干合并——分支≠天线（约束 1），同为
  related work；
- **Savaux**：本设计是它的同步前端补全，两者合成"单节点多副本接收机"完整故事；
- **Xhonneux**（低复杂度 STO 鲁棒同步）：解决 STO 校正精度，不解决低 SNR 可用性。

## 10. 一句话交接

> 把 `初始状态估计.md` 的相干估计器实现出来，但**输出 Top-M 模态候选列表而不是
> argmax**，候选经 sync word 软证据排序后喂给现有 ridge refinement + CRC 仲裁；
> 先在 10–12 dB 可用性坍塌区测 availability 与 hit@K，端到端生死线是
> PDR@11 dB +5 pts。
