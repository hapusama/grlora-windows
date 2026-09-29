# 多副本下探实验：目标 −20 dB（每样本）以下的同步+解码

**日期：** 2026-09-15（当晚更新：相干合并生效）
**目标坐标钉死：** SER 工具的 "−20 dB" = 1 MS/s **每样本 SNR**。
官方 Es/N0 协议满足每样本 SNR = Es/N0 − 36.1 dB。因此：
`−20 dB/样本 ↔ Es/N0 = +16 dB`。当前单包解调悬崖 ≈ Es/N0 10.5
（= 每样本 **−25.6 dB**，已在 −20 以下 5.6 dB）；建立有余量的
"−20 以下" 工作区 = 每样本 −28 ～ −32 dB（Es/N0 +8 → +4 dB）。

## 0. 生死实验结果（当晚）：相干跨副本合并判活

三方 agent 辩论一致指定的 kill-or-confirm 实验（决策规则：相干合并
≥2 dB/翻倍 = F2/F5 方向成立）。实现：每副本在共享 (axis, bin) 峰取
16 符号相干复音调值 → 相对相位 → 复数符号谱加权求和后取模（注意：
必须先复数求和再取模，先取模即退化为非相干）。结果（K=8，24 trial/点）：

| Es/N0 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| PDR | 0% | 4.2% | 20.8% | **75%** | **95.8%** |

**PDR50 ≈ 2.54 dB Es/N0 = 每样本 −33.6 dB**。相对单副本悬崖 10.5 dB
总增益 7.96 dB = **2.65 dB/翻倍**（非相干 1.2），稳过 2 dB 判据。
K-scaling 补全（K=4：5.54 dB，−30.6 dB/样本）：K=1→4 = 2.48 dB/翻倍，
**K=4→8 = 3.0 dB/翻倍 = 理想相干合并**；总效率 88%（7.96/9.03 dB），
增益随 K 趋近理想直线——共享 bin 相位估计在深处不退化。
关键机理：共享同步在已知 bin 上取单副本相位——单副本**寻峰**在 E≤6
失败（15 bin 误差）不等于单副本**相位**在已知 bin 不可估（16 符号
相干积累 + 已知共享 bin 消掉了 4096-bin 搜索损失）。
数据：`data/experiments/multicopy_replay_{k8_coh,k4_coh}_20260915/`。

**判决（按第二轮仲裁决策树）：≥2 dB/翻倍 → 提交 F2/F5 混合主线
"joint-sync combining below single-copy detection"**；诚实口径 =
回放上界（冻结信道、副本边界已知）；下一步 = OTA 固定 payload 重传
采集（H7 板 + `USE_FIXED_TEST_PAYLOAD=1`），真实信道的每副本 CFO/
STO 变化是唯一的墙。

## 1. 当天的三步验证（全部现有 8 包数据 + 同噪声协议）

### 第一步：K 副本重放架构成立（独立每副本同步）

K 份独立带限 AWGN 噪声重放同一包；每副本 PC-Ridge 同步（candidate-0）；
逐符号 Savaux 功率谱跨副本**平均**后单次软解 CRC。

| Es/N0 | K=1 | K=2 | K=4 | K=8 |
|---:|---:|---:|---:|---:|
| 6 | 0 | 0 | 0 | 0 |
| 8 | 0 | 0 | 41.7% | 83.3% |
| 10 | 4.2% | 83.3% | 95.8% | — |

PDR50：K=1≈10.2 → K=2≈9.0 → K=4≈7.9 → K=8≈7.5 dB。
数据：`multicopy_replay_{k124,k8}_20260915`。

### 第二步：每副本同步地板确诊并被跨副本共享同步消灭

E=6 时每副本 b_u 平均偏 **15 bin**（同步死亡，与解码无关）。副本共享
同一 b_u 真值：把 K 份 2-D 相干谱（chirp 轴 × bin 轴）**先平均再找峰**，
同步统计量获得 ~10log10(K) 增益。实测 E=4–6 共享 b_u 误差恒为
−0.20 bin（≈零，−0.2 是已知 sto 常数）。数据：`multicopy_replay_k8_shared_20260915`。

### 第三步：最后的天花板 = 非相干合并的 LLR 效率

共享同步后 E=6 仍 0%（状态：header_invalid/ok-错包各半）。原因：
功率谱平均下，信号均值与噪声均值之比**不随 K 增长**（只有方差 ÷K），
软判决得到的等效增益仅 ~1.2–1.5 dB/翻倍（实测每倍 ~1.2 dB），
不是相干合并的 3 dB/翻倍。E=6+K=8 的等效工作点约 14–15 dB 单包，
本应 ~100%，实际 0%——差距就是合并方式。

## 2. 结论与到 "−28 ～ −32 dB/样本" 的剩余路线

- 架构成立：`每副本 PC-Ridge 同步（或跨副本共享同步）→ 谱合并 → 单次软解 CRC`，
  今日非相干版已把全包解码悬崖从 Es/N0 10.5 推到 ~7.5（+3 dB）。
- **下一刀（明确且只有一个）：跨副本相干合并。** 每副本相位在 E≤6
  单副本不可观测（第一步的 15-bin 失败即证据），但跨副本联合可估：
  共享 b_u 下的副本间互谱（X_k·conj(X_ref)，16 chirp 积累 + K 副本平均）
  给出相对相位；按此对齐复数谱后相加，信号幅值 K 倍相干叠加，
  预期恢复 ~3 dB/翻倍 → K=8 悬崖 ≈ Es/N0 2–3 dB = **每样本 −34 dB**，
  深入目标区。若互谱相位方差过大，退回 K 更大 + 非相干（收益 ~1.2 dB/翻倍）。
- 与真实重传的差距（诚实）：本实验为独立噪声重放，**乐观**——真实
  重传间有信道变化/频偏漂移；固件 `USE_FIXED_TEST_PAYLOAD=1` 模式 +
  H7 板低 SNR 采集是必做的真实性检验（也正好是 HANDOFF Step 5 扩数据）。
- 定位张力（写论文时必须处理）：重传合并原语属 XCopy（MobiCom'23）；
  我们的差异点 = 极低 SNR 区（XCopy 评估范围之下）+ 跨副本共享同步
  （PC-Ridge 的自然延伸）+ 列表仲裁架构；且 §9.3 单包对比口径需与
  系统口径分开陈述。

## 3. 复现

```powershell
python -B weak_decoder/os_lora/experiments/evaluate_multicopy_replay_ota.py `
  --esn0-db 2,4,6,8,10 --copies 1,2,4,8 --workers 8 `
  --output-dir data/experiments/multicopy_replay_k124_20260915
# 跨副本共享同步
python -B weak_decoder/os_lora/experiments/evaluate_multicopy_replay_ota.py `
  --shared-sync --esn0-db 0,2,4,6 --copies 8 --workers 6 `
  --output-dir data/experiments/multicopy_replay_k8_shared_20260915
```

模块新增：`preamble_coherent_candidates.py::coherent_preamble_axes_power /
shared_bu_from_axes_power`（跨副本 2-D 谱平均接口）。
