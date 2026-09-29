# PC-Ridge 首片验证结果（PC-1/PC-2，2026-09-15）

**回答：** `preamble_coherent_list_sync_design_20260914.md` 的生死线是
"同 560-trial 协议 PDR@11dB +5pts 进主线"。实测 **+15.0 pts（18.8% → 33.75%）**，
同步缺口（oracle gap）在 11/12/13 dB 分别收回 86%/90%/85%。
**结论：通过生死线，PC-Ridge 进主线。**

## 1. 主结果（8 OTA × 10 seeds，与官方 run 同种子逐比特复现 AWGN）

| Es/N0 | PC 相干+仲裁 PDR | 官方 K4-soft 仲裁 | soft-oracle | 配对 修复/回退 | oracle 缺口剩余 |
|---:|---:|---:|---:|---:|---:|
| 10 dB | 2.5% | 0.0% | 1.2% | +2/0 | 0/80 |
| 11 dB | **33.8%** | 18.8% | 36.2% | **+13/1** | 4/80 |
| 12 dB | **77.5%** | 65.0% | 78.8% | **+12/2** | 3/80 |
| 13 dB | **96.3%** | 88.8% | 97.5% | **+7/1** | 1/80 |

端到端 PDR50：11.676 dB（官方 K4-soft）→ **约 11.37 dB**，逼近 soft-oracle
天花板 11.324 dB。candidate-0 命中率极高（mean hit rank 1.04–1.2），
平均解码尝试 ≈1 次/包；候选生成 ~110–150 ms/包（noisy FrameSync 为 ~13 s）。

数据：`data/experiments/preamble_coherent_candidates_ota_20260915/`；
复现：`python -B weak_decoder/os_lora/experiments/evaluate_preamble_coherent_candidates_ota.py`。

## 2. 实现要点（本次调试获得的三个机理结论）

1. **分数 CFO 段内对消**：4-chirp 分段相干在 CFO 小数部分 ≈0.27 时损失
   约 −14 dB（不同包的固定分数 CFO 造成"按包二值"的假象）。修复：
   逐 bin 对 16 个 chirp 做 16 点 FFT（ν 轴取最大），全相干且顺带给出
   分数估计；修复后 clean 包 b_u 误差 ≤0.06 bin（8/8）。
2. **耦合线容忍带随 SNR 收窄**：clean 上解码容忍带为 d∈[−1,+0.5] chip；
   13 dB 收窄到 d∈[0,+0.25]。因此 sto 枚举围绕精确锚点（包检测给出的
   payload 起点）以 0.5 chip 步长展开即可；SFD 下chirp 对（仅 2 符号）
   在低 SNR 中心误差 ±3 chip，**不配做枚举中心**，v1 弃用。
3. **SFO 模型必须用带符号 CFO**：sfo = signed_cfo·BW/fc ≈ −0.033 chip/符号
   （与 clean audit 的 sfo_hat 完全一致）；漏掉它时 payload 游标全程漂
   ~2.5 chip，CRC 必挂；用回绕正数算则 sfo=+1.02，全帧跑飞。
   `sfo_cum_initial=0`（前导漂移已折进锚点，与 ridge builder 约定一致）。

## 3. 诚实边界

- **粗锚点来自干净采集的 payload 起点**（包检测给定假设）：部署时需由
  检测级提供准确到 ±1 符号以内的锚点；±2 chip 枚举只吸收小误差。
  这是 PC-1 的显式假设，下一步要与 `preamble_detector` 的粗检测级联。
- 仍是 8 个物理 capture 上的重复噪声 trial（HANDOFF §11 Step 5 扩数据
  要求不变）。
- 我的列表没有与官方 ridge 候选并集；10–11 dB 官方 FrameSync 无估计时
  PC 完全接管，并集只会更好不会更差。
- 10 dB 以下 demod 本身死亡（oracle 1.2%），同步收益无意义。

## 4. 对论文主张的意义

- "premature synchronization rejection" 现象 + "decode to finish" 架构之外，
  PC-Ridge 补上了**同步级的方法贡献**：16 符号 chirp 轴全相干统计 +
  锚点耦合线枚举 + SFO 模型，把同步可用性从 FrameSync 的 ~12 dB 门限
  推到 demod 天花板（11.3 dB）。
- 与 Ameloot 的对比（§11 Step 4）仍必做：相干前导积累原语同源，
  差异在列表形态 + Savaux 前端复用 + CRC 仲裁。

## 5. 一句话交接

> `weak_decoder/os_lora/system/preamble_coherent_candidates.py` 已可用：
  输入整包 IQ + 粗锚点，输出 Top-9 候选（~120 ms）；接
  `arbitrate_sync_list_with_crc(decoder_mode='soft_hamming')` 即端到端。
  11 dB +15 pts 过生死线；下一刀是与粗检测级联 + 扩真实数据。
