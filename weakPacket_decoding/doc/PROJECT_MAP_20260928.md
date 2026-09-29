# weakPacket_decoding 项目地图（2026-09-28 整理）

> 目的：一眼知道哪里是哪里。按"对比方法 / 自研方法 / 主系统 / 实验数据 / 文档"组织。

## 1. 对比方法库（你说的"全是对比方法的文件夹"）

**`weak_decoder/baselines/`** —— 发表方法的忠实复现，全部符号级、FFT-bin 域：

| 文件夹 | 出处 | 方法本质 | battle 状态（2026-09-28） |
|---|---|---|---|
| `savaux_oversampled/` | Savaux, TIM 2022 | OSR 分支 DFT + Eq.34-37 相位相干合并 | 已打：全模式落后 TREL-N 1.5~2 dB；κ=0.5 崩（0.5 地板） |
| `loratrimmer/` | LoRaTrimmer, MobiCom 2024 | 按候选 bin 切分去斜符号、前后段投影能量和 | 已打：同上，略逊/互有胜负，κ=0.5 崩 |
| `unichirp/` | UniChirp（PDF 在文件夹内） | preamble 拟合线性相位模型 + 主/混叠双峰相干融合 | 已打：**唯一在半 bin 存活的基线**（其设计领地），仍落后 TREL-N ~2 dB；κ=0 时退化（无双峰→模型空→SER 0.5，实现特性） |
| `symfec/` | Sym-FEC, TMC 2026 | 符号级 FEC：谱似然 + 交织器映射到码字（解码层） | **已打（终局）**：作为共用裁判喂全员谱——证据层是唯一变量，我方谱喂它全模式第一（`symfec_pk_20260928/`）；测试床借此升级为真 LoRa PHY 编码 |
| `common.py` / `run_ser_comparison.py` | — | 数据集加载/SER 对比 harness | 复用其数据口径 |

集成约定（battle 时踩过的坑，复现必读）：bin→值映射常数 `(s+bin)%N=512`（探针校准）；
UniChirp 需 `UniChirpDemodConfig(cfo_correction_mode="none")`、符号后余量（包尾 guard）、
preamble 训练符号（我方约定值 s=512 ↔ 对方 bin0）；其相位观测只在双峰存在（分数偏移）时有效。

## 2. 自研解调探索（历史方法，已收纳退役）

**`weak_decoder/decoding/legacy/`**（2026-09-28 结构整理迁入，顶层 `decoding/__init__.py`
保留再导出，旧 import 兼容）：`adaptive_path_demod`（Savaux Top-K 上的非均匀路径 DP）、
`alias_trim`、`structured_path_demod`、`timing_path_demod`、`robust_sparse_demod`、
`phase_templates`。活跃部分留在 `decoding/`：`header_first_demod`、`payload_codec`
（Gray/交织/Hamming 编解码，symfec_pk 等在用）。
`weak_decoder/synchronization/`：frame_locator、preamble_detector、xcopy_sync。
`weak_decoder/rf_super_resolution/`：实验性前端（仅单测引用，未接主链，保留原位）。
`doc/history/`（含 phase_line 系列史）。

## 3. 主系统（当前论文主线）

**`weak_decoder/os_lora/system/`**：ambiguity-ridge 候选列表 → Savaux 似然 →
bounded soft-Hamming → CRC 仲裁（decode-to-finish）。入口
`decode_savaux_sync_candidate` / `arbitrate_sync_list_with_crc`。

## 4. 2026-09-28 新线（本日三连实验）

| 目录 | 内容 |
|---|---|
| `data/experiments/oversampling_accounting_20260928/` | 泄漏结构验证 + 过采样能量记账（反相关系精确验证；共享/独立噪声分集） |
| `data/experiments/enhanced_demod_chain_awgn_20260928/` | 增强链组装首测（+5 dB vs 旧链；半 bin 平局假象解剖） |
| `data/experiments/ablation_battle_20260928/` | **消融 + 全基线 battle**（含 UniChirp；方法结构修订为 C1+κ格Viterbi） |

论文方法稿：`doc/paper_method_fine_grid_demod_20260928.md`。

## 5. 关键历史实验（挑大的）

`data/experiments/`：`multicopy_replay_*`（副本分集，K=1/2/4/8，含 shared 对照）、
`ambiguity_ridge_*`（当前主线 K 消融）、`preamble_coherent_candidates_ota_*`（PC-Ridge）、
`coded_trellis_receiver / coded_sampling_*`（残差码精确 trellis 解码）、
`residual_cfo_revival`、`paper_audit_20260911 / paper_story_20260912`（论文审计）。

## 6. 数据与文档

- 原始 IQ：`../data/USRP_IQ/`（SF10/1MHz/8×，含 lab1_sf11_TP2）；`noisy_iq/`；
  `data/groundtruth/`、`data/weak_preamble_detections/`、`data/weak_sync_chain/`
- 交接史：根目录 `HANDOFF_2026-07-*.md / 2026072x / 20260820 / 20260928.md`（最新）
- 文档：`doc/`（本地图、论文方法稿、`phase_map_methodology`、`grlora_framesync`、
  `litemap+savaux`、`data数据说明`、`history/`）
- 根目录顶层：`信源待下载清单.md`（3 项待手动获取，Preprints 2026 = 唯一撞车风险）

## 7. 环境备忘

battle/消融脚本用 `D:/mysoft2/miniconda3/envs/gr-lora/python.exe`（3.10，numpy 1.26）；
系统 python3.6 跑不了 baselines（future annotations）。纯 numpy 验证脚本两者皆可。
