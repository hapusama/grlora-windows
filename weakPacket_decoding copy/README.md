# weakPacket_decoding 原型说明

这个目录用于弱包解码原型开发。当前主线已经从单纯的前导码检测推进到：

```text
raw complex64 IQ
  -> 弱前导码检测
  -> 粗帧定界 frame_locator
  -> gr-lora 风格 framesync 参数估计/验证
  -> header-first FFT demod
  -> 导出 header/payload symbol 级 FFT peak
```

当前主线已经具备离线 payload codec 工具和 symbol-level selector 评估。现阶段重点是验证：弱检测和同步结果能否稳定支撑 header 解码、payload FFT peak 导出，以及不依赖 payload byte 先验的低 SNR symbol 修复。

## 当前研究入口：symbol-level offset coherence

当前正式主线是单包、PHY-only 的 multi-offset + offset-coherence symbol selector：

```text
scripts/experiments/run_symbol_phase_threshold_sweep.py
weak_decoder/symbol_phase_two_stage.py
```

核心流程是：

```text
header-first timing/header
  -> payload multi-offset FFT evidence
  -> high-confidence Top-1 locking
  -> low-confidence Top-L candidates
  -> offset coherence + energy + small packet-line phase score
  -> selected raw FFT bin
  -> LoRa PHY hard-decision codec/CRC metrics for evaluation
```

这条线不使用 payload byte 先验，不使用 counter/template，不跨包，不用 CRC 参与 symbol 选择。CRC 只作为最终 hard-decision payload 是否自洽的评估字段。

第一阶段 Top-L 候选召回仍可用这个轻量脚本单独观察：

```text
scripts/experiments/evaluate_candidate_pruning_metric.py
weak_decoder/candidate_pruning.py
data/candidate_pruning/
```

早期 two-stage codec-block decoder 仍保留为历史对照：

```text
scripts/experiments/run_two_stage_weak_decoder.py
weak_decoder/two_stage_weak_decoder.py
data/two_stage_weak_decoder/
```

它的候选也只来自当前 packet 内的 FFT evidence、LoRa FEC/CRC 结构和重编码轨迹评分；旧的 learn-session / byte-template / residual-prior 实验入口已在 2026-06-17 清理。

## 已清理的 session-prior 线

2026-06-17 已移除基于已知包结构或跨包学习的旧实验线，包括：

```text
scripts/learn_session_*.py
scripts/evaluate_session_priors.py
scripts/reconstruct_session_payloads.py
scripts/sweep_phase_guided_session.py
scripts/sweep_joint_residual_session.py
scripts/joint_decode_residual_candidates.py
scripts/run_phase_guided_demod.py
scripts/reproduce_phase_map_paper_artifacts.py
scripts/make_phase_ablation_table.py
scripts/make_generalization_validation_table.py
scripts/make_joint_threshold_table.py
scripts/sweep_map_kappa_from_candidates.py
data/phase_guided/
```

后续实验默认按“单包 PHY evidence”口径理解：不使用 payload byte prior、payload template、counter model、cross-packet joint decoding 或重传信息。

## 当前主流程

### 1. 弱前导码检测

入口：

```text
scripts/detect_weak_preamble.py
```

核心逻辑：

```text
raw IQ
  -> 滑动窗口扫描
  -> 每个窗口包含 win_chirps 个 chirp
  -> 每个 chirp 分别 dechirp + FFT
  -> 窗口内 FFT 能量按 bin 累加
  -> 取 peak bin
  -> 连续窗口 peak bin 稳定超过 min_periodic_peaks
  -> 输出 detection event
```

这一层只判断“这里像不像 LoRa upchirp 前导码”，不检查 sync word / netID。

输出：

```text
data/weak_preamble_detections/*_events.csv
data/weak_preamble_detections/*_windows.csv
```

### 2. 一体化弱同步链

入口：

```text
scripts/run_weak_sync_chain.py
```

当前流程：

```text
弱前导码 detection event
  -> preamble anchor refinement（非严格 chirp boundary alignment）
  -> frame_locator 搜索 preamble + sync word + SFD
  -> 输出 frame_valid
  -> gr-lora 风格 CFO/STO/SFO 估计
  -> 前导码校正后检查是否回到 bin0
  -> netID / sync word 成组检查
  -> 输出 grlora_framesync_valid
```

注意字段含义：

```text
selected_events        检测阶段选出的候选事件数，不等于有效包数
frame_valid            粗帧定界是否成立
grlora_netid_valid     两个 sync word / netID 是否满足共同 offset 逻辑
grlora_framesync_valid 最终 framesync 是否通过
```

注意：当前脚本控制台里仍可能打印历史字段名 `selected_packets=...`。这里的含义其实是 `selected_events`，也就是检测后被选中的候选事件数量，不是最终有效包数量。

当前 `grlora_framesync_valid` 的判定为：

```text
grlora_framesync_valid =
    前导码经 CFO/STO/SFO 校正后全部回到 bin0
    && grlora_netid_valid
```

`grlora_sync1_distance` 和 `grlora_sync2_distance` 仍会输出，但只作为调试字段，不再要求二者都严格为 0。这样比“两个 sync word 绝对落在固定 bin”更合理，因为实际同步残差可能让两个 sync word 共同偏移。

### 接口契约与命名注意事项

`run_weak_sync_chain.py` 里的 `align_event_start` 不应理解成严格的 chirp boundary alignment。它是在检测事件附近，用多个 preamble upchirp 的累加 FFT peak power 寻找更好的 sample anchor。整数 chip 偏移时，preamble dechirp 后的能量仍然会集中，只是 FFT bin 整体平移，所以这一步不能唯一消除整数 chip ambiguity。后面的 `grlora_frame_sync.py` 会继续用：

```text
preamble_ref_bin -> signed bin -> coarse_offset_chips -> coarse_offset_samples
```

把这个 bin 平移解释成粗同步偏移，再得到 `grlora_synced_preamble_start_sample` / `grlora_fine_payload_start_sample`。因此这一步更建议描述成 `preamble anchor refinement` 或 `sample-level preamble anchor search`，不要写成“精确 chirp 边界对齐”。

弱检测和粗帧定界阶段使用过采样 FFT：

```text
fft_len = chirp_samples = 2^SF * os_factor
```

对应字段包括 `detected_reference_bin`、`align_peak_bin`、`preamble_ref_bin`、`sync1_bin`、`sync2_bin`、`sfd1_bin`、`sfd2_bin`。而 gr-lora 风格 framesync、header-first demod 和 corrected/no-offset chip-rate FFT 使用：

```text
fft_len = 2^SF
```

对应字段包括 `grlora_netid1_est`、`grlora_netid2_est`、`raw_fft_bin`、`signed_fft_bin`、`symbol_value`。后续新增模块时要先确认字段来自哪个阶段，避免把 oversampled bin 和 chip-rate bin 混用。

另外，`grlora_synced_payload_start_sample` 和 `grlora_fine_payload_start_sample` 里的 `payload_start` 是沿用历史命名。它在 explicit header 模式下实际表示 LoRa data region 的起点，也就是 PHY header 第 0 个 symbol 的起点，不是 MAC payload 第 0 个 symbol。`run_header_first_demod.py` 会从这里先解 8 个 explicit header symbol，再进入真正的 payload symbol。

弱前导码检测本身是高召回候选生成器，不是强判决器。`preamble_detector.py` 主要依靠连续窗口 peak bin 稳定性，不设置强绝对能量门限，因此窄带干扰或稳定杂散也可能形成 detection event。后面的 sync word / SFD / framesync / header checksum 才是逐级过滤。

### STFT 图的 valid 含义

`--stft-dir` 生成的图片标题里的 `valid` 表示：

```text
frame_valid
```

也就是粗帧定界是否有效，不代表 `grlora_framesync_valid`。如果要判断某张图对应事件是否最终通过 framesync，需要到 sync chain CSV 中按：

```text
packet_index + event_index
```

查看：

```text
grlora_netid_valid
grlora_framesync_valid
```

## 输出目录约定

`data/weak_sync_chain/` 根目录保持干净，当前只保留主流程 CSV；STFT、频谱图和趋势图按需重新生成，不默认留在 `data/` 里：

```text
sync_chain/                 run_weak_sync_chain.py 的主表 CSV
framesync_peaks/            framesync 后前导码 FFT peak 验证表
header_first/               header-first demod 的 frame / symbol CSV
payload_consistency/        payload FFT bin 内部一致性检查表
*_stft/                     按需生成的 STFT PNG / CSV，默认已清理
*_framesync_spectrum/       按需生成的 framesync 前后频谱对比图，默认已清理
*_payload_peak_trends/      按需生成的 payload peak 相位/幅度趋势图，默认已清理
```

## data 目录地图：脚本来源、意义与复现入口

`data/` 已瘦身：只保留主链/两阶段主流程相关 CSV 和少量正式摘要；大体积构造数据、诊断图、probe/sweep 中间结果由脚本按需再生成。

```text
主链输入/中间结果      weak_preamble_detections/, weak_sync_chain/
两阶段主流程结果      two_stage_weak_decoder/, symbol_phase_two_stage/
当前正式结论          ablation_offset_coherence_summary/, ablation_current_default/, phase_opportunity_space/
按需生成诊断数据      peak_groundtruth/, low_snr_gt_bin/, noisy_iq/, payload_* 等
默认不保留探针数据    probe_*/, symbol_phase_threshold_sweep_* 等
```

### 最常看的正式结果

| 目录 | 产出脚本 | 主要意义 | 备注 |
| --- | --- | --- | --- |
| `data/ablation_offset_coherence_summary/` | `scripts/experiments/make_offset_coherence_ablation_table.py` | 汇总 offset-coherence 消融实验，回答 multi-offset、Top-L、locking、coherence、packet-line 各自贡献 | 当前最清爽的论文式表格入口 |
| `data/phase_opportunity_space/` | `scripts/experiments/analyze_phase_opportunity_space.py` | 分析 multi-offset Top-L 给第二阶段留下多少可修复空间，统计 already_correct / repairable / unrecoverable、phase repair/damage | 判断瓶颈是候选漏 GT 还是 phase selector 不够强 |
| `data/ablation_current_default/` | `scripts/experiments/run_symbol_phase_threshold_sweep.py` | 当前默认 selector 的完整 SNR threshold sweep | A5/current default |
| `data/ablation_energy_only_top24/` | 同上 | energy-only selected，对照 multi-offset argmax | 证明 selected path 本身不带来额外收益 |
| `data/ablation_amp_coherence_no_line/` | 同上 | energy + offset coherence，但去掉 packet-line phase | 隔离 offset coherence 主增益 |
| `data/ablation_topL_8/16/24/32/` | 同上 | Top-L 候选规模消融 | 当前 Top-24 是低复杂度折中 |
| `data/ablation_no_high_conf_lock/` | 同上 | 关闭高置信 Top-1 locking | 验证 lock 是否保护强符号 |
| `data/ablation_coherence_only_top24/` | 同上 | 只看 offset coherence 评分 | 隔离 coherence 本身 |
| `data/ablation_packet_line_only/` | 同上 | 只看 packet-line phase 评分 | 当前不是主增益来源 |

当前最推荐打开：

```text
data/ablation_offset_coherence_summary/ablation_report.md
data/ablation_offset_coherence_summary/ablation_threshold_summary.csv
data/ablation_offset_coherence_summary/ablation_probe_curve_m20_m23.csv
data/phase_opportunity_space/phase_opportunity_summary.csv
data/phase_opportunity_space/phase_opportunity_summary.json
```

### 主链数据

| 目录 | 产出脚本 | 主要意义 | 常用下游 |
| --- | --- | --- | --- |
| `data/weak_preamble_detections/` | `scripts/detect_weak_preamble.py`，也可由 `scripts/run_weak_sync_chain.py` 顺手写出 | 弱前导码 detection event 和 sliding-window 调试表 | frame locator / sync chain |
| `data/weak_sync_chain/sync_chain/` | `scripts/run_weak_sync_chain.py` | detection -> frame locator -> gr-lora 风格 framesync 的主 CSV | header-first demod |
| `data/weak_sync_chain/framesync_peaks/` | `scripts/run_weak_sync_chain.py --framesync-peaks-csv` | framesync 后前导码 peak 是否回到 bin0 的验证 | sync 质量诊断 |
| `data/weak_sync_chain/header_first/` | `scripts/run_header_first_demod.py` | PHY header decode 与 payload symbol FFT peak CSV | low-SNR GT、threshold sweep |
| `data/weak_sync_chain/payload_consistency/` | `scripts/run_header_first_demod.py --consistency-output` | payload raw FFT bin 在候选包之间是否一致 | 检查 invalid candidate 偏离 |
| `data/weak_sync_chain/*_stft/` | `scripts/run_weak_sync_chain.py --stft-dir` | event 级 STFT 图 | 按需生成，默认已清理 |
| `data/weak_sync_chain/*_payload_peak_trends/` | `scripts/plot_payload_peak_trends.py` | 每包 payload selected peak 的相位/幅度趋势图 | 按需生成，默认已清理 |

### 物理诊断与数据构造

这些目录多数是按需生成产物，当前为了控制体积默认不保留。脚本和命令说明保留在 README 中。

| 目录 | 产出脚本 | 主要意义 |
| --- | --- | --- |
| `data/peak_groundtruth/` | `scripts/experiments/export_peak_groundtruth.py` | gr-lora_sdr 原链导出的 clean peak/symbol groundtruth |
| `data/low_snr_gt_bin/` | `scripts/experiments/run_low_snr_gt_bin_experiment.py` | clean GT bin 在加噪 IQ 中的相位/幅度是否仍可读 |
| `data/low_snr_gt_bin/*/wrong_bin_control/` | `scripts/experiments/run_low_snr_wrong_bin_experiment.py` | 低 SNR 下 GT bin 与 wrong bin 的相位/幅度对照 |
| `data/low_snr_gt_bin/*/sto_phase_jump_corrected/` | `scripts/experiments/run_low_snr_sto_phase_jump_experiment.py` | residual STO phase-jump 补偿实验 |
| `data/payload_feature_no_offset/` | `scripts/experiments/export_payload_no_offset_features.py`，`plot_corrected_phase_diagnostics.py` | no-offset/corrected payload FFT 特征与相位诊断图 |
| `data/payload_wrong_bin_diagnostics/` | `scripts/experiments/plot_wrong_bin_phase_diagnostics.py` | clean IQ 下 selected bin 与 wrong bin 的 phase/amplitude 对照 |
| `data/candidate_pruning/` | `scripts/experiments/evaluate_candidate_pruning_metric.py`，`run_candidate_pruning_sweep.py` | Top-L candidate recall 和 phase-aware pruning 指标评估 |
| `data/two_stage_weak_decoder/` | `scripts/experiments/run_two_stage_weak_decoder.py` | 早期 two-stage codec/phase-gated payload decoder 输出 |
| `data/symbol_phase_two_stage/` | `scripts/experiments/run_symbol_phase_two_stage.py` | symbol-level two-stage selector 单点实验 |
| `data/symbol_phase_model_diagnostics/` | `scripts/experiments/diagnose_symbol_phase_models.py` | GT-only phase model 排名诊断 |
| `data/phase_opportunity_space/` | `scripts/experiments/analyze_phase_opportunity_space.py` | Top-L repairable/unrecoverable 空间与当前 phase/coherence selector 救回率分析 |
| `data/phase_vs_argmax/` | `scripts/experiments/run_phase_vs_argmax_comparison.py` | phase-aware bin 选择与 argmax 对比 |

### 历史 sweep / probe 怎么看

`data/symbol_phase_threshold_sweep*` 和 `data/probe_*` 大多是 2026-06-16 的调参过程产物，已经默认清理。需要追溯时重新运行：

```text
scripts/experiments/run_symbol_phase_threshold_sweep.py
```

命名大致含义：

```text
symbol_phase_threshold_sweep_coherence_default/   当时的正式 coherence default 结果
symbol_phase_threshold_sweep_smooth_*/            smooth/trajectory beam 探针
symbol_phase_threshold_sweep_coherence_l*/        Top-L 探针
probe_e24_phase005_a050_q09_m22/                  手动调权重 probe，e/topL=24, phase=0.05, amp=0.50, coherence=0.90, max drop=22 dB
probe_smooth_*/                                   smooth beam 参数 probe
```

这些目录只作为调参追溯材料，写论文/汇报时优先引用当前保留的小表：

```text
data/ablation_offset_coherence_summary/
data/ablation_current_default/
notes/plans/THRESHOLD_GAIN_EVALUATION_2026-06-16.md
```

### 常用复现命令

以下命令建议在本目录运行：

```powershell
cd "d:\Desktop\proj\gr-lora_sdr\weakPacket_decoding copy"
```

生成 header-first clean GT：

```powershell
python scripts\run_header_first_demod.py `
  -i ..\data\USRP_IQ\0_0_0_10_14_16.bin `
  -s data\weak_sync_chain\sync_chain\0_0_0_10_14_16_sync_chain.csv `
  -o data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv `
  --frames-output data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_frames.csv `
  --sf 10 --bw 125000 --samp-rate 500000 --ldro-mode 2
```

生成低 SNR GT-bin 数据集：

```powershell
python scripts\experiments\run_low_snr_gt_bin_experiment.py `
  -i ..\data\USRP_IQ\0_0_0_10_14_16.bin `
  -g data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv `
  -o data\low_snr_gt_bin\0_0_0_10_14_16 `
  --target-snr-db -10 -15 -20 `
  --cfo-correction-mode continuous `
  --overwrite
```

跑当前默认 threshold sweep：

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --snr-start -12 --snr-stop -26 --snr-step -1 `
  --output-dir data\ablation_current_default
```

跑 offset coherence ablation 里的一个典型对照：energy + coherence，无 packet-line：

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --selection-mode coherence `
  --top-l-low-confidence 24 `
  --smooth-phase-weight 0.0 `
  --smooth-amp-weight 0.50 `
  --smooth-coherence-weight 0.90 `
  --smooth-max-energy-drop-db 20 `
  --snr-start -12 --snr-stop -26 --snr-step -1 `
  --output-dir data\ablation_amp_coherence_no_line
```

重新生成 ablation 汇总表：

```powershell
python scripts\experiments\make_offset_coherence_ablation_table.py
```

分析 phase 可发挥空间：

```powershell
python scripts\experiments\analyze_phase_opportunity_space.py
```

如果只想快速试探一个 selector，不想跑完整阈值曲线，可以先用：

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --snr-start -20 --snr-stop -23 --snr-step -1 `
  --output-dir data\my_quick_probe
```

## 推荐运行命令

### 16 前导码文件：检测 + 帧定界 + framesync + STFT

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\run_weak_sync_chain.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin -o gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\sync_chain\0_0_0_10_14_16_sync_chain.csv --events-csv gr-lora_sdr\weakPacket_decoding\data\weak_preamble_detections\0_0_0_10_14_16_events.csv --windows-csv gr-lora_sdr\weakPacket_decoding\data\weak_preamble_detections\0_0_0_10_14_16_windows.csv --bw 125000 --samp-rate 500000 --center-freq 487.7e6 --sync-word 0x34 --preamble-len 16 --win-chirps 4 --hop-chirps 1 --min-periodic-peaks 12 --frame-min-preamble-peaks 12 --stft-dir gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_16_stft --framesync-peaks-csv gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\framesync_peaks\0_0_0_10_14_16_framesync_peaks.csv --framesync-spectrum-dir gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_16_framesync_spectrum --framesync-spectrum-bin-span 96 --framesync-spectrum-chirps 8
```

主要输出：

```text
data/weak_sync_chain/sync_chain/0_0_0_10_14_16_sync_chain.csv
data/weak_sync_chain/framesync_peaks/0_0_0_10_14_16_framesync_peaks.csv
data/weak_sync_chain/0_0_0_10_14_16_framesync_spectrum/
data/weak_sync_chain/0_0_0_10_14_16_stft/
data/weak_preamble_detections/0_0_0_10_14_16_events.csv
data/weak_preamble_detections/0_0_0_10_14_16_windows.csv
```

### 8 前导码文件：检测 + 帧定界 + framesync

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\run_weak_sync_chain.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin -o gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\sync_chain\0_0_0_10_14_8_sync_chain.csv --events-csv gr-lora_sdr\weakPacket_decoding\data\weak_preamble_detections\0_0_0_10_14_8_events.csv --windows-csv gr-lora_sdr\weakPacket_decoding\data\weak_preamble_detections\0_0_0_10_14_8_windows.csv --bw 125000 --samp-rate 500000 --center-freq 487.7e6 --sync-word 0x34 --preamble-len 8 --win-chirps 2 --hop-chirps 1 --stft-dir gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_8_stft --framesync-peaks-csv gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\framesync_peaks\0_0_0_10_14_8_framesync_peaks.csv --framesync-spectrum-dir gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_8_framesync_spectrum --framesync-spectrum-bin-span 96 --framesync-spectrum-chirps 8
```

## Header-first FFT demod

入口：

```text
scripts/run_header_first_demod.py
```

这个脚本不重复做 sync word / netID 检测。它读取 `run_weak_sync_chain.py` 输出的 CSV，只处理：

```text
grlora_framesync_valid == 1
```

然后从 `grlora_fine_payload_start_sample` 开始，把这个位置作为 PHY header 第 0 个 symbol 的起点，执行：

```text
8 个 header symbol
  -> STO/SFO 采样对齐
  -> CFO_int/CFO_frac downchirp
  -> dechirp + FFT
  -> argmax raw_fft_bin
  -> symbol_value = mod(raw_fft_bin - 1, 2^SF) / 4
  -> gray demap
  -> deinterleave
  -> hamming decode
  -> header checksum
```

如果 `header_valid == 1`，继续根据 header 里的：

```text
payload_len
CR
has_crc
LDRO
```

计算 payload symbol 数，并导出每个 payload symbol 的 FFT peak。

示例命令：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\run_header_first_demod.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin -s gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\sync_chain\0_0_0_10_14_16_sync_chain.csv -o gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv --frames-output gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_frames.csv --sf 10 --bw 125000 --samp-rate 500000 --ldro-mode 2
```

默认 `--cfo-correction-mode continuous` 会在 FFT demod 前按整帧连续 chip 时间补偿公共 CFO 相位。这样既保留 CFO-aware downchirp 对 FFT 聚峰的作用，也会消掉从帧起点累计到当前 symbol 的 CFO phase accumulation，避免 selected peak phase 被固定 CFO 斜率主导。

```text
--cfo-correction-mode continuous
```

如果需要复现实验早期的 gr-lora_sdr-like 口径，可以显式传 `--cfo-correction-mode symbol`。该旧口径只在每个 symbol 内用 CFO-aware downchirp 聚峰，不额外消除跨 symbol 累积的公共 CFO 相位，因此 phase trend 中会保留明显线性漂移。两种口径理论上不改变 FFT argmax 选 bin，差异主要体现在 selected peak phase 的相位参考。

输出：

```text
*_header_first_frames.csv   每个 frame 的 header 解码摘要
*_header_first_symbols.csv  每个 header/payload symbol 的 FFT peak
```

`*_header_first_symbols.csv` 就是后续低 SNR / wrong-bin 实验里常用的 clean GT-bin 来源。该文件由 `scripts/run_header_first_demod.py` 的 `-o` 参数生成；如果只想取 payload 的正确 FFT bin，筛选：

```text
stage == payload
header_valid == 1
```

然后读取 `raw_fft_bin` 即可。这里的 GT 不是低 SNR 条件下重新判决出来的，而是 clean IQ 经过 header-first demod 后记录下来的 payload selected bin。

重要字段：

```text
header_valid
payload_len
cr
has_crc
ldro
payload_symbol_count
raw_fft_bin
symbol_value
peak_amp
peak_phase
peak_margin_db
```

### Payload FFT bin 一致性检查

如果要把 `frame_valid=0`、`netID invalid` 的候选也一起送进 FFT demod，可以放宽筛选条件：

```text
--frame-filter all
```

配合：

```text
--invalid-header-payload-policy mode
--consistency-output <payload_bin_consistency.csv>
```

脚本会先用 header 有效的帧推断最常见的 payload symbol 数；header 无效的候选也按这个长度继续解调 payload FFT peak。这样可以检查同一个 bin 文件里所有候选包的 payload raw FFT bin 是否一致。

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\run_header_first_demod.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin -s gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\sync_chain\0_0_0_10_14_16_sync_chain_netidvalid.csv -o gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols_all_candidates.csv --frames-output gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_frames_all_candidates.csv --consistency-output gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\payload_consistency\0_0_0_10_14_16_payload_bin_consistency_all_candidates.csv --sf 10 --bw 125000 --samp-rate 500000 --ldro-mode 2 --frame-filter all --invalid-header-payload-policy mode
```

一致性 CSV 是一行一个 payload symbol 位置，核心字段为：

```text
payload_symbol_index
frame_count
mode_raw_fft_bin
mode_raw_count
raw_mismatch_count
raw_unique_count
unique_raw_fft_bins
mismatch_frame_indices
```

如果某个位置 `raw_unique_count == 1`，说明所有候选在这个 payload symbol 上 FFT bin 完全一致；如果 `mode_raw_count` 等于最终有效帧数、但小于全部候选数，通常说明 invalid 候选在该位置偏离了主序列。

### Payload peak 相位/幅度趋势图

入口：

```text
scripts/plot_payload_peak_trends.py
```

该脚本读取 `run_header_first_demod.py` 导出的 symbol CSV，只保留：

```text
stage == payload
header_valid == 1
```

然后为每个 packet 单独保存一张 PNG，展示该 packet 的 selected FFT peak 相位和幅度趋势。

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\plot_payload_peak_trends.py -i gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv -o gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_16_payload_peak_trends --analysis-output gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_16_payload_peak_trends\0_0_0_10_14_16_payload_peak_trends.csv --dpi 220
```

## 非解码链实验脚本

这些脚本统一放在：

```text
scripts/experiments/
```

它们不是弱包解码主链的必要步骤，而是围绕某个科研问题做对照实验、可视化诊断或数据集构造。主链入口仍然保留在 `scripts/` 根目录，包括：

```text
scripts/detect_weak_preamble.py
scripts/run_weak_sync_chain.py
scripts/run_header_first_demod.py
scripts/plot_payload_peak_trends.py
```

### 近期新增实验脚本

| 脚本 | 输出目录 | 解决的问题 |
| --- | --- | --- |
| `scripts/experiments/run_symbol_phase_threshold_sweep.py` | `data/ablation_*`、`data/symbol_phase_threshold_sweep_*` | 对 symbol-level selector 跑 SNR threshold sweep，比较 multi-offset argmax、energy-only、offset coherence、Top-L、locking、packet-line 等配置 |
| `scripts/experiments/make_offset_coherence_ablation_table.py` | `data/ablation_offset_coherence_summary/` | 汇总多组 sweep/probe 成论文式 ablation 表和 `ablation_report.md`，避免手动翻一堆中间 CSV |
| `scripts/experiments/analyze_phase_opportunity_space.py` | `data/phase_opportunity_space/` | 把每个 payload symbol 分成 already_correct / repairable / unrecoverable，评估 Top-L 召回、phase_repair_rate、phase_damage_rate 和 selected_SER |
| `scripts/experiments/run_symbol_phase_two_stage.py` | `data/symbol_phase_two_stage/` | 单点运行当前 symbol-level two-stage selector，便于对某个 IQ/SNR/packet 做细查 |
| `scripts/experiments/diagnose_symbol_phase_models.py` | `data/symbol_phase_model_diagnostics/` | 用 GT-only 口径诊断 phase/coherence 模型本身能否把正确 bin 排高，不参与主解码 |
| `scripts/experiments/run_two_stage_weak_decoder.py` | `data/two_stage_weak_decoder/` | 早期 codec-block beam decoder，对照 LoRa FEC/CRC 约束的收益；不使用 session template/counter/cross-packet prior |

### Corrected phase 诊断图

入口：

```text
scripts/experiments/plot_corrected_phase_diagnostics.py
```

该脚本读取 `run_header_first_demod.py` 导出的 symbol CSV，只分析 `header_valid == 1` 的 payload symbol。它会为每个 packet 画四联图：wrapped phase、unwrap phase 与一次拟合、去线性趋势后的 residual、residual 与 `raw_fft_bin` 的关系；同时输出 summary CSV，用来判断锯齿相位主要来自 residual/global CFO、SFO/drift，还是 STO 与 bin 的耦合。

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\experiments\plot_corrected_phase_diagnostics.py -i gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv --packet 5 -o gr-lora_sdr\weakPacket_decoding\data\payload_feature_no_offset\plots
```

### Wrong-bin phase/amplitude 对照实验

入口：

```text
scripts/experiments/plot_wrong_bin_phase_diagnostics.py
```

该脚本用于检查 corrected FFT demod 后的相位平滑性是否只属于 selected/正确 bin。它读取 `run_header_first_demod.py` 的 symbol CSV，从原始 IQ 里按已经校正后的 `start_sample`、`CFO_int/CFO_frac` 和 chip-rate 采样口径重算每个 payload symbol 的完整 FFT，然后同时观察：

```text
selected bin
selected + offset 的错误 bin
固定 raw FFT bin
```

每个 packet 会输出一张四联图：wrapped phase、unwrap phase、amplitude、energy ratio。绘图时会故意选 wrong-bin 中 unwrap phase 线性 R2 最高的几条，直接验证“错误 bin 是否也能看起来很平滑”。summary CSV 会额外记录每条 candidate 的 phase R2、residual、平均幅度、平均能量占比、bin rank 以及相对 selected 的 dB 损失。

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\experiments\plot_wrong_bin_phase_diagnostics.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin -s gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv -o gr-lora_sdr\weakPacket_decoding\data\payload_wrong_bin_diagnostics
```

默认输出：

```text
data/payload_wrong_bin_diagnostics/<basename>_wrong_bin_symbol_features.csv
data/payload_wrong_bin_diagnostics/<basename>_wrong_bin_summary.csv
data/payload_wrong_bin_diagnostics/plots/packet_xxx_wrong_bin_phase_amplitude_compare.png
```

### No-offset payload FFT feature 对照导出

入口：

```text
scripts/experiments/export_payload_no_offset_features.py
```

该脚本只读取 `grlora_framesync_valid == 1` 的候选，不重新做弱检测、sync word / netID 检查，也不使用 CFO_int、CFO_frac、STO_frac、SFO 或 gr-lora_sdr corrected downchirp。它从原始 IQ 里按固定 raw symbol 起点切 payload symbol，但 FFT 口径和 corrected 版本一致：每个 chip 取中心样点，做 `2^SF` 点 chip-rate `dechirp + FFT`，导出 selected peak 的幅度、功率、相位、unwrap 相位、peak margin 和 peak energy ratio。

默认 `--anchor located` 使用 `frame_locator` 的 `located_payload_start_sample` 作为 PHY header 第 0 个 symbol 起点，然后跳过 8 个 header symbol 到真正 payload。这个模式更接近“完全不做 offsets 补偿”的消融。如果只想保持 gr-lora fine header 起点、消融 FFT 内部补偿，可以改用 `--anchor fine`。

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\experiments\export_payload_no_offset_features.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin -s gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\sync_chain\0_0_0_10_14_16_sync_chain.csv --sf 10 --bw 125000 --samp-rate 500000 --sync-word 0x34 --preamble-len 16 --peak-gt-csv gr-lora_sdr\weakPacket_decoding\data\peak_groundtruth\0_0_0_10_14_16_peak_gt_preamble8.csv
```

默认输出：

```text
data/payload_feature_no_offset/<basename>_payload_no_offset_features.csv
data/payload_feature_no_offset/plots/packet_xxx_no_offset_peak_trends.png
data/payload_feature_no_offset/plots/packet_xxx_raw_vs_corrected_peak_trends.png
```

### 低 SNR GT-bin 相位/幅度实验

入口：

```text
scripts/experiments/run_low_snr_gt_bin_experiment.py
```

这个脚本用于验证：在给 clean IQ 加入 AWGN 后，clean header-first CSV 里已经确认的 payload `raw_fft_bin` 是否仍然保留稳定的相位/幅度轨迹。它不会重新做弱检测、framesync 或低 SNR argmax 判决，而是把：

```text
stage == payload
header_valid == 1
raw_fft_bin
```

作为 GT bin，从 noisy IQ 的 corrected FFT 里强行读取该 bin 的复数值。这样可以把“正确 bin 的物理相位结构是否还在”和“低 SNR 下 argmax 是否选错”分开观察。

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\experiments\run_low_snr_gt_bin_experiment.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin -g gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv -o gr-lora_sdr\weakPacket_decoding\data\low_snr_gt_bin\0_0_0_10_14_16 --target-snr-db -10 -15 -20 --cfo-correction-mode continuous --overwrite
```

更低 SNR 的一组对照可以单独放到新目录，避免覆盖前一组 summary：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\experiments\run_low_snr_gt_bin_experiment.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin -g gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv -o gr-lora_sdr\weakPacket_decoding\data\low_snr_gt_bin\0_0_0_10_14_16_extreme_snr --target-snr-db -23 -25 -27 --cfo-correction-mode continuous --overwrite
```

主要输出：

```text
data/low_snr_gt_bin/<basename>/<basename>_snr_m10dB.bin
data/low_snr_gt_bin/<basename>/<basename>_snr_m10dB_gt_bin_features.csv
data/low_snr_gt_bin/<basename>/<basename>_low_snr_gt_bin_features_all.csv
data/low_snr_gt_bin/<basename>/<basename>_low_snr_gt_bin_summary.csv
data/low_snr_gt_bin/<basename>/plots/snr_mXXdB/packet_xxx_gt_bin_phase_amp_diagnostics.png
```

`*_summary.csv` 会额外给出 top-K 召回率：

```text
gt_top8_recall   = mean(gt_bin_rank <= 8)
gt_top16_recall  = mean(gt_bin_rank <= 16)
gt_top32_recall  = mean(gt_bin_rank <= 32)
```

这几个指标用于评估后续 peak rerank 算法只在 top-K 候选里搜索时，GT bin 是否还保留在候选集合中。

### 低 SNR wrong-bin 对照实验

入口：

```text
scripts/experiments/run_low_snr_wrong_bin_experiment.py
```

这个脚本复用上一节生成的 noisy IQ，对每个 payload symbol 同时观察 GT bin 和多类错误 bin：

```text
gt          clean CSV 中的正确 bin
argmax      低 SNR FFT hard decision 会选的最大峰
wrong_peak  除 GT bin 之外的最高峰
off+/-N     相对 GT bin 人为偏移的错误 bin
fixXXXX     固定 raw FFT bin
```

实验目标是判断：错误 bin 在低 SNR 下是否也能产生类似 GT bin 的相位 residual 曲线。如果 `wrong_peak` 的幅度/能量接近 GT，但 residual 结构明显更乱，则说明跨符号相位一致性可以为低 SNR FFT demod 提供额外判别力。

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\experiments\run_low_snr_wrong_bin_experiment.py -d gr-lora_sdr\weakPacket_decoding\data\low_snr_gt_bin\0_0_0_10_14_16 -g gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv --cfo-correction-mode continuous
```

如果只跑更低 SNR 组：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\experiments\run_low_snr_wrong_bin_experiment.py -d gr-lora_sdr\weakPacket_decoding\data\low_snr_gt_bin\0_0_0_10_14_16_extreme_snr -g gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv --snr-db -23 -25 -27 --cfo-correction-mode continuous
```

主要输出：

```text
data/low_snr_gt_bin/<basename>/wrong_bin_control/<basename>_low_snr_wrong_bin_features_all.csv
data/low_snr_gt_bin/<basename>/wrong_bin_control/<basename>_low_snr_wrong_bin_summary.csv
data/low_snr_gt_bin/<basename>/wrong_bin_control/plots/snr_mXXdB/packet_xxx_wrong_bin_phase_amp_control.png
```

wrong-bin summary 中对应的通用字段为：

```text
target_top8_hit_rate
target_top16_hit_rate
target_top32_hit_rate
```

当 `candidate_label == gt` 时，它们就是 GT bin 的 top-K recall。

### 低 SNR STO phase-jump 补偿实验

入口：

```text
scripts/experiments/run_low_snr_sto_phase_jump_experiment.py
```

这个脚本用于验证一个更细的 LoRa chirp 现象：residual STO 会让循环移位 chirp 在 wrap 前后出现常相位差，进而让 dechirp+FFT 的相干叠加有亏损；而 residual STO 又会随着 SFO 在 payload symbol 之间缓慢漂移。脚本复用低 SNR noisy IQ 和 clean header-first GT bin，不重新做检测或同步，只在 corrected FFT 前额外做：

```text
tau_s = sfo_cum_before
wrap_cut ~= 2^SF - gt_raw_fft_bin + tau_s
dechirped[wrap_cut:] *= exp(+j * 2*pi*tau_s)
```

这里 `tau_s` 的默认来源是 `*_header_first_symbols.csv` 里的 `sfo_cum_before`，单位是 chip。`phase-sign=plus` 是按当前代码里 `sfo_cum_before` 的符号定义设置的；如果要做符号 sanity check，可以改成 `--phase-sign minus` 对照。

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\experiments\run_low_snr_sto_phase_jump_experiment.py -d gr-lora_sdr\weakPacket_decoding\data\low_snr_gt_bin\0_0_0_10_14_16_extreme_snr -g gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv --snr-db -23 -25 -27 --cfo-correction-mode continuous --tau-source sfo_cum_before --phase-sign plus
```

主要输出：

```text
data/low_snr_gt_bin/<basename>_extreme_snr/sto_phase_jump_corrected/<basename>_sto_jump_gt_bin_features_all.csv
data/low_snr_gt_bin/<basename>_extreme_snr/sto_phase_jump_corrected/<basename>_sto_jump_gt_bin_summary.csv
data/low_snr_gt_bin/<basename>_extreme_snr/sto_phase_jump_corrected/plots/snr_mXXdB/packet_xxx_event_xxx_sto_jump_gt_bin_diagnostics.png
```

## gr-lora_sdr peak groundtruth 导出

入口：

```text
scripts/experiments/export_peak_groundtruth.py
```

该脚本运行 gr-lora_sdr 原始接收链，并监听 `fft_demod` 的 `peak_candidates` 消息端口，导出 high-SNR / clean IQ 的 FFT peak label。

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\experiments\export_peak_groundtruth.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin -o gr-lora_sdr\weakPacket_decoding\data\peak_groundtruth\0_0_0_10_14_16_peak_gt_preamble8.csv --summary-output gr-lora_sdr\weakPacket_decoding\data\peak_groundtruth\0_0_0_10_14_16_peak_gt_preamble8_summary.csv --sf 10 --bw 125000 --samp-rate 500000 --cr 1 --center-freq 487.7e6 --sync-word 0x34 --preamble-len 8 --ldro-mode 2 --crc-mode 0
```

注意：`frame_sync` 的 `preamble_len` 更像同步触发门限，不一定必须等于真实前导码长度。比如 `0_0_0_10_14_16.bin` 用 `--preamble-len 8` 更容易复现 gr-lora_sdr 历史 clean decode 结果。

## 当前实验认识

`run_weak_sync_chain.py` 的检测候选数和 gr-lora_sdr groundtruth 包数不一定一致，原因是二者口径不同：

```text
弱检测候选       LoRa-like 物理 burst，高召回
frame_valid      粗帧定界成立
framesync_valid  前导码补偿 + netID 成组检查通过
peak GT          gr-lora_sdr 完整链能走到 fft_demod/header/payload 的帧
```

因此，弱同步链发现更多候选并不一定是 bug。后续评估时建议分别统计：

```text
detections
selected_events
frame_valid
grlora_framesync_valid
header_valid
payload FFT peak rows
```

## 代码位置

```text
weak_decoder/chirp.py                  gr-lora_sdr 兼容 chirp / dechirp FFT
weak_decoder/preamble_detector.py      多 chirp 能量累加的弱前导码检测
weak_decoder/frame_locator.py          sync word + SFD 粗帧定界
weak_decoder/grlora_frame_sync.py      CFO/STO/SFO 估计与 framesync 验证
weak_decoder/header_first_demod.py     header-first FFT demod 与 header hard decode
weak_decoder/payload_codec.py          LoRa PHY payload 编解码/重编码离线工具
weak_decoder/candidate_pruning.py      phase-aware Top-L 候选筛选指标
weak_decoder/symbol_phase_two_stage.py 当前 symbol-level offset/coherence selector
weak_decoder/two_stage_weak_decoder.py 早期 codec-block beam decoder 对照
weak_decoder/phase_guided_demod.py     phase line/anchor/candidate scoring 基础工具

scripts/detect_weak_preamble.py     独立弱前导码检测入口
scripts/run_weak_sync_chain.py      检测、帧定界、framesync 一体化入口
scripts/run_header_first_demod.py   从 framesync 有效候选继续做 header/payload FFT demod
scripts/plot_payload_peak_trends.py header-first demod 后的 payload selected peak 趋势图
scripts/verify_payload_codec_alignment.py payload codec 与 header-first symbol 对齐检查

scripts/experiments/export_peak_groundtruth.py              gr-lora_sdr 原链 peak groundtruth 导出
scripts/experiments/export_payload_no_offset_features.py    no-offset payload FFT 消融导出
scripts/experiments/plot_corrected_phase_diagnostics.py     corrected selected peak 相位诊断四联图
scripts/experiments/plot_wrong_bin_phase_diagnostics.py     clean IQ corrected wrong-bin 对照实验
scripts/experiments/run_low_snr_gt_bin_experiment.py        低 SNR 下强行读取 GT bin 的相位/幅度实验
scripts/experiments/run_low_snr_wrong_bin_experiment.py     低 SNR 下 GT bin 与 wrong bin 对照实验
scripts/experiments/run_low_snr_sto_phase_jump_experiment.py 低 SNR 下 STO phase-jump 补偿实验
scripts/experiments/make_noisy_iq.py                        通用 AWGN 加噪 IQ 生成工具
scripts/experiments/make_failure_limit_iq.py                噪声失败边界扫描工具
scripts/experiments/analyze_peak_groundtruth.py             peak groundtruth CSV 统计分析
scripts/experiments/evaluate_candidate_pruning_metric.py    Top-L/phase-aware candidate recall 评估
scripts/experiments/run_candidate_pruning_sweep.py          candidate pruning 参数 sweep
scripts/experiments/run_symbol_phase_two_stage.py           symbol-level selector 单点实验
scripts/experiments/run_symbol_phase_threshold_sweep.py     symbol-level selector SNR threshold sweep
scripts/experiments/make_offset_coherence_ablation_table.py offset-coherence 消融汇总表生成
scripts/experiments/analyze_phase_opportunity_space.py      Top-L repairable/unrecoverable 空间分析
scripts/experiments/diagnose_symbol_phase_models.py         GT-only phase/coherence 排名诊断
scripts/experiments/run_two_stage_weak_decoder.py           早期 codec-block beam decoder 对照
```

## Legacy / 已清理内容

2026-06-17 已清理 learn-session / byte-template / packet-structure-prior 相关脚本和数据。当前工作区不再保留这些入口：

```text
scripts/learn_session_*.py
scripts/evaluate_session_priors.py
scripts/reconstruct_session_payloads.py
scripts/sweep_phase_guided_session.py
scripts/sweep_joint_residual_session.py
scripts/joint_decode_residual_candidates.py
scripts/run_phase_guided_demod.py
scripts/reproduce_phase_map_paper_artifacts.py
scripts/make_phase_ablation_table.py
scripts/make_generalization_validation_table.py
scripts/make_joint_threshold_table.py
scripts/sweep_map_kappa_from_candidates.py
data/phase_guided/
```

如果后面要复现实验结论，优先看 `data/ablation_offset_coherence_summary/`、`data/phase_opportunity_space/` 和 `notes/plans/THRESHOLD_GAIN_EVALUATION_2026-06-16.md`。
