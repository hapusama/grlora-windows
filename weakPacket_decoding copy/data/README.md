# data 目录说明

这个目录只保留 `weakPacket_decoding` 原型系统里最常用、体积较小的主链输出和正式摘要。总体原则是：

- 主解码链 CSV 保留在 `weak_preamble_detections/` 和 `weak_sync_chain/`。
- 当前 two-stage / offset-coherence 正式小表保留，便于快速查看结论。
- 低 SNR noisy IQ、诊断 PNG、probe/sweep 中间结果默认不保留，需要时按 README 命令重新生成。
- `.bin` 文件通常是原始或加噪 IQ，体积很大，当前 `.gitignore` 会忽略 `weakPacket_decoding/data/**/*.bin`。

当前保留的顶层子目录：

```text
weak_preamble_detections/       弱前导码检测事件和滑窗级调试表
weak_sync_chain/                弱检测 -> 帧定界 -> framesync -> header-first demod 主链输出
two_stage_weak_decoder/         早期 codec-block two-stage decoder 小型结果
symbol_phase_two_stage/         symbol-level selector 单点实验
phase_opportunity_space/        Top-L repairable/unrecoverable 空间摘要
ablation_current_default/       当前默认 offset-coherence selector threshold sweep
ablation_offset_coherence_summary/ offset-coherence 消融汇总表和报告
ablation_*/                     当前保留的正式 selector/Top-L/locking 消融小表
```

按需生成、默认已清理的目录：

```text
peak_groundtruth/               gr-lora_sdr 原链 peak/symbol groundtruth
noisy_iq/                       AWGN 加噪 IQ sweep，通常包含大体积 .bin
low_snr_gt_bin/                 低 SNR GT-bin / wrong-bin / STO phase-jump 实验
candidate_pruning/              Top-L candidate recall 和 phase-aware pruning sweep
payload_feature_no_offset/      no-offset/corrected payload FFT 诊断
payload_wrong_bin_diagnostics/  wrong-bin phase/amplitude 诊断图
phase_line/                     phase-line 可视化或诊断输出
symbol_phase_threshold_sweep*/  旧 selector sweep 中间结果
probe_*/                        手动调参 quick probe
```

对应脚本主要位于：

```text
scripts/
scripts/experiments/
```

其中 `scripts/` 是主链入口，`scripts/experiments/` 是非解码链科研实验脚本。

2026-06-17 已清理两类数据：

- 基于 learn-session、payload byte template、counter model 或 cross-packet joint prior 的旧实验数据，原 `phase_guided/` 中间结果不再保留。
- 和当前 framesync / two-stage 主流程无直接关系、且可由脚本再生的大体积中间结果，包括 noisy IQ、低 SNR 构造数据、诊断 PNG、probe/sweep 输出。

当前正式结论优先查看：

```text
ablation_offset_coherence_summary/
phase_opportunity_space/
ablation_current_default/
```
