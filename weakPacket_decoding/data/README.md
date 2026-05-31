# data 目录说明

这个目录保存 `weakPacket_decoding` 原型系统的中间结果、实验输出和可视化图。总体原则是：

- 主解码链输出放在 `weak_preamble_detections/` 和 `weak_sync_chain/`。
- 对照实验、消融实验、诊断图放在独立目录中，避免和主链 CSV 混在一起。
- `.bin` 文件通常是原始或加噪 IQ，体积很大，当前 `.gitignore` 会忽略 `weakPacket_decoding/data/**/*.bin`。
- CSV、JSON、PNG 默认保留，便于复现实验结论和画图。

顶层子目录用途：

```text
weak_preamble_detections/       弱前导码检测事件和滑窗级调试表
weak_sync_chain/                弱检测 -> 帧定界 -> framesync -> header-first demod 主链输出
peak_groundtruth/               gr-lora_sdr 原链导出的 peak/symbol 级 groundtruth
noisy_iq/                       通用 AWGN 加噪 IQ sweep 和早期 noisy 实验输出
payload_feature_no_offset/      no-offset payload FFT 特征和 raw/corrected 对比图
payload_wrong_bin_diagnostics/  clean IQ 下 corrected wrong-bin 对照实验
low_snr_gt_bin/                 低 SNR GT-bin 与 wrong-bin 相位/幅度实验
```

对应脚本主要位于：

```text
scripts/
scripts/experiments/
```

其中 `scripts/` 是主链入口，`scripts/experiments/` 是非解码链科研实验脚本。
