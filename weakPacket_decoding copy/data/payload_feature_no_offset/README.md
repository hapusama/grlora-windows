# payload_feature_no_offset 数据目录说明

这个目录保存 no-offset payload FFT 消融实验的输出，主要由：

```text
scripts/experiments/export_payload_no_offset_features.py
scripts/experiments/plot_corrected_phase_diagnostics.py
```

生成。

## 实验目的

该实验用已有同步链确定 packet 的位置，但在 payload FFT 特征提取时禁用 offsets compensation，用来观察：

```text
完全不做 CFO/STO/SFO 补偿时，payload selected peak 的幅度和相位如何变化
```

这不是为了正确解码，而是为了构造 offsets compensation 的消融对照。

## 文件类型

```text
*_payload_no_offset_features.csv       no-offset payload symbol 级 FFT feature
plots/*_no_offset_peak_trends.png      每个 packet 的 no-offset 相位/幅度趋势图
plots/*_raw_vs_corrected_peak_trends.png
                                       no-offset 与 corrected selected peak 对比图
plots/*_corrected_phase_diagnostics.png
                                       corrected selected peak 相位诊断四联图
plots/corrected_phase_diagnostics_summary.csv
                                       corrected 相位拟合/残差统计表
```

## 关键字段

```text
packet_id / event_id / frame_id
payload_symbol_idx
mode
raw_fft_bin
selected_peak_amp
selected_peak_power
selected_peak_phase
selected_peak_phase_unwrap
peak_margin_db
total_fft_energy
peak_energy_ratio
```

如果提供 peak GT，还会附加：

```text
gt_bin
is_argmax_correct
gt_bin_amp
gt_bin_phase
gt_bin_rank
```

## 注意事项

no-offset 模式只使用已有 frame/sync 边界作为切片锚点，不应偷偷使用：

```text
CFO_int 修正
CFO_frac 相位旋转
STO_frac fractional sample 对齐
SFO sample insertion/deletion
SFO cumulative correction
gr-lora_sdr corrected downchirp
```

如果使用 `--anchor fine`，仍然会借用 gr-lora fine header 起点作为切片起点；如果使用 `--anchor located`，更接近“完全不做 offsets compensation”的消融设置。
