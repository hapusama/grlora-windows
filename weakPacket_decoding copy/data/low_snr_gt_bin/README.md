# low_snr_gt_bin 数据目录说明

这个目录保存低 SNR 下的 GT-bin 和 wrong-bin 相位/幅度对照实验输出，主要由：

```text
scripts/experiments/run_low_snr_gt_bin_experiment.py
scripts/experiments/run_low_snr_wrong_bin_experiment.py
```

生成。

## 实验目的

该目录服务于一个核心问题：

```text
当低 SNR 下单符号 FFT argmax 开始选错时，
clean/high-SNR CSV 中确认的 GT bin 是否仍然保留跨符号相位/幅度结构？
错误 bin 是否也会出现类似结构？
```

因此这里的实验不会重新跑弱检测或 framesync，也不会把低 SNR argmax 当真值；而是使用 clean `*_header_first_symbols.csv` 中：

```text
stage == payload
header_valid == 1
raw_fft_bin
```

作为 GT bin。

## 目录结构

```text
<iq_stem>/
  <iq_stem>_snr_m10dB.bin
  <iq_stem>_snr_m10dB_gt_bin_features.csv
  <iq_stem>_low_snr_gt_bin_features_all.csv
  <iq_stem>_low_snr_gt_bin_summary.csv
  <iq_stem>_low_snr_gt_bin_metadata.json
  plots/
    snr_m10dB/
    snr_m15dB/
    snr_m20dB/
  wrong_bin_control/
```

## GT-bin 实验文件

```text
*_snr_mXXdB.bin                      加噪后的 complex64 IQ；体积大，默认被 git ignore
*_snr_mXXdB_gt_bin_features.csv      某一 SNR 档的 GT-bin symbol 级特征
*_low_snr_gt_bin_features_all.csv    所有 SNR 合并后的 GT-bin 特征
*_low_snr_gt_bin_summary.csv         packet/SNR 粒度统计表
*_low_snr_gt_bin_metadata.json       加噪参考功率、SNR 档位、随机种子等 metadata
plots/snr_mXXdB/*_gt_bin_phase_amp_diagnostics.png
                                      每包 GT-bin 四联诊断图
```

GT-bin 图中右下角的点全部表示 GT bin 幅度；蓝点表示该 symbol 的低 SNR argmax 仍选中 GT bin，红点表示低 SNR argmax 被其他 bin 抢走。

## wrong-bin control 文件

```text
wrong_bin_control/*_wrong_bin_features.csv
wrong_bin_control/*_low_snr_wrong_bin_features_all.csv
wrong_bin_control/*_low_snr_wrong_bin_summary.csv
wrong_bin_control/plots/snr_mXXdB/*_wrong_bin_phase_amp_control.png
```

常见 candidate：

```text
gt          clean CSV 中的正确 bin
argmax      低 SNR FFT hard decision 选出的最大峰
wrong_peak  除 GT bin 外的最高峰
off+/-N     相对 GT bin 人为偏移
fixXXXX     固定 raw FFT bin
```

## 关键指标

```text
argmax_correct_rate / argmax_gt_hit_rate
    低 SNR 下传统 argmax 选中 GT bin 的比例。

ER / target_energy_ratio_mean
    目标 bin 功率占完整 FFT 能量的比例：
    |Z[target_bin]|^2 / sum_b |Z[b]|^2

linear_fit_r2
    unwrap phase 被一次线性模型解释的程度。

residual_quad_r2
    unwrap phase 去掉线性趋势后，residual 被二次曲线解释的程度。

target_rank_median
    目标 bin 在当前 FFT 频谱中的排名，中位数越小越突出。
```

## 当前用途

该目录用于验证“跨符号相位 residual 结构”是否可以作为低 SNR FFT demod 的额外判据。当前初步现象是：在 `-20 dB` 附近，`wrong_peak` 的幅度/ER 可以接近 GT，但 residual 曲线通常明显更乱，这说明相位轨迹一致性有潜在判别价值。
