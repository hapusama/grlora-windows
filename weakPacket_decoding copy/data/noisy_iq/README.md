# noisy_iq 数据目录说明

这个目录保存通用 AWGN 加噪 IQ sweep 和早期 noisy-IQ 相关实验输出，主要由：

```text
scripts/experiments/make_noisy_iq.py
scripts/experiments/make_failure_limit_iq.py
```

以及早期 phase/sequence 原型脚本生成。

## 目录结构

常见结构如下：

```text
<iq_stem>/
  <iq_stem>_noise_rel_10p0dB.bin
  <iq_stem>_noise_rel_10p0dB.json
  <iq_stem>_noise_sweep_summary.csv
  <iq_stem>_noise_sweep_summary.json

_default_phase_test/
_live_phase_test/
```

其中 `<iq_stem>` 通常对应原始 IQ 文件名，例如：

```text
0_0_0_10_14_8
0_0_0_10_14_16
0_0_0_10_14_32
```

## 文件类型

```text
*_noise_rel_*.bin              加噪后的 complex64 IQ；体积大，默认被 git ignore
*_noise_rel_*.json             单个加噪文件的 metadata、噪声功率、测量摘要
*_noise_sweep_summary.csv      一组噪声 sweep 的表格摘要
*_noise_sweep_summary.json     一组噪声 sweep 的完整 JSON 摘要
*_sequence_observations.*      早期序列/相位观测原型输出
*_phase_*                      早期 phase anchor / rerank 实验输出
```

## 命名注意

`noise_rel_10p0dB` 里的 `10p0dB` 是“添加噪声功率相对参考功率”的 dB 值，不一定等价于最终文件的绝对 SNR。具体参考功率和测量结果请看同名 JSON 或 sweep summary。

## 与 `low_snr_gt_bin/` 的区别

`noisy_iq/` 是通用加噪数据池，通常仍服务于检测/解码链跑通性测试。

`low_snr_gt_bin/` 是专门为“GT bin 相位/幅度是否在低 SNR 下保留结构”设计的实验目录，会直接保存 GT-bin features、wrong-bin control 和对应四联图。
