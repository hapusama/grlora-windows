# peak_groundtruth 数据目录说明

本目录保存 `scripts/experiments/export_peak_groundtruth.py` 导出的 peak 级 groundtruth。每个 CSV 文件对应一个 IQ 输入文件。默认情况下，每一行对应一个 **payload chirp**；如果运行脚本时加 `--include-header`，则 PHY header 符号也会被写入 CSV。

这些文件来自 gr-lora_sdr 原始接收链，主要用途是给弱包原型实验提供 high-SNR / clean 条件下的 symbol-level 或 peak-level label。

## 文件类型

```text
*_peak_gt.csv              peak/symbol 级 groundtruth 主表
*_peak_gt_summary.csv      每个 frame/packet 的摘要
*_peak_gt_plots/           对 groundtruth phase/amplitude 的辅助可视化
```

注意：这里的 GT 是 gr-lora_sdr 原链在 clean/high-SNR IQ 上跑出来的 peak label，不等价于最终 payload CRC 解码结果。

## 固定列

| 列名 | 含义 |
| --- | --- |
| `input_file` | 本 CSV 对应的 raw complex64 IQ 输入文件路径。 |
| `packet_index` | 本 CSV 内按 `frame_count` 排序得到的 0-based 包序号，便于和数组/训练样本对齐。 |
| `frame_count` | gr-lora_sdr 检测到的帧编号，用于跨文件或跨噪声版本对齐。 |
| `frame_symbol_index` | 当前帧内的符号序号，从 `fft_demod` 收到的第一个 header 符号开始计数，通常 `0..7` 是 PHY header 符号。 |
| `payload_chirp_index` | 当前 payload 内的 chirp 序号，从 `0` 开始。默认 payload-only 导出时该列连续递增。 |
| `packet_total_symbols` | 当前包被 `fft_demod` 处理到的总符号数，包含 header 和 payload。 |
| `packet_header_symbols` | 当前包的 header 符号数，通常为 `8`。 |
| `packet_payload_chirps` | 当前包的 payload chirp 数。 |
| `is_header` | `1` 表示当前符号属于 PHY header，`0` 表示 payload/data 符号。 |
| `sf` | Spreading Factor。 |
| `cr` | 当前符号所属 payload block 的 coding-rate index。header 符号阶段可能仍为 `0`，因为 header 尚未解出。 |
| `ldro` | `1` 表示当前帧启用 low data rate optimization，`0` 表示未启用。 |
| `samples_per_symbol` | `fft_demod` 实际处理的每符号采样数。注意这是 frame_sync 降采样/校正后的符号长度，等于 `2^SF`。 |
| `label_fft_bin` | 当前 payload chirp 的 FFT bin label。高 SNR clean 文件中它等于 gr-lora_sdr 的 Top1 / argmax bin。 |
| `label_symbol` | `label_fft_bin` 经 gr-lora_sdr 符号映射后的 symbol label。 |
| `label_real` / `label_imag` | label peak 对应的复数 FFT 值。 |
| `label_power` | label peak 功率。 |
| `label_phase` | label peak 相位，单位弧度。 |
| `label_confidence_db` | label peak 与第二强 peak 的功率比，单位 dB。 |
| `label_reliable` | `label_confidence_db >= --min-label-confidence-db` 时为 `1`，默认阈值为 6 dB。该字段只用于标记可靠性，不会自动丢弃行。 |
| `top_k` | 每个符号导出的候选 peak 数量，当前默认为 `16`。 |
| `hard_bin` | gr-lora_sdr 传统 FFT argmax 选出的频域 bin。clean/high-SNR 文件中可作为 peak 级 groundtruth。 |
| `hard_symbol` | `hard_bin` 经 gr-lora_sdr 符号映射后的值。基本映射为 `(hard_bin - 1) mod 2^SF`，header 或 LDRO 情况下还会除以 `4`。 |
| `confidence_db` | Top1 与 Top2 peak 功率比，单位 dB。值越大，传统 argmax 越可靠。 |
| `total_power` | 当前符号完整 FFT 频谱的功率和。 |
| `noise_power_est` | 除 Top1 外其余 bin 的平均功率估计，可作为粗噪声底参考。 |
| `cfo_int` | frame_sync 估计并传给 fft_demod 的整数 CFO。 |
| `cfo_frac` | frame_sync 估计并传给 fft_demod 的小数 CFO。 |

## Top-K 候选列

每个候选 peak 展开成一组列：

```text
topN_bin
topN_symbol
topN_real
topN_imag
topN_power
topN_phase
```

其中 `N=1..top_k`，当前通常是 `1..16`。

| 列名模式 | 含义 |
| --- | --- |
| `topN_bin` | 第 N 强 FFT peak 的 bin index。`top1_bin` 等于 `hard_bin`。 |
| `topN_symbol` | 第 N 强 peak 按 gr-lora_sdr 规则映射后的符号值。 |
| `topN_real` | 第 N 强 peak 对应复数 FFT 值的实部。 |
| `topN_imag` | 第 N 强 peak 对应复数 FFT 值的虚部。 |
| `topN_power` | 第 N 强 peak 的功率，即 `real^2 + imag^2`。 |
| `topN_phase` | 第 N 强 peak 的复数相位，单位弧度，范围约为 `[-pi, pi]`。 |

## 推荐用法

1. 用 clean 或 high-SNR IQ 导出 CSV，把 `label_fft_bin` / `label_symbol` 当作 payload chirp 的 symbol-level groundtruth。
2. 对同一份 clean IQ 手动加噪，再导出 noisy peak CSV 或运行弱包 peak 搜索算法。
3. 用 `frame_count + payload_chirp_index` 或 `packet_index + payload_chirp_index` 对齐 clean 与 noisy 结果。
4. 常用指标：
   - `noisy_top1_bin == clean_label_fft_bin`：传统 argmax 是否正确。
   - `clean_label_fft_bin in noisy_top1..topK_bin`：正确 peak 是否仍在候选集合中。
   - `reranked_bin == clean_label_fft_bin`：弱包重排序是否救回正确 peak。

注意：本 CSV 是 peak/符号级数据，不是 payload 级解码结果；它不直接表示 CRC 是否通过。
