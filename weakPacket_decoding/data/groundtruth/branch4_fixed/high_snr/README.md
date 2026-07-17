# Branch4 high-SNR FFT-bin ground truth

源 IQ：

```text
USRP_collector/data/branch4_fixed/high_snr/sf10_bw125_fs500_pre32_sw34_r001.bin
```

配置：SF10、BW 125 kHz、采样率 500 ksample/s、前导码 32、sync word 0x34、
explicit header、CR 4/7、PHY CRC enabled、PHY payload 33 bytes、LDRO off。

## 正式真值

`sf10_bw125_fs500_pre32_sw34_r001_fft_bin_groundtruth.csv` 是后续 SER 实验应使用的
57 行 consensus 表：前 8 行是 PHY header，后 49 行是 payload 编码 symbols。

核心字段：

```text
frame_symbol_index       data region 内的位置，0..56
stage                    header 或 payload
stage_symbol_index       各 stage 内的位置
groundtruth_fft_bin      dechirp + 1024 点 FFT 的真值 bin，范围 0..1023
groundtruth_symbol       gr-lora_sdr hard symbol 映射结果
support_frames           原生 gr-lora_sdr 重复帧支持数
agreement_ratio          重复帧对该 bin 的一致比例
consensus_reliable       该位置是否满足一致性和单帧置信度要求
```

本次原生 `gr-lora_sdr::fft_demod` 解出 10 帧，全部 57 个位置均为 10/10 一致，
最低 peak confidence 为 13.86 dB；`weak_decoder` 另外得到 17 个同步有效帧，其
逐位置众数与原生 gr-lora_sdr 在 57/57 个位置完全一致。

注意：`groundtruth_fft_bin` 是 FFT 数组下标。payload 的映射约定为
`groundtruth_symbol = (groundtruth_fft_bin - 1) mod 1024`；header 还会按 LoRa
显式头规则除以 4。做 FFT-bin SER 时比较 `groundtruth_fft_bin`，不要混用
`groundtruth_symbol`。

## 诊断文件

```text
*_sync.csv                    weak_decoder 检测与 frame-sync 结果
*_fft_symbols.csv             weak_decoder 逐帧逐 symbol FFT 结果
*_fft_frames.csv              weak_decoder 显式头摘要
*_payload_bin_consistency.csv weak_decoder payload 跨帧一致性
*_grlora_fft_peaks.csv        原生 gr-lora_sdr 逐帧 Top-K FFT peaks
*_grlora_fft_summary.csv      原生 gr-lora_sdr 逐帧摘要
```
