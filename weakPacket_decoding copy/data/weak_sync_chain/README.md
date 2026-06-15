# weak_sync_chain 数据目录说明

这个目录保存弱包检测、帧定界、framesync、header-first FFT demod 相关输出。

根目录尽量不直接放 CSV；CSV 按用途放在子目录里：

```text
sync_chain/                 run_weak_sync_chain.py 主表
framesync_peaks/            framesync 后前导码 peak 验证表
header_first/               run_header_first_demod.py 的 frame / symbol 表
payload_consistency/        payload FFT bin 内部一致性检查表
legacy_experiments/         历史或对照实验结果
```

图像类输出目前仍按实验名独立成目录：

```text
*_stft/                     帧定界 STFT 图
*_framesync_spectrum/       framesync 前后 dechirp+FFT 对比图
*_payload_peak_trends/      每包 payload peak 相位/幅度趋势图
```

已清理的临时文件：

```text
0_0_0_10_14_16_payload_peak_trends.png
```

这张图是旧版“所有 packet 叠在一张图”的过渡产物，当前已经改为每个 packet 单独保存一张 PNG。

## 接口契约与命名注意事项

当前弱同步链路是分模块开发的，模块之间的输入输出大致如下：

```text
raw IQ
  -> weak preamble detection
  -> preamble anchor refinement
  -> sync word + SFD coarse frame locator
  -> gr-lora_sdr 风格 frame sync validation
  -> header-first FFT demod
```

### `align_event_start` 不是严格 chirp boundary alignment

`run_weak_sync_chain.py` 里的 `align_event_start()` 只是用多个 preamble upchirp 的累加 FFT peak power，在检测事件附近找一个更稳定的 sample anchor。

整数 chip 级偏移不会让 preamble peak 消失；它通常只会让 dechirp 后的 peak bin 整体平移。因此这一步不能唯一消除整数 chip ambiguity，也不应描述为精确 chirp 边界对齐。

推荐术语：

```text
preamble anchor refinement
sample-level preamble anchor search
```

整数 chip ambiguity 由后续 `grlora_frame_sync.py` 里的 k_hat / coarse sync 处理：

```text
preamble_ref_bin
  -> signed bin
  -> coarse_offset_chips
  -> coarse_offset_samples
  -> synced_preamble_start_sample
```

### FFT bin 坐标系

前半段 weak detector 和 coarse frame locator 使用过采样 FFT：

```text
fft_len = chirp_samples = 2^SF * os_factor
```

因此这些字段属于 oversampled FFT bin 坐标：

```text
detected_reference_bin
align_peak_bin
preamble_ref_bin
sync1_bin
sync2_bin
sfd1_bin
sfd2_bin
```

后半段 gr-lora_sdr 风格 framesync / header-first demod 使用 chip-rate FFT：

```text
fft_len = 2^SF
```

因此这些字段属于 chip-rate / gr-lora_sdr demod 坐标：

```text
grlora_netid1_est
grlora_netid2_est
raw_fft_bin
signed_fft_bin
symbol_value
```

注意：当前很多字段名都叫 `*_bin`，但没有在字段名里显式区分 oversampled bin 和 chip-rate bin。后续新增模块时要先确认该字段来自哪个阶段，不要直接混用。

### `payload_start` 命名

`grlora_synced_payload_start_sample` 和 `grlora_fine_payload_start_sample` 沿用了 gr-lora_sdr / frame_sync 语义，实际表示 LoRa data 区起点。

对于 explicit header 模式，这个位置是：

```text
PHY header 第 0 个 symbol 起点
```

不是 MAC payload 的第 0 个 payload symbol 起点。

`run_header_first_demod.py` 会把这个位置作为 `header_start_sample` 使用，先 demod 8 个 explicit header symbol，再根据 header 结果继续 demod payload symbols。

### 弱检测是高召回策略

`preamble_detector.py` 的 weak preamble detection 主要看连续滑窗中的 FFT peak bin 是否稳定，不设置强绝对能量门限。

这有利于弱包召回，但稳定窄带干扰或其它周期结构也可能形成 detection event。因此后续必须继续使用：

```text
sync word 检查
SFD 检查
gr-lora_sdr frame sync validation
header checksum
```

来逐级过滤候选。
