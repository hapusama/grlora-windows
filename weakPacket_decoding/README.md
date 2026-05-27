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

当前还没有接完整的 dewhitening / CRC payload 解码链。现阶段重点是验证：弱检测和同步结果能否稳定支撑 header 解码与 payload FFT peak 导出。

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
  -> chirp 起点粗对齐
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

`data/weak_sync_chain/` 根目录保持干净，当前 CSV 按用途放在这些子目录里：

```text
sync_chain/                 run_weak_sync_chain.py 的主表 CSV
framesync_peaks/            framesync 后前导码 FFT peak 验证表
header_first/               header-first demod 的 frame / symbol CSV
payload_consistency/        payload FFT bin 内部一致性检查表
legacy_experiments/         早期或对照实验输出
*_stft/                     STFT PNG / CSV
*_framesync_spectrum/       framesync 前后频谱对比图
*_payload_peak_trends/      每包 payload peak 相位/幅度趋势图
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

输出：

```text
*_header_first_frames.csv   每个 frame 的 header 解码摘要
*_header_first_symbols.csv  每个 header/payload symbol 的 FFT peak
```

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

### No-offset payload FFT feature 对照导出

入口：

```text
scripts/export_payload_no_offset_features.py
```

该脚本只读取 `grlora_framesync_valid == 1` 的候选，不重新做弱检测、sync word / netID 检查，也不使用 CFO_int、CFO_frac、STO_frac、SFO 或 gr-lora_sdr corrected downchirp。它从原始 IQ 里按固定 raw symbol 起点切 payload symbol，但 FFT 口径和 corrected 版本一致：每个 chip 取中心样点，做 `2^SF` 点 chip-rate `dechirp + FFT`，导出 selected peak 的幅度、功率、相位、unwrap 相位、peak margin 和 peak energy ratio。

默认 `--anchor located` 使用 `frame_locator` 的 `located_payload_start_sample` 作为 PHY header 第 0 个 symbol 起点，然后跳过 8 个 header symbol 到真正 payload。这个模式更接近“完全不做 offsets 补偿”的消融。如果只想保持 gr-lora fine header 起点、消融 FFT 内部补偿，可以改用 `--anchor fine`。

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\export_payload_no_offset_features.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin -s gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\sync_chain\0_0_0_10_14_16_sync_chain.csv --sf 10 --bw 125000 --samp-rate 500000 --sync-word 0x34 --preamble-len 16 --peak-gt-csv gr-lora_sdr\weakPacket_decoding\data\peak_groundtruth\0_0_0_10_14_16_peak_gt_preamble8.csv
```

默认输出：

```text
data/payload_feature_no_offset/<basename>_payload_no_offset_features.csv
data/payload_feature_no_offset/plots/packet_xxx_no_offset_peak_trends.png
data/payload_feature_no_offset/plots/packet_xxx_raw_vs_corrected_peak_trends.png
```

## gr-lora_sdr peak groundtruth 导出

入口：

```text
scripts/export_peak_groundtruth.py
```

该脚本运行 gr-lora_sdr 原始接收链，并监听 `fft_demod` 的 `peak_candidates` 消息端口，导出 high-SNR / clean IQ 的 FFT peak label。

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\export_peak_groundtruth.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin -o gr-lora_sdr\weakPacket_decoding\data\peak_groundtruth\0_0_0_10_14_16_peak_gt_preamble8.csv --summary-output gr-lora_sdr\weakPacket_decoding\data\peak_groundtruth\0_0_0_10_14_16_peak_gt_preamble8_summary.csv --sf 10 --bw 125000 --samp-rate 500000 --cr 1 --center-freq 487.7e6 --sync-word 0x34 --preamble-len 8 --ldro-mode 2 --crc-mode 0
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
weak_decoder/chirp.py              gr-lora_sdr 兼容 chirp / dechirp FFT
weak_decoder/preamble_detector.py  多 chirp 能量累加的弱前导码检测
weak_decoder/frame_locator.py      sync word + SFD 粗帧定界
weak_decoder/grlora_frame_sync.py  CFO/STO/SFO 估计与 framesync 验证
weak_decoder/header_first_demod.py header-first FFT demod 与 header hard decode

scripts/detect_weak_preamble.py    独立弱前导码检测入口
scripts/run_weak_sync_chain.py     检测、帧定界、framesync 一体化入口
scripts/run_header_first_demod.py  从 framesync 有效候选继续做 header/payload FFT demod
scripts/export_peak_groundtruth.py gr-lora_sdr 原链 peak groundtruth 导出
```

## Legacy / 历史实验

以下模块属于早期探索，当前主流程暂不依赖：

```text
scripts/estimate_initial_state.py
weak_decoder/initial_state.py
scripts/decode_weak_packet.py
weak_decoder/phase_model.py
weak_decoder/observations.py
```

它们主要用于早期相干初始状态估计、phase anchor rerank 和 noisy IQ peak 搜索实验。当前主线以 `run_weak_sync_chain.py` 和 `run_header_first_demod.py` 为准。
