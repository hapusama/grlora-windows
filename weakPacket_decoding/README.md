# weakPacket_decoding 原型说明

这个目录当前新增了一个弱包解码 Python 原型，主线对应 `doc/弱包解码方案粗设计.md`：

```text
gr-lora_sdr 包检测/粗同步 metadata
  -> raw IQ packet window
  -> dechirp + FFT 复数 Top-K 候选
  -> 高置信度 anchor 相位拟合
  -> 低置信度符号 Top-L 轻量重排序
  -> symbol likelihood / bit LLR JSON
```

## 快速运行

在仓库根目录运行：

```powershell
python gr-lora_sdr\weakPacket_decoding\scripts\decode_weak_packet.py `
  -m gr-lora_sdr\weakPacket_decoding\data\noisy_iq\0_0_0_10_14_8\0_0_0_10_14_8_noise_rel_30p0dB.json `
  --top-k 8 `
  -o gr-lora_sdr\weakPacket_decoding\data\noisy_iq\0_0_0_10_14_8\_phase_anchor_v1
```

输出：

```text
*_observations.json  # 每个符号的复数 FFT Top-K 候选
*_rerank.json        # anchor 相位模型、重排序分数、候选概率、bit LLR
*_symbols.csv        # 一行一个符号，便于快速看 hard decision 是否变化
```

## peak 级 groundtruth 导出

已经在 `gr-lora_sdr/lib/fft_demod_impl.cc` 新增消息端口：

```text
peak_candidates
```

每个进入 `fft_demod` 的 LoRa 符号都会发布一条消息，包含：

```text
frame_count, symbol_index, is_header, sf, cr, ldro
hard_bin, hard_symbol, confidence_db
candidate_bins, candidate_symbols
candidate_values, candidate_powers, candidate_phases
```

导出 high-SNR / clean IQ 的 peak 级 groundtruth CSV：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe `
  gr-lora_sdr\weakPacket_decoding\scripts\export_peak_groundtruth.py `
  -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin `
  -o gr-lora_sdr\weakPacket_decoding\data\peak_groundtruth\0_0_0_10_14_8_peak_gt.csv `
  --sf 10 --samp-rate 500000 --bw 125000 --cr 1 `
  --sync-word 0x34 --preamble-len 8 --ldro-mode 2 --crc-mode 0
```

推荐实验流程：

```text
clean/high-SNR IQ
  -> export_peak_groundtruth.py 得到 peak_gt.csv
  -> 对同一份 clean IQ 手动加噪
  -> 弱包 peak 搜索/重排序
  -> 用 frame_count + symbol_index 对齐比较 hard_bin / candidate_bins
```

## 当前实现边界

- 入口依赖已有 `noisy_iq/*.json` 里的 `packet_measurements`，即默认复用 gr-lora_sdr 做包检测和粗同步。
- FFT 候选提取用 NumPy 复刻 gr-lora_sdr 的 chirp 公式，不需要重新编译 C++。
- 当前相位模型是第一版轻量 baseline：高置信度符号作为 anchor，做加权线性相位拟合，再用幅度项 + 相位一致性项重排 Top-K。
- 已输出 bit LLR，但还没有接回 gr-lora_sdr 的 deinterleaver/hamming/dewhitening/CRC 链路；下一步应把 `_rerank.json` 的 LLR 接到编码层验证 payload/CRC 是否被救回。
- CFO/STO/SFO 目前使用 gr-lora_sdr metadata 做粗补偿，尚未实现 Hi2LoRa 式的两段 STO 相位对齐项和包内 SFO 漂移细化。

## 弱包前导码滑窗检测

`scripts/detect_weak_preamble.py` 是一个独立于 GNU Radio 的前导码检测原型，对应 `doc/弱包检测逻辑.md`：

```text
raw IQ
  -> 按 win_chirps 个 chirp 做滑窗
  -> 每个 chirp 保留过采样长度做 dechirp+FFT
  -> 按 bin 累加窗口内所有 chirp 的 FFT 能量
  -> 记录每个窗口最大 peak bin
  -> 连续窗口 peak bin 稳定则报 preamble 检测
```

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe `
  gr-lora_sdr\weakPacket_decoding\scripts\detect_weak_preamble.py `
  -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin `
  -o gr-lora_sdr\weakPacket_decoding\data\weak_preamble_detections\0_0_0_10_14_8_events.csv `
  --windows-csv gr-lora_sdr\weakPacket_decoding\data\weak_preamble_detections\0_0_0_10_14_8_windows.csv `
  --sf 10 --bw 125000 --samp-rate 500000 `
  --win-chirps 4 --min-periodic-peaks 6 --bin-tol 2
```

当前版本只做检测，不输出 payload 起点精修、CFO/STO/SFO 估计，也不接后续解码链。

## 前导码初始状态估计

`scripts/estimate_initial_state.py` 接在检测事件后面，对每个 `events.csv` 里的粗前导码起点做相干同步估计：

```text
检测事件 start_sample
  -> 取连续 estimate_chirps 个过采样 upchirp
  -> 搜索 tau0(chip) 和 beta(CFO bin)
  -> 用 bin0 相干叠加目标 |sum_s Z_s[0]|^2 选最大值
  -> 输出粗 payload_start_sample、tau0、beta、zeta
```

示例：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe `
  gr-lora_sdr\weakPacket_decoding\scripts\estimate_initial_state.py `
  -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin `
  -d gr-lora_sdr\weakPacket_decoding\data\weak_preamble_detections\0_0_0_10_14_8_events.csv `
  -o gr-lora_sdr\weakPacket_decoding\data\initial_state_estimates\0_0_0_10_14_8_initial_state.csv `
  --sf 10 --bw 125000 --samp-rate 500000 --preamble-len 8 `
  --estimate-chirps 6 --tau-step 4 --beta-min -64 --beta-max 64
```

第一版默认 `zeta=0`，也就是暂时不估 SFO；需要实验 SFO 时再加 `--zeta-span` 和 `--zeta-step` 做局部细搜。

## 一体化同步链入口

`scripts/run_weak_sync_chain.py` 会把“前导码滑窗检测 -> chirp 粗对齐 -> sync word/SFD 帧定位 -> 初始状态估计”串起来，输出一个总表。默认按文件名 `experiment_corridor_position_sf_txpower_preamble.bin` 推断 `SF` 和 `preamble_len`，并且每个检测事件独立估计，避免不同包混在一起。

按当前弱包检测设计，短前导码建议先用 2 个 chirp 的检测窗口、1 个 chirp 的滑动步长：

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe `
  gr-lora_sdr\weakPacket_decoding\scripts\run_weak_sync_chain.py `
  -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin `
  -o gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_8_sync_chain.csv `
  --bw 125000 --samp-rate 500000 `
  --win-chirps 2 --hop-chirps 1
```

如需把定位出的 `preamble + sync word + SFD` 区间画成 STFT 验证图，可额外加：

```powershell
--stft-dir gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_8_stft
```

如需检查中间过程，可额外加：

```powershell
--events-csv gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_8_events.csv `
--windows-csv gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_8_windows.csv
```

## 代码位置

```text
weak_decoder/chirp.py          # gr-lora_sdr 兼容 chirp / dechirp FFT
weak_decoder/observations.py   # 从 packet window 抽取每符号 Top-K 复数候选
weak_decoder/phase_model.py    # anchor 相位拟合与 Top-L rerank
weak_decoder/preamble_detector.py # 多 chirp 能量累加的弱包前导码检测
weak_decoder/frame_locator.py  # sync word + SFD 帧定位
weak_decoder/initial_state.py  # 前导码相干叠加初始 CFO/STO/SFO 状态估计
weak_decoder/decode_packet.py  # CLI 主逻辑
scripts/decode_weak_packet.py  # 仓库内薄包装入口
scripts/detect_weak_preamble.py # 弱包前导码滑窗检测入口
scripts/estimate_initial_state.py # 检测事件后的初始状态估计入口
scripts/run_weak_sync_chain.py # 检测、chirp 对齐、初始状态估计一体化入口
```
