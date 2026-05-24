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

## 代码位置

```text
weak_decoder/chirp.py          # gr-lora_sdr 兼容 chirp / dechirp FFT
weak_decoder/observations.py   # 从 packet window 抽取每符号 Top-K 复数候选
weak_decoder/phase_model.py    # anchor 相位拟合与 Top-L rerank
weak_decoder/decode_packet.py  # CLI 主逻辑
scripts/decode_weak_packet.py  # 仓库内薄包装入口
```
