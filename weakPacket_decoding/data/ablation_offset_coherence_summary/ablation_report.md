# Offset-Coherence 消融实验汇总

范围：`gr-lora_sdr/weakPacket_decoding copy`

正式完整 sweep 在三个验证数据集上使用 `--snr-start -12 --snr-stop -26 --snr-step -1`。
Probe 行只使用 `-20..-23 dB`，因此应比较曲线数值，而不是完整阈值增益。

## 主要完整 Sweep 消融

| ID | 变体 | Multi | Top-L | Lock | Coherence | Line | SER 增益 | CRC90 增益 | 相对 Multi 的 SER 增益 | SER@-22 | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | center argmax | no | 1 | no | no | no | 0.00 | 0.00 |  | 0.646 |  |
| A1 | multi-offset argmax | yes | 1 | no | no | no | 3.38 | 2.65 | 0.00 | 0.204 |  |
| A2 | energy-only selected Top-24 | yes | 24 | yes | no | no | 3.38 | 2.65 | 0.00 | 0.204 | 0.924 |
| A3 | offset coherence only Top-24 | yes | 24 | yes | yes | no | 3.20 |  | -0.18 | 0.203 | 0.924 |
| A4 | energy + offset coherence, no line | yes | 24 | yes | yes | no | 4.69 | 4.14 | 1.31 | 0.103 | 0.924 |
| A5 | current default | yes | 24 | yes | yes | small | 4.74 | 4.14 | 1.36 | 0.101 | 0.924 |
| A6 | packet-line phase only Top-24 | yes | 24 | yes | no | yes |  |  |  | 0.576 | 0.924 |
| A8 | no high-confidence lock | yes | 24 | no | yes | small | 4.65 | 4.14 | 1.27 | 0.104 | 0.940 |

## Top-L 完整 Sweep 消融

| ID | Top-L | SER 增益 | CRC90 增益 | 相对 Multi 的 SER 增益 | SER@-20 | SER@-21 | SER@-22 | SER@-23 | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A7-L8 | 8 | 4.60 | 4.03 | 1.22 | 0.046 | 0.063 | 0.108 | 0.212 | 0.885 |
| A7-L16 | 16 | 4.73 | 4.14 | 1.34 | 0.044 | 0.063 | 0.102 | 0.188 | 0.911 |
| A7-L24 | 24 | 4.74 | 4.14 | 1.36 | 0.044 | 0.060 | 0.101 | 0.184 | 0.924 |
| A7-L32 | 32 | 4.71 | 4.14 | 1.32 | 0.044 | 0.060 | 0.103 | 0.182 | 0.933 |

## 快速 Probe（-20..-23 dB）

| ID | 变体 | 平均 SER | 平均 CRC/PRR | 相对默认的 SER 变化 | 相对默认的 CRC 变化 |
| --- | --- | --- | --- | --- | --- |
| A5 | current default | 0.097 | 0.509 | 0.000 | 0.000 |
| A9-C32 | coherence candidate Top-32 | 0.099 | 0.493 | 0.002 | -0.016 |
| A9-C64 | coherence candidate Top-64 | 0.099 | 0.493 | 0.002 | -0.016 |
| A9-C128 | coherence candidate Top-128 | 0.099 | 0.493 | 0.002 | -0.016 |
| A10 | smooth trajectory beam probe | 0.123 | 0.369 | 0.026 | -0.140 |

## 解释

- Energy-only selected 与 multi-offset argmax 完全一致，说明 selected path 机制本身不会凭空带来额外增益。
- Offset coherence 在 multi-offset energy 之上带来明确的阈值增益；但 amplitude/energy 保护很重要，因为 coherence-only 更弱。
- Packet-line phase 单独使用时没有竞争力。当前默认配置只把它作为一个很小的辅助项。
- 在这组矩阵里，Top-24 是完整 sweep 下最好的低复杂度折中；Top-32 虽然提高了 recall，但没有改善正式 SER/CRC 阈值。
- 关闭 high-confidence lock 会降低阈值表现，支持“lock 保护已经可靠的符号，避免被过度 rerank”的判断。
- 在 `-20..-23 dB` probe 范围里，coherence-candidate 扩展和 smooth beam probe 都不足以替代当前默认配置。
