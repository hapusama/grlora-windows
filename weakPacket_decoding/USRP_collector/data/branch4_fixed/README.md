# Branch4 固定 PHY 配置 IQ 数据集

这里保存当前 STM32 Branch4 固件循环发送 120 种确定性 payload 时采集的原始
`complex64` IQ。PHY 配置和除应用 payload 外的 33 字节 frame 字段保持固定。

## 目录

```text
high_snr/       强而干净的 120 种帧，用于生成/校验 clean reference
low_snr/        相同 120 种帧的低 SNR 接收，用于训练和评估
noise_only/     发射机关闭的同配置噪声，用于估计有色噪声协方差
interference/   相同帧与干扰源同时存在
```

## 文件名

```text
sf12_bw125_fs1000_pre16_sw12_r001.cfile
```

- `sf12`：扩频因子 12
- `bw125`：LoRa 带宽 125 kHz
- `fs1000`：IQ 采样率 1 Msample/s
- `pre16`：前导码 16 symbols
- `sw12`：private sync word `0x12`
- `r001`：该条件下第 1 次采集，只用于区分重复文件

文件名只记录解码前端需要的信息。不要再把中心频率、CR、payload 长度、FCnt、
TX power、RX gain 或 SNR 条件塞进文件名；这些信息记录在本文件、各条件目录的
README，以及 Python 采集脚本生成的同名 `.cfile.json` 中。

## 固定发射与接收配置

```text
固件工程          LoraSTMacH7_2026.07.19_移植STM32H743VIT6_实现classA_通用版(Branch4)
中心频率          487.7 MHz
SF                12
LoRa 带宽         125 kHz
编码率            4/8
前导码            16 symbols
Sync word         0x12（private）
PHY header        explicit
PHY CRC           enabled
应用 payload      20 bytes，120 种确定性伪随机内容
PHY payload       33 bytes
FCnt              1（固定）
TX power          2 dBm
发射周期          6 s
IQ 采样率         1 Msample/s（相对 LoRa BW 的 OSR=8）
RF-SR 输入/标签   250 kS/s -> 1 MS/s（倍率 4）
USRP 前端带宽     1 MHz
USRP RX gain      20 dB，首轮实验保持固定
USRP 天线         RX2
```

如果后续改变任何固定配置，另建一个数据集目录并复制修改这份 README，不要把两套
配置混在 `branch4_fixed` 中。正式采集优先使用 `collect_usrp_iq.py`，这样每个 `.cfile`
旁边都会有记录实际 UHD 参数的 `.cfile.json`。
