# Branch4 固定帧 IQ 数据集

这里保存当前 STM32 Branch4 固件重复发送固定帧时采集的原始 `complex64` IQ。

## 目录

```text
high_snr/       强而干净的固定帧，用于建立 FFT-bin ground truth
low_snr/        低 SNR 固定帧，用于 SER 和 GLS 对比
noise_only/     发射机关闭的同配置噪声，用于估计有色噪声协方差
interference/   固定帧与干扰源同时存在
```

## 文件名

```text
sf10_bw125_fs500_pre32_sw34_r001.bin
```

- `sf10`：扩频因子 10
- `bw125`：LoRa 带宽 125 kHz
- `fs500`：IQ 采样率 500 ksample/s
- `pre32`：前导码 32 symbols
- `sw34`：sync word `0x34`
- `r001`：该条件下第 1 次采集，只用于区分重复文件

文件名只记录解码前端需要的信息。不要再把中心频率、CR、payload 长度、FCnt、
TX power、RX gain 或 SNR 条件塞进文件名；这些信息记录在本文件、各条件目录的
README，以及 Python 采集脚本生成的同名 `.bin.json` 中。

## 固定发射与接收配置

```text
固件工程          LoraSTMacL1_2019.03.28_修改main函数_实现classA_通用版(Branch4)
中心频率          487.7 MHz
SF                10
LoRa 带宽         125 kHz
编码率            4/7
PHY header        explicit
PHY CRC           enabled
应用 payload      20 bytes
PHY payload       33 bytes
FCnt              1（固定）
TX power          2 dBm
发射周期          3 s
IQ 采样率         500 ksample/s（OSR=4）
USRP 前端带宽     500 kHz
USRP RX gain      20 dB，首轮实验保持固定
USRP 天线         RX2
```

如果后续改变任何固定配置，另建一个数据集目录并复制修改这份 README，不要把两套
配置混在 `branch4_fixed` 中。正式采集优先使用 `collect_usrp_iq.py`，这样每个 `.bin`
旁边都会有记录实际 UHD 参数的 `.bin.json`。
