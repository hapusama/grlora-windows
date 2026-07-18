# low_snr

发射机开启，采集与 `high_snr` 相同固定帧的低 SNR 数据，用于计算 demod SER 并比较
单 branch、普通合并和 GLS。通过射频衰减或距离改变信号强度，USRP RX gain 默认保持
20 dB，不启用 AGC。

记录每轮实验的衰减、距离和目标 SNR；IQ 文件使用
`sf10_bw125_fs500_pre32_sw34_lowNNN.bin` 命名。
