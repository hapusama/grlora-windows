# low_snr

发射机开启，采集与 `high_snr` 相同 120 种 payload 的低 SNR 数据，用于模型训练、
demod SER 和 weak-decoder 对比。通过射频衰减或距离改变信号强度，USRP RX gain
默认保持 20 dB，不启用 AGC。

记录每轮实验的衰减、距离和目标 SNR；IQ 文件使用
`sf12_bw125_fs1000_pre16_sw12_lowNNN.cfile` 命名。
