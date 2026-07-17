# high_snr

发射机开启，采集强而干净的 Branch4 固定帧。先用这里的数据完整解码并确认每个
payload symbol 的 FFT bin，形成后续低 SNR SER 评估使用的 ground truth。

记录每轮实验的发射机距离、衰减器配置和异常情况；USRP RX gain 默认固定为 20 dB。
IQ 文件使用 `sf10_bw125_fs500_pre32_sw34_rNNN.bin` 命名。
