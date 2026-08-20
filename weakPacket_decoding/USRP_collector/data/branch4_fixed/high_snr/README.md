# high_snr

发射机开启，采集强而干净的 120 种确定性帧。保存同一轮 UART 日志，并用其中的
`seq/id` 和完整 33 字节 `[TX Frame]` 生成、配对 clean reference。

记录每轮实验的发射机距离、衰减器配置和异常情况；USRP RX gain 默认固定为 20 dB。
IQ 文件使用 `sf12_bw125_fs1000_pre16_sw12_rNNN.cfile` 命名。
