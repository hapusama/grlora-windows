# interference

LoRa 固定帧发射机与干扰源同时开启。USRP 配置与 `high_snr`、`low_snr` 和
`noise_only` 保持一致。

每轮必须记录干扰类型、中心频率、带宽、功率、占空比和相对位置；IQ 文件使用
`sf10_bw125_fs500_pre32_sw34_rNNN.bin` 命名。
