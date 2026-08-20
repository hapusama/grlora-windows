# noise_only

关闭 LoRa 发射机，以与信号数据完全相同的 USRP 频率、采样率、前端带宽、天线和
20 dB RX gain 采集背景噪声。该数据用于估计真实有色噪声的协方差矩阵。

记录现场环境和附近已知干扰源状态；IQ 文件使用
`sf12_bw125_fs1000_pre16_sw12_rNNN.cfile` 命名。
