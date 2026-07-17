# USRP 离线 IQ 采集器

这里提供两种采集方式：

- `collect_usrp_iq.py`：正式采集推荐使用。它会保存原始 IQ，同时生成一份记录实际 UHD 参数和实验配置的 JSON 元数据。
- `usrp_iq_collector.grc`：可在 GNU Radio Companion 中打开的可视化流图，适合查看和临时调整采集链路；它只输出原始 IQ，不生成 JSON。

输出的 `.bin` 是无文件头的 GNU Radio `gr_complex` 样点，也就是 `numpy.complex64` IQ，可以直接交给现有弱包分析脚本。

## 默认实验参数

默认值与当前 STM32/SX1276 固定帧实验一致：

```text
中心频率          487.7 MHz
采样率            500 ksample/s
LoRa 带宽         125 kHz
编码率            4/7
前导码            32 symbols
Sync word         0x34
PHY payload       33 bytes，CRC enabled
过采样倍数        4
USRP 前端带宽     500 kHz
接收增益          20 dB，手动固定
天线端口          RX2
采集时长          60 秒（GRC 中默认为 120 秒）
启动丢弃时长      1 秒
```

500 ksample/s 的原始 complex64 数据约占用 4 MB/s，即每分钟约 229 MiB。

## 在 RadioConda 中打开

本机的 `gr-lora` 环境能运行离线 LoRa 模块，但没有 GNU Radio UHD Python 绑定。USRP 采集请使用已经安装好的 RadioConda：

```powershell
conda activate D:\mysoft2\radioconda
uhd_find_devices
uhd_usrp_probe
```

用 GRC 打开可视化流图。先进入 `USRP_collector` 目录，可以确保相对输出路径
`data/branch4_fixed/high_snr/` 正确落到本目录下的 `data` 文件夹：

```powershell
Set-Location "D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\USRP_collector"
gnuradio-companion ".\usrp_iq_collector.grc"
```

打开后可以双击顶部变量修改中心频率、采样率、增益、采集时长和输出文件。当前 `device_args` 已填写本机 B210 的序列号 `2603160`；如果更换 USRP，需要同步修改。

当前 UHD Source 只启用数字通道 0，并把该通道的天线端口设为 `RX2`。运行后
`QT GUI Sink` 会在同一个窗口提供频谱、瀑布图和时域 IQ 三个标签页；瀑布图最适合
现场确认 STM32 发包、中心频率和干扰情况。显示支路接在定时 `Head` 之后，因此不会
改变写入 `.bin` 的样点，也不会破坏 120 秒后自动结束采集的行为。

GRC 默认输出到
`data/branch4_fixed/high_snr/sf10_bw125_fs500_pre32_sw34_r001.bin`。
这是相对于启动流图时工作目录的路径；如果不从
`USRP_collector` 目录启动，它也会跟着工作目录变化。GRC 的 `File Sink` 会覆盖同名文件，
开始运行前先检查 `output_file`。高 SNR、低 SNR 和纯噪声数据之间要保持 USRP 增益不变；
用于噪声协方差分析时不要启用 AGC。

## 使用 Python 脚本定时采集

在 `weakPacket_decoding` 目录运行：

```powershell
python USRP_collector\collect_usrp_iq.py `
  --output USRP_collector\data\branch4_fixed\high_snr\sf10_bw125_fs500_pre32_sw34_r001.bin `
  --duration 120 `
  --center-freq 487.7e6 `
  --samp-rate 500e3 `
  --lora-bandwidth 125e3 `
  --rf-bandwidth 500e3 `
  --gain 20 `
  --antenna RX2 `
  --device-args "serial=2603160"
```

当前发射周期是 3 秒，因此 120 秒大约可采到 40 个包。脚本默认拒绝覆盖已有文件；确认需要覆盖时添加 `--overwrite`。

## 持续采集到手动停止

```powershell
python USRP_collector\collect_usrp_iq.py `
  --output USRP_collector\data\branch4_fixed\high_snr\sf10_bw125_fs500_pre32_sw34_r002.bin `
  --duration 0 `
  --gain 20 `
  --device-args "serial=2603160"
```

按一次 `Ctrl+C` 后，脚本会停止流图并刷新输出文件。

## 输出文件

若输出名称为 `capture.bin`，Python 脚本会生成：

```text
capture.bin       原始 complex64 IQ
capture.bin.json  UHD 请求值、实际值和实验元数据
```

读取方式：

```python
import numpy as np

iq = np.fromfile("capture.bin", dtype=np.complex64)
```

第一轮实验至少应使用完全相同的 USRP 设置采集下面三组数据：

```text
branch4_fixed/high_snr/       发射机开启，强而干净的信号
branch4_fixed/low_snr/        发射机开启，低 SNR 信号
branch4_fixed/noise_only/     发射机关闭，只有接收机噪声
branch4_fixed/interference/   发射机和干扰源同时开启
```

各目录内统一使用 `sf10_bw125_fs500_pre32_sw34_rNNN.bin`。文件名只携带解码前端
需要的信息，中心频率、编码率、payload、FCnt、收发增益等由数据集 README 和
Python 采集脚本生成的 `.bin.json` 记录。

用高 SNR 下重复发送的固定帧建立“每个符号对应 FFT bin”的真值模板。该真值只供离线评估 SER 使用，不应作为 GLS 检测器本身的输入。
