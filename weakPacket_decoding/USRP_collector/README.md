# USRP LoRa 实时 CRC 可视化采集器

`usrp_iq_collector.grc` 在同一条 USRP 样点流上并行完成频谱/瀑布图显示、
LoRa 解调与 CRC 统计，以及由界面开关控制的原始 IQ 写盘。预检期间录制开关默认
关闭，不会向正式 `.cfile` 写入样点。

输出的 `.cfile` 是无文件头的 GNU Radio `gr_complex` 样点，也就是 `numpy.complex64` IQ，可以直接交给现有弱包分析脚本。这里的 `.cfile` 与此前的 `.bin` 只是扩展名不同，字节布局和文件大小完全相同。

## 默认实验参数

默认值与当前 STM32/SX1276 固定帧实验一致：

```text
中心频率          487.7 MHz
采样率            2 Msample/s
LoRa 带宽         125 kHz
SF                12
编码率            4/8
前导码            16 symbols
Sync word         0x12（private）
PHY payload       33 bytes，CRC enabled
应用 payload      20 bytes，120 种确定性伪随机内容
采集端 LoRa OSR   16
RF-SR 输入/标签   250 kS/s -> 1 MS/s（倍率 4）
USRP 前端带宽     1 MHz
接收增益          20 dB，手动固定
天线端口          RX2
发射周期          6 秒
采集时长          780 秒
启动丢弃时长      1 秒
```

2 Msample/s 的原始 complex64 数据约占用 16 MB/s。780 秒产生 12.48 GB
（约 11.62 GiB）数据。后处理会把每个 2 MS/s packet 拆成两个 1 MS/s
ADC phase，再从每个 phase 动态取得四个 250 kS/s 训练视图。

## 已知旧捕获的参数陷阱

现有 `high_snr/sf12_bw125_fs1000_pre16_sw12_r001.bin` 的文件名与实际波形不符。
功率突发约每 `3.01 s` 出现一次、每次约 `0.77 s`；用 SF10、1 MS/s、sync word
`0x34` 可以得到 `13/13` CRC PASS，而按文件名中的 SF12、`0x12` 解码为 0 包。
它是旧固件行为，不能用于验证本轮 SF12 流图。现场 CRC 全为零时，先检查烧录版本、
SF、sync word、前导码和发包周期，再判断位置是否不可用。

## 一轮 120 包的采集顺序

1. 启动 GRC 流图，保持“正式录制”开关为灰色关闭状态。
2. 在当前位置观察至少 5–10 包的 CRC；高 SNR 参考采集应接近 100% PASS。
3. 确认输出文件名和现场质量后，将“正式录制”开关切换为绿色。
4. 打开录制后再复位 STM32，使正式文件从 `seq=0/id=0` 附近开始。
5. 文件支路写满 `duration × samp_rate` 个样点后停止写入；频谱和解码界面仍会继续
   运行，需要手动关闭流图并确认文件已刷新。

正式数据统一输出到 `lora-rfsr-savaux/data/raw/ota/`。连续原始捕获使用
可被 trim 脚本直接解析的名称：

```text
rxcap_exp000_sess000_loclab1_condhighsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.cfile
rxcap_exp000_sess000_loclab1_condhighsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.uart.log
rxcap_exp000_sess000_loclab1_condhighsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.usrp.log
rxcap_exp000_sess000_loclab1_condhighsnr_run000_sf12_bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_rxg20.cfile.json
```

字段中的 `exp` 是每个连续 cfile 的唯一编号，每次新采集都应递增；
`sess/loc/cond/run` 用来描述实验分组。`rxg` 是手动 USRP RX gain，不是 SNR。
文件名用于快速发现参数，JSON/manifest 才是程序使用的权威记录。

在 `lora-rfsr-savaux` 根目录先运行下面的命令，可以生成与 GRC 默认值一致的
规范文件名和 JSON sidecar：

```powershell
python tools\build_rfsr_ota_dataset.py init-capture `
  --experiment-id 0 `
  --session-id 0 `
  --location-id lab1 `
  --condition highsnr `
  --run-id 0
```

然后把打印出的路径与 GRC 顶部
`capture_experiment/session/location/condition/run` 核对一致。sidecar 中继续补充
USRP 序列号、实际起止时间和现场说明。

## 在 RadioConda 中打开

本机的 `gr-lora` 环境包含当前 LoRa 解码器，但没有 GNU Radio UHD Python 绑定；
RadioConda 有 UHD，但自带的是 2025 年旧版 `gr-lora_sdr`。旧版在已知真实捕获上
不稳定，不应直接用于现场 CRC 门控。先为 RadioConda 补齐 Boost 开发头文件，再在
仓库内独立编译当前模块：

```powershell
D:\mysoft2\radioconda\Scripts\conda.exe install `
  --prefix D:\mysoft2\radioconda libboost-devel=1.86

Set-Location D:\Desktop\proj\gr-lora_sdr
.\build_grlora_radioconda.bat
```

构建结果留在被 Git 忽略的 `build-radioconda/`，不会覆盖 RadioConda 的系统安装。
此后用专用启动器打开 GRC，它会让运行时优先加载仓库内的新模块：

```powershell
Set-Location "D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\USRP_collector"
.\open_live_collector_grc.bat
```

打开后可以双击顶部变量修改中心频率、采样率、增益、采集时长和输出文件。
`device_args` 默认留空，让 UHD 自动选择当前唯一的 USRP；只有同时连接多台设备时，
才填写 `serial=XXXXXXXX`。

当前 UHD Source 只启用数字通道 0，并把该通道的天线端口设为 `RX2`。运行后
`QT GUI Sink` 会在同一个窗口提供频谱、瀑布图和时域 IQ 三个标签页；瀑布图最适合
现场确认 STM32 发包、中心频率和干扰情况。CRC 面板显示本包 PASS/FAIL、已解码包数、
成功数、累计成功率和最近 10 包成功率。显示和解码支路不经过录制开关，因此预检期间
始终工作；只有 `Copy -> Head -> File Sink` 文件支路受开关控制。

GRC 默认输出到
`lora-rfsr-savaux/data/raw/ota/rxcap_exp000_..._rxg20.cfile`。
`capture_root` 是相对于 `USRP_collector` 工作目录的路径，因此应从该目录打开和运行
GRC。开始运行前必须确认 `output_file` 展开的完整名称，并为每个新连续 cfile
递增 `capture_experiment` 或 `capture_run`。高 SNR、低 SNR 和纯噪声数据之间要保持
USRP 增益不变；用于噪声协方差分析时不要启用 AGC。

GNU Radio `File Sink` 在流图启动时就会创建或截断目标文件，即使录制开关仍关闭。
因此每次启动前必须使用新的输出文件名。一次流图运行只对应一轮正式录制；开关变绿后
不要再关闭并重新打开，否则文件中的两段 IQ 之间会存在未记录的时间缺口。

## 输出文件

若输出名称为 `rxcap_....cfile`，GRC 写出：

```text
rxcap_....cfile       原始 2 MS/s complex64 IQ
```

GRC 本身不写 JSON；应在采集前用 `init-capture` 创建同名 `.cfile.json`，
采集后补充 USRP 序列号、实际 RF 参数、录制起止时间和 CRC 预检结果。

读取方式：

```python
import numpy as np

iq = np.fromfile("capture.cfile", dtype=np.complex64)
```

第一轮实验至少应使用完全相同的 USRP 设置采集下面四组条件：

```text
condhighsnr       发射机开启，强而干净的信号
condlowsnr        发射机开启，低 SNR 信号
condnoiseonly     发射机关闭，只有接收机噪声
condinterference  发射机和干扰源同时开启
```

条件由文件名中的 `condhighsnr`、`condlowsnr`、`condnoiseonly`、
`condinterference` 表示，不再分别依赖 `branch4_fixed` 子目录。原始 IQ、UART/USRP
日志和 sidecar 均放在 `lora-rfsr-savaux/data/raw/ota/`，逐包 fulltrim 输出由
`tools/build_rfsr_ota_dataset.py` 写入
`lora-rfsr-savaux/data/reference_phy/rfsr_db/`。

用高 SNR 下重复发送的固定帧建立“每个符号对应 FFT bin”的真值模板。该真值只供离线评估 SER 使用，不应作为 GLS 检测器本身的输入。

## XCopy 式多副本同步

Branch4 固定帧数据可以直接交给
`weakPacket_decoding/scripts/run_xcopy_sync.py`。脚本利用不同发射时刻的真实重传包做
整帧共轭时延/CFO/相位估计和相干合并，再把帧边界映射回未合并的原始低 SNR 副本。
它不会把 OSR 分支误当成独立副本。

默认 `--detection-mode paper` 使用逐包 4-chirp 长窗检测，不需要已知发包周期。高 SNR
旧 SF10/500 kS/s 数据测得的 `1,500,365`-sample 间隔不能用于本轮数据；本轮应重新
测量 1 MS/s、6 秒周期下的实际间隔。`--detection-mode periodic` 仅在显式提供新测量值时使用。
sync word、header checksum 和 gr-lora_sdr `framesync_valid` 均不作为原始 payload 导出的
硬门。完整命令、输出字段、实测成功/失败边界见
[`../doc/xcopy_sync.md`](../doc/xcopy_sync.md)。
