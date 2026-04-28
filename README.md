# gr-lora_sdr Windows 构建与调试指南

> 本文面向本地工程 `d:\Desktop\proj\gr-lora_sdr`，重点覆盖 **Windows 11 + conda + VS 2022** 环境下的构建、安装验证、CRC 双模式、PHY header 频谱图，以及离线前导码特征导出。

默认命令在 `d:\Desktop\proj\gr-lora_sdr` 目录下执行，除非代码块中另有说明。

## 目录

- [1. 快速开始](#1-快速开始)
- [2. 环境准备](#2-环境准备)
- [3. 构建与安装](#3-构建与安装)
- [4. 安装验证](#4-安装验证)
- [5. CRC 校验说明](#5-crc-校验说明)
- [6. PHY Header 频谱图](#6-phy-header-频谱图)
- [7. 前导码特征批处理](#7-前导码特征批处理)
- [8. 常见问题与排查](#8-常见问题与排查)
- [9. 本地改动清单](#9-本地改动清单)
- [10. 更新日志](#10-更新日志)

---

## 1. 快速开始

### 1.1 一键重新编译

```powershell
cd d:\Desktop\proj\gr-lora_sdr
.\build_grlora.bat
```

`build_grlora.bat` 会清理旧的 `build/` 目录，重新配置 CMake，执行 `nmake`，并安装到脚本中配置的 conda 环境。

注意：当前脚本默认写死 conda 环境为 `gr-lora`。如果实际使用 GNU Radio 的环境是 `radioconda` 或其他环境，需要先修改脚本中的 conda 路径、Python 路径和安装前缀。

### 1.2 验证安装

```powershell
conda activate gr-lora
python -c "import gnuradio.lora_sdr as lora; print('Import OK'); print('Crc_mode:', lora.Crc_mode.GRLORA, lora.Crc_mode.SX1276)"
```

如果能输出 `Import OK`，并显示 `Crc_mode.GRLORA` 和 `Crc_mode.SX1276`，说明 Python 绑定和模块加载正常。

### 1.3 离线解码示例

```powershell
python examples\lora_file_RX.py `
  -f data\USRP_IQ\1_1_6_10_2_16.bin `
  --sf 10 `
  --bw 125000 `
  --samp-rate 500000 `
  --cr 1 `
  --sync-word 0x34 `
  --preamble-len 8 `
  --ldro-mode 2 `
  --has-crc `
  --crc-mode 0
```

`--crc-mode 0` 是默认的 `GRLORA` 模式，可以省略。解码 SX1276 / LoraSTMacL1 发出的 LoRa PHY 包时，优先使用该模式。

### 1.4 导出 PHY Header 频谱图

```powershell
python examples\lora_file_RX.py `
  -f data\USRP_IQ\1_1_6_10_2_16.bin `
  --sf 10 `
  --bw 125000 `
  --samp-rate 500000 `
  --cr 1 `
  --center-freq 487.7e6 `
  --sync-word 0x34 `
  --preamble-len 8 `
  --ldro-mode 2 `
  --crc-mode 0 `
  --plot-phy-header `
  --preamble-plot-max 3
```

输出 PNG 默认保存到 `examples\preamble_plots`。

---

## 2. 环境准备

### 2.1 安装 Visual Studio 2022

安装 **Visual Studio Community 2022** 或 **Build Tools for Visual Studio 2022**，并勾选工作负载：

- `使用 C++ 的桌面开发`

确认 `vcvarsall.bat` 路径，例如：

```text
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat
```

### 2.2 创建 conda 环境

```powershell
conda create -n gr-lora python=3.10
conda activate gr-lora
conda install -c conda-forge gnuradio-core boost-cpp volk cmake pybind11
```

关键点：`pybind11` 版本必须与 `gnuradio-core` 的 Python 绑定 ABI 兼容。如果后续遇到：

```text
ImportError: generic_type: type "..." referenced unknown base type "gr::block"
```

通常说明 `pybind11` 版本不匹配，需要调整版本后重新编译。常用修复见 [8.1 pybind11 ABI 不匹配](#81-pybind11-abi-不匹配)。

---

## 3. 构建与安装

### 3.1 使用一键脚本

推荐优先使用仓库中的 [`build_grlora.bat`](build_grlora.bat)：

```powershell
cd d:\Desktop\proj\gr-lora_sdr
.\build_grlora.bat
```

脚本会执行：

1. 调用 `vcvarsall.bat x64` 设置 MSVC 环境。
2. 删除旧的 `build/` 目录并重新创建。
3. 使用 `NMake Makefiles` 生成器运行 CMake 配置。
4. 执行 `nmake` 编译。
5. 执行 `nmake install`，安装到 conda 环境的 `Library` 目录。

### 3.2 手动分步构建

排查构建问题时，可以手动执行同等步骤：

```powershell
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

cd /d d:\Desktop\proj\gr-lora_sdr
rmdir /s /q build
mkdir build
cd build

set CMAKE_PREFIX_PATH=D:\mysoft2\miniconda3\envs\gr-lora\Library

cmake .. -G "NMake Makefiles" ^
    -DCMAKE_INSTALL_PREFIX=D:\mysoft2\miniconda3\envs\gr-lora\Library ^
    -DPYTHON_EXECUTABLE=D:\mysoft2\miniconda3\envs\gr-lora\python.exe ^
    -DGR_PYTHON_DIR=D:\mysoft2\miniconda3\envs\gr-lora\Lib\site-packages

nmake
nmake install
```

---

## 4. 安装验证

### 4.1 检查 Python 导入

```powershell
conda activate gr-lora
python -c "import gnuradio.lora_sdr as lora; print('Import OK'); print('Crc_mode:', lora.Crc_mode.GRLORA, lora.Crc_mode.SX1276)"
```

预期能看到：

```text
Import OK
Crc_mode: Crc_mode.GRLORA Crc_mode.SX1276
```

### 4.2 检查模块路径

确认 Python 加载的是当前安装的新版本，而不是旧环境中的残留版本：

```powershell
python -c "import gnuradio.lora_sdr; import os; print(os.path.dirname(gnuradio.lora_sdr.__file__))"
```

预期路径类似：

```text
D:\mysoft2\miniconda3\envs\gr-lora\lib\site-packages\gnuradio\lora_sdr
```

---

## 5. CRC 校验说明

### 5.1 先读结论

`gr-lora_sdr` 默认 CRC 校验逻辑本身没有错。如果离线解码 LoraSTMacL1 / SX1276 发出的 LoRa 包时出现 `invalid CRC`，不要直接归因于 CRC 算法错误。更常见原因是：

- 解调后仍有残留 bit 错。
- `SF`、`CR`、`SyncWord`、`LDRO`、前导码长度等参数不匹配。
- 同步、CFO 或 STO 校正不够好。
- IQ 文件采样率与命令行 `--samp-rate` 不一致。

LoraSTMacL1 工程中的 `Radio.SetTxConfig(..., crcOn=true, ...)` 只是打开 SX1276 的硬件 LoRa PHY CRC。MCU 发送进 FIFO 的 LoRaWAN `PHYPayload` 不包含这两个 PHY CRC 字节，CRC 由 SX1276 在空口帧尾自动追加。

因此，代码或文档中出现的 `0x12345678` 一类字段属于 LoRaWAN MIC / 测试占位字段，不是 LoRa PHY CRC。

### 5.2 CRC 模式

`crc_verif` 当前保留两种计算模式，用于对照排查：

| 模式 | 枚举值 | 适用场景 | 算法说明 |
|------|--------|----------|----------|
| `GRLORA` 默认 | `lora_sdr.Crc_mode.GRLORA` | gr-lora_sdr 软件发射机；LoraSTMacL1 / SX1276 LoRa PHY 包优先使用此模式 | 对 `payload_len - 2` 字节计算 CRC-16，再与最后 2 字节 XOR；这是 gr-lora_sdr 论文和代码采用的 LoRa PHY CRC 形式 |
| `SX1276` | `lora_sdr.Crc_mode.SX1276` | 仅作为对照 / 实验模式 | 对全部 payload 字节直接计算 CRC-16-CCITT；名字来自早期排查假设，不建议作为 LoraSTMacL1 / SX1276 LoRa PHY 包默认选择 |

### 5.3 Python 中使用

```python
import gnuradio.lora_sdr as lora_sdr

# 推荐：使用默认 GRLORA 模式
crc = lora_sdr.crc_verif(1, False, lora_sdr.Crc_mode.GRLORA)

# 兼容旧代码风格
crc = lora_sdr.crc_verif(1, False, lora_sdr.crc_verif.GRLORA())
```

### 5.4 `lora_file_RX.py` 中使用

```powershell
# 接收 LoraSTMacL1 / SX1276 硬件发射的 LoRa PHY 帧，优先使用默认 GRLORA 模式
python examples\lora_file_RX.py `
  -f data\USRP_IQ\1_1_6_12_2_16.bin `
  --sf 12 `
  --bw 125000 `
  --samp-rate 500000 `
  --cr 2 `
  --center-freq 487.7e6 `
  --sync-word 0x12 `
  --has-crc `
  --crc-mode 0
```

`--crc-mode 0` 可以省略，因为默认就是 `GRLORA` 模式。

如果 `--crc-mode 0` 仍然显示 `invalid CRC`，下一步应检查解出的 payload 是否已经有 bit 错。例如 ASCII 测试载荷中出现 `0x39 -> 0x59`、`0x43 -> 0x13` 这种一位或多位错误时，CRC 必然不通过。

### 5.5 判断 CRC 失败原因

可以对同一个文件分别运行两次：

```powershell
# 默认 GRLORA 模式
python examples\lora_file_RX.py ... --crc-mode 0

# 对照模式
python examples\lora_file_RX.py ... --crc-mode 1
```

判断方式：

- 模式 0 显示 `CRC valid`：当前 LoRa PHY CRC 校验正常。
- 模式 0 `invalid`，但 payload 明显有 bit 错：CRC 失败是残留误码的正常结果。
- 模式 0 和模式 1 都 `invalid`：优先排查 `SF`、`CR`、`SyncWord`、采样率、`LDRO`、前导码长度、CFO / STO，以及 IQ 文件中是否有过长静默段。

运行时 `crc_verif_impl.cc` 会打印当前模式，例如：

```text
[crc_verif] CRC mode: GRLORA, payload_len=33
```

或：

```text
[crc_verif] CRC mode: SX1276, payload_len=33
```

---

## 6. PHY Header 频谱图

`examples/lora_file_RX.py` 可以把原始 IQ 文件中的 LoRa PHY 非 payload 片段保存为频谱图，用于检查 preamble、sync word 和 SFD 的真实时频结构。

### 6.1 数据流设计

- C++ 层 `frame_sync_impl.cc` 在完成前导码、sync word、SFD 精同步后，计算原始 IQ 文件中对齐的 `preamble + sync word + SFD` 样本索引范围。
- C++ 层不会立即发布该范围；只有 `header_decoder` 回传 header checksum valid 后，`frame_sync` 才把范围发给 Python。
- Python 层 `lora_file_RX.py` 根据 `start_sample:end_sample` 从真实 `.bin` IQ 中切片、画图并保存 PNG。
- C++ 不负责画图，避免把 `numpy` / `Pillow` 这类分析逻辑塞进实时信号处理块。

### 6.2 `frame_sync` 输出消息

`frame_sync` 新增 message output port：

```text
preamble
```

header checksum 通过后会发送 PMT dict，主要字段如下：

| 字段 | 含义 |
|------|------|
| `frame_count` | 当前帧序号 |
| `sf` | 扩频因子 |
| `bw` | 带宽 |
| `sample_rate` | 原始 IQ 文件采样率，等于 `bw * os_factor` |
| `samples_per_symbol` | 原始 IQ 中每个 LoRa symbol 的样本数，等于 `2^sf * os_factor` |
| `start_sample`, `end_sample` | 原始 IQ 文件中对齐后的 `preamble + sync word + SFD` 范围 |
| `n_samples`, `n_symbols` | 范围长度；`n_symbols = preamble_len + 4.25` |
| `preamble_len` | 本次 `frame_sync` 使用的前导码 symbol 数，来自脚本参数 `--preamble-len` |
| `sync_word_symbols`, `sfd_symbols` | 固定为 `2` 和 `2.25` |
| `header_valid` | header checksum 已通过时为 `true` |
| `source` | 当前为 `preamble_sync_sfd` |
| `snr_db` | `frame_sync` 估计的前导码 SNR |
| `cfo` | CFO 估计值，单位为 bins |
| `sto` | STO 估计值 |
| `sfo` | SFO 估计值 |
| `netid1`, `netid2` | 解出的两个 sync word / network identifier symbol |
| `cr`, `pay_len`, `crc`, `ldro_mode`, `err` | header_decoder 回传的 PHY header 元数据，用于脚本侧和 header/payload 消息按同一 frame 对齐 |

消息不携带拼接后的绘图 IQ。Python 端直接从输入 `.bin` 文件按索引读取连续原始 IQ，不再做 dechirp + FFT 二次细化。

### 6.3 脚本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--plot-preamble`, `--plot-phy-header` | 关闭 | 开启 LoRa PHY 非 payload 片段频谱图保存 |
| `--preamble-plot-dir` | `examples/preamble_plots` | PNG 输出目录 |
| `--preamble-plot-max` | `3` | 最多保存多少帧，`0` 表示不限制 |
| `--preamble-plot-dpi` | `150` | 输出图片 DPI |

默认只画前三帧，避免长 IQ 文件生成大量 PNG。如果要画全部帧：

```powershell
--preamble-plot-max 0
```

注意：`--preamble-len` 不只是绘图时切出多少个 preamble symbol，它也会原样传给 C++ `frame_sync` 作为检测阈值，并在内部形成 `m_n_up_req = preamble_len - 3`。低信噪比或前导码质量较差的 IQ 文件可能在标称值下检测不到，需要适当降低该值做排查。例如 `0_0_0_10_6_16.bin` 用 `--preamble-len 16` 没有有效输出，但用 `--preamble-len 12` 可以正常解出 1 个 CRC-valid 包。

### 6.4 输出示例

```text
[phy_header_plot] queued frame 1 (samples 645341:695517, 12.25 symbols)
[phy_header_plot] saving frame 1 (12.25 symbols, 50176 samples)
[phy_header_plot] saved D:\Desktop\proj\gr-lora_sdr\examples\preamble_plots\phy_header_frame_001.png
```

### 6.5 图像生成方式

Python 端使用 `numpy + Pillow` 生成频谱图，不依赖 `scipy`，也不依赖 matplotlib 的绘图后端，从而避免 Windows 下 GNU Radio 消息线程触发 matplotlib 底层崩溃。

图像内容：

- 标题：`LoRa Preamble + Sync + SFD Spectrogram`
- x 轴：`Time (ms)`
- y 轴：`Frequency (kHz)`
- 频率范围：`-bw/2` 到 `+bw/2`
- 色图：`viridis`
- 右侧 colorbar 显示 dB 刻度
- 竖向虚线标出 LoRa symbol 边界

PHY header 图按 `--preamble-len + 2 + 2.25` 个 symbol 从原始 IQ 中连续切出 preamble、2 个 sync word symbol 和 2.25 symbol SFD，不包含 payload。比如 `--preamble-len 8` 时，输出长度就是 `12.25` symbols。

---

## 7. 前导码特征批处理

`examples/lora_file_preamble_fft.py` 用于从离线 IQ 文件中批量导出 packet 级信号特征，尤其适合处理 `data\USRP_IQ` 下按 lab 分组的采集数据。

### 7.1 批处理入口

```powershell
python examples\lora_file_preamble_fft.py `
  --all-bin `
  --input-dir data\USRP_IQ
```

脚本会按 lab 子文件夹分组处理，每个 lab 单独写回：

- `packet_features.csv`
- `preamble_features.npz`

### 7.2 输出特征

当前输出收敛为 packet 级特征：

- 平均 RSSI
- 平均 SNR
- SX1276 风格 SNR register 值
- 前导码主峰 `-3 dB` 宽度
- 主峰对齐后的局部平均幅度谱

旧版 per-frame 明细 CSV 不再生成。脚本会清理 `rssi_samples_5ms.csv`、`preamble_symbol_features.csv`、`position_summary.csv` 等过期输出，避免误读旧文件。

### 7.3 默认不解析 payload

默认模式下，流图在 `header_decoder` 后直接连接 `null_sink`，只依赖 `frame_sync` 的对齐范围和 `header_decoder` 的 PHY header 信息计算信号特征。

因此，末尾残包、payload 解码失败或 CRC invalid 不会导致 `decoded payload count != detected packet count` 这类 warning，也不会影响 RSSI / SNR / preamble FFT 特征导出。

### 7.4 frame/header/payload 元数据合并

脚本不再按 list index 直接合并 `frame_sync`、`header_decoder` 和 `crc_verif` 三路消息。C++ 链路现在会把同一个包的 `frame_count` 以及 `start_sample:end_sample` 沿着 tag/message 传递：

```text
frame_sync -> header_decoder -> crc_verif
```

Python 端优先按 `frame_count` 合并 frame/header/payload；如果遇到旧版已安装模块没有这些 ID，会打印 warning 并退回 detection-order merge。这样可以避免 frame 排序、无效 header、payload 只对 CRC-valid 包产生消息时造成错配。

### 7.5 需要 `FCnt` 时

如果确实需要 LoRaWAN `FCnt`，可以启用：

```powershell
--require-valid-payload
```

或别名：

```powershell
--with-fcnt
```

启用后，脚本会重新接回 `header_decoder -> dewhitening -> crc_verif`，只保留 CRC-valid payload 对应的数据包，并尝试从 `PHYPayload` / `FHDR` 中解析 `FCnt`。末尾残包、CRC invalid 包或 payload 不完整包会被整包舍弃。`--crc-mode` 和 `--print-payload` 仅在该模式下有意义。

### 7.6 批处理隔离模式

长批量任务中，如果担心单个 native 崩溃拖垮整个 lab，可以启用子进程隔离：

```powershell
--isolated-workers
```

worker 失败时可通过以下参数打印尾部日志：

```powershell
--worker-log-lines 80
```

---

## 8. 常见问题与排查

### 8.1 pybind11 ABI 不匹配

错误示例：

```text
ImportError: generic_type: type "add_crc" referenced unknown base type "gr::block"
```

原因：`gnuradio-core` 的 Python 绑定和 `lora_sdr_python.pyd` 的 `pybind11 internals` 版本不匹配。

可以检查两个二进制文件中的 pybind11 内部版本号：

```python
import re

with open(r'D:\mysoft2\miniconda3\envs\gr-lora\Lib\site-packages\gnuradio\gr\gr_python.cp310-win_amd64.pyd', 'rb') as f:
    print('gr_python:', re.findall(rb'__pybind11_internals_v\d+', f.read()))

with open(r'D:\mysoft2\miniconda3\envs\gr-lora\Lib\site-packages\gnuradio\lora_sdr\lora_sdr_python.cp310-win_amd64.pyd', 'rb') as f:
    print('lora_sdr_python:', re.findall(rb'__pybind11_internals_v\d+', f.read()))
```

两者必须显示相同版本，例如都是 `v5`。

常用修复：

```powershell
conda activate gr-lora
conda install -c conda-forge pybind11=2.13.6
rmdir /s /q d:\Desktop\proj\gr-lora_sdr\build
cd d:\Desktop\proj\gr-lora_sdr
.\build_grlora.bat
```

### 8.2 Python bindings out of sync

原因：修改了 C++ 头文件，例如 `crc_verif.h`，但对应绑定文件中的 `BINDTOOL_HEADER_FILE_HASH(...)` 没有更新。

修复方式：

1. 计算新头文件的 MD5 哈希。
2. 更新对应 `*_python.cc` 文件中的 `BINDTOOL_HEADER_FILE_HASH(...)` 宏。
3. 重新编译并安装。

### 8.3 CRC invalid

排查顺序：

1. 确认 `--crc-mode 0` 是否已使用默认 `GRLORA` 模式。
2. 检查 payload 是否存在明显 bit 错。
3. 确认 `--sf`、`--bw`、`--cr`、`--sync-word`、`--preamble-len`、`--ldro-mode` 是否与发射端一致。
4. 检查 IQ 文件真实采样率是否与 `--samp-rate` 一致。
5. 再考虑同步、CFO、STO 或文件截断问题。

采样率以采集时的真实参数为准；文件名中的 SF 字段修正后，`--sf` 直接按文件名或实际发射参数填写即可。

### 8.4 中文路径导致 `file_source` 读取失败

Windows 下 GNU Radio C++ `file_source` 读取包含中文字符的路径时可能失败。当前脚本已增加 ASCII hardlink staging：

- GNU Radio C++ `file_source` 读取 ASCII-only 临时硬链接。
- Python 侧仍从真实路径读取 IQ 和元数据。

### 8.5 native 崩溃 `0xC0000005`

旧版 `frame_sync_impl.cc` 在 `DETECT -> SYNC` 阶段可能因为负索引访问输入缓冲区之前的内存，Windows 下表现为：

```text
exit=3221225477
0xC0000005
```

当前代码已在拷贝窗口前检查 `k_hat` 和边界。非法检测会直接回到 `DETECT`，避免越界访问。

---

## 9. 本地改动清单

### 9.1 CRC 双模式

| 文件 | 改动内容 |
|------|----------|
| `include/gnuradio/lora_sdr/crc_verif.h` | 新增 `Crc_mode` 枚举（`GRLORA`, `SX1276`）和 `make()` 的第三个参数 |
| `lib/crc_verif_impl.h` | 新增 `m_crc_mode` 成员和 `crc16_sx1276()` 声明 |
| `lib/crc_verif_impl.cc` | 新增 `crc16_sx1276()` 实现，运行时根据 `m_crc_mode` 分支 |
| `python/lora_sdr/bindings/crc_verif_python.cc` | pybind11 绑定：`py::enum_<Crc_mode>` 和 `.def_static` 兼容层 |
| `python/lora_sdr/lora_sdr_lora_rx.py` | hier block 增加 `crc_mode` 参数 |
| `examples/lora_file_RX.py` | 新增 `--crc-mode` CLI 参数和 int 到 enum 的转换 |
| `grc/lora_sdr_crc_verif.block.yml` | GRC 块定义增加 CRC 模式下拉框 |
| `build_grlora.bat` | Windows 一键编译脚本 |

### 9.2 PHY Header 频谱图

| 文件 | 改动内容 |
|------|----------|
| `lib/frame_sync_impl.h` | 新增 PHY header 样本范围发布所需状态和 `publish_phy_header(...)` / `try_publish_phy_header()` 声明 |
| `lib/frame_sync_impl.cc` | 新增 `preamble` message output port；精同步后记录 `preamble + sync word + SFD` 原始 IQ 索引，header checksum 通过后再发给 Python |
| `examples/lora_file_RX.py` | 新增 `phy_header_spectrogram_sink`，接收 `frame_sync` 的 `preamble` 消息并保存频谱图 |
| `examples/lora_file_RX.py` | 新增 `--plot-preamble`、`--plot-phy-header`、`--preamble-plot-dir`、`--preamble-plot-max`、`--preamble-plot-dpi` 参数 |

### 9.3 前导码特征导出

| 文件 | 改动内容 |
|------|----------|
| `examples/lora_file_preamble_fft.py` | 扩展为离线批处理特征导出脚本，支持 lab 分组、packet 级 CSV / NPZ 输出、payload 可选解析、Windows 路径兼容和 worker 隔离 |
| `examples/lora_file_preamble_fft.py` | frame/header/payload 元数据优先按 `frame_count` 合并，并保留旧版 detection-order fallback |
| `lib/frame_sync_impl.cc` / `lib/frame_sync_impl.h` | `frame_info` tag 增加 `frame_count` 和原始 IQ 样本范围，供后续 header/payload 元数据对齐 |
| `lib/header_decoder_impl.cc` / `lib/header_decoder_impl.h` | 发布 `frame_info` 时保留上游 frame 元数据，避免 Python 侧只能按消息顺序合并 |
| `lib/crc_verif_impl.cc` | 保留原 `msg` payload 输出，并新增 `payload_metadata` message port，输出 payload、CRC 结果和 frame ID |
| `grc/lora_sdr_crc_verif.block.yml` | GRC 块定义增加可选 `payload_metadata` 消息口 |
| `lib/frame_sync_impl.cc` | 修复 `DETECT -> SYNC` 阶段边界检查，避免 SF12 等场景下负索引导致 native 崩溃 |

---

## 10. 更新日志

### 2026-04-28

- `examples/lora_file_preamble_fft.py` 从单帧前导码 FFT 导出脚本扩展为离线批处理特征导出脚本：支持 `--all-bin --input-dir ...` 按 USRP_IQ 下的 lab 子文件夹分组处理，每个 lab 单独写回 `packet_features.csv` 和 `preamble_features.npz`。
- `lora_file_preamble_fft.py` 会从文件名解析实验编号、走廊编号、位置编号、SF、发射功率和前导码长度；如果 lab 目录存在 `补充.txt`，会读取其中的说明和参数覆盖信息，并把 lab 元数据写入输出结果。
- 输出收敛为 packet 级特征：平均 RSSI、平均 SNR、SX1276 风格 SNR register 值、前导码主峰 `-3 dB` 宽度、以及主峰对齐后的局部平均幅度谱。旧版 per-frame 明细 CSV 不再生成，并会清理 `rssi_samples_5ms.csv`、`preamble_symbol_features.csv`、`position_summary.csv` 等过期输出，避免误读旧文件。
- 默认运行模式不再解析 payload。流图在 `header_decoder` 后直接接 `null_sink`，只依赖 `frame_sync` 的对齐范围和 `header_decoder` 的 PHY header 信息计算信号特征。因此末尾残包、payload 解码失败或 CRC invalid 不会再导致 `decoded payload count != detected packet count` 这类 warning，也不会影响 RSSI/SNR/preamble FFT 特征导出。
- 如确实需要 LoRaWAN `FCnt`，新增可选开关 `--require-valid-payload`（别名 `--with-fcnt`）。启用后脚本会重新接回 `header_decoder -> dewhitening -> crc_verif`，只保留 CRC-valid payload 对应的数据包并尝试从 PHYPayload/FHDR 中解析 `FCnt`；末尾残包、CRC invalid 包或 payload 不完整包会被整包舍弃。`--crc-mode` 和 `--print-payload` 仅在该模式下有意义。
- 修复 `lora_file_preamble_fft.py` 的 frame/header/payload 按 index 合并风险：`frame_sync` 现在把 `frame_count` 和 `start_sample:end_sample` 写入 `frame_info` tag，`header_decoder` 会保留这些字段，`crc_verif` 额外发布 `payload_metadata` 消息；Python 端优先按 `frame_count` 合并三路元数据，避免无效 header、排序或 CRC-valid payload 过滤造成错配。
- `lora_file_RX.py` 的前导码检测阈值未改变，仍由 `--preamble-len` 原样传入 `frame_sync`。已验证 `0_0_0_10_14_8.bin --preamble-len 8` 可正常解码和画图；`0_0_0_10_6_16.bin` 在 `--preamble-len 16` 下无有效输出，但降低到 `--preamble-len 12` 可解出 1 个 CRC-valid 包。
- 修复 `lora_file_preamble_fft.py` 在遇到非数字位置字段时写 NPZ 失败的问题。例如 `1_0_yidong_12_14_16.bin` 的 `position_id` 不能转成整数，现在 NPZ 中数字版 `position_id` 写为 `-1`，同时新增 `position_labels` 保留原始字符串 `yidong`。
- 为 Windows 路径兼容性增加 ASCII hardlink staging：GNU Radio C++ `file_source` 读取包含中文字符的路径时可能失败，脚本现在会为输入文件创建 ASCII-only 临时硬链接供 `file_source` 使用，Python 侧仍从真实路径读取 IQ 和元数据。
- 新增 `--isolated-workers` 批处理隔离模式和隐藏的 `--detect-only-json` worker 模式。批量处理时可让每个 `.bin` 在子进程中单独检测，避免单个 native 崩溃拖垮整个 lab 批处理；worker 失败时可通过 `--worker-log-lines` 打印尾部日志。
- 修复 `lib/frame_sync_impl.cc` 中 `DETECT -> SYNC` 阶段的越界崩溃：原代码在检测到 preamble 后用 `0.75 * m_samples_per_symbol - k_hat * m_os_factor` 作为 `in[]` 起点拷贝 `net_id_samp`，当 SF12 且 `k_hat > 3072` 时索引会变成负数，Windows 下表现为 `exit=3221225477` / `0xC0000005`。现在会检查 `k_hat` 和拷贝窗口边界，非法检测直接回到 `DETECT`，避免访问输入缓冲区之前的内存。
- 已按 Windows VS2022 + conda 环境增量编译并安装；用 `lab1-SF12-TP10/1_0_10_12_10_16.bin` 验证 `frame_sync` 不再触发 `0xC0000005`，默认模式可收集 9 个包；用 `lab1-SF12-TP14` 验证 18 个文件可完整写出 180 个 packet 的 CSV 和 NPZ。

### 2026-04-22

- `lora_file_RX.py` 新增 `--plot-phy-header` 别名，用于绘制 LoRa PHY 非 payload 片段：完整 preamble upchirp、2 个 sync word symbol、以及 2.25 symbol SFD。
- `frame_sync` 的绘图消息改为发布原始 IQ 文件中的对齐 `start_sample` / `end_sample`，范围为 `preamble_len + 2 sync + 2.25 SFD`，并且只在 header checksum valid 后发布。
- Python 端不再用 dechirp + FFT 细化起点；它直接从输入 `.bin` 连续切片绘图。已用 `0_0_0_10_14_8.bin` 验证：`--preamble-len 8` 时范围 `645341:695517`，长度 `12.25` symbols / `50176` samples。
- 修正 PHY header 频谱图纵轴缩放：绘图前把原始过采样 IQ 按 `sample_rate / bw` 抽取到 LoRa 带宽采样率，再按 `±bw/2` 生成频率轴，使一个 chirp 正好铺满纵轴高度。
- 明确 `--preamble-len` 同时决定图中导出的 preamble upchirp 数；总绘图长度为 `--preamble-len + 4.25` symbols。
- `lora_file_RX.py` 直接把 `--preamble-len` 原值传给 `frame_sync`，不再在 Python 层执行 `//2`，避免同步阈值被意外放低。
- 频谱图从 `matplotlib` 改为 `numpy + Pillow` 直接生成 PNG，避免 Windows 下 GNU Radio 消息线程触发 matplotlib 底层崩溃。
- `lora_file_RX.py` 的绘图消息处理改为先缓存索引/元数据，流图结束后统一从原始 IQ 文件切片并保存图片，提升离线批量处理稳定性。
- 已重新运行 `build_grlora.bat` 编译安装，并用 `0_0_0_10_14_8.bin` 验证：CRC valid，PHY header 图成功保存。
