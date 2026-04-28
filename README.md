# gr-lora_sdr Windows 编译、CRC 校验与前导码频谱图使用指南

> 本指南专门针对本项目（`d:\Desktop\proj\gr-lora_sdr`）在 **Windows 11 + conda + VS 2022** 环境下的编译、安装和调试。

---

## 一、环境准备（只需首次配置）

### 1.1 安装 Visual Studio 2022

- 下载并安装 **Visual Studio Community 2022**（或 Build Tools）
- 安装时必须勾选：**"使用 C++ 的桌面开发"** 工作负载
- 记录 `vcvarsall.bat` 路径，例如：
  ```
  C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat
  ```

### 1.2 创建 conda 环境并安装依赖

```powershell
conda create -n gr-lora python=3.10
conda activate gr-lora
conda install -c conda-forge gnuradio-core boost-cpp volk cmake pybind11
```

> ⚠️ **关键**：`pybind11` 版本必须与 `gnuradio-core` 的 Python 绑定 ABI 兼容。如果后续出现 `ImportError: generic_type: type "..." referenced unknown base type "gr::block"`，说明 pybind11 版本不匹配，需要升级/降级后重新编译。

---

## 二、编译与安装（每次修改源码后执行）

### 2.1 使用一键编译脚本（推荐）

本项目已提供 `build_grlora.bat`，路径：(注意，目前是把绑定的conda写死成gr-lora，但是实际使用gnuradio的时候是radioconda环境，所以如果要在gnuradio中使用这个模块，需要把这个路径改成radioconda的路径)
```
d:\Desktop\proj\gr-lora_sdr\build_grlora.bat
```

直接双击运行，或在 **x64 Native Tools Command Prompt for VS 2022** 中执行：

```powershell
cd d:\Desktop\proj\gr-lora_sdr
.\build_grlora.bat
```

脚本会自动完成以下步骤：
1. 调用 `vcvarsall.bat x64` 设置 MSVC 环境
2. 删除旧的 `build` 目录，创建新的
3. 运行 CMake 配置（使用 NMake Makefiles 生成器）
4. 编译（`nmake`）
5. 安装到 conda 环境的 `Library` 目录

### 2.2 手动分步编译（如需排查问题）

```powershell
# 1. 打开 VS 2022 x64 命令行工具，或手动调用 vcvarsall
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

# 2. 进入项目目录
cd /d d:\Desktop\proj\gr-lora_sdr

# 3. 清理并创建 build 目录
rmdir /s /q build
mkdir build
cd build

# 4. 设置 CMake 前缀路径
set CMAKE_PREFIX_PATH=D:\mysoft2\miniconda3\envs\gr-lora\Library

# 5. 配置
cmake .. -G "NMake Makefiles" ^
    -DCMAKE_INSTALL_PREFIX=D:\mysoft2\miniconda3\envs\gr-lora\Library ^
    -DPYTHON_EXECUTABLE=D:\mysoft2\miniconda3\envs\gr-lora\python.exe ^
    -DGR_PYTHON_DIR=D:\mysoft2\miniconda3\envs\gr-lora\Lib\site-packages

# 6. 编译
nmake

# 7. 安装
nmake install
```

---

## 三、验证编译是否成功

### 3.1 检查 Python 导入

```powershell
conda activate gr-lora
python -c "import gnuradio.lora_sdr as lora; print('Import OK'); print('Crc_mode:', lora.Crc_mode.GRLORA, lora.Crc_mode.SX1276)"
```

如果输出 `Import OK` 并且显示了 `Crc_mode.GRLORA` 和 `Crc_mode.SX1276`，说明编译和安装都成功了。

### 3.2 检查模块路径（确认不是旧版本）

```powershell
python -c "import gnuradio.lora_sdr; import os; print(os.path.dirname(gnuradio.lora_sdr.__file__))"
```

应输出类似：
```
D:\mysoft2\miniconda3\envs\gr-lora\lib\site-packages\gnuradio\lora_sdr
```

---

## 四、常见问题与排查

### 4.1 `ImportError: generic_type: type "add_crc" referenced unknown base type "gr::block"`

**原因**：`gnuradio-core` 的 Python 绑定和 `lora_sdr_python.pyd` 的 **pybind11 internals 版本不匹配**。

**诊断**：检查两个二进制文件中的 pybind11 内部版本号：
```python
import re
with open(r'D:\mysoft2\miniconda3\envs\gr-lora\Lib\site-packages\gnuradio\gr\gr_python.cp310-win_amd64.pyd','rb') as f:
    print('gr_python:', re.findall(rb'__pybind11_internals_v\d+', f.read()))
with open(r'D:\mysoft2\miniconda3\envs\gr-lora\Lib\site-packages\gnuradio\lora_sdr\lora_sdr_python.cp310-win_amd64.pyd','rb') as f:
    print('lora_sdr_python:', re.findall(rb'__pybind11_internals_v\d+', f.read()))
```

两者必须显示 **相同** 的版本（如都是 `v5`）。

**修复**：
```powershell
conda activate gr-lora
conda install -c conda-forge pybind11=2.13.6   # 安装与 gnuradio-core 匹配的版本
rmdir /s /q d:\Desktop\proj\gr-lora_sdr\build   # 清理旧构建
# 然后重新运行 build_grlora.bat
```

### 4.2 `Python bindings out of sync` 错误

**原因**：修改了 C++ 头文件（如 `crc_verif.h`），但 `crc_verif_python.cc` 中的 `BINDTOOL_HEADER_FILE_HASH` 与新的头文件哈希不匹配。

**修复**：计算新头文件的 MD5 哈希，更新到 `crc_verif_python.cc` 中的 `BINDTOOL_HEADER_FILE_HASH(...)` 宏。

---

## 五、CRC 校验说明

结论先写清楚：**gr-lora_sdr 默认的 CRC 校验逻辑本身没有错**。如果离线解码
LoraSTMacL1 / SX1276 发出来的 LoRa 包时出现 `invalid CRC`，不能直接归因于
gr-lora_sdr 的 CRC 算法错误；更常见的原因是解调后仍有残留 bit 错、SF/CR/SyncWord/
LDRO/前导码长度等参数不匹配，或者同步、CFO、STO 校正不够好。

`crc_verif` 当前保留了两种计算模式，便于对照排查：

| 模式 | 枚举值 | 适用场景 | 算法说明 |
|------|--------|---------|---------|
| **GRLORA**（默认） | `lora_sdr.Crc_mode.GRLORA` | gr-lora_sdr 软件发射机；LoraSTMacL1/SX1276 LoRa PHY 包优先使用此模式 | 对 `payload_len - 2` 字节计算 CRC-16，再与最后 2 字节 XOR；这是 gr-lora_sdr 论文和代码采用的 LoRa PHY CRC 形式 |
| **SX1276** | `lora_sdr.Crc_mode.SX1276` | 仅作为对照/实验模式 | 对**全部 payload** 字节直接计算 CRC-16-CCITT；名字来自早期排查假设，不建议作为 LoraSTMacL1/SX1276 LoRa PHY 包的默认选择 |

注意：LoraSTMacL1 工程里的 `Radio.SetTxConfig(..., crcOn=true, ...)` 只是打开
SX1276 的硬件 LoRa PHY CRC。MCU 代码发送进 FIFO 的 LoRaWAN PHYPayload 不包含
这两个 PHY CRC 字节，CRC 由 SX1276 在空口帧尾自动追加。因此，README 中提到的
`0x12345678` 一类字段如果存在，属于 LoRaWAN MIC/测试占位字段，不是 LoRa PHY CRC。

### 5.1 在 Python 脚本中使用

```python
import gnuradio.lora_sdr as lora_sdr

# 推荐：使用默认 GRLORA 模式
crc = lora_sdr.crc_verif(1, False, lora_sdr.Crc_mode.GRLORA)

# 方式 2：使用兼容的静态方法（旧代码风格）
crc = lora_sdr.crc_verif(1, False, lora_sdr.crc_verif.GRLORA())
```

### 5.2 在 lora_file_RX.py 中使用

```powershell
# 接收 LoraSTMacL1 / SX1276 硬件发射的 LoRa PHY 帧，优先使用默认 GRLORA 模式
python gr-lora_sdr\examples\lora_file_RX.py `
  -f "gr-lora_sdr\data\USRP_IQ\1_1_6_12_2_16.bin" `
  --sf 12 --bw 125000 --samp-rate 500000 --cr 2 `
  --center-freq 487.7e6 --sync-word 0x12 `
  --has-crc --crc-mode 0

# `--crc-mode 0` 可省略，因为默认就是 GRLORA 模式
python gr-lora_sdr\examples\lora_file_RX.py `
  -f "gr-lora_sdr\data\USRP_IQ\xxx.bin" `
  --sf 12 --bw 125000 --samp-rate 500000 --cr 2 `
  --has-crc
```

如果 `--crc-mode 0` 仍然显示 `invalid CRC`，下一步应检查解出来的 payload 是否已经有
bit 错。例如 ASCII 测试载荷中出现 `0x39 -> 0x59`、`0x43 -> 0x13` 这种一位或多位错误时，
CRC 必然不通过；这说明 FEC 之后仍有残留误码，而不是 CRC 算法错了。

---

## 六、调试技巧

### 6.1 确认 CRC 模式实际生效

`crc_verif_impl.cc` 中已加入调试输出，运行时会在终端打印：
```
[crc_verif] CRC mode: SX1276, payload_len=33
```
或
```
[crc_verif] CRC mode: GRLORA, payload_len=33
```

看到这一行，即可 100% 确认当前使用的是哪种 CRC 算法。

### 6.2 对比两种模式的校验结果

同一个文件跑两次：
```powershell
# 第一次：默认 GRLORA 模式
python ... --crc-mode 0

# 第二次：对照模式
python ... --crc-mode 1
```

- 如果 **模式 0 显示 CRC valid** → 当前 LoRa PHY CRC 校验正常。
- 如果 **模式 0 invalid，但 payload 明显有 bit 错** → CRC 失败是残留误码的正常结果。
- 如果 **模式 0 和模式 1 都 invalid** → 优先排查 SF/CR/SyncWord/采样率/LDRO/前导码长度、CFO/STO、IQ 文件中静默段等问题。

---

## 七、对齐前导码频谱图功能

`examples/lora_file_RX.py` 现在可以把原始 IQ 文件中的 LoRa PHY 非 payload 片段画成频谱图，用于检查 preamble、sync word 和 SFD 的真实时频结构。

### 7.1 设计边界

- C++ 层 `frame_sync_impl.cc` 在完成前导码、sync word、SFD 精同步后，计算原始 IQ 文件中对齐的 `preamble + sync word + SFD` 样本索引范围。
- C++ 层不会立刻发布该范围；只有 `header_decoder` 回传 header checksum valid 后，`frame_sync` 才把该范围发给 Python。
- Python 层 `lora_file_RX.py` 只负责按 `start_sample:end_sample` 从真实 `.bin` IQ 中切片、画图、保存 PNG，未来也可以继续扩展为 `.npy/.npz` 导出或更复杂的分析。
- C++ 不负责画图，避免把 numpy/Pillow 这类分析逻辑塞进实时信号处理块。

### 7.2 C++ 输出内容

`frame_sync` 新增了一个 message output port：

```text
preamble
```

header checksum 通过后会发送一个 PMT dict，主要字段如下：

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
| `snr_db` | frame_sync 估计的前导码 SNR |
| `cfo` | CFO 估计值，单位为 bins |
| `sto` | STO 估计值 |
| `sfo` | SFO 估计值 |
| `netid1`, `netid2` | 解出的两个 sync word / network identifier symbol |

注意：该消息不携带拼接后的绘图 IQ。Python 端根据 `start_sample` / `end_sample` 直接从输入 `.bin` 文件读取连续原始 IQ，不再做 dechirp + FFT 二次细化。

### 7.3 Python 脚本参数

`lora_file_RX.py` 新增以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--plot-preamble`, `--plot-phy-header` | 关闭 | 开启 LoRa PHY 非 payload 片段频谱图保存 |
| `--preamble-plot-dir` | `examples/preamble_plots` | PNG 输出目录 |
| `--preamble-plot-max` | `3` | 最多保存多少帧，`0` 表示不限制 |
| `--preamble-plot-dpi` | `150` | 输出图片 DPI |

默认只画前三帧，是为了避免长 IQ 文件生成大量 PNG。如果要画全部帧：

```powershell
--preamble-plot-max 0
```

### 7.4 使用示例

```powershell
python examples\lora_file_RX.py `
  -f data\USRP_IQ\1_1_6_10_2_16.bin `
  --sf 11 `
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

输出示例：

```text
[phy_header_plot] queued frame 1 (samples 645341:695517, 12.25 symbols)
[phy_header_plot] saving frame 1 (12.25 symbols, 50176 samples)
[phy_header_plot] saved D:\Desktop\proj\gr-lora_sdr\examples\preamble_plots\phy_header_frame_001.png
```

### 7.5 绘图实现说明

Python 端使用 `numpy + Pillow` 生成频谱图，不依赖 `scipy`，也不依赖 matplotlib 的绘图后端。这样可以避免 Windows 下 GNU Radio 消息线程触发 matplotlib 底层崩溃。图像风格包括：

- 标题：`LoRa Preamble + Sync + SFD Spectrogram`
- x 轴：`Time (ms)`
- y 轴：`Frequency (kHz)`
- 频率范围：`-bw/2` 到 `+bw/2`
- 色图：`viridis`
- 右侧 colorbar 显示 dB 刻度
- 竖向虚线标出 LoRa symbol 边界

注意：PHY header 图按 `--preamble-len + 2 + 2.25` 个 symbol 从原始 IQ 中连续切出 preamble、2 个 sync word symbol 和 2.25 symbol SFD，不包含 payload。比如 `--preamble-len 8` 时，输出长度就是 `12.25` symbols。

---

## 八、文件修改清单（本项目的改动）

以下文件已被修改以支持 CRC 双模式：

| 文件 | 改动内容 |
|------|---------|
| `include/gnuradio/lora_sdr/crc_verif.h` | 新增 `Crc_mode` 枚举（`GRLORA`, `SX1276`）和 `make()` 的第三个参数 |
| `lib/crc_verif_impl.h` | 新增 `m_crc_mode` 成员和 `crc16_sx1276()` 声明 |
| `lib/crc_verif_impl.cc` | 新增 `crc16_sx1276()` 实现，运行时根据 `m_crc_mode` 分支 |
| `python/lora_sdr/bindings/crc_verif_python.cc` | pybind11 绑定：`py::enum_<Crc_mode>` + `.def_static` 兼容层 |
| `python/lora_sdr/lora_sdr_lora_rx.py` | hier block 增加 `crc_mode` 参数 |
| `examples/lora_file_RX.py` | 新增 `--crc-mode` CLI 参数，int→enum 转换 |
| `grc/lora_sdr_crc_verif.block.yml` | GRC 块定义增加 CRC 模式下拉框 |
| `build_grlora.bat` | Windows 一键编译脚本 |

以下文件已被修改以支持对齐前导码频谱图：

| 文件 | 改动内容 |
|------|---------|
| `lib/frame_sync_impl.h` | 新增 PHY header 样本范围发布所需状态和 `publish_phy_header(...)` / `try_publish_phy_header()` 声明 |
| `lib/frame_sync_impl.cc` | 新增 `preamble` message output port；精同步后记录 `preamble + sync word + SFD` 原始 IQ 索引，header checksum 通过后再发给 Python |
| `examples/lora_file_RX.py` | 新增 `phy_header_spectrogram_sink`，接收 `frame_sync` 的 `preamble` 消息并保存频谱图 |
| `examples/lora_file_RX.py` | 新增 `--plot-preamble`、`--preamble-plot-dir`、`--preamble-plot-max`、`--preamble-plot-dpi` 参数 |

---

## 九、更新日志

### 2026-04-28

- `examples/lora_file_preamble_fft.py` 从单帧前导码 FFT 导出脚本扩展为离线批处理特征导出脚本：支持 `--all-bin --input-dir ...` 按 USRP_IQ 下的 lab 子文件夹分组处理，每个 lab 单独写回 `packet_features.csv` 和 `preamble_features.npz`。
- `lora_file_preamble_fft.py` 会从文件名解析实验编号、走廊编号、位置编号、SF、发射功率和前导码长度；如果 lab 目录存在 `补充.txt`，会读取其中的说明和参数覆盖信息，并把 lab 元数据写入输出结果。
- 输出收敛为 packet 级特征：平均 RSSI、平均 SNR、SX1276 风格 SNR register 值、前导码主峰 `-3 dB` 宽度、以及主峰对齐后的局部平均幅度谱。旧版 per-frame 明细 CSV 不再生成，并会清理 `rssi_samples_5ms.csv`、`preamble_symbol_features.csv`、`position_summary.csv` 等过期输出，避免误读旧文件。
- 默认运行模式不再解析 payload。流图在 `header_decoder` 后直接接 `null_sink`，只依赖 `frame_sync` 的对齐范围和 `header_decoder` 的 PHY header 信息计算信号特征。因此末尾残包、payload 解码失败或 CRC invalid 不会再导致 `decoded payload count != detected packet count` 这类 warning，也不会影响 RSSI/SNR/preamble FFT 特征导出。
- 如确实需要 LoRaWAN `FCnt`，新增可选开关 `--require-valid-payload`（别名 `--with-fcnt`）。启用后脚本会重新接回 `header_decoder → dewhitening → crc_verif`，只保留 CRC-valid payload 对应的数据包并尝试从 PHYPayload/FHDR 中解析 `FCnt`；末尾残包、CRC invalid 包或 payload 不完整包会被整包舍弃。`--crc-mode` 和 `--print-payload` 仅在该模式下有意义。
- 修复 `lora_file_preamble_fft.py` 在遇到非数字位置字段时写 NPZ 失败的问题。例如 `1_0_yidong_12_14_16.bin` 的 `position_id` 不能转成整数，现在 NPZ 中数字版 `position_id` 写为 `-1`，同时新增 `position_labels` 保留原始字符串 `yidong`。
- 为 Windows 路径兼容性增加 ASCII hardlink staging：GNU Radio C++ `file_source` 读取包含中文字符的路径时可能失败，脚本现在会为输入文件创建 ASCII-only 临时硬链接供 `file_source` 使用，Python 侧仍从真实路径读取 IQ 和元数据。
- 新增 `--isolated-workers` 批处理隔离模式和隐藏的 `--detect-only-json` worker 模式。批量处理时可让每个 `.bin` 在子进程中单独检测，避免单个 native 崩溃拖垮整个 lab 批处理；worker 失败时可通过 `--worker-log-lines` 打印尾部日志。
- 修复 `lib/frame_sync_impl.cc` 中 `DETECT → SYNC` 阶段的越界崩溃：原代码在检测到 preamble 后用 `0.75 * m_samples_per_symbol - k_hat * m_os_factor` 作为 `in[]` 起点拷贝 `net_id_samp`，当 SF12 且 `k_hat > 3072` 时索引会变成负数，Windows 下表现为 `exit=3221225477` / `0xC0000005`。现在会检查 `k_hat` 和拷贝窗口边界，非法检测直接回到 `DETECT`，避免访问输入缓冲区之前的内存。
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

---

## 十、快速参考：一条命令重新编译

```powershell
cd d:\Desktop\proj\gr-lora_sdr && .\build_grlora.bat
```

编译完成后，直接运行 `lora_file_RX.py` 即可使用最新的 CRC 功能。
