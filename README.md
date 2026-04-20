# gr-lora_sdr Windows 编译、CRC 双模式与前导码频谱图使用指南

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

本项目已提供 `build_grlora.bat`，路径：
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

## 五、CRC 双模式功能说明

本项目对 `crc_verif` 块进行了扩展，支持两种 CRC 校验算法：

| 模式 | 枚举值 | 适用场景 | 算法说明 |
|------|--------|---------|---------|
| **GRLORA**（默认） | `lora_sdr.Crc_mode.GRLORA` | gr-lora_sdr 软件发射机 ↔ 软件接收机 | 对 `payload_len - 2` 字节计算 CRC-16，再与最后 2 字节 XOR |
| **SX1276** | `lora_sdr.Crc_mode.SX1276` | SX1276/RFM95/SX1262 硬件发射机 → 软件接收机 | 标准 CRC-16-CCITT，对**全部 payload** 字节计算 |

### 5.1 在 Python 脚本中使用

```python
import gnuradio.lora_sdr as lora_sdr

# 方式 1：显式传入枚举值（推荐，清晰明了）
crc = lora_sdr.crc_verif(1, False, lora_sdr.Crc_mode.SX1276)

# 方式 2：使用兼容的静态方法（旧代码风格）
crc = lora_sdr.crc_verif(1, False, lora_sdr.crc_verif.SX1276())
```

### 5.2 在 lora_file_RX.py 中使用

```powershell
# 接收 SX1276 硬件发射的帧（标准 CRC-16）
python gr-lora_sdr\examples\lora_file_RX.py `
  -f "gr-lora_sdr\data\USRP_IQ\1_1_6_12_2_16.bin" `
  --sf 12 --bw 125000 --samp-rate 500000 --cr 2 `
  --center-freq 487.7e6 --sync-word 0x12 `
  --has-crc --crc-mode 1

# 接收 gr-lora_sdr 软件发射的帧（自定义 CRC）
python gr-lora_sdr\examples\lora_file_RX.py `
  -f "gr-lora_sdr\data\USRP_IQ\xxx.bin" `
  --sf 12 --bw 125000 --samp-rate 500000 --cr 2 `
  --has-crc --crc-mode 0
```

> `--crc-mode 0` 可省略，因为默认就是 GRLORA 模式。

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
# 第一次：SX1276 模式
python ... --crc-mode 1

# 第二次：GRLORA 模式
python ... --crc-mode 0
```

- 如果 **模式 1 显示 CRC valid，模式 0 显示 CRC invalid** → 发射端是标准 SX1276 CRC，新功能工作正常。
- 如果 **两者都 invalid** → 除了 CRC 算法外，SF/CR/SyncWord/采样率等参数可能还有不匹配。

---

## 七、对齐前导码频谱图功能

`examples/lora_file_RX.py` 现在可以把 `frame_sync` 内部已经同步、校正后的前导码 IQ 数据画成频谱图，用于检查前导码对齐、采样率/SF 是否匹配，以及后续做信号分析。

### 7.1 设计边界

- C++ 层 `frame_sync_impl.cc` 负责在同步成功后导出可信的前导码 IQ 数据。
- Python 层 `lora_file_RX.py` 负责接收 IQ 数据、画图、保存 PNG，未来也可以继续扩展为 `.npy/.npz` 导出或更复杂的分析。
- C++ 不负责画图，避免把 matplotlib/numpy 这类分析逻辑塞进实时信号处理块。

### 7.2 C++ 输出内容

`frame_sync` 新增了一个 message output port：

```text
preamble
```

同步成功后会发送一个 PMT dict，主要字段如下：

| 字段 | 含义 |
|------|------|
| `preamble_iq` | 对齐并校正后的前导码 IQ，PMT `c32vector`，Python 端可直接转为 `numpy.complex64` |
| `frame_count` | 当前帧序号 |
| `sf` | 扩频因子 |
| `bw` | 带宽 |
| `sample_rate` | 此处为校正后前导码的等效采样率，即 `bw` |
| `samples_per_symbol` | 每个 LoRa symbol 的样本数，等于 `2^sf` |
| `n_symbols` | 导出的前导码 symbol 数 |
| `snr_db` | frame_sync 估计的前导码 SNR |
| `cfo` | CFO 估计值，单位为 bins |
| `sto` | STO 估计值 |
| `sfo` | SFO 估计值 |
| `netid1`, `netid2` | 解出的两个 sync word / network identifier symbol |

这里使用 `pmt::init_c32vector(...)`，而不是 `pmt::make_blob(...)`。原因是 Windows 的 GNU Radio Python 绑定中 `pmt.blob_data()` 可能返回 `PyCapsule`，不方便直接转 `bytes`；`c32vector` 可以在 Python 中用 `pmt.c32vector_elements()` 稳定还原。

### 7.3 Python 脚本参数

`lora_file_RX.py` 新增以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--plot-preamble` | 关闭 | 开启前导码频谱图保存 |
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
  --crc-mode 1 `
  --plot-preamble `
  --preamble-plot-max 3
```

输出示例：

```text
[preamble_plot] saved D:\Desktop\proj\gr-lora_sdr\examples\preamble_plots\preamble_frame_001.png
```

### 7.5 绘图实现说明

Python 端使用 `numpy + matplotlib` 生成频谱图，不依赖 `scipy`。图像风格包括：

- 标题：`Preamble Symbol Spectrogram`
- x 轴：`Time (ms)`
- y 轴：`Frequency (kHz)`
- 频率范围：`-bw/2` 到 `+bw/2`
- 色图：`viridis`
- 右侧 colorbar 显示 dB 刻度
- 竖向虚线标出 LoRa symbol 边界

注意：`--preamble-len` 应尽量与发射端真实前导码长度一致。STM32/SX1276 默认常见值是 8，如果发射端是 8，就建议使用 `--preamble-len 8`。

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
| `lib/frame_sync_impl.h` | 新增 `publish_preamble(...)` 声明 |
| `lib/frame_sync_impl.cc` | 新增 `preamble` message output port；同步成功后把对齐/校正后的前导码 IQ 以 PMT `c32vector` 发给 Python |
| `examples/lora_file_RX.py` | 新增 `preamble_spectrogram_sink`，接收 `frame_sync` 的 `preamble` 消息并保存频谱图 |
| `examples/lora_file_RX.py` | 新增 `--plot-preamble`、`--preamble-plot-dir`、`--preamble-plot-max`、`--preamble-plot-dpi` 参数 |

---

## 九、快速参考：一条命令重新编译

```powershell
cd d:\Desktop\proj\gr-lora_sdr && .\build_grlora.bat
```

编译完成后，直接运行 `lora_file_RX.py` 即可使用最新的 CRC 功能。
