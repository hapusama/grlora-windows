# gr-lora_sdr Windows 安装与复用指南

> 记录时间：2026-04-15  
> 安装方式：Conda (Miniconda)  
> 操作系统：Windows 10/11  
> 项目原生支持 Windows（2023 年由 ryanvolz 完成移植）

---

## 1. 环境概况

| 项目 | 版本/说明 |
|------|----------|
| Python | 3.10.20 |
| GNU Radio | 3.10.11.0 |
| gr-lora_sdr | 0.0.0.20260105 (dev) |
| libvolk | 3.1.2 |
| Boost | 1.84.0 |
| numpy | 2.2.6 |
| pybind11 | abi=4 |

环境路径：`D:\mysoft2\miniconda3\envs\gr-lora`

---

## 2. 快速安装（复用步骤）

### 2.1 前置要求
- 已安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda（Windows 版）
- 网络可访问 `conda-forge` 和 `tapparelj` 频道

### 2.2 一键安装

```powershell
# 1. 创建虚拟环境
conda create -n gr-lora python=3.10 -y

# 2. 安装 gr-lora_sdr（会自动拉取 GNU Radio 3.10 及依赖）
conda install -n gr-lora -c tapparelj -c conda-forge gnuradio-lora_sdr -y

# 3. （可选）安装完整 GNU Radio 元包
conda install -n gr-lora -c conda-forge gnuradio -y
```

> 说明：步骤 2 已包含 `gnuradio-core`、`libvolk`、`boost`、`numpy` 等核心依赖。步骤 3 会额外安装 Qt/GUI 相关组件，但在 Windows 的 Conda 包中 **GNU Radio Companion (GRC) 图形界面并未完整分发**，因此无法使用 `gnuradio-companion` 命令。所有功能需通过 Python 脚本调用。

---

## 3. 验证安装

激活环境后，执行以下 Python 代码验证模块是否加载成功：

```python
# 验证核心 GNU Radio
import gnuradio
print("GNU Radio version:", gnuradio.VERSION_MAJOR)  # 应输出 3

# 验证 gr-lora_sdr 模块
from gnuradio.lora_sdr import (
    frame_sync, fft_demod, crc_verif,
    lora_sdr_lora_rx, lora_sdr_lora_tx
)
print("gr-lora_sdr blocks loaded successfully")
```

运行官方示例：

```powershell
conda activate gr-lora
cd examples
python tx_rx_functionality_check.py
```

预期输出包含 `modulate :info: set_min_output_buffer...` 等信息，表示收发链路已正常工作。

---

## 4. 项目目录结构（源码 + 环境配置）

```
gr-lora_sdr/
├── WINDOWS_SETUP.md          # 本文件：Windows 安装与复用说明
├── environment-windows.yml   # 简化版环境配置（可直接用于 conda env create）
├── environment-lock.yml      # 完整锁定环境导出（含所有子依赖版本）
├── conda_env_packages.txt    # conda list --export 的完整包列表
├── examples/                 # 官方示例脚本（.py + .grc）
│   ├── tx_rx_functionality_check.py
│   ├── tx_rx_simulation.py
│   ├── tx_rx_usrp.py
│   ├── lora_TX.py
│   └── lora_RX.py
├── lib/                      # C++ 核心实现
├── python/                   # Python 绑定
└── ...
```

---

## 5. 已知问题与注意事项

### 5.1 GNU Radio Companion (GRC) 不可用
Windows Conda 包未包含 `gnuradio-companion` 可执行文件，因此 `.grc` 流程图需通过以下两种方式使用：
- **方式 A**：在已安装 GRC 的 Linux/macOS 上编辑 `.grc`，生成 `.py` 后再拿到 Windows 运行。
- **方式 B**：直接阅读/修改 `examples/` 下已生成的 `.py` 文件。

### 5.2 部分示例脚本含 Linux 绝对路径
例如 `tx_rx_simulation.py` 中硬编码了：
```python
'/home/jtappare/Documents/gr-lora_sdr/data/GRC_default/example_tx_source.txt'
```
在 Windows 上运行前，需手动修改为本地实际路径，例如：
```python
r'D:\Desktop\proj\gr-lora_sdr\data\GRC_default\example_tx_source.txt'
```

### 5.3 gdk-pixbuf 安装警告（可忽略）
安装 `gnuradio` 元包时，`gdk-pixbuf` 的 post-link 脚本可能会因为 Windows 字符编码问题抛出 `UnicodeDecodeError`，但不影响 `gr-lora_sdr` 的正常运行。

---

## 6. 核心模块清单（C++ 源码位置）

| 文件 | 功能 |
|------|------|
| `lib/frame_sync_impl.cc` | 帧同步、STO/CFO 估计与校正 |
| `lib/fft_demod_impl.cc` | FFT 解调 |
| `lib/gray_demap_impl.cc` | Gray 解映射 |
| `lib/deinterleaver_impl.cc` | 解交织 |
| `lib/hamming_dec_impl.cc` | Hamming 解码 |
| `lib/dewhitening_impl.cc` | 去白化 |
| `lib/crc_verif_impl.cc` | CRC 校验 |

---

## 7. 参考链接

- 项目主页：https://github.com/tapparelj/gr-lora_sdr
- Conda 包页面：https://anaconda.org/tapparelj/gnuradio-lora_sdr
- 原始 README：[README.md](./README.md)
