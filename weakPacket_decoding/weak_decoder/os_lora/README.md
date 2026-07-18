# OS-LoRa 弱包解码模块

本目录按“系统实现不能依赖实验代码、实验入口不能互相依赖”的原则拆分。
后续接入实时或离线解码链时只需要依赖 `system/`；复现实验、生成表格和绘图时
才使用 `experiments/`。

## 目录职责

```text
os_lora/
├── system/       可复用的解码算法与数据结构
├── experiment_support/  多个实验共享的非在线基础设施
├── experiments/  可独立删除和运行的评估、消融、诊断、标定与绘图入口
├── doc/          算法说明与历史实验记录
└── __init__.py   稳定的公共接口
```

`system/` 当前包含：

- `nonuniform_sampling.py`：非均匀采样 pattern、频谱打分、GLS、条件检测器等核心算法。
- `chirp_svd.py`：ChirpSVD 的配置、训练与候选打分实现。
- `noise.py`：背景频点选择等共享噪声处理工具。

`experiments/` 中的脚本可以依赖 `system/` 和 `experiment_support/`，但不得导入
另一个实验入口；因此删除任意一个实验脚本不会导致其他实验出现导入错误。
`system/` 不得依赖另外两个目录。实验脚本中的 CSV 字段名和算法标识仍保留英文，
以免破坏已有结果、绘图脚本和论文数据处理流程；源代码注释与说明统一使用中文。

## 系统代码的导入方式

新代码优先从稳定公共接口导入：

```python
from weak_decoder.os_lora import build_pattern_bank, conditional_lora_gls_detect
```

需要明确模块来源时也可以直接导入：

```python
from weak_decoder.os_lora.system.nonuniform_sampling import build_pattern_bank
from weak_decoder.os_lora.system.noise import select_background_bins
```

## 运行实验

请从 `weakPacket_decoding` 目录以模块方式运行，避免脚本移动后出现相对路径问题：

```powershell
python -m weak_decoder.os_lora.experiments.evaluate_real_capture_gls --help
python -m weak_decoder.os_lora.experiments.evaluate_low_complexity_gls --help
python -m weak_decoder.os_lora.experiments.evaluate_nonuniform_sampling --help
```

实验入口按用途大致分为：

- `evaluate_*.py`：性能评估与解码对比。
- `analyze_*.py`：矩阵、候选、噪声协方差和 oracle 上限分析。
- `calibrate_*.py`：阈值标定。
- `compare_*.py`：条件检测器基线对比。
- `diagnose_*.py`：候选失败诊断。
- `plot_*.py`：结果可视化。

新增可部署算法时放入 `system/` 并通过 `system/__init__.py` 和顶层
`__init__.py` 导出；多个实验共用但不属于在线解码器的代码放入
`experiment_support/`；一次性评估、数据扫描和画图入口放入 `experiments/`。

## 真实 capture 汇总表

`evaluate_real_capture_gls` 每次运行都会在输出目录生成 `capture_summary.csv`，
其中包含包检测时刻、检测数、strict sync 数、`dechirp_gt_packet_snr_db`、
dechirp PNR、FFT errors，以及普通 FFT、Savaux 和 GLS 的 SER。传入以下参数可以把多次运行
更新到同一张表中：

```powershell
python -m weak_decoder.os_lora.experiments.evaluate_real_capture_gls `
  <其余参数> `
  --capture-summary-csv data\experiments\capture_summary.csv
```

`dechirp_gt_packet_snr_db` 使用传统单 branch dechirp FFT：先在每个 payload symbol
中读取外部 GT bin 的能量，再按包分别累加 GT-bin 能量与其余 `2^SF - 1` 个 bin 的能量，
计算 `10*log10(sum(E_gt) / sum(E_other))`。`capture_summary.csv` 中的主字段取逐包值的
中位数，完整逐包结果保存在 `packet_snr.csv`。外部 GT 只用于三种解调器完成判决后的
离线评分，不参与同步或 hard-bin 判决。dechirp PNR 是传统单 branch dechirp FFT 的
峰值功率与背景频点中位功率之比，并在全部评分 symbol 上取中位数。
`gls_ser` 默认对应 `gls_crossfit`，可用 `--summary-gls-method gls_offpacket` 改为
固定包外协方差 GLS。

## 架构约束测试

目录依赖由自动测试固定下来：

```powershell
python -m unittest `
  weak_decoder.os_lora.tests.test_architecture `
  weak_decoder.os_lora.tests.test_capture_summary -v
```

测试会检查实验入口之间没有直接导入、`system/` 没有反向依赖、系统模块没有
根目录同名副本，并逐个导入所有现存实验入口。
