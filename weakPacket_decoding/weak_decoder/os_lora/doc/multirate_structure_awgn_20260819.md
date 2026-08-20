# 单 Symbol Multi-rate LoRa Structure：纯 AWGN 结果

## 问题

实验只研究一个已经完美同步的 LoRa symbol。源输入为 `Fs = 8B`，并从同一份
noisy IQ 确定性抽取得到 `q = 8, 4, 2, 1` 四个视图。整个过程不做插值、低通、
重新同步或噪声训练，也不为不同视图重新生成独立噪声。

对于候选 bin `m`，`q > 1` 的完整 dechirped FFT 中，合法分量位于

```text
m
m - N (mod qN)
```

二者相隔严格的 `N = 2^SF` 个 FFT bin。在完美同步下，两处分量的理想幅度比为

```text
(N - m) : m
```

`q = 1` 时两处分量 alias 到同一个 bin。

## 实现

在线算法位于 `system/multirate_structure.py`，提供：

- nested-rate dechirp FFT；
- full-rate 强峰 Argmax；
- 间隔 `N` 的双峰能量和；
- 使用 `(N-m):m` 的 fold-profile 投影；
- 直接多率分数相加；
- 正确考虑 nested-view 相关性的 AWGN GLRT；
- 利用 LoRa 循环移位关系实现的完整高采样 coherent ML。

公共 `detect_multirate_structure()` 默认使用 `fusion_mode="awgn_glrt"`，不会把相关
视图当成独立信息重复相加；失败的直接相加仅能通过显式 `fusion_mode="equal_sum"`
用于消融复现。

完整 coherent ML 只需要一次输入 FFT、与缓存 reference spectrum 相乘、再做一次
IFFT，不需要建立 `N x qN` 模板矩阵。小规模数值测试确认它与项目现有 Savaux
oversampled AWGN metric 的归一化候选功率一致。

## 实验配置

```text
SF                         10
BW                         125 kHz
source Fs                  1 MS/s (q=8)
derived Fs                 500/250/125 kS/s
noise                      circular complex AWGN at source rate
SNR                        per-source-sample signal/noise power
synchronization/CFO/SFO    perfect
common symbol phase        randomized
trials                     2000 per SNR, paired across methods
seed                       20260819
```

命令：

```powershell
python -m weak_decoder.os_lora.experiments.archive.evaluate_multirate_structure_awgn `
  --sf 10 --bandwidth 125000 --source-os-factor 8 --rates 8 4 2 1 `
  --snrs -34 -32 -30 -28 -26 -24 -22 -21 -20 `
  --trials-per-snr 2000 --seed 20260819 `
  --output-dir data/experiments/multirate_structure_awgn_20260819
```

## SER

| SNR | 1M 强峰 Argmax | 1M 双峰能量 | 1M fold-profile | 四率直接相加 | 四率 AWGN-GLRT | 完整 1M ML |
|---:|---:|---:|---:|---:|---:|---:|
| -34 | 0.9590 | 0.9575 | 0.9285 | 0.9710 | 0.9285 | 0.8650 |
| -32 | 0.8960 | 0.8950 | 0.8460 | 0.9350 | 0.8460 | 0.6875 |
| -30 | 0.7340 | 0.7370 | 0.6335 | 0.8250 | 0.6335 | 0.3690 |
| -28 | 0.5080 | 0.4865 | 0.3525 | 0.6245 | 0.3525 | 0.0895 |
| -26 | 0.2575 | 0.1945 | 0.1140 | 0.3215 | 0.1140 | 0.0055 |
| -24 | 0.0775 | 0.0525 | 0.0105 | 0.0925 | 0.0105 | 0.0000 |
| -22 | 0.0250 | 0.0140 | 0.0005 | 0.0060 | 0.0005 | 0.0000 |
| -21 | 0.0140 | 0.0075 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| -20 | 0.0080 | 0.0055 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

这里的 `1M 强峰 Argmax` 已经不是只查看前 `N` 个 bin：它在两个合法位置中取较强者，
然后 modulo `N` 映射。因此 fold-profile 的收益不是来自修复一个明显错误的 bin 映射。

## 主要结论

### 1. 直接 multi-rate 相加失败

四个视图包含重复样点和相关噪声。等权相加在主要 weak-error 区间持续变差，例如
`-26 dB` 从 1M Argmax 的 `25.75%` 变成 `32.15%`。按 `q` 或 `q^2` 加权的消融也没有
稳定超过 1M fold-profile。

### 2. 正确白化后，multi-rate GLRT 退化为 1M fold-profile

把 `q=8,4,2,1` 的七个复分量组成小向量，并用源率白噪声诱导出的完整跨 rate
协方差白化后，低速视图的 conditional residual 不再包含候选信号。因此 multi-rate
AWGN-GLRT 与 1M fold-profile 在每个 trial 上给出相同判决。

这不是额外的 6 dB 合并增益；它是“确定性降采样没有创造信息”的可执行版本。

### 3. 两峰比例本身有用

`(N-m):m` fold-profile 明显优于只取强峰：

- `-28 dB`：`50.80% -> 35.25%`，paired fixes/breaks 为 `348/37`；
- `-26 dB`：`25.75% -> 11.40%`，paired fixes/breaks 为 `300/13`；
- `-24 dB`：`7.75% -> 1.05%`，paired fixes/breaks 为 `135/1`。

按 log-SER 插值，fold-profile 相对强峰 Argmax 的增益约为：

- SER `10%`：`1.47 dB`；
- SER `1%`：`3.57 dB`。

### 4. 收益确实集中在 wrap 信息最强的 symbol

`-26 dB` 按 `min(m, N-m)/N` 分桶：

| wrap 区域 | 1M Argmax SER | fold-profile SER | 完整 ML SER |
|---|---:|---:|---:|
| edge，均值 0.063 | 0.0183 | 0.0061 | 0.0041 |
| outer，均值 0.187 | 0.1102 | 0.0673 | 0.0020 |
| inner，均值 0.311 | 0.2923 | 0.1552 | 0.0081 |
| center，均值 0.436 | 0.5870 | 0.2199 | 0.0076 |

当 `m` 接近边缘时，本来就只有一个强分段；当 `m` 接近中间时，两段都重要，比例模板的
收益最大。这支持“wrap length 是 FFT peak location 之外的第二个 symbol evidence”。

### 5. 两峰仍不是完整充分统计量

完整 coherent ML 在 `-26 dB` 的 SER 是 `0.55%`，而 fold-profile 是 `11.40%`。
原因是两个合法中心 bin 之外仍有由有限分段产生的确定性旁瓣；完整时域模板利用了全部
`8N` 样点，fold-profile 只使用两个复投影。

## 落地建议

纯 AWGN 下不继续堆 multi-rate 投票。可部署路径按复杂度分两级：

1. 在已有 full-rate dechirp FFT 后增加 `(N-m):m` fold-profile，作为很便宜的
   Top-K 重排证据；
2. 如果允许约两次 `8N` FFT 的代价，直接使用循环相关 coherent ML，也就是当前
   Savaux AWGN metric 的快速等价实现。

下一步真实 IQ 验证必须冻结同步/CFO，并分别报告 edge/center symbol；否则同步误差会与
wrap-profile 误差混在一起。若加入 fractional STO，只能在预先声明的小 timing grid 上
最大化，并同时报告多重假设带来的 AWGN false-selection 代价。

## 产物

- `summary.csv`：逐 SNR、逐方法 SER 与 paired fixes/breaks；
- `trials.csv`：18000 个配对 trial；
- `wrap_regions.csv`：按 wrap 位置分桶；
- `invariants.csv`：双峰间隔与幅度比例审计；
- `RESULTS.md`：实验脚本自动生成的配置和总表。

无噪声 audit 的 20 个 `(symbol, rate)` 组合中，错误间隔数为 0；最大主峰幅度误差
`1.55e-6`，最大副峰幅度误差 `7.75e-7`，最大 wrap-ratio 误差 `3.96e-9`。
