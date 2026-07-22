# Full colored-ML / Savaux added-AWGN audit

本实验使用冻结的真实 USRP payload 切片和冻结 GT；AWGN 只在内存中的 symbol 副本上加入。
`white_ml` 使用已知的 added-AWGN 协方差 `sigma^2 I`，因此在数学和实现上都与 Savaux 同判。
`colored_ml` 使用独立 continuous low1 包外噪声的 WSS/circulant PSD，再加 `sigma^2 I`；它是结构化估计下的 full-sample ML，不是已知真协方差 oracle。

## Global SER

| added SNR | decisions | Savaux errors | colored-ML errors | fixes / breaks | decision p | seed W/T/L | seed p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1176 | 3 | 3 | 0 / 0 | 1 | 0/3/0 | 1 |
| 5 | 1176 | 4 | 3 | 1 / 0 | 1 | 1/2/0 | 1 |
| 0 | 1176 | 7 | 7 | 0 / 0 | 1 | 0/3/0 | 1 |
| -2 | 1176 | 15 | 16 | 1 / 2 | 1 | 0/2/1 | 1 |
| -4 | 1176 | 25 | 25 | 1 / 1 | 1 | 0/3/0 | 1 |
| -6 | 1176 | 100 | 95 | 7 / 2 | 0.18 | 3/0/0 | 0.25 |
| -8 | 1176 | 263 | 263 | 5 / 5 | 1 | 1/1/1 | 1 |
| -10 | 1176 | 587 | 587 | 7 / 7 | 1 | 1/1/1 | 1 |
| clean | 392 | 1 | 1 | 0 / 0 | 1 | 0/1/0 | 1 |

## IQ storage audit

| dataset | whole-file exact zeros | header zeros | payload zeros | header→payload corr |
|---|---:|---:|---:|---:|
| collector_low4_train_low1 | 0.0019 | 0.0020 | 0.0019 | 0.652 |

`collector_low4` 是连续 IQ；其 colored covariance 只由独立 low1 capture 的检测保护区外噪声窗口估计，low4 payload GT 不参与协方差。

## Why the covariance becomes diagonal

对每个 added-SNR 点，总协方差模型是 `R_total = R_estimated + sigma^2 I`。下面的 color CV 越接近 0，矩阵越接近缩放单位阵。

| dataset | added SNR | AWGN / residual (dB) | color CV | lag-1 corr | offdiag/diag Fro |
|---|---:|---:|---:|---:|---:|
| collector_low4_train_low1 | 10 | -9.52 | 0.2428 | 0.1168 | 0.2428 |
| collector_low4_train_low1 | 5 | -4.52 | 0.1994 | 0.09598 | 0.1994 |
| collector_low4_train_low1 | 0 | 0.48 | 0.1275 | 0.06135 | 0.1275 |
| collector_low4_train_low1 | -2 | 2.48 | 0.09742 | 0.04688 | 0.09742 |
| collector_low4_train_low1 | -4 | 4.48 | 0.07091 | 0.03413 | 0.07091 |
| collector_low4_train_low1 | -6 | 6.48 | 0.04955 | 0.02384 | 0.04955 |
| collector_low4_train_low1 | -8 | 8.48 | 0.03353 | 0.01614 | 0.03353 |
| collector_low4_train_low1 | -10 | 10.48 | 0.02217 | 0.01067 | 0.02217 |

## Interpretation boundary

- `colored_ml > Savaux` 只说明完整样点中有可利用的二阶统计结构；它不自动证明 NUS。
- 若 weak-error 区域由 added AWGN 主导，`sigma^2 I` 会淹没原始有色项，full colored-ML 应收敛到 Savaux。
- 独立 low1 包外噪声能检验一般时间有色性；要支持 NUS，还必须进一步证明这种异常能定位到可预测的 `(p,q)` 采样可靠性，而不只是一般 WSS color。
