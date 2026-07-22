# CFR covariance NUS/GLRT synthetic SER

> 这是已知 GT 理想 LoRa symbol 上的 AWGN/AR(1) 方法学实验，不是自然 capture 结论。

## Selected patterns

| noise model | order | pattern | fixed | conditional gain vs Savaux | exact replicas |
|---|---:|---|---:|---:|---:|
| awgn | 0 | fixed_0 | 1 | 0 | 1 |
| awgn | 1 | fixed_1 | 1 | 0 | 2 |
| awgn | 2 | fixed_2 | 1 | 0 | 3 |
| awgn | 3 | fixed_3 | 1 | 0 | 4 |
| awgn | 4 | dyadic_w1_s1_b0 | 0 | 0 | 4 |
| awgn | 5 | dyadic_w1_s1_b1 | 0 | 0 | 4 |
| awgn | 6 | dyadic_w1_s1_b2 | 0 | 0 | 4 |
| awgn | 7 | dyadic_w1_s1_b3 | 0 | 0 | 4 |
| ar1_rho0.8 | 0 | fixed_0 | 1 | 0 | 1 |
| ar1_rho0.8 | 1 | fixed_1 | 1 | 0 | 1.1086 |
| ar1_rho0.8 | 2 | fixed_2 | 1 | 0 | 1.17756 |
| ar1_rho0.8 | 3 | fixed_3 | 1 | 0 | 1.19982 |
| ar1_rho0.8 | 4 | dyadic_w1_s2_b1 | 0 | 0.00223205 | 1.19982 |
| ar1_rho0.8 | 5 | dyadic_w16_s1_b0 | 0 | 0.00210216 | 1.19982 |
| ar1_rho0.8 | 6 | dyadic_w8_s1_b0 | 0 | 0.00194086 | 1.19982 |
| ar1_rho0.8 | 7 | dyadic_w4_s1_b0 | 0 | 0.00160124 | 1.19982 |

## SER

| dataset | noise | SNR (dB) | symbols | Savaux equal | Savaux GLS | Savaux+NUS GLS | full CFR-ML | S-GLS fixes/breaks | NUS fixes/breaks vs S-GLS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0_0_0_10_14_32 | ar1_rho0.8 | -26.000 | 384 | 0.856771 | 0.856771 | 0.856771 | 0.617188 | 0/0 | 0/0 |
| 0_0_0_10_14_32 | ar1_rho0.8 | -24.000 | 384 | 0.679688 | 0.679688 | 0.679688 | 0.351562 | 0/0 | 0/0 |
| 0_0_0_10_14_32 | ar1_rho0.8 | -22.000 | 384 | 0.408854 | 0.408854 | 0.408854 | 0.075521 | 0/0 | 0/0 |
| 0_0_0_10_14_32 | awgn | -26.000 | 384 | 0.229167 | 0.229167 | 0.229167 | 0.229167 | 0/0 | 0/0 |
| 0_0_0_10_14_32 | awgn | -24.000 | 384 | 0.023438 | 0.023438 | 0.023438 | 0.023438 | 0/0 | 0/0 |
| 0_0_0_10_14_32 | awgn | -22.000 | 384 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0/0 | 0/0 |

主要判据是 Savaux+NUS 联合 GLS 是否稳定超过 Savaux 四支路 GLS，而不是是否只超过等权 Savaux。
AWGN 下条件增益应为零；AR(1) 结果只验证已知 CFR 与测试噪声模型匹配时的可行性。
