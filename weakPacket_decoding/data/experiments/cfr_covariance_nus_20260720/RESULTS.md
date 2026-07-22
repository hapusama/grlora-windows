# CFR covariance NUS/GLRT synthetic SER

> 这是已知 GT 理想 LoRa symbol 上的 AWGN/AR(1) 方法学实验，不是自然 capture 结论。

## Selected patterns

| noise model | order | pattern | fixed | design marginal | exact replicas |
|---|---:|---|---:|---:|---:|
| awgn | 0 | fixed_0 | 1 | 1024 | 1 |
| awgn | 1 | fixed_1 | 1 | 1024 | 2 |
| awgn | 2 | fixed_2 | 1 | 1024 | 3 |
| awgn | 3 | fixed_3 | 1 | 1024 | 4 |
| awgn | 4 | dyadic_w1_s1_b0 | 0 | 0 | 4 |
| awgn | 5 | dyadic_w1_s3_b0 | 0 | 6.36646e-12 | 4 |
| awgn | 6 | dyadic_w1_s2_b1 | 0 | 0 | 4 |
| awgn | 7 | dyadic_w2_s2_b1 | 0 | 9.09495e-13 | 4 |
| ar1_rho0.8 | 0 | fixed_0 | 1 | 1010.23 | 1 |
| ar1_rho0.8 | 1 | fixed_1 | 1 | 111.965 | 1.1086 |
| ar1_rho0.8 | 2 | fixed_2 | 1 | 73.2607 | 1.17756 |
| ar1_rho0.8 | 3 | fixed_3 | 1 | 23.6866 | 1.19982 |
| ar1_rho0.8 | 4 | dyadic_w16_s1_b2 | 0 | 0.13217 | 1.19982 |
| ar1_rho0.8 | 5 | dyadic_w2_s3_b0 | 0 | 0.11918 | 1.19982 |
| ar1_rho0.8 | 6 | dyadic_w32_s3_b2 | 0 | 0.11784 | 1.19982 |
| ar1_rho0.8 | 7 | dyadic_w128_s3_b0 | 0 | 0.0862392 | 1.19982 |

## SER

| dataset | noise | SNR (dB) | symbols | Savaux SER | CFR-NUS-GLRT SER | full CFR-ML SER | NUS fixes/breaks | ML fixes/breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0_0_0_10_14_32 | ar1_rho0.8 | -26.000 | 128 | 0.875000 | 0.875000 | 0.609375 | 0/0 | 39/5 |
| 0_0_0_10_14_32 | ar1_rho0.8 | -24.000 | 128 | 0.671875 | 0.671875 | 0.351562 | 0/0 | 44/3 |
| 0_0_0_10_14_32 | awgn | -26.000 | 128 | 0.156250 | 0.156250 | 0.156250 | 0/0 | 0/0 |
| 0_0_0_10_14_32 | awgn | -24.000 | 128 | 0.023438 | 0.023438 | 0.023438 | 0/0 | 0/0 |

AWGN 下 CFR-NUS-GLRT 不应系统性超过 Savaux；两者应趋于同一个全样本匹配判决。
AR(1) 结果只验证已知 CFR 与测试噪声模型匹配时的可行性。
