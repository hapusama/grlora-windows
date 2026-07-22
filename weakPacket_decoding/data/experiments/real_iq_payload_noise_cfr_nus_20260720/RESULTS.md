# Real IQ payload-only added-noise GLS/NUS experiment

> 信号波形来自 USRP 实采 IQ；AWGN/AR(1) 是只叠加到 payload 符号副本的受控噪声。AR(1) 不是自然环境噪声拟合。

## Sources

| dataset | packets | audited symbols | min clean margin (dB) | reference power |
|---|---:|---:|---:|---:|
| 0_0_0_10_14_8 | 10 | 300 | 14.317 | 9.80178e-05 |
| 0_0_0_10_14_16 | 11 | 330 | 8.668 | 7.29068e-05 |
| 0_0_0_10_14_32 | 7 | 210 | 10.967 | 6.43783e-05 |

Only CRC-valid packets whose header and deterministic payload prefix re-encode exactly are used. Padding-only suffix symbols are excluded.

## Covariance information

| noise | Savaux GLS | full + NUS | full conditional | segmented Savaux | segmented + NUS | segmented conditional |
|---|---:|---:|---:|---:|---:|---:|
| awgn | 4096 | 4096 | 0 | 4096 | 4096 | 0 |
| ar1_rho0.8 | 1228.62 | 1228.62 | 0.00341657 | 2037.25 | 2037.42 | 0.169201 |

## Global SER

| noise | added SNR (dB) | symbols | Savaux equal | Savaux GLS | full + NUS | segmented Savaux | segmented + NUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| ar1_rho0.8 | -28.0 | 2520 | 0.962302 | 0.962302 | 0.961905 | 0.907540 | 0.907937 |
| ar1_rho0.8 | -26.0 | 2520 | 0.911111 | 0.911508 | 0.911508 | 0.776587 | 0.776984 |
| ar1_rho0.8 | -24.0 | 2520 | 0.796429 | 0.796429 | 0.796032 | 0.554365 | 0.554365 |
| ar1_rho0.8 | -22.0 | 2520 | 0.572222 | 0.570635 | 0.571429 | 0.279365 | 0.279365 |
| ar1_rho0.8 | -20.0 | 2520 | 0.296825 | 0.296825 | 0.296825 | 0.092460 | 0.092460 |
| awgn | -28.0 | 2520 | 0.678175 | 0.678175 | 0.678175 | 0.678175 | 0.678175 |
| awgn | -26.0 | 2520 | 0.403968 | 0.403968 | 0.403968 | 0.403968 | 0.403968 |
| awgn | -24.0 | 2520 | 0.150794 | 0.150794 | 0.150794 | 0.150794 | 0.150794 |
| awgn | -22.0 | 2520 | 0.043254 | 0.043254 | 0.043254 | 0.043254 | 0.043254 |
| awgn | -20.0 | 2520 | 0.025397 | 0.025397 | 0.025397 | 0.025397 | 0.025397 |

## Attribution

`segmented Savaux - Savaux GLS` measures information recovered by delaying the four-branch compression. `segmented + NUS - segmented Savaux` is the conditional contribution attributable to NUS. All methods use the same noise draw and a full-bin argmax.

## Paired attribution

| noise | SNR | segmented Savaux fixes/breaks | exact p | segmented NUS fixes/breaks | exact p | NUS runs W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| ar1_rho0.8 | -28 | 174/36 | 6.76e-23 | 3/4 | 1 | 1/5/3 |
| ar1_rho0.8 | -26 | 387/47 | 1.37e-67 | 3/4 | 1 | 0/8/1 |
| ar1_rho0.8 | -24 | 661/51 | 3.10e-136 | 2/2 | 1 | 1/7/1 |
| ar1_rho0.8 | -22 | 776/42 | 6.39e-176 | 3/3 | 1 | 1/7/1 |
| ar1_rho0.8 | -20 | 522/7 | 2.55e-144 | 1/1 | 1 | 1/7/1 |
| awgn | -28 | 0/0 | 1 | 0/0 | 1 | 0/9/0 |
| awgn | -26 | 0/0 | 1 | 0/0 | 1 | 0/9/0 |
| awgn | -24 | 0/0 | 1 | 0/0 | 1 | 0/9/0 |
| awgn | -22 | 0/0 | 1 | 0/0 | 1 | 0/9/0 |
| awgn | -20 | 0/0 | 1 | 0/0 | 1 | 0/9/0 |

## Verdict

- AWGN control passes exactly: all five detectors make the same decision on every trial. Neither GLS nor NUS has conditional information beyond Savaux in white noise.
- Under matched added AR(1) noise, 4-block fixed-branch GLS is stably better than full-symbol Savaux+GLS on every dataset, seed and tested SNR. At -22/-24/-26 dB, global SER changes from 0.5706/0.7964/0.9115 to 0.2794/0.5544/0.7766.
- NUS does not add a stable gain after the 4-block Savaux representation. Its paired fixes and breaks are balanced at every SNR, all exact p-values are 1, and the aggregate SER is equal or slightly worse at four of five colored-noise points.
- The usable method from this experiment is therefore covariance-guided time-localized Savaux GLS. Calling the gain a NUS gain would be incorrect.
