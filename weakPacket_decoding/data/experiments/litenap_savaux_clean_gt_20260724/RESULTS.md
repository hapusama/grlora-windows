# LiteNap-Savaux clean-GT added-noise comparison

Source set: 17 clean-synchronized packets, 833 payload symbols.
Complex AWGN uses the noisy_iq convention: I/Q variance is half the requested total added-noise power.
Clean consensus FFT bins are scoring-only; synchronization, CFO, header-bin calibration, alias selection, steering, and phase-jump scores do not read payload GT.
Payload reference power: `0.00904977765`.
Full-view component spectrum versus original Savaux maximum absolute error: `1.92219e-06`.

## Outcome

Original Savaux has the lowest SER at every added-noise condition.
K1/K2 therefore provide sample-rate and computation trade-offs in this AWGN experiment, not a decoding gain over full-sample Savaux.
The phase variants are a training-free phase-jump diagnostic, not a full reproduction of LiteNap's transmitter-specific, preamble-calibrated hardware fingerprint.

| added SNR | Savaux | Savaux + phase | K2 (1/2 samples) | K1 (1/4 samples) |
|---:|---:|---:|---:|---:|
| -22 | 1.60% | 4.32% | 22.05% | 64.55% |
| -24 | 11.20% | 18.53% | 47.58% | 83.31% |
| -26 | 33.17% | 43.98% | 72.51% | 92.44% |
| -28 | 60.02% | 71.55% | 87.76% | 97.20% |

## Full comparison

| added SNR | method | samples | fraction | errors / decisions | SER | fixes / breaks | paired p | GT margin |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| clean | litenap_savaux_k1 | 1024 | 0.250 | 0 / 833 | 0.000000 | 0 / 0 | 1 | +3.868 dB |
| clean | litenap_savaux_k1_phase | 1024 | 0.250 | 0 / 833 | 0.000000 | 0 / 0 | 1 | +3.868 dB |
| clean | litenap_savaux_k2 | 2048 | 0.500 | 0 / 833 | 0.000000 | 0 / 0 | 1 | +6.889 dB |
| clean | litenap_savaux_k2_phase | 2048 | 0.500 | 0 / 833 | 0.000000 | 0 / 0 | 1 | +6.889 dB |
| clean | litenap_single | 256 | 0.062 | 714 / 833 | 0.857143 | 0 / 714 | 2.321e-215 | +0.000 dB |
| clean | litenap_single_phase | 256 | 0.062 | 341 / 833 | 0.409364 | 0 / 341 | 4.465e-103 | +0.000 dB |
| clean | savaux | 4096 | 1.000 | 0 / 833 | 0.000000 | 0 / 0 | 1 | +19.214 dB |
| clean | savaux_phase | 4096 | 1.000 | 0 / 833 | 0.000000 | 0 / 0 | 1 | +19.214 dB |
| -28 | litenap_savaux_k1 | 1024 | 0.250 | 2429 / 2499 | 0.971989 | 9 / 938 | 2.758e-264 | -6.592 dB |
| -28 | litenap_savaux_k1_phase | 1024 | 0.250 | 2440 / 2499 | 0.976391 | 10 / 950 | 3.625e-266 | -6.592 dB |
| -28 | litenap_savaux_k2 | 2048 | 0.500 | 2193 / 2499 | 0.877551 | 43 / 736 | 7.36e-164 | -3.993 dB |
| -28 | litenap_savaux_k2_phase | 2048 | 0.500 | 2250 / 2499 | 0.900360 | 30 / 780 | 1.198e-189 | -3.993 dB |
| -28 | litenap_single | 256 | 0.062 | 2490 / 2499 | 0.996399 | 5 / 995 | 1.548e-288 | -8.844 dB |
| -28 | litenap_single_phase | 256 | 0.062 | 2493 / 2499 | 0.997599 | 3 / 996 | 6.203e-293 | -8.844 dB |
| -28 | savaux | 4096 | 1.000 | 1500 / 2499 | 0.600240 | 0 / 0 | 1 | -1.185 dB |
| -28 | savaux_phase | 4096 | 1.000 | 1788 / 2499 | 0.715486 | 53 / 341 | 1.28e-52 | -1.185 dB |
| -26 | litenap_savaux_k1 | 1024 | 0.250 | 2310 / 2499 | 0.924370 | 15 / 1496 | 0 | -4.923 dB |
| -26 | litenap_savaux_k1_phase | 1024 | 0.250 | 2340 / 2499 | 0.936375 | 14 / 1525 | 0 | -4.923 dB |
| -26 | litenap_savaux_k2 | 2048 | 0.500 | 1812 / 2499 | 0.725090 | 33 / 1016 | 1.151e-253 | -2.175 dB |
| -26 | litenap_savaux_k2_phase | 2048 | 0.500 | 1989 / 2499 | 0.795918 | 18 / 1178 | 6.502e-321 | -2.175 dB |
| -26 | litenap_single | 256 | 0.062 | 2494 / 2499 | 0.997999 | 1 / 1666 | 0 | -8.110 dB |
| -26 | litenap_single_phase | 256 | 0.062 | 2482 / 2499 | 0.993197 | 4 / 1657 | 0 | -8.110 dB |
| -26 | savaux | 4096 | 1.000 | 829 / 2499 | 0.331733 | 0 / 0 | 1 | +0.823 dB |
| -26 | savaux_phase | 4096 | 1.000 | 1099 / 2499 | 0.439776 | 54 / 324 | 4.828e-48 | +0.823 dB |
| -24 | litenap_savaux_k1 | 1024 | 0.250 | 2082 / 2499 | 0.833133 | 4 / 1806 | 0 | -3.208 dB |
| -24 | litenap_savaux_k1_phase | 1024 | 0.250 | 2142 / 2499 | 0.857143 | 3 / 1865 | 0 | -3.208 dB |
| -24 | litenap_savaux_k2 | 2048 | 0.500 | 1189 / 2499 | 0.475790 | 10 / 919 | 5.6e-257 | -0.223 dB |
| -24 | litenap_savaux_k2_phase | 2048 | 0.500 | 1429 / 2499 | 0.571829 | 12 / 1161 | 0 | -0.223 dB |
| -24 | litenap_single | 256 | 0.062 | 2483 / 2499 | 0.993597 | 1 / 2204 | 0 | -6.984 dB |
| -24 | litenap_single_phase | 256 | 0.062 | 2469 / 2499 | 0.987995 | 1 / 2190 | 0 | -6.984 dB |
| -24 | savaux | 4096 | 1.000 | 280 / 2499 | 0.112045 | 0 / 0 | 1 | +2.826 dB |
| -24 | savaux_phase | 4096 | 1.000 | 463 / 2499 | 0.185274 | 27 / 210 | 2.683e-36 | +2.826 dB |
| -22 | litenap_savaux_k1 | 1024 | 0.250 | 1613 / 2499 | 0.645458 | 1 / 1574 | 0 | -1.422 dB |
| -22 | litenap_savaux_k1_phase | 1024 | 0.250 | 1709 / 2499 | 0.683874 | 2 / 1671 | 0 | -1.422 dB |
| -22 | litenap_savaux_k2 | 2048 | 0.500 | 551 / 2499 | 0.220488 | 3 / 514 | 1.074e-148 | +1.575 dB |
| -22 | litenap_savaux_k2_phase | 2048 | 0.500 | 762 / 2499 | 0.304922 | 2 / 724 | 1.495e-213 | +1.575 dB |
| -22 | litenap_single | 256 | 0.062 | 2473 / 2499 | 0.989596 | 0 / 2433 | 0 | -5.589 dB |
| -22 | litenap_single_phase | 256 | 0.062 | 2461 / 2499 | 0.984794 | 0 / 2421 | 0 | -5.589 dB |
| -22 | savaux | 4096 | 1.000 | 40 / 2499 | 0.016006 | 0 / 0 | 1 | +4.786 dB |
| -22 | savaux_phase | 4096 | 1.000 | 108 / 2499 | 0.043217 | 3 / 71 | 7.157e-18 | +4.786 dB |

## Interpretation guardrails

- `savaux` uses all `R*N` samples.
- `savaux_phase` uses the same samples and only adds LiteNap phase-jump reranking.
- `litenap_single` uses one `N/D` observation and retains alias ambiguity.
- `litenap_savaux_kK` uses all R oversampling phases for K downsampling views.
- `_phase` variants add the training-free phase-jump timing statistic; no clean waveform template is used.
- Every method shares the same explicit-header modulo-four residual-bin calibration.
- Added SNR is signal-reference power divided by added AWGN power, not the final measured capture SNR.
