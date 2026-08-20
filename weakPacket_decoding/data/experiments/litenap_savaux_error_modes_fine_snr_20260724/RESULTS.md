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

| added SNR | Savaux | K2 (1/2 samples) | K1 (1/4 samples) |
|---:|---:|---:|---:|
| -16 | 0.00% | 0.00% | 2.64% |
| -17 | 0.00% | 0.12% | 6.96% |
| -18 | 0.00% | 0.56% | 13.37% |
| -19 | 0.00% | 1.68% | 23.49% |
| -20 | 0.08% | 5.52% | 36.73% |
| -21 | 0.32% | 12.16% | 50.30% |
| -22 | 1.72% | 20.97% | 62.71% |
| -23 | 5.36% | 34.33% | 75.19% |
| -24 | 10.20% | 46.42% | 81.71% |
| -25 | 20.97% | 61.10% | 89.24% |
| -26 | 33.29% | 72.11% | 92.28% |
| -27 | 46.86% | 81.11% | 95.68% |
| -28 | 60.70% | 88.76% | 95.76% |

## Full comparison

| added SNR | method | samples | fraction | errors / decisions | SER | fixes / breaks | paired p | GT margin |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| clean | litenap_savaux_k1 | 1024 | 0.250 | 0 / 833 | 0.000000 | 0 / 0 | 1 | +3.868 dB |
| clean | litenap_savaux_k2 | 2048 | 0.500 | 0 / 833 | 0.000000 | 0 / 0 | 1 | +6.889 dB |
| clean | litenap_single | 256 | 0.062 | 714 / 833 | 0.857143 | 0 / 714 | 2.321e-215 | +0.000 dB |
| clean | savaux | 4096 | 1.000 | 0 / 833 | 0.000000 | 0 / 0 | 1 | +19.214 dB |
| -28 | litenap_savaux_k1 | 1024 | 0.250 | 2393 / 2499 | 0.957583 | 15 / 891 | 5.821e-241 | -6.500 dB |
| -28 | litenap_savaux_k2 | 2048 | 0.500 | 2218 / 2499 | 0.887555 | 38 / 739 | 1.386e-169 | -4.142 dB |
| -28 | litenap_single | 256 | 0.062 | 2495 / 2499 | 0.998399 | 3 / 981 | 1.942e-288 | -8.813 dB |
| -28 | savaux | 4096 | 1.000 | 1517 / 2499 | 0.607043 | 0 / 0 | 1 | -1.269 dB |
| -27 | litenap_savaux_k1 | 1024 | 0.250 | 2391 / 2499 | 0.956783 | 7 / 1227 | 0 | -5.633 dB |
| -27 | litenap_savaux_k2 | 2048 | 0.500 | 2027 / 2499 | 0.811124 | 44 / 900 | 1.522e-208 | -3.082 dB |
| -27 | litenap_single | 256 | 0.062 | 2496 / 2499 | 0.998800 | 1 / 1326 | 0 | -8.305 dB |
| -27 | savaux | 4096 | 1.000 | 1171 / 2499 | 0.468587 | 0 / 0 | 1 | -0.185 dB |
| -26 | litenap_savaux_k1 | 1024 | 0.250 | 2306 / 2499 | 0.922769 | 8 / 1482 | 0 | -4.899 dB |
| -26 | litenap_savaux_k2 | 2048 | 0.500 | 1802 / 2499 | 0.721088 | 28 / 998 | 1.328e-254 | -2.167 dB |
| -26 | litenap_single | 256 | 0.062 | 2490 / 2499 | 0.996399 | 1 / 1659 | 0 | -8.007 dB |
| -26 | savaux | 4096 | 1.000 | 832 / 2499 | 0.332933 | 0 / 0 | 1 | +0.793 dB |
| -25 | litenap_savaux_k1 | 1024 | 0.250 | 2230 / 2499 | 0.892357 | 9 / 1715 | 0 | -4.236 dB |
| -25 | litenap_savaux_k2 | 2048 | 0.500 | 1527 / 2499 | 0.611044 | 19 / 1022 | 1.293e-273 | -1.202 dB |
| -25 | litenap_single | 256 | 0.062 | 2484 / 2499 | 0.993998 | 0 / 1960 | 0 | -7.491 dB |
| -25 | savaux | 4096 | 1.000 | 524 / 2499 | 0.209684 | 0 / 0 | 1 | +1.771 dB |
| -24 | litenap_savaux_k1 | 1024 | 0.250 | 2042 / 2499 | 0.817127 | 5 / 1792 | 0 | -3.035 dB |
| -24 | litenap_savaux_k2 | 2048 | 0.500 | 1160 / 2499 | 0.464186 | 12 / 917 | 3.587e-253 | -0.137 dB |
| -24 | litenap_single | 256 | 0.062 | 2483 / 2499 | 0.993597 | 1 / 2229 | 0 | -6.942 dB |
| -24 | savaux | 4096 | 1.000 | 255 / 2499 | 0.102041 | 0 / 0 | 1 | +2.910 dB |
| -23 | litenap_savaux_k1 | 1024 | 0.250 | 1879 / 2499 | 0.751901 | 2 / 1747 | 0 | -2.384 dB |
| -23 | litenap_savaux_k2 | 2048 | 0.500 | 858 / 2499 | 0.343337 | 9 / 733 | 1.566e-203 | +0.599 dB |
| -23 | litenap_single | 256 | 0.062 | 2481 / 2499 | 0.992797 | 0 / 2347 | 0 | -6.586 dB |
| -23 | savaux | 4096 | 1.000 | 134 / 2499 | 0.053621 | 0 / 0 | 1 | +3.737 dB |
| -22 | litenap_savaux_k1 | 1024 | 0.250 | 1567 / 2499 | 0.627051 | 0 / 1524 | 0 | -1.423 dB |
| -22 | litenap_savaux_k2 | 2048 | 0.500 | 524 / 2499 | 0.209684 | 4 / 485 | 2.969e-138 | +1.614 dB |
| -22 | litenap_single | 256 | 0.062 | 2470 / 2499 | 0.988395 | 0 / 2427 | 0 | -5.696 dB |
| -22 | savaux | 4096 | 1.000 | 43 / 2499 | 0.017207 | 0 / 0 | 1 | +4.784 dB |
| -21 | litenap_savaux_k1 | 1024 | 0.250 | 1257 / 2499 | 0.503001 | 0 / 1249 | 0 | -0.455 dB |
| -21 | litenap_savaux_k2 | 2048 | 0.500 | 304 / 2499 | 0.121649 | 1 / 297 | 1.174e-87 | +2.530 dB |
| -21 | litenap_single | 256 | 0.062 | 2461 / 2499 | 0.984794 | 0 / 2453 | 0 | -4.998 dB |
| -21 | savaux | 4096 | 1.000 | 8 / 2499 | 0.003201 | 0 / 0 | 1 | +5.784 dB |
| -20 | litenap_savaux_k1 | 1024 | 0.250 | 918 / 2499 | 0.367347 | 0 / 916 | 3.61e-276 | +0.308 dB |
| -20 | litenap_savaux_k2 | 2048 | 0.500 | 138 / 2499 | 0.055222 | 0 / 136 | 2.296e-41 | +3.391 dB |
| -20 | litenap_single | 256 | 0.062 | 2446 / 2499 | 0.978792 | 0 / 2444 | 0 | -4.257 dB |
| -20 | savaux | 4096 | 1.000 | 2 / 2499 | 0.000800 | 0 / 0 | 1 | +6.791 dB |
| -19 | litenap_savaux_k1 | 1024 | 0.250 | 587 / 2499 | 0.234894 | 0 / 587 | 3.948e-177 | +1.075 dB |
| -19 | litenap_savaux_k2 | 2048 | 0.500 | 42 / 2499 | 0.016807 | 0 / 42 | 4.547e-13 | +4.147 dB |
| -19 | litenap_single | 256 | 0.062 | 2419 / 2499 | 0.967987 | 0 / 2419 | 0 | -3.298 dB |
| -19 | savaux | 4096 | 1.000 | 0 / 2499 | 0.000000 | 0 / 0 | 1 | +7.792 dB |
| -18 | litenap_savaux_k1 | 1024 | 0.250 | 334 / 2499 | 0.133653 | 0 / 334 | 5.715e-101 | +1.732 dB |
| -18 | litenap_savaux_k2 | 2048 | 0.500 | 14 / 2499 | 0.005602 | 0 / 14 | 0.0001221 | +4.742 dB |
| -18 | litenap_single | 256 | 0.062 | 2407 / 2499 | 0.963185 | 0 / 2407 | 0 | -2.575 dB |
| -18 | savaux | 4096 | 1.000 | 0 / 2499 | 0.000000 | 0 / 0 | 1 | +8.766 dB |
| -17 | litenap_savaux_k1 | 1024 | 0.250 | 174 / 2499 | 0.069628 | 0 / 174 | 8.352e-53 | +2.202 dB |
| -17 | litenap_savaux_k2 | 2048 | 0.500 | 3 / 2499 | 0.001200 | 0 / 3 | 0.25 | +5.202 dB |
| -17 | litenap_single | 256 | 0.062 | 2339 / 2499 | 0.935974 | 0 / 2339 | 0 | -1.821 dB |
| -17 | savaux | 4096 | 1.000 | 0 / 2499 | 0.000000 | 0 / 0 | 1 | +9.764 dB |
| -16 | litenap_savaux_k1 | 1024 | 0.250 | 66 / 2499 | 0.026411 | 0 / 66 | 2.711e-20 | +2.591 dB |
| -16 | litenap_savaux_k2 | 2048 | 0.500 | 0 / 2499 | 0.000000 | 0 / 0 | 1 | +5.582 dB |
| -16 | litenap_single | 256 | 0.062 | 2307 / 2499 | 0.923169 | 0 / 2307 | 0 | -1.200 dB |
| -16 | savaux | 4096 | 1.000 | 0 / 2499 | 0.000000 | 0 / 0 | 1 | +10.708 dB |

## Interpretation guardrails

- `savaux` uses all `R*N` samples.
- `savaux_phase` uses the same samples and only adds LiteNap phase-jump reranking.
- `litenap_single` uses one `N/D` observation and retains alias ambiguity.
- `litenap_savaux_kK` uses all R oversampling phases for K downsampling views.
- `_phase` variants add the training-free phase-jump timing statistic; no clean waveform template is used.
- Every method shares the same explicit-header modulo-four residual-bin calibration.
- Added SNR is signal-reference power divided by added AWGN power, not the final measured capture SNR.
