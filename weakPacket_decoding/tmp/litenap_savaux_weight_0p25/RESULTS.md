# LiteNap-Savaux clean-GT added-noise comparison

Source set: 4 clean-synchronized packets, 196 payload symbols.
Complex AWGN uses the noisy_iq convention: I/Q variance is half the requested total added-noise power.
Clean consensus FFT bins are scoring-only; synchronization, CFO, alias selection, steering, and phase-jump scores do not read GT.
Payload reference power: `0.0111065404`.
Full-view component spectrum versus original Savaux maximum absolute error: `0`.

| added SNR | method | samples | fraction | errors / decisions | SER | fixes / breaks | paired p | GT margin |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| clean | litenap_savaux_k1 | 1024 | 0.250 | 0 / 196 | 0.000000 | 0 / 0 | 1 | +3.815 dB |
| clean | litenap_savaux_k1_phase | 1024 | 0.250 | 16 / 196 | 0.081633 | 0 / 16 | 3.052e-05 | +3.815 dB |
| clean | litenap_savaux_k2 | 2048 | 0.500 | 0 / 196 | 0.000000 | 0 / 0 | 1 | +6.841 dB |
| clean | litenap_savaux_k2_phase | 2048 | 0.500 | 0 / 196 | 0.000000 | 0 / 0 | 1 | +6.841 dB |
| clean | litenap_single | 256 | 0.062 | 168 / 196 | 0.857143 | 0 / 168 | 5.346e-51 | +0.000 dB |
| clean | litenap_single_phase | 256 | 0.062 | 98 / 196 | 0.500000 | 0 / 98 | 6.311e-30 | +0.000 dB |
| clean | savaux | 4096 | 1.000 | 0 / 196 | 0.000000 | 0 / 0 | 1 | +19.513 dB |
| clean | savaux_phase | 4096 | 1.000 | 0 / 196 | 0.000000 | 0 / 0 | 1 | +19.513 dB |
| -28 | litenap_savaux_k1 | 1024 | 0.250 | 187 / 196 | 0.954082 | 3 / 69 | 2.637e-17 | -6.686 dB |
| -28 | litenap_savaux_k1_phase | 1024 | 0.250 | 190 / 196 | 0.969388 | 1 / 70 | 6.099e-20 | -6.686 dB |
| -28 | litenap_savaux_k2 | 2048 | 0.500 | 175 / 196 | 0.892857 | 1 / 55 | 1.582e-15 | -4.244 dB |
| -28 | litenap_savaux_k2_phase | 2048 | 0.500 | 179 / 196 | 0.913265 | 0 / 58 | 6.939e-18 | -4.244 dB |
| -28 | litenap_single | 256 | 0.062 | 196 / 196 | 1.000000 | 0 / 75 | 5.294e-23 | -8.861 dB |
| -28 | litenap_single_phase | 256 | 0.062 | 194 / 196 | 0.989796 | 1 / 74 | 4.023e-21 | -8.861 dB |
| -28 | savaux | 4096 | 1.000 | 121 / 196 | 0.617347 | 0 / 0 | 1 | -1.084 dB |
| -28 | savaux_phase | 4096 | 1.000 | 139 / 196 | 0.709184 | 5 / 23 | 0.0009122 | -1.084 dB |
| -26 | litenap_savaux_k1 | 1024 | 0.250 | 184 / 196 | 0.938776 | 1 / 129 | 1.925e-37 | -4.992 dB |
| -26 | litenap_savaux_k1_phase | 1024 | 0.250 | 186 / 196 | 0.948980 | 2 / 132 | 8.307e-37 | -4.992 dB |
| -26 | litenap_savaux_k2 | 2048 | 0.500 | 137 / 196 | 0.698980 | 3 / 84 | 1.419e-21 | -1.899 dB |
| -26 | litenap_savaux_k2_phase | 2048 | 0.500 | 154 / 196 | 0.785714 | 1 / 99 | 1.593e-28 | -1.899 dB |
| -26 | litenap_single | 256 | 0.062 | 196 / 196 | 1.000000 | 0 / 140 | 1.435e-42 | -7.889 dB |
| -26 | litenap_single_phase | 256 | 0.062 | 195 / 196 | 0.994898 | 0 / 139 | 2.87e-42 | -7.889 dB |
| -26 | savaux | 4096 | 1.000 | 56 / 196 | 0.285714 | 0 / 0 | 1 | +1.058 dB |
| -26 | savaux_phase | 4096 | 1.000 | 78 / 196 | 0.397959 | 4 / 26 | 5.948e-05 | +1.058 dB |
| -24 | litenap_savaux_k1 | 1024 | 0.250 | 172 / 196 | 0.877551 | 0 / 161 | 6.842e-49 | -3.501 dB |
| -24 | litenap_savaux_k1_phase | 1024 | 0.250 | 179 / 196 | 0.913265 | 1 / 169 | 2.285e-49 | -3.501 dB |
| -24 | litenap_savaux_k2 | 2048 | 0.500 | 96 / 196 | 0.489796 | 0 / 85 | 5.17e-26 | -0.241 dB |
| -24 | litenap_savaux_k2_phase | 2048 | 0.500 | 111 / 196 | 0.566327 | 0 / 100 | 1.578e-30 | -0.241 dB |
| -24 | litenap_single | 256 | 0.062 | 196 / 196 | 1.000000 | 0 / 185 | 4.078e-56 | -7.307 dB |
| -24 | litenap_single_phase | 256 | 0.062 | 195 / 196 | 0.994898 | 1 / 185 | 3.813e-54 | -7.307 dB |
| -24 | savaux | 4096 | 1.000 | 11 / 196 | 0.056122 | 0 / 0 | 1 | +2.926 dB |
| -24 | savaux_phase | 4096 | 1.000 | 19 / 196 | 0.096939 | 4 / 12 | 0.07681 | +2.926 dB |

## Interpretation guardrails

- `savaux` uses all `R*N` samples.
- `savaux_phase` uses the same samples and only adds LiteNap phase-jump reranking.
- `litenap_single` uses one `N/D` observation and retains alias ambiguity.
- `litenap_savaux_kK` uses all R oversampling phases for K downsampling views.
- `_phase` variants add the training-free phase-jump timing statistic; no clean waveform template is used.
- Added SNR is signal-reference power divided by added AWGN power, not the final measured capture SNR.
