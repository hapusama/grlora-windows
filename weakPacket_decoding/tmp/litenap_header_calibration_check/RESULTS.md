# LiteNap-Savaux clean-GT added-noise comparison

Source set: 17 clean-synchronized packets, 833 payload symbols.
Complex AWGN uses the noisy_iq convention: I/Q variance is half the requested total added-noise power.
Clean consensus FFT bins are scoring-only; synchronization, CFO, header-bin calibration, alias selection, steering, and phase-jump scores do not read payload GT.
Payload reference power: `0.00904977765`.
Full-view component spectrum versus original Savaux maximum absolute error: `1.92219e-06`.

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

## Interpretation guardrails

- `savaux` uses all `R*N` samples.
- `savaux_phase` uses the same samples and only adds LiteNap phase-jump reranking.
- `litenap_single` uses one `N/D` observation and retains alias ambiguity.
- `litenap_savaux_kK` uses all R oversampling phases for K downsampling views.
- `_phase` variants add the training-free phase-jump timing statistic; no clean waveform template is used.
- Every method shares the same explicit-header modulo-four residual-bin calibration.
- Added SNR is signal-reference power divided by added AWGN power, not the final measured capture SNR.
