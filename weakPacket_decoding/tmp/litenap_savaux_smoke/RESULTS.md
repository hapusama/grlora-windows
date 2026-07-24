# LiteNap-Savaux clean-GT added-noise comparison

Source set: 1 clean-synchronized packets, 4 payload symbols.
Complex AWGN uses the noisy_iq convention: I/Q variance is half the requested total added-noise power.
Clean consensus FFT bins are scoring-only; synchronization, CFO, alias selection, steering, and phase-jump scores do not read GT.
Payload reference power: `0.0100002668`.
Full-view component spectrum versus original Savaux maximum absolute error: `1.3487e-06`.

| added SNR | method | samples | fraction | errors / decisions | SER | fixes / breaks | paired p | GT margin |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| clean | litenap_savaux_k1 | 1024 | 0.250 | 0 / 4 | 0.000000 | 0 / 0 | 1 | +3.864 dB |
| clean | litenap_savaux_k1_phase | 1024 | 0.250 | 0 / 4 | 0.000000 | 0 / 0 | 1 | +3.864 dB |
| clean | litenap_savaux_k2 | 2048 | 0.500 | 0 / 4 | 0.000000 | 0 / 0 | 1 | +6.905 dB |
| clean | litenap_savaux_k2_phase | 2048 | 0.500 | 0 / 4 | 0.000000 | 0 / 0 | 1 | +6.905 dB |
| clean | litenap_single | 256 | 0.062 | 4 / 4 | 1.000000 | 0 / 4 | 0.125 | +0.000 dB |
| clean | litenap_single_phase | 256 | 0.062 | 0 / 4 | 0.000000 | 0 / 0 | 1 | +0.000 dB |
| clean | savaux | 4096 | 1.000 | 0 / 4 | 0.000000 | 0 / 0 | 1 | +24.413 dB |
| -24 | litenap_savaux_k1 | 1024 | 0.250 | 4 / 4 | 1.000000 | 0 / 4 | 0.125 | -4.696 dB |
| -24 | litenap_savaux_k1_phase | 1024 | 0.250 | 4 / 4 | 1.000000 | 0 / 4 | 0.125 | -4.696 dB |
| -24 | litenap_savaux_k2 | 2048 | 0.500 | 4 / 4 | 1.000000 | 0 / 4 | 0.125 | -1.571 dB |
| -24 | litenap_savaux_k2_phase | 2048 | 0.500 | 3 / 4 | 0.750000 | 0 / 3 | 0.25 | -1.571 dB |
| -24 | litenap_single | 256 | 0.062 | 4 / 4 | 1.000000 | 0 / 4 | 0.125 | -6.514 dB |
| -24 | litenap_single_phase | 256 | 0.062 | 4 / 4 | 1.000000 | 0 / 4 | 0.125 | -6.514 dB |
| -24 | savaux | 4096 | 1.000 | 0 / 4 | 0.000000 | 0 / 0 | 1 | +1.725 dB |

## Interpretation guardrails

- `savaux` uses all `R*N` samples.
- `litenap_single` uses one `N/D` observation and retains alias ambiguity.
- `litenap_savaux_kK` uses all R oversampling phases for K downsampling views.
- `_phase` variants add the training-free phase-jump timing statistic; no clean waveform template is used.
- Added SNR is signal-reference power divided by added AWGN power, not the final measured capture SNR.
