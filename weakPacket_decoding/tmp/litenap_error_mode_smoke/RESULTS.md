# LiteNap-Savaux clean-GT added-noise comparison

Source set: 1 clean-synchronized packets, 32 payload symbols.
Complex AWGN uses the noisy_iq convention: I/Q variance is half the requested total added-noise power.
Clean consensus FFT bins are scoring-only; synchronization, CFO, header-bin calibration, alias selection, steering, and phase-jump scores do not read payload GT.
Payload reference power: `0.010386325`.
Full-view component spectrum versus original Savaux maximum absolute error: `1.92219e-06`.

## Outcome

Original Savaux has the lowest SER at every added-noise condition.
K1/K2 therefore provide sample-rate and computation trade-offs in this AWGN experiment, not a decoding gain over full-sample Savaux.
The phase variants are a training-free phase-jump diagnostic, not a full reproduction of LiteNap's transmitter-specific, preamble-calibrated hardware fingerprint.

| added SNR | Savaux | K2 (1/2 samples) | K1 (1/4 samples) |
|---:|---:|---:|---:|
| -20 | 0.00% | 0.00% | 34.38% |
| -21 | 0.00% | 3.12% | 56.25% |

## Full comparison

| added SNR | method | samples | fraction | errors / decisions | SER | fixes / breaks | paired p | GT margin |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| clean | litenap_savaux_k1 | 1024 | 0.250 | 0 / 32 | 0.000000 | 0 / 0 | 1 | +3.740 dB |
| clean | litenap_savaux_k2 | 2048 | 0.500 | 0 / 32 | 0.000000 | 0 / 0 | 1 | +6.766 dB |
| clean | litenap_single | 256 | 0.062 | 27 / 32 | 0.843750 | 0 / 27 | 1.49e-08 | +0.000 dB |
| clean | savaux | 4096 | 1.000 | 0 / 32 | 0.000000 | 0 / 0 | 1 | +20.132 dB |
| -21 | litenap_savaux_k1 | 1024 | 0.250 | 18 / 32 | 0.562500 | 0 / 18 | 7.629e-06 | -0.357 dB |
| -21 | litenap_savaux_k2 | 2048 | 0.500 | 1 / 32 | 0.031250 | 0 / 1 | 1 | +3.324 dB |
| -21 | litenap_single | 256 | 0.062 | 32 / 32 | 1.000000 | 0 / 32 | 4.657e-10 | -4.662 dB |
| -21 | savaux | 4096 | 1.000 | 0 / 32 | 0.000000 | 0 / 0 | 1 | +6.137 dB |
| -20 | litenap_savaux_k1 | 1024 | 0.250 | 11 / 32 | 0.343750 | 0 / 11 | 0.0009766 | +0.367 dB |
| -20 | litenap_savaux_k2 | 2048 | 0.500 | 0 / 32 | 0.000000 | 0 / 0 | 1 | +3.566 dB |
| -20 | litenap_single | 256 | 0.062 | 31 / 32 | 0.968750 | 0 / 31 | 9.313e-10 | -3.823 dB |
| -20 | savaux | 4096 | 1.000 | 0 / 32 | 0.000000 | 0 / 0 | 1 | +6.779 dB |

## Interpretation guardrails

- `savaux` uses all `R*N` samples.
- `savaux_phase` uses the same samples and only adds LiteNap phase-jump reranking.
- `litenap_single` uses one `N/D` observation and retains alias ambiguity.
- `litenap_savaux_kK` uses all R oversampling phases for K downsampling views.
- `_phase` variants add the training-free phase-jump timing statistic; no clean waveform template is used.
- Every method shares the same explicit-header modulo-four residual-bin calibration.
- Added SNR is signal-reference power divided by added AWGN power, not the final measured capture SNR.
