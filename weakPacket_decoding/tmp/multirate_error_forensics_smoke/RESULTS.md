# Multi-rate error-bin forensic results

This is a statistical exploration, not a decoder evaluation. FrameSync was
estimated once from each clean OTA packet and then frozen for all noisy trials.
One B-wide AWGN realization was added at 1 MS/s; q=4 and q=2 are nested
decimations of that same waveform.

- Clean payload symbols tested: 8
- Clean symbols admitted at all three rates: 8
- Clean exclusions: 0
- FFT coordinate convention: `u=(S-1) mod N`
- Noise convention: `Es/N0 = N * P_signal / P_noise`, after limiting noise to B

| Es/N0 (dB) | q=2 SER | errors | true Top-8 | P(C1M true > wrong) | P(Cmulti true > wrong) |
|---:|---:|---:|---:|---:|---:|
| 6 | 1.0000 | 8 | 0.1250 | 0.1250 | 0.1250 |
| 10 | 0.7500 | 6 | 0.0000 | 0.1667 | 0.1667 |

Diagnostic plots use Es/N0=10 dB, selected as the point
with at least 20 errors whose q=2 SER is closest to 0.3.
