# Multi-rate error-bin forensic results

This is a statistical exploration, not a decoder evaluation. FrameSync was
estimated once from each clean OTA packet and then frozen for all noisy trials.
One B-wide AWGN realization was added at 1 MS/s; q=4 and q=2 are nested
decimations of that same waveform.

- Clean payload symbols tested: 16
- Clean symbols admitted at all three rates: 16
- Clean exclusions: 0
- FFT coordinate convention: `u=(S-1) mod N`
- Noise convention: `Es/N0 = N * P_signal / P_noise`, after limiting noise to B

| Es/N0 (dB) | q=2 SER | errors | true Top-8 | P(C1M true > wrong) | P(Cmulti true > wrong) |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.6875 | 11 | 0.6364 | 0.1818 | 0.1818 |
| 12 | 0.3750 | 6 | 0.6667 | 0.0000 | 0.3333 |
| 14 | 0.1875 | 3 | 1.0000 | 0.0000 | 0.0000 |
| 16 | 0.0000 | 0 | nan | nan | nan |
| 18 | 0.0000 | 0 | nan | nan | nan |
| 20 | 0.0000 | 0 | nan | nan | nan |

Diagnostic plots use Es/N0=12 dB, selected as the point
with at least 20 errors whose q=2 SER is closest to 0.3.
