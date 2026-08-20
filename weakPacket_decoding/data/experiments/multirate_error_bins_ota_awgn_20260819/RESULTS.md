# Multi-rate error-bin forensic results

This is a statistical exploration, not a decoder evaluation. FrameSync was
estimated once from each clean OTA packet and then frozen for all noisy trials.
One B-wide AWGN realization was added at 1 MS/s; q=4 and q=2 are nested
decimations of that same waveform.

- Clean payload symbols tested: 256
- Clean symbols admitted at all three rates: 249
- Clean exclusions: 7
- FFT coordinate convention: `u=(S-1) mod N`
- Noise convention: `Es/N0 = N * P_signal / P_noise`, after limiting noise to B

| Es/N0 (dB) | q=2 SER | errors | true Top-8 | P(C1M true > wrong) | P(Cmulti true > wrong) |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.8233 | 615 | 0.2748 | 0.3203 | 0.3203 |
| 11 | 0.7229 | 540 | 0.3759 | 0.3278 | 0.3315 |
| 12 | 0.5944 | 444 | 0.5068 | 0.3221 | 0.3288 |
| 13 | 0.4632 | 346 | 0.6879 | 0.3353 | 0.3324 |
| 14 | 0.3333 | 249 | 0.8032 | 0.3414 | 0.3414 |
| 15 | 0.2463 | 184 | 0.9511 | 0.3261 | 0.3043 |

Diagnostic plots use Es/N0=14 dB, selected as the point
with at least 20 errors whose q=2 SER is closest to 0.3.
