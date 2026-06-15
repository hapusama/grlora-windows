# Argmax vs Phase-MAP Joint Threshold Table

| SNR | Argmax SER | MAP residual SER | SER reduction | Independent app | Joint app | Joint margin |
|---:|---:|---:|---:|---:|---:|---:|
| -10 dB | 0.0000 | 0.0000 | 0.0000 | 5/5 | 5/5 | 0.511667 |
| -15 dB | 0.0114 | 0.0000 | 0.0114 | 5/5 | 5/5 | 0.414201 |
| -20 dB | 0.4000 | 0.0400 | 0.3600 | 5/5 | 5/5 | 0.315587 |
| -23 dB | 0.7657 | 0.1086 | 0.6571 | 5/5 | 5/5 | 0.265137 |
| -25 dB | 0.9486 | 0.1657 | 0.7829 | 2/5 | 5/5 | 0.222221 |
| -27 dB | 0.9886 | 0.2000 | 0.7886 | 2/5 | 5/5 | 0.145620 |

Notes:
- MAP residual SER is the independent codec-projected residual search result.
- Joint app is the hard affine session trajectory result.
- The target claim should emphasize robust 3 dB+ improvement over ordinary argmax, not the lowest SNR point alone.

## Rough Threshold Estimates

| SER target | Argmax crossing | MAP residual crossing | Estimated gain |
|---:|---:|---:|---:|
| 0.100 | -16.14 dB | -22.62 dB | 6.48 dB |

These are rough linear interpolations over sparse SNR sweep points.
