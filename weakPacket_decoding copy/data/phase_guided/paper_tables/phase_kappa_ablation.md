# Phase Kappa Ablation

| SNR | k=0 app | k=0.5 app | k=1 app | k=2 app |
|---:|---:|---:|---:|---:|
| -10 dB | 5/5 | 5/5 | 5/5 | 5/5 |
| -15 dB | 5/5 | 5/5 | 5/5 | 5/5 |
| -20 dB | 5/5 | 5/5 | 5/5 | 5/5 |
| -23 dB | 5/5 | 5/5 | 5/5 | 5/5 |
| -25 dB | 1/5 | 5/5 | 5/5 | 5/5 |
| -27 dB | 1/5 | 1/5 | 5/5 | 5/5 |

Margins:

| SNR | k=0 margin | k=0.5 margin | k=1 margin | k=2 margin |
|---:|---:|---:|---:|---:|
| -10 dB | 0.401172 | 0.428796 | 0.456420 | 0.511667 |
| -15 dB | 0.302475 | 0.330407 | 0.358338 | 0.414201 |
| -20 dB | 0.202684 | 0.230910 | 0.259136 | 0.315587 |
| -23 dB | 0.146581 | 0.176220 | 0.205859 | 0.265137 |
| -25 dB | 0.034117 | 0.062188 | 0.158493 | 0.222221 |
| -27 dB | 0.079958 | 0.055442 | 0.011579 | 0.145620 |

Notes:
- k=0 removes the phase likelihood term.
- App exactness is joint affine session decoding.
- The deeper weak points show when phase becomes necessary rather than only confidence-improving.
