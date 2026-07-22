# Phase-compensated NUS ordinary-FFT audit

Pattern bank: `canonical_only` (4 patterns).
The compensation bin is obtained from the plain NUS FFT bank; payload GT is scoring-only.
`mean_gt_to_mean_energy` is GT power divided by the mean candidate power, not an H0-normalized Lambda.

| added SNR | method | decisions | errors | fixes / breaks | GT/mean energy | GT margin | GT/floor |
|---:|---|---:|---:|---:|---:|---:|---:|
| clean | savaux | 8 | 0 | 0 / 0 | 120.9210 | +12.7301 dB | 22.9320 dB |
| clean | plain_nus_mean | 8 | 1 | 0 / 1 | 20.7944 | +4.0453 dB | 12.5297 dB |
| clean | phase_fft_1 | 8 | 0 | 0 / 0 | 115.4960 | +11.9714 dB | 22.6728 dB |
| clean | phase_fft_2 | 8 | 0 | 0 / 0 | 126.2650 | +12.8135 dB | 23.1508 dB |
| clean | exact_nus_coherent | 8 | 0 | 0 / 0 | 120.9210 | +12.7301 dB | 22.9320 dB |
