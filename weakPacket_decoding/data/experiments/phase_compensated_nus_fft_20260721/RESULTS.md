# Phase-compensated NUS ordinary-FFT audit

Pattern bank: `canonical_only` (4 patterns).
The compensation bin is obtained from the plain NUS FFT bank; payload GT is scoring-only.
`mean_gt_to_mean_energy` is GT power divided by the mean candidate power, not an H0-normalized Lambda.

| added SNR | method | decisions | errors | fixes / breaks | GT/mean energy | GT margin | GT/floor |
|---:|---|---:|---:|---:|---:|---:|---:|
| clean | savaux | 392 | 1 | 0 / 0 | 74.0855 | +9.8050 dB | 20.1781 dB |
| clean | plain_nus_mean | 392 | 129 | 0 / 128 | 11.7818 | +2.2196 dB | 9.6082 dB |
| clean | phase_fft_1 | 392 | 10 | 0 / 9 | 66.4273 | +8.5280 dB | 19.1984 dB |
| clean | phase_fft_2 | 392 | 6 | 0 / 5 | 77.6822 | +9.8805 dB | 20.2457 dB |
| clean | exact_nus_coherent | 392 | 1 | 0 / 0 | 74.0855 | +9.8050 dB | 20.1781 dB |
| -3 | savaux | 7840 | 128 | 0 / 0 | 26.3399 | +5.2745 dB | 15.5647 dB |
| -3 | plain_nus_mean | 7840 | 3962 | 4 / 3838 | 4.0777 | -0.4379 dB | 5.4536 dB |
| -3 | phase_fft_1 | 7840 | 1641 | 13 / 1526 | 20.4366 | +2.8304 dB | 13.1868 dB |
| -3 | phase_fft_2 | 7840 | 811 | 22 / 705 | 25.0300 | +4.4087 dB | 14.7182 dB |
| -3 | exact_nus_coherent | 7840 | 128 | 0 / 0 | 26.3399 | +5.2745 dB | 15.5647 dB |
| -6 | savaux | 7840 | 614 | 0 / 0 | 16.5303 | +3.1401 dB | 13.4355 dB |
| -6 | plain_nus_mean | 7840 | 5413 | 12 / 4811 | 2.8137 | -1.7623 dB | 4.0047 dB |
| -6 | phase_fft_1 | 7840 | 3251 | 44 / 2681 | 11.3397 | -0.1096 dB | 10.2133 dB |
| -6 | phase_fft_2 | 7840 | 2214 | 50 / 1650 | 13.9677 | +1.1427 dB | 11.4535 dB |
| -6 | exact_nus_coherent | 7840 | 614 | 0 / 0 | 16.5303 | +3.1401 dB | 13.4355 dB |
| -9 | savaux | 7840 | 2763 | 0 / 0 | 9.7919 | +0.6628 dB | 10.9501 dB |
| -9 | plain_nus_mean | 7840 | 6845 | 53 / 4135 | 2.0103 | -3.0648 dB | 2.6726 dB |
| -9 | phase_fft_1 | 7840 | 5493 | 117 / 2847 | 5.8043 | -3.3536 dB | 6.9439 dB |
| -9 | phase_fft_2 | 7840 | 4663 | 169 / 2069 | 7.0639 | -2.3957 dB | 7.9081 dB |
| -9 | exact_nus_coherent | 7840 | 2763 | 0 / 0 | 9.7919 | +0.6628 dB | 10.9501 dB |
