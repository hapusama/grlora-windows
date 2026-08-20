# Savaux STO/CFO sensitivity on OTA symbols

This experiment perturbs only the synchronization hypothesis presented to
Savaux. Clean FrameSync is frozen before the same B-wide AWGN realization is
reused across every STO/CFO grid point.

- Es/N0: 14 dB
- Clean payload symbols tested/admitted: 64/64
- Trials per grid cell: 128
- Oracle-hypothesis SER at (0,0): 0.0078
- LoRa bin spacing: 30.517578 Hz
- FrameSync-to-Savaux origin conversion: -4 ADC samples

## CFO slice at zero STO error

| CFO error (bin) | CFO error (Hz) | SER | mean true margin (dB) |
|---:|---:|---:|---:|
| -2 | -61.035 | 1.0000 | -15.596 |
| -1 | -30.518 | 1.0000 | -15.202 |
| -0.75 | -22.888 | 1.0000 | -7.208 |
| -0.5 | -15.259 | 0.3750 | 0.440 |
| -0.25 | -7.629 | 0.0078 | 3.629 |
| 0 | 0.000 | 0.0078 | 4.204 |
| 0.25 | 7.629 | 0.0312 | 2.967 |
| 0.5 | 15.259 | 0.7578 | -1.920 |
| 0.75 | 22.888 | 1.0000 | -10.551 |
| 1 | 30.518 | 1.0000 | -16.825 |
| 2 | 61.035 | 1.0000 | -15.510 |

## STO slice at zero CFO error

| STO error (1M sample) | STO error (chip) | SER | mean true margin (dB) |
|---:|---:|---:|---:|
| -4 | -0.5 | 0.8125 | -1.827 |
| -2 | -0.25 | 0.0234 | 2.950 |
| -1 | -0.125 | 0.0000 | 3.811 |
| -0.5 | -0.0625 | 0.0000 | 4.053 |
| 0 | 0 | 0.0078 | 4.204 |
| 0.5 | 0.0625 | 0.0078 | 4.234 |
| 1 | 0.125 | 0.0078 | 4.142 |
| 2 | 0.25 | 0.0078 | 3.619 |
| 4 | 0.5 | 0.3828 | 0.398 |

## Wide CFO slice

| CFO error (Hz) | CFO error (bin) | SER |
|---:|---:|---:|
| -2000 | -65.536 | 1.0000 |
| -1000 | -32.768 | 1.0000 |
| -500 | -16.384 | 1.0000 |
| -250 | -8.192 | 1.0000 |
| -125 | -4.096 | 1.0000 |
| 0 | 0.000 | 0.0078 |
| 125 | 4.096 | 1.0000 |
| 250 | 8.192 | 1.0000 |
| 500 | 16.384 | 1.0000 |
| 1000 | 32.768 | 1.0000 |
| 2000 | 65.536 | 1.0000 |
