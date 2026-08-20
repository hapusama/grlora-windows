# Savaux STO/CFO sensitivity on OTA symbols

This experiment perturbs only the synchronization hypothesis presented to
Savaux. Clean FrameSync is frozen before the same B-wide AWGN realization is
reused across every STO/CFO grid point.

- Es/N0: 14 dB
- Clean payload symbols tested/admitted: 4/4
- Trials per grid cell: 4
- Oracle-hypothesis SER at (0,0): 0.0000
- LoRa bin spacing: 30.517578 Hz

## CFO slice at zero STO error

| CFO error (bin) | CFO error (Hz) | SER | mean true margin (dB) |
|---:|---:|---:|---:|
| -0.5 | -15.259 | 0.5000 | -0.069 |
| 0 | 0.000 | 0.0000 | 1.367 |
| 0.5 | 15.259 | 0.5000 | -3.862 |

## STO slice at zero CFO error

| STO error (1M sample) | STO error (chip) | SER | mean true margin (dB) |
|---:|---:|---:|---:|
| -0.5 | -0.0625 | 0.0000 | 2.001 |
| 0 | 0 | 0.0000 | 1.367 |
| 0.5 | 0.0625 | 0.5000 | 0.485 |

## Wide CFO slice

| CFO error (Hz) | CFO error (bin) | SER |
|---:|---:|---:|
| -500 | -16.384 | 1.0000 |
| 0 | 0.000 | 0.0000 |
| 500 | 16.384 | 1.0000 |
