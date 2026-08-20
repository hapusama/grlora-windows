# Noisy FrameSync headroom for Savaux

AWGN is added to the complete 1 MS/s packet before FrameSync. Clean-sync
and noisy-sync Savaux operate on the same noisy IQ. A returned estimate is
reported separately from strict FrameSync validation success.

- Clean packets admitted: 8
- LoRa bin spacing: 30.517578 Hz
- Fatal sensitivity band from the preceding experiment: |CFO error|=0.25--0.75 bin
- Coordinate-local definition: |CFO error|<=2 bin and |STO error|<=32 samples
- Coupled residual proxy: CFO error - STO error / OSR (LoRa timing/frequency ambiguity)

| Es/N0 | strict sync | estimate | coord-local | strict / local | raw local median |CFO| | raw CFO in 0.25--0.75 | coupled median | coupled in 0.25--0.75 | oracle SER | noisy SER (local) | local gap | noisy SER (all) | strict E2E SER |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.938 | 1.000 | 1.000 | 0.938 | 0.0026 | 0.000 | 0.0035 | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0625 |
| 15 | 0.750 | 1.000 | 0.938 | 0.733 | 0.0027 | 0.000 | 0.0042 | 0.062 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2500 |
| 14 | 0.625 | 1.000 | 0.750 | 0.667 | 0.0046 | 0.000 | 0.0065 | 0.125 | 0.0000 | 0.0000 | 0.0000 | 0.1875 | 0.4375 |
| 13 | 0.438 | 1.000 | 0.750 | 0.500 | 0.0055 | 0.000 | 0.0089 | 0.062 | 0.0234 | 0.0156 | 0.0000 | 0.2070 | 0.5742 |
| 12 | 0.125 | 0.938 | 0.625 | 0.200 | 0.0061 | 0.000 | 0.1155 | 0.133 | 0.1172 | 0.1062 | 0.0062 | 0.4000 | 0.8984 |
