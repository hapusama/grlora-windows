# Noisy FrameSync headroom for Savaux

AWGN is added to the complete 1 MS/s packet before FrameSync. Clean-sync
and noisy-sync Savaux operate on the same noisy IQ. A returned estimate is
reported separately from strict FrameSync validation success.

- Clean packets admitted: 2
- LoRa bin spacing: 30.517578 Hz
- Fatal sensitivity band from the preceding experiment: |CFO error|=0.25--0.75 bin
- Local-lock definition: |CFO error|<=2 bin and |STO error|<=32 samples

| Es/N0 | strict sync | estimate available | local estimate | catastrophic / estimate | local median |CFO| | local CFO in 0.25--0.75 | oracle SER | noisy SER (local) | local gap | noisy SER (all estimates) | strict E2E SER |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | 0.000 | 1.000 | 1.000 | 0.000 | 0.4986 | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
