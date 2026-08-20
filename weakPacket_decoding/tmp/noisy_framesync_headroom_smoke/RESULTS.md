# Noisy FrameSync headroom for Savaux

AWGN is added to the complete 1 MS/s packet before FrameSync. Clean-sync
and noisy-sync Savaux operate on the same noisy IQ. A returned estimate is
reported separately from strict FrameSync validation success.

- Clean packets admitted: 1
- LoRa bin spacing: 30.517578 Hz
- Fatal sensitivity band from the preceding experiment: |CFO error|=0.25--0.75 bin

| Es/N0 | strict sync | estimate available | median |CFO| bin | p90 |CFO| bin | CFO in 0.25--0.75 | oracle SER | noisy-estimate SER | conditional gap | strict E2E SER |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | 0.000 | 1.000 | 653.0126 | 653.0126 | 0.000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
