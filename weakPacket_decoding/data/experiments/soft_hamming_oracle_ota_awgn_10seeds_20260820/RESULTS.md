# Oracle-sync Savaux hard versus soft-Hamming CRC

Both decoders use identical noisy IQ and clean synchronization.

| Es/N0 | trials | hard PDR | soft-Hamming PDR | fixes | breaks | false CRC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 80 | 0.000 | 0.013 | 1 | 0 | 1 |
| 11 | 80 | 0.000 | 0.362 | 29 | 0 | 0 |
| 12 | 80 | 0.350 | 0.787 | 35 | 0 | 0 |
| 13 | 80 | 0.787 | 0.975 | 15 | 0 | 0 |
| 14 | 80 | 0.950 | 1.000 | 4 | 0 | 0 |
| 15 | 80 | 1.000 | 1.000 | 0 | 0 | 0 |
| 16 | 80 | 1.000 | 1.000 | 0 | 0 | 0 |

Expected bytes are used only after CRC for exact-delivery audit.
