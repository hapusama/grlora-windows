# Oracle-sync Savaux hard versus soft-Hamming CRC

Both decoders use identical noisy IQ and clean synchronization.

| Es/N0 | trials | hard PDR | soft-Hamming PDR | fixes | breaks | false CRC |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 80 | 0.350 | 0.787 | 35 | 0 | 0 |

Expected bytes are used only after CRC for exact-delivery audit.
