# CRC-aided ambiguity-ridge list synchronization

SFD Top-L integer-CFO peaks are mapped to coupled payload boundaries; only Top-K candidates enter the unchanged Savaux/FEC/CRC decoder.

| Es/N0 | strict | soft gate | list K=1 | list K=2 | list K=4 | list K=8 | list K=16 | list K=32 | oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| 14 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

## Arbitration audit

- K=1: 0 rescues over soft gate, 0 regressions, 0 CRC false deliveries.
- K=2: 0 rescues over soft gate, 0 regressions, 0 CRC false deliveries.
- K=4: 0 rescues over soft gate, 0 regressions, 0 CRC false deliveries.
- K=8: 0 rescues over soft gate, 0 regressions, 0 CRC false deliveries.
- K=16: 1 rescues over soft gate, 0 regressions, 0 CRC false deliveries.
- K=32: 1 rescues over soft gate, 0 regressions, 0 CRC false deliveries.
- Reproduction mismatches versus baseline: CFO=0, payload start=0.

Ground-truth bytes are used only after CRC to score exact delivery and detect CRC collisions.
