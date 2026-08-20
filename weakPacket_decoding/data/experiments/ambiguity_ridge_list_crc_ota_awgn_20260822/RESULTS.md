# CRC-aided ambiguity-ridge list synchronization

SFD Top-L integer-CFO peaks are mapped to coupled payload boundaries; only Top-K candidates enter the unchanged Savaux/FEC/CRC decoder.

| Es/N0 | strict | soft gate | list K=1 | list K=2 | list K=4 | oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.062 | 0.250 | 0.312 | 0.438 | 0.500 | 0.500 |
| 13 | 0.375 | 0.625 | 0.688 | 0.750 | 0.812 | 0.875 |
| 14 | 0.500 | 0.750 | 0.750 | 0.875 | 0.938 | 1.000 |
| 15 | 0.688 | 0.938 | 0.938 | 1.000 | 1.000 | 1.000 |
| 16 | 0.938 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Across 80 trials: strict=41, soft gate=57, oracle=70 exact CRC-valid packets.
K=1 delivers 59/80, closes 2/13 (15.4%) of the soft-to-oracle gap, and needs 0.988 decoder attempts per input trial with early stopping.
K=2 delivers 65/80, closes 8/13 (61.5%) of the soft-to-oracle gap, and needs 1.238 decoder attempts per input trial with early stopping.
K=4 delivers 68/80, closes 11/13 (84.6%) of the soft-to-oracle gap, and needs 1.562 decoder attempts per input trial with early stopping.

## Arbitration audit

- K=1: 2 rescues over soft gate, 0 regressions, 0 CRC false deliveries.
- K=2: 8 rescues over soft gate, 0 regressions, 0 CRC false deliveries.
- K=4: 11 rescues over soft gate, 0 regressions, 0 CRC false deliveries.
- Reproduction mismatches versus baseline: CFO=0, payload start=0.
- Full-list audit: 95 CRC-valid candidate decodes, 0 non-exact deliveries.

Ground-truth bytes are used only after CRC to score exact delivery and detect CRC collisions.
