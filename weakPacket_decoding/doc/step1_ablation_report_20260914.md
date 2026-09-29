# Step-1 ablation panel (auto-generated)

## Cross-run determinism audit (shared seeds must agree)

- all shared trials agree

## Per-arm PDR by Es/N0

| arm | n | 10dB | 11dB | 12dB | 13dB | 14dB | 15dB | 16dB |
|---|---|---|---|---|---|---|---|---|
| A0 strict (cand-0 hard, strict gates) | 560 | 0.0% | 0.0% | 2.5% | 31.2% | 57.5% | 77.5% | 91.2% |
| A1 relaxed hard (cand-0) | 560 | 0.0% | 0.0% | 13.8% | 50.0% | 75.0% | 96.2% | 98.8% |
| A2 ridge K=1 hard (cand-0 + refine) | 560 | 0.0% | 0.0% | 17.5% | 55.0% | 78.8% | 96.2% | 98.8% |
| B  ridge K=1 upstream soft (cand-0) | 280 | 0.0% | 2.5% | 45.0% | 57.5% | 82.5% | 92.5% | 97.5% |
| C  ridge K=1 current soft (cand-0) | 560 | 0.0% | 6.2% | 43.8% | 65.0% | 82.5% | 96.2% | 98.8% |
| D  ridge K=4 upstream soft | 560 | 0.0% | 18.8% | 63.7% | 90.0% | 93.8% | 98.8% | 98.8% |
| E  ridge K=4 current soft | 560 | 0.0% | 18.8% | 65.0% | 88.8% | 92.5% | 98.8% | 98.8% |
| F1 hard oracle | 560 | 0.0% | 0.0% | 35.0% | 78.8% | 95.0% | 100.0% | 100.0% |
| F2 current-soft oracle | 560 | 1.2% | 36.2% | 78.8% | 97.5% | 100.0% | 100.0% | 100.0% |
| F3 upstream-soft oracle | 560 | 1.2% | 33.8% | 75.0% | 98.8% | 100.0% | 100.0% | 100.0% |

## Exact-delivery totals and thresholds

| arm | exact/total | PDR50 dB | PDR80 dB |
|---|---|---|---|
| A0 strict (cand-0 hard, strict gates) | 208/560 | 13.714 | 15.182 |
| A1 relaxed hard (cand-0) | 267/560 | 13.000 | 14.235 |
| A2 ridge K=1 hard (cand-0 + refine) | 277/560 | 12.867 | 14.071 |
| B  ridge K=1 upstream soft (cand-0) | 151/280 | 12.400 | 13.900 |
| C  ridge K=1 current soft (cand-0) | 314/560 | 12.294 | 13.857 |
| D  ridge K=4 upstream soft | 371/560 | 11.694 | 12.619 |
| E  ridge K=4 current soft | 370/560 | 11.676 | 12.632 |
| F1 hard oracle | 327/560 | 12.343 | 13.077 |
| F2 current-soft oracle | 411/560 | 11.324 | 12.067 |
| F3 upstream-soft oracle | 407/560 | 11.394 | 12.211 |

## False deliveries (CRC-valid wrong packets)

- K=1 current soft: 1
- K=4 upstream soft: 1
- K=1 upstream soft: 1
