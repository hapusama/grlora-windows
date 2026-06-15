# Extra-Capture Phase-MAP Validation

| Capture | SNR | Argmax SER | MAP residual SER | SER reduction | Independent app | Joint app | Joint rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0_0_0_10_14_32 | -20 dB | 0.4686 | 0.0914 | 0.3772 | 3/5 | 5/5 | 16.400 |
| 0_0_0_10_14_32 | -23 dB | 0.8000 | 0.1943 | 0.6057 | 3/5 | 3/5 | 30.400 |
| 0_0_0_10_14_8 | -20 dB | 0.2000 | 0.0286 | 0.1714 | 5/5 | 5/5 | 1.000 |
| 0_0_0_10_14_8 | -23 dB | 0.6914 | 0.0800 | 0.6114 | 5/5 | 5/5 | 1.000 |

Notes:
- These are sanity/generalization checks on additional USRP_IQ captures.
- They reuse the same Phase-MAP residual candidate scoring and hard affine joint layer.
- The preamble-32 -23 dB point is intentionally kept as a boundary case: symbol SER improves, but app-level joint recovery is not perfect.
