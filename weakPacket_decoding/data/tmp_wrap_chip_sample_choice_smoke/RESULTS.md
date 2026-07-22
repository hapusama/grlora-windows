# LoRa wrap-chip sample-choice audit

All chips use fixed q=0 except one candidate-specific chip near p_wrap=N-k.
The changed sample is phase-translated back to q=0 before candidate scoring.
GT is used only for energy and head/tail diagnostics after every candidate score is formed.

| SNR | delta | q | fixed errors | choice errors | Savaux errors | fixes/breaks | GT energy gain | head gain | tail gain | coherence delta | phase-error reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | -1 | 0 | 0 | 0 | 0 | 0/0 | +0.000000 dB | +0.000000 dB | +0.000000 dB | +0.000e+00 | +0.000000 deg |
| clean | -1 | 1 | 0 | 0 | 0 | 0/0 | +0.013188 dB | +0.189665 dB | +0.000000 dB | +1.069e-03 | +1.245051 deg |
| clean | -1 | 2 | 0 | 0 | 0 | 0/0 | +0.007148 dB | +0.205428 dB | +0.000000 dB | +9.105e-04 | +1.043378 deg |
| clean | -1 | 3 | 0 | 0 | 0 | 0/0 | +0.010740 dB | +0.208112 dB | +0.000000 dB | +2.884e-04 | +0.471926 deg |
| clean | 0 | 0 | 0 | 0 | 0 | 0/0 | +0.000000 dB | +0.000000 dB | +0.000000 dB | +0.000e+00 | +0.000000 deg |
| clean | 0 | 1 | 0 | 0 | 0 | 0/0 | -0.003028 dB | +0.000000 dB | -0.000655 dB | +8.619e-05 | +0.325348 deg |
| clean | 0 | 2 | 0 | 0 | 0 | 0/0 | -0.019549 dB | +0.000000 dB | -0.011597 dB | -2.147e-04 | -0.182549 deg |
| clean | 0 | 3 | 0 | 0 | 0 | 0/0 | -0.015279 dB | +0.000000 dB | -0.029267 dB | -2.750e-04 | -0.237976 deg |
| clean | 1 | 0 | 0 | 0 | 0 | 0/0 | +0.000000 dB | +0.000000 dB | +0.000000 dB | +0.000e+00 | +0.000000 deg |
| clean | 1 | 1 | 0 | 0 | 0 | 0/0 | +0.009649 dB | +0.000000 dB | +0.006473 dB | -4.274e-06 | -0.150107 deg |
| clean | 1 | 2 | 0 | 0 | 0 | 0/0 | +0.007209 dB | +0.000000 dB | +0.025333 dB | -1.261e-04 | -0.166002 deg |
| clean | 1 | 3 | 0 | 0 | 0 | 0/0 | +0.010745 dB | +0.000000 dB | +0.024795 dB | +1.113e-05 | -0.149403 deg |
