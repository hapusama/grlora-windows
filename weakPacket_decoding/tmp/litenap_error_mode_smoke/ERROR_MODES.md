# LiteNap-Savaux error-mode decomposition

`wrong_group` means the aliased bin is correct modulo `N/D`, but the full-frequency alias group is wrong.
`wrong_alias_bin` means even the modulo-`N/D` aliased bin is wrong. The two modes partition all hard-decision errors.

| added SNR | method | SER | wrong group | wrong alias bin | group share | alias-bin share |
|---:|---|---:|---:|---:|---:|---:|
| -20 | litenap_savaux_k1 | 0.343750 | 1 | 10 | 0.091 | 0.909 |
| -20 | litenap_savaux_k2 | 0.000000 | 0 | 0 | 0.000 | 0.000 |
| -21 | litenap_savaux_k1 | 0.562500 | 2 | 16 | 0.111 | 0.889 |
| -21 | litenap_savaux_k2 | 0.031250 | 0 | 1 | 0.000 | 1.000 |
