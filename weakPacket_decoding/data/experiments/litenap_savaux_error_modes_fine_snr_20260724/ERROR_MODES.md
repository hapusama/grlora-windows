# LiteNap-Savaux error-mode decomposition

`wrong_group` means the aliased bin is correct modulo `N/D`, but the full-frequency alias group is wrong.
`wrong_alias_bin` means even the modulo-`N/D` aliased bin is wrong. The two modes partition all hard-decision errors.

| added SNR | method | SER | oracle group floor | wrong group | wrong alias bin | group share | group error given alias | far alias share |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| -16 | litenap_savaux_k1 | 0.026411 | 0.019608 | 17 | 49 | 0.258 | 0.007 | 0.837 |
| -16 | litenap_savaux_k2 | 0.000000 | 0.000000 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| -17 | litenap_savaux_k1 | 0.069628 | 0.056423 | 33 | 141 | 0.190 | 0.014 | 0.844 |
| -17 | litenap_savaux_k2 | 0.001200 | 0.001200 | 0 | 3 | 0.000 | 0.000 | 1.000 |
| -18 | litenap_savaux_k1 | 0.133653 | 0.110044 | 59 | 275 | 0.177 | 0.027 | 0.880 |
| -18 | litenap_savaux_k2 | 0.005602 | 0.004802 | 2 | 12 | 0.143 | 0.001 | 0.750 |
| -19 | litenap_savaux_k1 | 0.234894 | 0.207683 | 68 | 519 | 0.116 | 0.034 | 0.904 |
| -19 | litenap_savaux_k2 | 0.016807 | 0.016807 | 0 | 42 | 0.000 | 0.000 | 1.000 |
| -20 | litenap_savaux_k1 | 0.367347 | 0.330132 | 93 | 825 | 0.101 | 0.056 | 0.888 |
| -20 | litenap_savaux_k2 | 0.055222 | 0.052021 | 8 | 130 | 0.058 | 0.003 | 0.923 |
| -21 | litenap_savaux_k1 | 0.503001 | 0.466186 | 92 | 1165 | 0.073 | 0.069 | 0.910 |
| -21 | litenap_savaux_k2 | 0.121649 | 0.115246 | 16 | 288 | 0.053 | 0.007 | 0.920 |
| -22 | litenap_savaux_k1 | 0.627051 | 0.587835 | 98 | 1469 | 0.063 | 0.095 | 0.914 |
| -22 | litenap_savaux_k2 | 0.209684 | 0.205282 | 11 | 513 | 0.021 | 0.006 | 0.918 |
| -23 | litenap_savaux_k1 | 0.751901 | 0.720288 | 79 | 1800 | 0.042 | 0.113 | 0.904 |
| -23 | litenap_savaux_k2 | 0.343337 | 0.333333 | 25 | 833 | 0.029 | 0.015 | 0.936 |
| -24 | litenap_savaux_k1 | 0.817127 | 0.792317 | 62 | 1980 | 0.030 | 0.119 | 0.928 |
| -24 | litenap_savaux_k2 | 0.464186 | 0.455382 | 22 | 1138 | 0.019 | 0.016 | 0.924 |
| -25 | litenap_savaux_k1 | 0.892357 | 0.871949 | 51 | 2179 | 0.023 | 0.159 | 0.923 |
| -25 | litenap_savaux_k2 | 0.611044 | 0.601841 | 23 | 1504 | 0.015 | 0.023 | 0.930 |
| -26 | litenap_savaux_k1 | 0.922769 | 0.903962 | 47 | 2259 | 0.020 | 0.196 | 0.923 |
| -26 | litenap_savaux_k2 | 0.721088 | 0.709484 | 29 | 1773 | 0.016 | 0.040 | 0.924 |
| -27 | litenap_savaux_k1 | 0.956783 | 0.941176 | 39 | 2352 | 0.016 | 0.265 | 0.921 |
| -27 | litenap_savaux_k2 | 0.811124 | 0.799120 | 30 | 1997 | 0.015 | 0.060 | 0.934 |
| -28 | litenap_savaux_k1 | 0.957583 | 0.945978 | 29 | 2364 | 0.012 | 0.215 | 0.933 |
| -28 | litenap_savaux_k2 | 0.887555 | 0.873950 | 34 | 2184 | 0.015 | 0.108 | 0.942 |
