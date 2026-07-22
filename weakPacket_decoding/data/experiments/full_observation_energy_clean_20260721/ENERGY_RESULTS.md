# Full-observation candidate-energy audit

Savaux 和 full colored-ML 使用同一个完整 OSR symbol 与全部候选模板。
Savaux 能量按 `s^H R_v s` 标定，colored-ML 按 `s^H R_v^-1 s` 标定，
因此两者在各自的噪声输出上都是 H0 均值为 1 的无量纲 Lambda。
Savaux hard decision 仍使用原始匹配能量，不使用该标定量重新判决。

| dataset | condition | symbols | Savaux GT Lambda | colored GT Lambda | Savaux margin | colored margin | margin delta | model SNR gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| collector_low4_train_low1 | clean | 392 | 74.3573 | 77.482 | 9.802150 dB | 9.889008 dB | +0.086858 dB | +0.166323 dB |

- GT Lambda increased/decreased: 341/51.
- GT-vs-false margin improved/worsened: 239/153; symbol-level sign p=1.63747e-05.
- Packet mean margin improved/tied/worsened: 8/0/0.
- Model GT-bin SNR gain range: 0.137231 to 0.169236 dB.
- Hard decisions: Savaux errors=1, colored-ML errors=1, fixes/breaks=0/0.

The full per-symbol, per-candidate arrays are stored in `candidate_scores.npz`;
`candidate_summary.csv` contains a 1024-bin aggregate view.
