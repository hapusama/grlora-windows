# phase_line

这里保存相位线构建与可靠性验证实验输出。

当前主要子目录：

```text
preamble_only/
```

其中 `preamble_only/<capture>/` 保存 `scripts/experiments/phase_line/run_preamble_phase_line_experiment.py`
的输出：

```text
*_preamble_phase_anchor_features.csv       每个 preamble anchor 的 FFT 相位/幅度/ER
*_preamble_phase_payload_validation.csv    preamble-only phase line 外推到 payload GT bin 后的 residual
*_preamble_phase_line_summary.csv          每个 packet 的 phase-line 拟合和可靠性摘要
plots/packet_xxx_event_xxx_preamble_phase_line.png
```

该实验不做 FFT bin 重选，只用于判断可靠 anchor 是否能构造 packet-level phase line，以及这根线是否能迁移到 payload。

