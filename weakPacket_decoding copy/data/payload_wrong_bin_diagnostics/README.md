# payload_wrong_bin_diagnostics 数据目录说明

这个目录保存 clean/high-SNR IQ 下的 corrected wrong-bin 对照实验输出，主要由：

```text
scripts/experiments/plot_wrong_bin_phase_diagnostics.py
```

生成。

## 实验目的

该实验用于回答：

```text
corrected FFT demod 后 observed phase/amplitude smoothness
是只存在于 selected/正确 bin，
还是错误 bin 上也同样存在？
```

脚本会按 header-first symbol CSV 中的同步/补偿参数重算每个 payload symbol 的完整 corrected FFT，然后同时读取：

```text
selected bin
selected + offset 的 wrong bin
固定 raw FFT bin
```

并比较相位 unwrap、线性拟合、幅度、能量占比和 rank。

## 文件类型

```text
*_wrong_bin_symbol_features.csv        symbol/candidate 粒度特征表
*_wrong_bin_summary.csv                packet/candidate 粒度统计表
plots/packet_xxx_wrong_bin_phase_amplitude_compare.png
                                       selected vs wrong-bin 四联图
```

## 常见 candidate

```text
selected   当前 corrected FFT argmax bin
off+1      selected + 1
off-1      selected - 1
off+16     selected + 16
fix0000    固定 raw bin 0
fix0256    固定 raw bin 256
fix0512    固定 raw bin 512
fix0768    固定 raw bin 768
```

## 读图要点

如果 wrong bin 的 unwrap phase 也能给出高 R2，但幅度、energy ratio、rank 明显差于 selected，则说明相位平滑性本身需要和能量/排名联合使用。

如果 wrong bin 的 residual 曲线也和 selected 一样稳定，则相位 residual 不具备单独判别力。当前实验中，多数 wrong bin 的能量占比和 rank 明显较差，residual 结构也不如 selected 稳定。
