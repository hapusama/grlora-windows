# weak_preamble_detections 数据目录说明

这个目录保存弱前导码检测阶段的输出，主要由：

```text
scripts/detect_weak_preamble.py
scripts/run_weak_sync_chain.py
```

生成。

## 文件类型

```text
*_events.csv   检测事件表，一行对应一个 weak preamble candidate event
*_windows.csv  滑动窗口级调试表，一行对应一个检测窗口
```

## `*_events.csv`

事件表用于把弱前导码检测结果传给后续 `frame_locator` / `run_weak_sync_chain.py`。常见字段含义：

```text
event_index          检测事件编号
start_sample         事件粗起点
end_sample           事件粗终点
reference_bin        检测阶段的参考 peak bin
window_count         该事件聚合了多少个滑动窗口
```

这里的 event 只是“疑似 LoRa preamble”的高召回候选，不代表已经完成 sync word / SFD / framesync 验证。

## `*_windows.csv`

窗口表用于调试弱检测本身，记录每个滑动窗口的 FFT peak、能量、稳定性等信息。它通常比事件表大很多，适合分析：

```text
连续窗口 peak bin 是否稳定
候选事件是如何被合并出来的
窄带干扰或稳定杂散是否触发了检测
```

## 注意事项

弱检测阶段使用过采样 FFT：

```text
fft_len = 2^SF * os_factor
```

因此这里出现的 bin 字段属于 oversampled bin 坐标，不能直接和 header-first demod 中的 `raw_fft_bin` 混用。后者是 chip-rate FFT：

```text
fft_len = 2^SF
```
