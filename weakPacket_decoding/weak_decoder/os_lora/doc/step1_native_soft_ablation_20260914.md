# Step-1 native-soft ablation: gr-lora_sdr 自带软解码是否吃掉 soft-Hamming 增益

**日期：** 2026-09-14
**回答的问题：** HANDOFF_20260820 §16 "第一刀"——把 gr-lora_sdr 原生
`soft_decoding=True` 链接到相同的 noisy candidate 0 和 ridge K=4 上，判断
bounded soft-Hamming 的增益是否独立于上游已有的软解码。
**答案：不独立。** 原生软解码链在同一 ridge 列表上复现了全部软解码增益；
论文可主张的贡献收敛为 ambiguity-ridge list synchronization +
decoder-in-the-loop CRC arbitration。

## 1. 结论表（560 trial，8 OTA × 10 seeds × 7 Es/N0）

| 臂 | exact/total | PDR50 dB | PDR80 dB |
|---|---:|---:|---:|
| A0 strict（candidate-0 硬解 + strict gate） | 208/560 | 13.714 | 15.182 |
| A1 relaxed 硬解 | 267/560 | 13.000 | 14.235 |
| A2 ridge K=1 硬解（candidate-0 + 精细化） | 277/560 | 12.867 | 14.071 |
| B ridge K=1 upstream soft（5 seeds, n=280） | 151/280 | 12.400 | 13.900 |
| C ridge K=1 current soft | 314/560 | 12.294 | 13.857 |
| D ridge K=4 upstream soft | **371/560** | 11.694 | 12.619 |
| E ridge K=4 current soft | **370/560** | 11.676 | 12.632 |
| F1 硬解 oracle | 327/560 | 12.343 | 13.077 |
| F2 current-soft oracle | 411/560 | 11.324 | 12.067 |
| F3 upstream-soft oracle | 407/560 | 11.394 | 12.211 |

自动生成的完整面板（含逐 SNR PDR）：`doc/step1_ablation_report_20260914.md`。

## 2. 三个决定性证据

1. **D vs E：** 371 对 370；配对比较 current-only 胜 4 次、upstream-only 胜
   5 次。PDR50 相差 0.018 dB。两条软解码链在同一 ridge 列表上统计等价。
2. **B vs C（共享 280 trial）：** current 149 对 upstream 151；配对
   current-only 0、upstream-only 2——candidate-0 上 upstream 甚至略优。
   oracle 完全相同（206=206）。
3. **CRC 错误交付同源：** D 与 E 的唯一 false delivery 都发生在
   `exp0_000005_rxg20_0_fulltrim:11:20260821`，与 20260820 官方 run 的 soft
   collision 是同一 trial。两个解码器的失败模式也一致。

## 3. 增益归因（PDR50 口径）

```text
strict 13.714
  -> gate 放松（decoder-aware）          13.000  (0.71 dB)
  -> candidate-0 精细化                   12.867  (0.13 dB)
  -> 任意软解码（current 或 upstream）    12.294  (0.57 dB，非创新)
  -> ridge K=4 列表 + CRC 仲裁            11.676  (0.62 dB，本文贡献)
软 oracle 天花板                          11.324
```

配对口径：列表在硬解码上 +32 包（0 回退）；在软解码上 +56 包（0 回退）。
错误的同步坐标无法被任何解码器恢复，这 0.62 dB 是结构性的。

## 4. 实现与验证

- 新模块 `weak_decoder/os_lora/system/upstream_soft_crc.py`：忠实移植
  `lib/fft_demod_impl.cc::get_LLRs`（Bessel-I0 max-log，含 N-1 模数怪癖与
  713 溢出回退）、`lib/deinterleaver_impl.cc` 软支路、
  `lib/hamming_dec_impl.cc` 16-LUT 软支路。`log(I0)` 为 numpy 级
  级数/渐近展开，无 scipy 依赖。nibble 约定换算：upstream data nibble 是
  wire nibble 的 4-bit 反转（对应 C++ 输出行的位反转）。
- `decode_soft_hamming_sync_candidate` 新增可注入 `repair_block` 参数；
  `arbitrate_sync_list_with_crc` 新增 `decoder_mode="upstream_soft"`；
  实验入口新增 `--ridge-upstream-soft`（含 upstream oracle 臂）。
- `weak_decoder/os_lora/tests/test_upstream_soft_crc.py`：10 个测试全过，
  含 SF12/CR1-4/LDRO 理想频谱块往返、整帧往返，以及**真实 clean OTA 包
  端到端精确解码**（header/CRC/payload 逐字节）。
- 跨运行确定性审计：三次新 run 的 strict/relaxed 列与 20260820 官方 run
  在全部共享 trial 上逐项一致。

## 5. 局限（写论文时必须声明）

- 所有软解码臂共享同一个 OSR=8 Savaux 匹配滤波频谱前端；这正确隔离了
  "后端度量"这一变量，但不等于字面运行 gr-lora_sdr 的 C++ 流图
  （其前端为自身 frame_sync + OSR=1 FFT）。
- 仍是 8 个物理 capture 上的重复 AWGN trial；HANDOFF §11 Step 5 的扩数据
  （≥100 包、多 session）要求不变。
- B 臂只跑了 5 个 seed（n=280）；与 C 的比较以共享 seed 配对为准。
- upstream 溢出回退在高 SNR 生效时等价于 |Y|^2 线性度量；理想频谱测试
  已覆盖该路径，但真实低 SNR 下两族度量的差异已被 D/E 等价性吸收。

## 6. 对论文主张的影响

按 HANDOFF §12 的判据，"bounded soft-Hamming" 退出贡献列表（当前
soft-Hamming 保留为实现载体与复杂度论证）。存活的主张：

1. premature synchronization rejection 现象量化（A0/A1 与 oracle 的缺口）；
2. ambiguity-ridge list synchronization（+56/0 配对增益，0.62 dB）；
3. decoder-in-the-loop CRC arbitration（decode to finish synchronization）；
4. 复杂度：K=4、平均 <2 次解码尝试/包（见各 run summary 的
   ridge_mean_decoder_attempts 列）。

下一刀（§11 Step 2-4）不变：统一 harness 正面对比 Ameloot / NELoRa 族 /
LZn single-user。

## 7. 复现

```powershell
# K=4 双软臂（本文件主结果）
python -B weak_decoder/os_lora/experiments/evaluate_decoder_aware_crc_pdr_ota.py `
  --max-packets 8 --esn0-db 10,11,12,13,14,15,16 `
  --seeds 20260821,20260822,20260823,20260824,20260825,20260826,20260827,20260828,20260829,20260830 `
  --workers 10 --ridge-top-k 4 --ridge-sfd-peak-pool 32 `
  --ridge-soft-hamming --ridge-upstream-soft `
  --output-dir data/experiments/ambiguity_ridge_upstream_soft_k4_ota_awgn_10seeds_20260914

# K=1 candidate-0 臂
#   data/experiments/ambiguity_ridge_soft_hamming_k1_ota_awgn_10seeds_20260914
# K=1 upstream 臂（5 seeds）
#   data/experiments/ambiguity_ridge_upstream_soft_k1_ota_awgn_20260914

# 汇总面板
python -B scripts/summarize_step1_ablation.py `
  --k4   data/experiments/ambiguity_ridge_soft_hamming_ota_awgn_10seeds_20260820 `
  --k1   data/experiments/ambiguity_ridge_soft_hamming_k1_ota_awgn_10seeds_20260914 `
  --k4-upstream data/experiments/ambiguity_ridge_upstream_soft_k4_ota_awgn_10seeds_20260914 `
  --k1-upstream data/experiments/ambiguity_ridge_upstream_soft_k1_ota_awgn_20260914 `
  --output doc/step1_ablation_report_20260914.md
```
