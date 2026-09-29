# ota_replay_20260929 — RESULTS

> **v6（2026-09-29 下午）判据修复后，OTA 定量对比恢复有效。**
> v1–v5 全部作废（根因见下，已在 v6 修复并验证）。数据本身（USRP_IQ 三个
> capture，谱峰比 ~37 dB ≈ 7 dB 带内 SNR）没有问题。

## 一、v5 作废的真正根因（2026-09-29 审计实锤，勘误原诊断）

v5 的结论是"CRC 空真（自洽环）→ 判据不可用 → 退回合成测试床"。
审计（`data/smoke_tmp/audit_*.py`）推翻了该诊断的机制描述：

1. **CRC 判据本身有效**：随机谱 200 次 0 过、单符号损坏 30/30 被拒。
   "自洽环"的说法不成立——对解码字节重算 CRC 与解码出的 CRC 字节比对
   是标准独立校验。
2. **真因 R1**：`payload_codec.decode_explicit_frame_symbols` 把
   `crc_valid` 初始化为 `not header.has_crc`，且从不检查
   `header_valid`（PHY header checksum）。header 被噪声打坏、has_crc
   位解成 0 时，crc_valid 无条件为 True——payload 完全未被校验。
   实测 +60 dB 噪声下全链"通过"即此旁路。
3. **真因 R2**：8 条链共享同一份 header 解调，且 wm 的 header argmax
   在 OTA 上从未对过——真实 header 符号只带 SF−2 比特，wm argmax =
   **4×真值**（v5 直接 `%1024` 当值用，错得离谱）。
4. **真因 R3**：因此 v5 的 native "28/28 全过"也全是旁路通过，
   "native 共识 GT"从未真正建立；8 链同数与深端非单调
   （+30dB 9/28 → +36dB 18/28）均为 has_crc 比特随机翻转所致。
5. **结论修订**：OTA 定量对比**可救**，无需退回"只能合成"。判据修复 +
   GT 重建后重跑即 v6。

## 二、v6 修复内容（ota_v6_fixed.py）

| 项 | 修复 |
|---|---|
| 判据 | `payload_codec`：crc_valid 以 header_valid 为前提，has_crc=0 旁路封堵（回归测试 4 例全过） |
| GT | `demod_symbol_sequence`（CSV 生成器，OTA 验证过）原生段解出 header/payload 符号，CRC 必须通过，字节冻结为 GT |
| 受控对比 | 所有链共用 GT header（同合成测试床"真值 header"约定），唯一变量 = payload 证据链 |
| 约定 | 每链探针校准 δ = argmax−真值（多数投票）；证据 roll(−1+δ) 喂 SymFEC；payload 上 wm argmax = 真值已实证（8/8 符号精确） |
| 回绕 | OTA 实测整 bin 回绕（部分帧 κ_eff>0.5，tone 在 v+1；合成床扫 κ∈[−0.5,0.5] 未覆盖）→ 多 δ∈{−1,0,+1} 候选 + CRC 仲裁（同 bcjr_marginal 的镜像仲裁原语） |
| 门禁 | 合成正控（cr=1/4）+ 负控 A（整交织块损坏必须全拒）+ 负控 B（坏 header 必须全拒，R1 回归护栏），不过不碰真实数据 |

## 三、v6 战表（28 帧 ×3 capture，SF10/CR4=5/33B/8×preamble 变体）

| 噪声 | PLAIN | OLD-A | OLD-A+F | SAVAUX | TRIMMER | UNICHIRP | NEW-0 | BCJR |
|---|---|---|---|---|---|---|---|---|
| native | 3 | 5 | 5 | 5 | 22 | 9 | 22 | **23** |
| +18dB | 0 | 0 | 5 | 4 | **22** | 0 | **22** | 21 |
| +21dB | 0 | 0 | 1 | 1 | **22** | 1 | **22** | **22** |
| +24dB | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| ≥+27dB | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

与 v5 的"8 链同数 + 非单调"对照：v6 各链真实分化、单调、物理合理
（这些帧为 CR4/5 弱 FEC + 33B 长包，悬崖在 −14~−17 dB 有效 SNR 处，
比 bcjr_marginal 的 CR4/8+1B 场景早 ~7 dB，符合编码与包长预期）。

## 四、如实判读与遗留

1. **判据与 harness 已修复**：负控双门禁保证无空真回归；GT 字节判据
   可审计（GT 经原生 CRC 校验冻结）。
2. **native 残余失败反映方法覆盖极限，非 harness bug**：5-6 帧需要
   δ=±1 与非中心列**同时**成立（整 bin 回绕 × 分数列的联合），5 列
   κ 格（±0.5 bin）不覆盖——这是方法侧的自然升级点：
   **跨 bin 15 态格（δ∈{−1,0,1}×5 列）**，OTA 数据直接提供了证据。
3. OLD-A/PLAIN native 仅 3-5/28：裸抽取在真实分数偏移
   （cfo_frac+sto_frac ≈ ±0.5）下的已知弱点再次独立确认
   （κ 分裂，与 battle 结论一致）。
4. UniChirp native 9/28：OTA 段内只有 4 个干净 preamble 符号可训练
   （合成床有 8 个），训练质量受限；如需公平可再引入 sync word
   符号（值已知）扩充训练集。
5. 深端（≥+24dB）信息量有限：CR4/5+33B 的悬崖太陡。后续若要 OTA 深端
   对比，建议用 CR4/8 或短包采集补一档。

## 五、文件

```
ota_v6_fixed.py     # 修复版重放（自包含，自检门禁焊死）
results_v6.json     # v6 战表 + 每链探针 δ
v6_run.log          # 运行记录（829s）
ota_final.py 等 v1-v5 # 保留作历史勘误参照（其数字作废）
data/smoke_tmp/audit_*.py  # 审计证据脚本（旁路实锤、SNR、回绕推导）
```
