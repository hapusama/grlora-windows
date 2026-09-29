# 弱包接收恢复：可运行入口与残余频偏实验

日期：2026-09-13。研究对象为 `gr-lora_sdr/weakPacket_decoding`。

## 本轮实际恢复了什么

新增 `scripts/recover_weak_packets.py`：读取原始 complex64 IQ 和已有的接收信号检测/定位 CSV，重新估计同步参数，经过 Savaux、软 Hamming、显式头、PHY CRC，输出逐事件 JSONL。它不读取干净同步参数、发送字节或数据集参考文件。默认保留原 CFO，失败后尝试 CFO−1、CFO+1；三者保持最终时间位置和 SFO 不变。

新增 `system/residual_cfo.py`：把这个三候选流程从一次性实验中抽出，同时提供仅利用收到的已知前导码进行候选排序的实验选项。原有同步器、解码器及其默认行为未修改。

已有 GNU Radio 3.10.11.0 和 `gnuradio.lora_sdr` 可以导入。本轮入口和实验使用工作区 `.venvs/rfsr/Scripts/python.exe`，不需要重新编译 GNU Radio。

## 可核对的机制

重放 ID89、seed91612、Es/N0=13 dB 时，原估计为 −127.371869 bin，完整包 CRC 失败。保持首个编码符号起点 676486 不变，CFO−1 后恢复全部参考字节且 CRC 通过；干净同步诊断为 −128.368194 bin、相同起点。

把最终编码符号时间原点反推回已知前导码，用相同 Savaux 接口计算已知 bin 的功率，可以在不读取载荷真值的条件下区分这三个候选。本轮冻结规则为：使用前导码内部索引 4–11 的八个 upchirp，在原 CFO、CFO−1、CFO+1 下分别计算

`score(d) = sum_m |Z_m,d[0]|² / sum_m,k |Z_m,d[k]|²`。

按分数降序排列；完全相同时保留原顺序；前导码不足时保留原顺序。该分数没有被标定为检测概率。这里沿用标称符号间隔，较大采样时钟漂移、整 chirp 定位错误及近似平票均可能使排序出错。

不能据此认定“只需修改某个取整函数”或“CFO 总要减去 netid_offset”：重放数据中也有 netid_offset=0 而 CFO 偏一的情况，另有时间和 CFO 同时偏移但仍可解码的情况。

## 旧数据重放：12 个物理包，72 次试验

ID81–92，seed91611/91612，Es/N0=12/13/14 dB；SF12、BW125 kHz、Fs1 MHz、前导16、sync0x12、显式头、软纠错。目标信道 SNR 为约 −24.12/−23.12/−22.12 dB。整段 IQ 添加同一带限噪声，所有方法使用相同随机前缀和噪声。

| 流程 | 字节正确且 CRC 通过 | 解码调用总数 | 相对原单候选的配对增益/退步 |
|---|---:|---:|---:|
| 原同步点，软解码一次 | 38/72 | 57 | — |
| 固定三候选 | 44/72 | 89 | +6 / −0 |
| 前导码排序，只解一次 | 43/72 | 57 | +6 / −1 |
| 前导码排序，最多三次 | 44/72 | 84 | +6 / −0 |
| 干净同步 oracle，软解码 | 59/72 | 72 | 干净同步对照 |

原单候选和固定三候选的全部72条 exact、false_delivery、attempts，均与 `paper_story_20260912/frequency_holdout/trials.csv` 精确一致。这是代码抽取与复现实验验证，不是新增独立性能证据。

前导码单次选择的净增益是5次，但确有1次退步：ID85、seed91612、Es/N0=13 dB。最高两个 score 约0.002907和0.002840，选择最高项反而失败，原候选能成功。因此本轮未把单次排序设为默认，也未根据这次失败临时添加阈值后再声称无退步。

## 另一批包与新噪声

验证规则在运行前冻结：ID65–76，seed91301/91302，其他协议相同。物理包清单和噪声种子均与上述重放不重叠；两轮核心模块 SHA-256 相同。这仍是同一采集 session，不能称为跨场景验证，也不保证仓库历史研究未使用这些包。

| 流程 | 字节正确且 CRC 通过 | 解码调用总数 | 相对原单候选的配对增益/退步 |
|---|---:|---:|---:|
| 原同步点，软解码一次 | 46/72 | 62 | — |
| 固定三候选 | 48/72 | 92 | +2 / −0 |
| 前导码排序，只解一次 | 46/72 | 62 | +1 / −1 |
| 前导码排序，最多三次 | 48/72 | 93 | +2 / −0 |
| 干净同步 oracle，软解码 | 62/72 | 72 | +17 / −1 |

全部12包均通过干净IQ的CRC/字节资格检查，无本轮排除；所有方法均无CRC错误交付。
固定三候选的增益缩小到2次，绝对增加2.78个百分点，按物理包聚类的探索性95%区间为0至6.94个百分点；不足以声称稳定显著改进。前导码单次选择没有净增益，排序三候选还比固定三候选多一次解码调用。因此默认仍为固定原点优先的三候选，不把新排序替换为默认。

干净同步oracle也有1次相对原估计的退步：它是参考坐标下的诊断对照，不是每个有限噪声实现都严格占优的理论上界。

完整结果位于 `data/experiments/residual_cfo_revival_20260913/validation/summary.json`。主张应限定为“默认恢复入口在两组数据中保留原单候选的全部正确交付，并补回少量包”，而不是宣布已找到统一优胜的新算法。

## 自然弱录音：7 段，68 个共享检测事件

SF10、BW125 kHz、Fs500 kHz、前导32、sync0x34、无新增噪声。沿用历史接收信号定位事件，重新进行 full/virtual-phase SFD 同步，运行新入口。解码选择完成后，才与高 SNR 共识参考字节比对。

| 录音 | 检测事件 | 固定三候选正确包 | 前导码单次正确包 | 固定三候选调用 | 前导码单次调用 |
|---|---:|---:|---:|---:|---:|
| low1 | 15 | 11 | 11 | 23 | 15 |
| low2 | 18 | 15 | 15 | 26 | 18 |
| low3 | 16 | 10 | 10 | 28 | 16 |
| low4 | 19 | 8 | 8 | 44 | 19 |
| low5/6/7 | 0 | 0 | 0 | 0 | 0 |
| 合计 | 68 | 44 | 44 | 121 | 68 |

所有接受包均逐字节正确。单次排序保持相同44包，减少53次解码调用（43.8%）。这不等于端到端耗时下降43.8%，因为排序增加了前导码频谱计算。上述录音此前已经用于研究，不作为新的外部测试集。

原记录中的 full 单候选软解码为40/68，chip严格流程为42/68；历史 chip四候选软解码为45/68，本轮仍未超过后者。实际发送总数未知，后三段没有任何检测事件，所以44/68是共享检测事件中的恢复数，不是整段录音PDR。

## 运行方法

在 `D:/Desktop/proj` 执行。输出文件必须使用新名字，脚本会拒绝覆盖已有结果。

```powershell
& ./.venvs/rfsr/Scripts/python.exe gr-lora_sdr/weakPacket_decoding/scripts/recover_weak_packets.py `
  -i gr-lora_sdr/weakPacket_decoding/USRP_collector/data/branch4_fixed/low_snr/sf10_bw125_fs500_pre32_sw34_low2.bin `
  -s gr-lora_sdr/weakPacket_decoding/data/experiments/real_low_snr_20260717/low2_win4/sync.csv `
  -o gr-lora_sdr/weakPacket_decoding/data/experiments/residual_cfo_revival_20260913/low2_rerun.jsonl `
  --sf 10 --samp-rate 500000 --preamble-len 32 --sync-word 0x34 `
  --allow-gate-failed --policy fixed --max-attempts 3
```

`--allow-gate-failed` 允许已有定位结果但严格同步门控失败的候选进入完整头部/CRC验证。不提供时保留严格拒绝行为。每条JSONL记录保留候选状态；只有头有效、CRC开启且通过时输出 `payload_hex`。CLI本身只知道CRC接受，不能声称字节完全正确。

实验性前导码单次选择：改用 `--policy preamble --max-attempts 1` 和新的输出文件。没有检测CSV时，先运行 `scripts/run_weak_sync_chain.py`；本轮没有解决原检测器漏检的问题。

新包验证复现命令：

```powershell
& ./.venvs/rfsr/Scripts/python.exe gr-lora_sdr/weakPacket_decoding/scripts/evaluate_residual_cfo_recovery.py `
  --offset 64 --packets 12 --seeds 91301,91302 --snrs 12,13,14 --workers 3 `
  --label frozen_same_session_replay `
  --output gr-lora_sdr/weakPacket_decoding/data/experiments/residual_cfo_revival_20260913/validation_rerun
```

相关测试17项通过，包括正负整数偏差的合成IQ选择、保持时间/SFO、前导不足/全零回退、解码预算、CRC关闭不能接受，以及现有软纠错和Savaux同步敏感性测试。

结果目录的 `recovery_evidence.png`/`.svg` 汇总两批加噪试验和自然录音，`integrity.json` 记录完整配对数量、包/seed不重叠、源码哈希一致及旧结果逐条复现检查。重新核对和绘图：

```powershell
& ./.venvs/rfsr/Scripts/python.exe gr-lora_sdr/weakPacket_decoding/scripts/summarize_residual_cfo_recovery.py `
  --output gr-lora_sdr/weakPacket_decoding/data/experiments/residual_cfo_revival_20260913
```

## 接下来值得推进的范围

本轮救回的是可独立使用的离线完整收包入口，以及一个能解释、能复现的残余频偏控制。它不是已经证明的新同步理论或可投稿算法。

下一步应在相同完整包指标和计算预算下，对照已有的前导码阶段SFO补偿方法：[LoRa Fine Synchronization with Two-Pass Time and Frequency Offset Estimation](https://arxiv.org/abs/2502.08485)。该工作已经处理前导码阶段SFO对CFO/STO估计的影响，因此本轮简单重选频偏不能直接作为新颖性依据。

检测层还需与使用频谱交集的现代同步方法比较，例如在三个真实数据集上评估的 [LZn](https://arxiv.org/abs/2604.27672)。本轮未实现这些方法，也未证明优于它们。

真正待补的证据是不同采集session、带发送日志的连续弱信号PDR、相同误报预算，以及接收机真实运行成本。当前小样本没有错误交付不能外推为零误报率。
