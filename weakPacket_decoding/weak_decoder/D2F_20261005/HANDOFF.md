# D2F HANDOFF — 交接文档（2026-10-06，P 列+消融+DeRa 消融后）

> 读序：**05_docs/D2F_接收流程说明.md（人类可读步骤说明，新读者入口）**
> → README.md（包结构与规则）→ **FINDINGS.md**（F1-F12 认知沉淀）
> → 本文件（现状+怎么继续）。规范唯一权威 =
> `weak_decoder/EXPERIMENT_PROTOCOL.md` v1.1（§5A：检测验收只认
> 下游 CRC；半格点 δ 必测；oracle 必打标签；三层分报）。

## 1. 这条线是什么（一句话）

**D2F（Decode-to-Finish）**：弱 LoRa 包的检测+同步+解码一体链。
贡献在检测/参数估计层（证书式检测、连续锚、**fast 时间基修复**），
解码器可插拔（A 列 = DeRa port 收编，P 列 = 两段斜坡相位补齐，
B 列 = 自有 trellis），CRC 逐级仲裁收口。系统名已定（用户拍板），
论文/战表/代码统一叫 D2F。

## 2. 战绩快照（对 DeRa 全链 port，SF10 OTA 28 帧，20 种子满额）

| 场景 | 结果 | 归档 |
|---|---|---|
| δ=0 | 10%PER **−23.3 vs DeRa −22.2**；−24 .174 vs .672；−18 .000 vs .002 | RESULTS_PHASE.md §3 |
| δ=0.02 | **全档首胜**：−24 **.197 vs .663**；10%PER ≈−23.0 vs 全档不达（对 m3p2 +2.2dB） | 同上 + RESULTS_FAST_TMPL §4 |
| 模块消融（2026-10-06 定案） | δ=0.02 −24 贡献：**fast .652 ≫ 相干合并 .401 > 梯 .084 > B .019 > P .007**；两段相干合并价值 +2.2dB/.40PER | RESULTS_PHASE.md §3 |
| 检测层（诊断级） | cert +4.0~4.2 / FF_cert8 +0.6~3.7 / dep4 +0.63~2.18（半格点补考欠着） | 仓外 fullfield 系 |

已知口径注意：DeRa 检测级 port 比论文瘦身（缺 20 候选×25 点精化），
δ=0 的漏检优势按"待确认"持有——**v14 重构决议在案（P1-P4），重构后
如实改判、报双版本**。跨 checkpoint 比较必须同分母（join 缺口单元
多为失败，F11 工程律）。

## 3. 当前 shipping 链（m3p2+fast+P，全部已验证）

```
门扫(top-3事件) → dep4k16 两级检测(13.74dB@FAR1e-3 池化单门限)
→ 连续锚(提名抛物线精化 ν̂) + δ̂(0.01格)
→ ν_rot 居中 + 斜率梯{0,.01,.02,.03}逐符 de-walk
→ A列(DeRa port) → B列(TREL-5 走格) → Δ0 CRC 逐级早退
→ [全败] fast 救援：δ梯{0,±.01,±.02}×前导盲τ̂₀(含噪段逐候选)
   → ε精确逆重采样+advance(τ̂₀) → A→P→B 三列 → Δ0 CRC（零回归）
   （P=逐符号两段斜坡相位补齐：盲拟合斜率+护栏退回常数）
```

## 4. 主线状态与下一刀

**两段相位补齐已完成关账（2026-10-06）+ DeRa 解码器消融定案（同日）**：
P 列调通入列（`01_core/phase_align_core.py`）+ 机制/链战/消融满额。
**余量定律（F11）：段内相干叠加已榨干**；模块贡献分解定案（fast 主力
.652 ≫ 相干合并 .401）。**DeRa 消融（F12）：解码增益 = ML φ̂₀（全档
.14-.46）≈ 相干合并（门限 +2.2dB）＞ 两段结构（必需地基，混叠定律）
＞ CRC 重试（必需品）；系统大头在其前端**。

**下一刀候选（按价值排）**：
1. **行3-5 跨符号相干（榨取相干能量的唯一出路）**：块级跟踪+
   前后向平滑（公共相位 φ̂₀ 跨符号 ML 联合、τ/φ 转移模型；入口 =
   e3 turbo 断点差分 §6-D）；软反馈；上界对照。
2. κ̃-GLRT 跨 bin 相干（Dirichlet 泄漏回收，oversampling 线接续）。
3. τ 细梯（±0.5×3）：−24 天花板 .057 vs 现状（GT 扫描裁定）。
4. SF11 扩展（runner 现成，改 SEED_CONST=20261006+数据源）。
5. 浅端地板（DeRa δ=0 −20 略优 .005 vs .011）。
6. 学术点：段幅度不平衡解析式 / τ̂ 仪器偏低 2 采样机理（F10/F11）。

## 5. 并行注意（其他窗口/会话在动的东西）

- **v14 DeRa 检测器重构**（P1-P4）在另一个线推进——重构落地后：
  ①δ=0 战表可能收窄（预注册如实改判）；②D 列 turbo 重评；③
  m3p/m3p2/fast_tmpl 若重跑，DeRa 臂要换 v2 检测器并双版本报。
- 同级 idea 目录：`decode_to_finish_rescue/`、`splice_dera_savaux/`
  （其他会话），互不依赖。
- 共享 canonical `weak_decoder/decoding/kappa_trellis/` 被多线使用，
  **不动原件**；D2F 一律 `from decode_trellis import ...`（包内
  副本，逐位一致验证过）。

## 6. 坑清单（省下一个窗口的调试时间）

- **A. 路径/解释器**：跑实验用 `D:/mysoft2/miniconda3/envs/gr-lora/
  python.exe`（系统 `python` 是 3.6 会炸 `from __future__`）；包内
  脚本自带 sys.path 注入，但 m3p 系原目录原件仍解析到
  `data/experiments/e2e_final_20261004/`（join 一致性，别改）。
- **B. Mimosa 钩子**：bash 里写源码会被拦（用 Write/Edit 通道）；
  绝对路径写 jsonl 可能误报路径穿越（改 stdout+重定向）；heredoc
  写记忆文件会把 shell 尾巴带进去（用 python 追加）。
- **C. battle 纪律**：--budget 分钟分片 + jsonl 断点（house style）；
  换臂集合要清 checkpoint。**同噪声 join 公式**（m3p 系）：
  `(20261004·7919+(lv+100)·131+sd·101+fi·7919+di·104729) mod 2³¹`；
  fast_tmpl 战复用此派生 join m3p/m3p2/dera——换噪声必须新 SEED_CONST。
- **D. m3p3（D 列 turbo）现状**：已集成但**默认停用**——e1-harness
  10.6% 增益不向链上转移（δ=0.02 0/20、δ=0 0/15、GT 也败）。死因
  未定位；下一刀 = 同帧同噪 e1 vs 链上中间量逐层差分。教训：
  try/except 兜底会把异常吞成"0 救回"假象——兜底必须留 err 串。
- **E. demap/判决约定**：行必须功率域；φ 测量 bin 必须用跟踪 bin；
  参数补偿门 0.04 会把 δ=0.02 拦掉（历史咬过两次）。
- **F. FR 帧无 seg 字段**（只有 iq），切段式：
  `iq[hs−(pre+6)·NF : hs+(8+psym+2)·NF+64]`，注入 origin =
  `(lead−(pre+4.25))·NF`（场首起累计）。
- **G. fast_tmpl 三定律（F9）**：①机制对拍统计量必须读
  **mode(argmax−gt−1) bin**（去斜 STO 免疫：整段延迟不挪 tone 分数
  位、只挪 fold；整数记账 |d|≈0.75 处翻转，Δ0 吸收；读固定 bin 得
  −24dB 假 FAIL——γ_k/增益读数同理，F11 已二次踩中）；②修复符号 =
  **advance(−τ̂₀)**，tone SNR 平台 |d|∈[1.5,3]（native 两侧同高，
  低 SNR 才分胜负）；③**payload GT 码值 est_tau0 偏低 ~2 采样不可信**
  （F10 仪器翻转），τ̂ 仪器只认前导盲估/下游解码。
- **H. 相位补齐/消融坑（F11）**：①demod_phased 的 rows_const 与
  port 逐位一致是消融锚点（窗切片 complex64/ML 全符号口径勿改）；
  ②盲斜率拟合必须强符号掩模+护栏（残差 >0.35 rad 退回常数），
  port 历史 unwrap 全符号失败教训；③跨 checkpoint 比较**必须同分母**
  （join 缺口单元多为失败，混入虚抬对照差距）；④消融臂用
  `fast_arm_sets`（一次投影多套列组合，省 60% 算力）。

## 7. 数据与口径备忘

- 数据：SF10 `data/USRP_IQ/0_0_0_10_14_{8,16,32}.bin`（28 帧，
  P∈{8,16,32}）；SF11 lab1_sf11_TP2（120 帧 LDRO）未用——扩展时
  runner 现成。
- 种子派生：m3p 系 `(20261004·7919+(lv+100)·131+sd·101+fi·7919+
  di·104729) mod 2³¹`；fast_tmpl 机制战用 sfo_sto 派生
  `(20261005·7919+…+3·104729) mod 2³¹`（join sfo_sto 用）。
- δ 注入：**整段 IQ（含前导）先注入后加噪，同步解码面对漂移后
  波形**（全部 RESULTS 已声明）。
- 指标：PER=CRC16（唯一胜负判据）；SER 会骗人（F4 解离现象）。

## 8. 文件地图（本包）

```
FINDINGS.md          认知沉淀 F1-F11（先读）
README.md            包结构/依赖地图/复现命令/归因/未决
01_core/  m3p_core   dep4k16 盲检测+连续锚（shipping 检测）
          m3p2_core  斜率梯仲裁（shipping 仲裁）
          m3p3_core  turbo D 列（停用，§6-D）
          ref_template  精确模板相关器（行1/2 基准）
          fast_tmpl_core 快实现接收机（shipping 第三级，2026-10-05）
          phase_align_core P 列相位补齐+N 消融臂+fast_arm_sets（2026-10-06）
decode_trellis/      B 列自有副本（去网格化演进在此做）
02_detector/         dep4/dep5 检测器库（含"搜索税构成定律"实验）
03_battle/           战与汇总脚本（sfo_sto/sfo_phase_decomp/
                     fast_tmpl_check/mech/battle、phase_align_check/
                     phase_ab_battle 消融链战）
04_results/          RESULTS_M3P/M3P2/FAST_TMPL/PHASE + 全量 checkpoint
                     （m3p 8484 / m3p2 5656 / mech 700 / fast 4536 /
                     phase 728 / phase_ab 4536）
05_docs/             dep5 设计 / 方法突破存档 / 用户下一代接收机设计
```
