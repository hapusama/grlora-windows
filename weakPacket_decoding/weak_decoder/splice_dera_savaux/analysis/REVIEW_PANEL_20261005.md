# 三方评审会审（2026-10-05：MobiCom 审稿人 / SP 审稿人 / 红队）

> 用户问"有没有发 CCF-A 的潜力"。三个独立 agent 各自核对文献后给出
> 评审。本文件为归档摘要；完整原文在会话记录。

## 三方评分与裁定

| 评审 | 评分（现状） | venue 判定 |
|---|---|---|
| R1（MobiCom 风） | 2/6 weak reject | MobiCom ~85%+ reject / MobiSys ~80% / SenSys 50:50；故事够 A、地基不够 |
| R2（SP/TMC 风） | 4/6 borderline（major revision） | **TMC 首选**（4）；SECON 5（但 B 类降价出手）；INFOCOM 3（缺网络级）；MobiCom 2-3 |
| 红队 | 死亡概率 92-95%（MobiCom'27）/ 85%（INFOCOM'28） | 全套补齐后 ~27-35%（正常竞争风险） |

## 新颖性裁定（R2 独立检索后）

- **拼接失效机制：首创（经验机制发现）**——检索无先例；三条否证实验
  使其"系统类论文里罕见地干净"；升 A 需形式化为"残余免疫分类"定理。
- **γ-链序列级跟踪：LoRa 域内首创（维持成立）**——域外是 PSP'95 模式
  移植，必须引 PSP + 与 DeRa 二阶段（帧级前馈拟合）显式划界。
- **fd_c：增量**——Savaux IoT-J'22 同步 ML/Two-Pass fine sync/Marquet
  谱系占估计量类别；卖点只能写"可移植补丁"（装到纯 Savaux 全档胜）。
- **"软保留统一框架"：目前是叙事不是贡献**——做 κ 网格软边缘化之前
  不许当主张写。

## 红队三条关键情报

1. **DeRa 官方仓库已存在**（github.com/daibiaoxuwu/DeRa，占位 2 commits
   + gen_signal.py），论文自称 opensourced，MobiCom'26 会期 2026-10-26~30
   ——**"全球唯一实现"只剩 ~19 天独占期**，之后任何人可拿官方码打
   我们的 port。去混血不是可选项是时间窗铁律。
2. **DeRa 原文评测 = OTA 捕获 + 校准合成噪声注入（§5.1）**——与我方
   方法论同款，AWGN 注入本身不是死罪；差距在 DeRa 另有 SF7-12×2 硬件
   ×3 环境×现场 PDR×移动×CFO 五档×干扰×Pi4 实时×开源（我方 0 项）。
3. **统计功效攻击成立**：28 帧/3 capture = 3 个信道实现（伪重复）；
   PER 头条 −20 档 0/84 vs 2/84（Fisher p≈0.5）、−22 档 5/84 vs 9/84
   （p≈0.4）**均不显著**；SER（千级判决）才是统计硬轴——报告重心
   必须换到 SER 主指标 + PER 带 CI 辅助。

## 补齐清单（三方交叉印证后的优先级）

| 级 | 项 | 杀伤力/理由 |
|---|---|---|
| P0 | **去混血**：cert 检测层接入主链 + 机制/增益在 ≥2 前端复现 | 单项最大杠杆（红队 −18%）；同时杀 W1 与"寄生"攻击 |
| P0 | **融合 D2F**：帧内漂移成主实验轴（δ=0.02 ≥+2.7dB 证据已存在） | γ-链主场；机制叙事闭合 |
| P1 | **软读出 LLR**（统一度量→Hamming 软解码） | 翻 −24 SER + "soft"从口号变交付 |
| P1 | **数据规模**：≥500 帧/≥15 capture/≥2 环境/SF7+11+12 + bootstrap CI + CMH | 伪重复死穴 |
| P1 | **理论四件套**：平台/格点充分性引理、fd_c CRLB+虚警证书、WLS=ML 条件+软硬 gap、SER–κ 闭式界；终极=残余免疫分类定理 | A 级理论肉 |
| P2 | 鲁棒性矩阵（离线衰落/CFO 扫描/off-bin/时变 CFO/干扰）+ 真实远距 OTA + Pi4 实时性 + 全开源 | DeRa 对齐标尺 |
| P2 | 基线补齐：LoRaTrimmer（**有开源码**，快赢）、UniChirp、Ostinato、Xhonneux 式估计-校正、Two-Pass fine sync | 廉价防御 |

## 叙事改造（三方一致）

- 标题级主张改为 **"超灵敏检测与相干解调之间缺失的交接（missing
  handoff）+ 把同步不确定性保留进解码器"**；DeRa 从"对手"降格为
  "超灵敏检测类的代表实例"；正文不出现对 DeRa 的 dB 胜负（数字交给
  第二前端/自有前端）。
- R2 给的 abstract 骨架（照抄级）："Every current LoRa receiver collapses
  synchronization uncertainty before decoding; we show this early collapse
  is not benign—it breaks oversampled coherent demodulators at
  fractional-bin wrap boundaries—and we keep the residual uncertainty
  inside the decoder, with WLS-as-ML and plateau certificates."
- 时间窗：19 天后官方码落地。**5 个月（MobiCom'27 截稿）做不完全套就
  投 INFOCOM'28 / TMC，SenSys'27（B）保底。**

## 引用勘误（R2 核出）

- Xhonneux STO 论文 = IoT-J 9(5), 2021/2022（DOI 10.1109/JIOT.2021.3101002），
  非 '20；须补引同组双用户 ML 接收机（2022）。
- LZn 机制按原文 = spectral intersection（谱交集），**非 Zoom-FFT**——
  我方 LIT_COMPETITORS.md 里技术扫描 agent 的转述有误，以此为准。
- 必引：PSP（Raheli'95）、Two-Pass fine sync（fd_c 近邻）、FCLoRa。
