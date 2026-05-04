可以，但投稿 INFOCOM 时，你需要把这项工作从"一个更强的 LoRa 接收算法"提升为"一个**有清晰模型、可证明增益、可分析复杂度的统计推断问题**"。INFOCOM 官方定位本来就同时覆盖 theoretical 和 systems research，所以仅有系统增益通常不够，必须把核心贡献写成一个 formal problem，再给出 theorem-level 的结果。

结合你现有文档里的思路，你最有潜力的主线不是"利用 CFO 这些缺陷解码"，而是：**将残余 CFO、相位漂移、时偏失配统一建模为随符号演化的隐状态，并在单包、单天线、极弱信号条件下做联合序列估计与软解码**。你文档里已经有"相位状态网格、动态规划、软输出、Hamming 联动"的雏形，但现在还偏工程直觉，理论包装还不够硬。

先看近两年的高水平论文给出的信号。XCopy（MobiCom 2023）把增益建立在**跨重传副本的相干组合**上；ChirpTransformer（MobiSys 2024）通过**改编码空间**来换 coverage / throughput / energy；LoRaTrimmer（MobiCom 2024）已经开始把弱包失败归因到**符号内频跳导致的噪声与相位问题**，并引入 probabilistic modeling；LoRaSeek（MobiCom 2025）继续走**神经去噪接收机**；SLoRa+（2025）则把**symbol recovery 和 soft decoding**系统性结合起来；B2LoRa（2025）在卫星 LoRa 里进一步把 Doppler、CFO、phase drift 纳入 blind coherent combining。换言之，前沿已经从"单点 FFT 寻峰"走向"结构化建模、软信息、相干对齐、学习式接收机"，但**单包单天线场景下、围绕残余硬件缺陷建立理论化序列推断模型**这一块仍然是可打的空档。

所以，如果你要冲 INFOCOM，我建议你把论文主线收敛成下面这个版本：

---

## 1. 把研究问题改写成"隐状态辅助的 MAP 序列解码"
不要写"用 CFO 缺陷帮助解码"，这个说法不学术，也容易被质疑为 heuristic。更好的表述是：

> **在 LoRa 弱包接收中，残余 CFO、公共相位误差、慢变漂移与时偏失配构成了跨符号相关的 nuisance process。传统逐符号 FFT 判决忽略了这一相关性；本文将其建模为 partially observed latent state，并据此构造 single-packet MAP / approximate-MAP decoder。**
>

这个表述的优点是，你的创新立刻从"经验算法"变成了"统计推断问题"。这比单说 Viterbi、卡尔曼或相位链，要更符合 INFOCOM 审稿口味。

---

## 2. 你最应该追求的，不是一个"更复杂的 Viterbi"，而是三个可证明的问题
最可行的理论化路线有三条。

### 第一条，也是最推荐的一条：可证明优于逐符号判决的序列估计框架
你需要定义观测模型

$ z_{k,m} = A_k \cdot g(m - s_k, \tau_k) \cdot e^{j\phi_k} + w_{k,m}, $

以及状态演化

$ \phi_{k+1} = \phi_k + 2\pi \Delta f_k T + u_k, \qquad \Delta f_{k+1} = \rho \Delta f_k + v_k. $

其中 $ s_k $ 是离散符号，$ (\phi_k, \Delta f_k, \tau_k) $ 是隐状态。然后你的算法是对 $ (s_{1:K}, x_{1:K}) $ 做 MAP 或近似 MAP。理论上最值得证明的是：在某个"峰值分离度 + 状态平稳度"条件下，序列解码的 error probability 严格优于 symbol-wise FFT hard decision，或者至少能给出一个 upper bound gap。只要你把这个 gap 定理做出来，这篇文章立刻就从"技巧改进"变成"有理论贡献的 receiver design"。

### 第二条：候选裁剪后的近似最优性
INFOCOM 很吃"复杂度—性能 tradeoff"这种东西。你现在的 Top-K 候选 + 状态网格很适合写成一个 theorem：当正确 bin 的后验质量满足某个 tail condition 时，仅保留 Top-K 候选不会以超过 $ \delta $ 的概率丢失最优路径；于是复杂度从 $ O(KM \cdot 2^{SF}) $ 降到 $ O(KMK) $。LoRaTrimmer 已经说明"FFT 感知域可以被裁剪，而且 probabilistic modeling 是有效的"，但它主要还是 per-symbol 处理；你可以把这个思想推进到**sequence level candidate pruning with guarantee**，这会更像 INFOCOM。

### 第三条：前端 soft metric 到后端软译码的失配有界性
你文档里已经在做 soft symbol $ \to $ LLR $ \to $ soft Hamming，这一点非常值得保留，但 INFOCOM 不会满足于"工程上能用"。你要证明的是：当前端 branch metric 存在相位估计误差 $ \epsilon_\phi $ 和频偏估计误差 $ \epsilon_f $ 时，映射得到的 bit-level LLR 偏差是可界的，或者该失配不会破坏 message passing 的单调改进。SLoRa+ 已经把 symbol recovery 和 soft decoding 结合起来了，所以你再写"我也做软判决"不够；你必须回答**为什么这个 soft 信息在你的 receiver 里是 statistically meaningful 的**。

---

## 3. 哪些方向适合做，哪些不适合做
你现在有几个分叉，但不是每个都适合 INFOCOM。

### 最适合做主线的
"单包、单天线、残余缺陷隐状态建模 + 序列级 MAP/近似 MAP + soft decoding 反馈"。

因为这条线最容易形成：模型、算法、证明、复杂度、消融，全都完整。XCopy 和 B2LoRa 已经把"多副本/重传/重复广播 + coherent combining"做得很强；ChirpTransformer 则把"改发射端编码空间"做得很强；LoRaSeek 又占了"纯神经弱包解码"方向。你的差异化空间，更像是**不改 PHY、不依赖多副本、不依赖重训练，仅靠接收端的理论化推断提升弱包极限**。

### 不建议作为主线的
**第一，纯 AI 接收机。** LoRaSeek 已经在 neural-enhanced weak decoding 上继续往前推进，如果你没有明确的 generalization bound、physics-informed architecture proof 或 sample complexity 分析，INFOCOM 上这条线很容易显得既不够理论，也不如 MobiCom 方向新。

**第二，改 payload 插 anchor/downchirp 作为主版本。** 这个可以做 extension，但主线最好保持标准 LoRa 兼容，否则审稿人会把增益解释成"靠额外冗余换来的"。ChirpTransformer 已经说明 waveform/encoding redesign 是一条独立路线，你没必要和它正面重合。

---

## 4. 我更建议你在论文里写出的三类 theorem
你现在最缺的其实不是算法，而是 theorem menu。下面这三类非常适合你的题目。

### Theorem A：可辨识性 / identifiability 条件
在给定残余 CFO 漂移上界、相位噪声方差上界和候选谱峰最小间隔条件下，正确符号序列在观测分布意义下是可辨识的。

这个定理的意义是：先证明"为什么这个问题理论上可做"。

### Theorem B：序列解码增益下界
在 Gauss-Markov 漂移模型或 bounded-innovation 模型下，你的 sequence decoder 相比 symbol-wise hard decision 至少能获得一个非负增益下界，或者错误概率满足更紧的 exponent/bound。

这个定理的意义是：回答"为什么用相关性一定有价值"。

### Theorem C：近似算法误差界
对 Top-K candidate pruning、相位网格量化、beam search / M-algorithm 的近似误差给出界，写清楚复杂度和性能之间的 tradeoff。

这个定理的意义是：回答"为什么这个算法不是只在理论上成立，而是 practical"。

只要这三个里你能扎实做出两个，INFOCOM 的理论味道就基本够了。

---

## 5. 实验也要按 INFOCOM 的逻辑重排
如果走这条路线，实验不应先上 PRR，而应先验证理论假设。

先做三组"机制实验"：

1. **粗同步后，相邻符号的 residual phase / CFO estimate 是否具有短时相关性；**
2. **正确候选 bin 与错误候选 bin 在 phase-consistency metric 上是否 statistically separable；**
3. **soft metric 经 Gray 逆映射后的 LLR，是否与真实 bit posterior 单调相关。**

这三组是 theorem 的实验支撑，比单纯报 -2 dB、-3 dB 更重要。然后再做系统实验：SER/BER/PRR、复杂度、时延、不同 SF/BW/CR、不同 CFO/SFO/STO、多径与动态信道。这样论文逻辑会非常顺。

---

## 6. 一个更适合 INFOCOM 的标题和摘要核心句
你现在最好不要用"phase chain"这种偏直觉的说法。更适合的标题方向是：

> **Hardware-Impairment-Aware Sequence Decoding for Single-Packet Weak LoRa Reception**
>

或者

> **Latent-State MAP Decoding for Weak LoRa Packets under Residual CFO and Phase Drift**
>

摘要核心句可以写成：

> **We formulate weak LoRa reception as a latent-state sequence inference problem, where residual CFO, common phase error, and slow impairment drift are modeled as symbol-evolving nuisance states. Based on this model, we design an approximate MAP decoder with candidate pruning and soft-information feedback, and establish identifiability / performance-complexity guarantees under mild impairment dynamics.**
>

这才是 INFOCOM 会觉得"这篇文章有理论骨架"的表达。

---

## 总结
我的判断很明确：**你最可行、也最有 INFOCOM 风格的方向，不是"CFO 帮我解码"，而是"残余硬件缺陷诱导的跨符号相关性，可被 formalize 为 latent-state inference，并带来可证明的序列解码增益"**。这条线和你现有草稿高度兼容，而且和最近几篇高水平 LoRa 论文的方向错位是合理的。

下一步最值得做的不是继续扩展点子，而是把这条主线写成"**问题定义 — 观测模型 — 状态模型 — MAP 目标 — 近似算法 — 三条 theorem — 验证实验**"的论文骨架。

---

**参考文献**

+ XCopy（MobiCom 2023）: [https://www4.comp.polyu.edu.hk/~csyqzheng/papers/XCopy-MobiCom23.pdf](https://www4.comp.polyu.edu.hk/~csyqzheng/papers/XCopy-MobiCom23.pdf)
+ ChirpTransformer（MobiSys 2024）: [https://www.cse.msu.edu/~caozc/papers/mobisys24-ren.pdf](https://www.cse.msu.edu/~caozc/papers/mobisys24-ren.pdf)
+ LoRaTrimmer（MobiCom 2024）: [https://cse.msu.edu/~caozc/papers/mobicom24-du.pdf](https://cse.msu.edu/~caozc/papers/mobicom24-du.pdf)
+ SLoRa+（2025）: [https://dl.acm.org/doi/10.1145/3767338](https://dl.acm.org/doi/10.1145/3767338)
+ LoRaSeek（MobiCom 2025）: [https://www.cse.msu.edu/~caozc/papers/mobicom25-khang.pdf](https://www.cse.msu.edu/~caozc/papers/mobicom25-khang.pdf)
+ B2LoRa（2025）: [https://www.cse.msu.edu/~caozc/papers/...](https://www.cse.msu.edu/~caozc/papers/...)

