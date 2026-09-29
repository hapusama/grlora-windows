# Fine-Grid GLRT Soft Demodulation for Oversampled LoRa —— 方法稿 v0.1

> 2026-09-28。数据与复现：`data/experiments/enhanced_demod_chain_awgn_20260928/`、
> `data/experiments/oversampling_accounting_20260928/`。本稿按论文骨架组织，数字全部来自已跑实验。

## 0. 定位：解决的是什么问题

**本体是解码侧问题**：在同步估计已不可靠的深度弱信号区，逐符号软证据（LLR 的原料）
被分数 bin 失配系统性污染。本文不解分数 CFO/STO，而是**把分数失配作为滋扰参数吸收进
解调统计量**（GLRT/边际化），让软证据对对齐不确定性免疫。

与同步的关系（三层，各有归属）：

| 层 | 内容 | 归属 |
|---|---|---|
| 同步·整数部分 | CFO/STO 整数 bin 歧义 → 候选列表 | 已有工作（ambiguity-ridge + decode-to-finish） |
| **同步·分数部分** | frac CFO/STO、SFO 包内漂移、子 bin 多径 | **本文吸收进解调，免估计** |
| 同步·粗检测 | 弱信号包发现（preamble detection） | 已诊断（同步−解码 gap 6.1 dB = 10log10(4)，副本分集没进同步），检测级联已设计、未实现 |

一句话故事：**候选列表管住"哪个 bin"，细网格 GLRT 管住"bin 内哪里"——合起来是完整的
"在同步不确定性下解码"。**

## 1. 系统模型与问题

去斜后，整数部分已对齐、残余分数偏移 $\kappa\in(-\tfrac12,\tfrac12]$ 的第 $k$ 符号在
OS 率（$L=N\cdot\mathrm{OS}$ 样本）上为

$$y[m] = A e^{j\phi}\exp\Big(j2\pi\big(\tfrac{(u_m-\kappa)^2}{2N}-\tfrac{s\,(u_m-\kappa)}{N}\big)\Big)+w[m],\quad u_m=m/\mathrm{OS}$$

经 chip 网格参考去斜后为位于 $-(s+\kappa)\Delta f$ 的单频音。四种物理失配——分数 CFO、
分数 STO、SFO 漂移、子 bin 多径时延——**在接收机看来是同一个标量滋扰 $\kappa$**
（chirp 对偶性把别处二维的 delay-Doppler 塌缩成一维，这是 CSS 特有的结构）。

现有接收机的三笔损失（均已实测）：

1. **抽样丢样本**：chip 网格单相位抽取丢弃 $K-1$ 份独立噪声观测——噪声带宽覆盖
   过采样带（前端未滤波）时，等于把 $10\log_{10}K$ dB 的非相干分集留在桌上；
2. **量化税**：$\kappa$ 残留导致 Dirichlet 泄漏，chip 网格 argmax 付 1.5–2 dB，
   $\kappa\to\pm0.5$ 处 SER ×3.7（悬崖）；
3. **argmax 丢结构**：泄漏图样（反相双峰，相位差恒 $-\pi(N{-}1)/N$、幅值比
   $\kappa/(1{-}\kappa)$）本身编码 $\kappa$，硬判决全部扔掉，软证据失准。

**命题（半 bin 不可分）**：$|\kappa|=\tfrac12$ 时符号谱关于 $s+\tfrac12$ 严格对称，
任何仅依赖当前符号的判决规则 $P(\text{err})=\tfrac12$（与 SNR 无关）；可分性只能来自
跨符号的公共 $\kappa$ 信息或信道编码。仿真已给出构造性证明（κ=0.5 时音落在相邻两
假设窗口的公共边界，统计逐 bit 相等；细网格不解决，$\kappa$ 信息才解决）。

## 2. 方法：Fine-Grid GLRT Demodulation（FGD）

**C1 相位对齐获取（一次 FFT 吃下全部分集）**：在 OS 域直接去斜并补零到 $M=16N$ 点：

$$Z[j]=\sum_{m<L} y[m]\,e^{-j2\pi u_m^2/(2N)}\,e^{-j2\pi jm/M}$$

单个 16N-FFT 等价于对全部 $K$ 个抽取相位做相干对齐合并——对比单相位抽取获得
$\le 10\log_{10}K$ dB（未滤波噪声条件下实测 +5~6 dB），对比朴素功率平均（错开损失
~3 dB）白捡回大头。

**C2 跨符号 κ-格联合（消融后的正确原语）**：把 $\kappa$ 离散成 5 态格
$\{-\tfrac12,-\tfrac14,0,\tfrac14,\tfrac12\}$（恰为补零网格列），在漂移约束
$|\kappa_{k+1}-\kappa_k|\le\tfrac14$ 下做 Viterbi 联合检测：

$$T_{\rm trel}(s_k)=\big|Z_k[-4(s_k{+}\kappa_k^*)]\big|^2,\quad
\{\kappa_k^*\}=\arg\max_{\kappa\text{-path}}\sum_k\max_s\big|Z_k[-4(s{+}\kappa_k)]\big|^2$$

这是**带漂移先验的 κ 边际化**。消融否决了两个看似自然的替代原语：逐符号
窗口 max（$T_1(s)=\max_{j\in\mathcal{G}_s}|Z_j|^2$，max-of-5 噪声膨胀，负贡献）、
逐符号指纹 $\hat\kappa$（补零网格主瓣宽，相邻幅值比对 κ 近乎不敏感，偏差未修）。
半 bin 处 $|\kappa|=\tfrac12$ 的逐符号统计严格对称（命题），而 κ-列跨符号一致性
**真正破平**：TREL-N 在 κ=0.5 处 PDR50=−18.7 dB，距 oracle 0.3 dB——
命题的"逃生口"被工程兑现。

**C3 软输出进 FEC**：全谱能量直接作为符号后验，边缘化出比特 LLR：

$$P(s\mid y)\propto T(s),\qquad \mathrm{LLR}_i=\log\frac{\sum_{s:\,b_i(s)=1}T(s)+\epsilon}{\sum_{s:\,b_i(s)=0}T(s)+\epsilon}$$

喂 Chase-16 软 Hamming（对齐现有 weak_decoder 链的 bounded soft-Hamming 接口）。

**复杂度**：每符号 1 次 16N-FFT + $O(N\!\cdot\!W)$ 窗口运算（$W{=}5$），
约为旧链（N-FFT）的 4~5 倍 / 朴素副本链（4×N-FFT）的 1~2 倍。

## 3. 实验设置

SF10、OS=4、AWGN（**os 域 iid 噪声 = 未滤波宽带噪声模型**，per-sample SNR 口径）；
可控旋钮 $\kappa\in\{0,\,0.25,\,0.5,\,\text{drift}(+0.02/\text{sym})\}$；120 包/点，
8 符号/包；FEC stand-in：10×Hamming(8,4)+总校验、对角交织、Gray、Chase-16（软）/
最近码字（硬）——非完整 LoRa PHY，仅用于链间公平对比。基线：OLD-A（frame_sync 风格
单相位 chip 网格）、OLD-B（4 相位功率平均）；参考：真 $\kappa$ 相干 ML oracle。

## 4. 结果

**真 LoRa PHY 终局战（`symfec_pk_20260928/`）**：TX 升级为真编码链
（白化+Hamming CR4/8+对角交织+Gray+CRC），RX 全员谱统一喂 SymFEC（TMC'26）
自己的解码器——裁判与 FEC 层同一，唯一变量是证据层。PDR50（κ=0.25/0.5/drift）：
TREL −24.9/−22.6/−24.8 vs 最强基线 −23.5/−22.4/−23.3 vs OLD-A <−20；深端 −26 dB
处 TREL 0.4~0.63 对基线 0~0.28。两个新发现：真交织+Gray 把半 bin 掷硬币
"可纠化"（SER 0.56 而 PDR 0.98——命题的编码逃生口定量兑现）；好硬判决≠好软证据
（UniChirp 硬 SER 最低但 PDR 不领先）。κ 格在 |κ|=0.5 的符号二义由镜像双解+CRC
仲裁解决（decode-to-finish 内战内行）。

**Battle（soft-FEC PDR50，与仓库基线真实现同场，SF10/OS=4/120 包/点）**：

| 模式 | OLD-A | SAVAUX(TIM'22) | LoRaTrimmer(MobiCom'24) | UniChirp | **TREL-N(本文)** | ORACLE |
|---|---|---|---|---|---|---|
| κ=0 | −16.5 | −19.2 | −19.2 | 退化* | **−20.3** | −20.3 |
| κ=0.25 | −15.8 | −18.5 | −18.2 | −18.1 | **−19.9** | −19.9 |
| κ=0.5 | <−16 | 失败(0.5地板) | 失败(0.5地板) | −18.0 | **−19.9** | −20.0 |
| SFO drift | −16.5 | −18.1 | −18.0 | −18.0 | **−20.1** | −20.1 |

\* UniChirp（preamble 相位模型+主/混叠双峰相干融合）为最强基线：其相位观测本质是
泄漏指纹的另一种读法，是唯一在半 bin 存活的发表方法，但 κ=0 时无双峰→模型空→退化，
且全模式落后本文 ~2 dB（注意其额外消费 8 个已知 preamble 符号，本文零训练）。
对两个发表基线常规领先 1.5~2 dB，半 bin 模式断崖式胜出，漂移领先 ~2 dB；
对 frame_sync 风格旧链 +3.5~5 dB。图：`ablation_battle_20260928/battle_pdr.png`。

**消融（对 OLD-A 的 PDR50 增量）**：C1 细网格精确读 +3.4~4.3 dB（主力，
其中相干对齐 vs 朴素相位平均占 +2~3.5）；κ-格 Viterbi +0~0.1 dB 常规模式
+ **破半 bin 死区**；逐符号窗口 max = **负贡献 −0.8 dB**（噪声极值膨胀，
消融否决）；逐符号指纹 κ̂ 未兑现（δ̂ 偏差，待 5 点核拟合）；软 FEC +2~3 dB。
消融直接改写了方法结构：正确原语 = 细网格谱 + 跨符号 κ-格 Viterbi。

## 5. 边界（诚实声明）与下一步

- 增益条件（2026-09-28 噪声对照实验定量，`noise_model_control_20260928/`）：
  C1 相位对齐部分是**条件性增益**——噪声铺满过采样带（前端未滤波，即本部署链
  实测情形）时 +2~4 dB；加 2BW 固定滤波后 +0.4~1 dB；加信道 BW 滤波后 ≈0。
  论文按此分解报告，并增加"OLD-A+固定滤波"基线。**κ 结构部分（量化税/半bin/
  漂移 trellis）与噪声带宽无关**：最 adversarial 噪声臂下 TREL 仍 +2 dB（k25）。
- FEC 为 stand-in，LoRa 完整交织/CR 族未建模；单 SF10/OS=4，未含多径；
- 下一步：①κ̂ 5 点核拟合（让逐符号指纹超过精确读，当前被 TREL 功能性替代）；
  ②TREL-N 的 T(s) 接入 weak_decoder 全链（ridge-list + soft-Hamming + CRC 仲裁）出真链 PDR；
  ③OTA capture 重放；④检测级联（同步粗捕获，把 6.1 dB 结构 gap 压掉）；
  ⑤与 SymFEC 组合（其符号级 FEC 消费加性符号得分，与本文证据层正交互补）。

## 6. 相关工作定位（一句话版）

估计-then-校正路线（两步同步等）在深度弱信号区受估计地板约束；GLRT 检测已有
（前导级）；过采样降复杂度已有（Savaux TIM'22，把过采样当成本；其官方形式实现
经同场对比落后本文 1~2 dB，理想式是否等价于补零 FFT 待全文核对）；LoRaTrimmer
（MobiCom'24）与 SymFEC（TMC'26）已同场/定位对比（Trimmer 被全面超越且在半 bin
崩塌；SymFEC 正交可组合）。本文首次把 GLRT/边际化落到 **CSS 符号级软解调 + FEC +
弱包区**，并给出半 bin 不可分命题（含跨符号逃生口的工程兑现）与相位对齐分集的
记账。撞车风险清单见项目根目录《信源待下载清单.md》。
