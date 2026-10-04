---
title: "Lecture 09 — Scaling Laws I"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-04-27"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf"
  - "../experiments/topics/scaling-laws.md"
  - "../assignments/spring2026/assignment3-scaling/"
---

# Lecture 09 — Scaling Laws I：固定算力下如何分配模型与数据

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述；面向自学者、训练工程师与 scaling-law 研究者

## 摘要

Scaling laws 用低成本实验预测高成本训练的 loss、compute-optimal model size 与 data budget，
其价值来自资源配置而非仅拟合漂亮直线。本文以 decoder-only language model 为对象，
从 \(C\approx6ND\) 的训练 FLOPs 近似出发，推导固定 compute 下参数量 \(N\) 与训练 tokens
\(D\) 的最优分配，并比较 Kaplan-style 与 Chinchilla-style scaling。我们系统讨论
IsoFLOP curve、离散 minimum、联合 loss law、log-space regression、边界 optimum、
heteroscedastic noise、bootstrap、held-out prediction 与外推不确定性；进一步分析 data
quality/repetition、optimizer recipe、tokenizer、hardware utilization 和 wall-clock 对“算力”
定义的影响。工程部分给出 experiment manifest、budget alignment、failure logging 与可复现
拟合流程；研究部分强调 winner’s curse、functional-form risk 与 sequential experimental design。

**关键词：** Scaling Laws；Compute-Optimal Training；IsoFLOPs；Chinchilla；
Power Law；Extrapolation；Bootstrap；Experimental Design

## 本文贡献

1. 从 Transformer train FLOPs 推导 \(C\approx6ND\) 及其适用边界；
2. 推导联合 loss law 下 \(N_{\mathrm{opt}},D_{\mathrm{opt}}\) 的幂律指数；
3. 建立离散 IsoFLOP、联合拟合、held-out 与 bootstrap 的完整证据链；
4. 系统分析 data quality、repetition、hardware efficiency 与 recipe 的 confounders；
5. 提供面向真实预算规划的实验设计、故障排查与论文式结论规范。

## 学习目标

- 从 Transformer 的训练 FLOPs 推出一阶成本模型 \(C\approx6ND\)，并说明它何时失真。
- 理解经验 power law、IsoFLOP profile 与 compute-optimal envelope 分别回答什么问题。
- 设计能识别内部最优点、而不是制造边界最优点的实验矩阵。
- 把课程公式映射到 A3 的数据、脚本、运行约束和报告证据。

## 先修知识

- Lecture 02：\(C\approx6ND\) 的 FLOPs 推导（scaling 记账的单位）。
- Lecture 01：tokenizer 决定 token 数与 loss 的计量单位（bytes-per-token 换算）。
- 统计基础：log-space 回归、异方差噪声、bootstrap 置信区间。

## 相关工作与争议脉络

Kaplan 等发现 language-model loss 对 model size、dataset size 和 compute 呈稳定 power law，
并提出偏向更大模型、较少数据的 compute-optimal recipe [[1]](#ref-1)。Hoffmann 等通过
更密集 IsoFLOP experiments 得到不同结论：compute-optimal 情况下 model parameters 与
training tokens 应近似等比例增长，即 Chinchilla recipe [[2]](#ref-2)。

Scaling 现象并不限于文本。Henighan 等在图像、视频、数学等 autoregressive domains 观察到
类似规律 [[3]](#ref-3)；Bahri 等从统计/谱角度解释 power-law learning curves 的可能来源
[[4]](#ref-4)。后续研究强调幂律并非单一普适形式：Alabdulmohsin 等讨论 broken/neural
scaling regimes [[5]](#ref-5)，Muennighoff 等研究 data-constrained、重复训练下的最优分配
[[6]](#ref-6)，Porian 等重新分析 Kaplan/Chinchilla 差异与实验 recipe
[[7]](#ref-7)。
Replication work 进一步显示原始 Chinchilla 数据/拟合选择会显著影响系数
[[8]](#ref-8)；routed/MoE 与 transfer learning 又需要扩展标准 dense pretraining law
[[9]](#ref-9)[[10]](#ref-10)。

这些工作说明 exponent 是“指定 architecture/data/optimizer/scale range 下的经验参数”，
不是自然常数。研究重点应是数据质量、拟合诊断与外推风险，而非只复述 \(a\approx0.5\)。

## 1. 先定义问题，而不是先拟合直线

记：

- \(N\)：参与主要矩阵乘法的有效参数量，通常采用 non-embedding parameters；
- \(D\)：训练所消费的 tokens，而非原始文档数或唯一 tokens；
- \(C\)：训练 FLOPs；
- \(L\)：同一 tokenizer、数据分布和评估口径下的 loss。

Scaling law 是在指定实验域内，用少量小规模观测预测

\[
(N,D,C)\longmapsto L
\]

或固定 \(C\) 后求

\[
(N_{\mathrm{opt}},D_{\mathrm{opt}})
=\arg\min_{N,D:\,\operatorname{cost}(N,D)=C}L(N,D).
\]

它不是“模型越大越好”的理论证明。架构、tokenizer、数据质量、优化器或成本口径变化后，旧指数没有自动迁移性。

## 2. 关键推导：为什么 \(C\approx6ND\)

对一个权重 \(W\)，每处理一个 token：

1. forward 使用一次矩阵乘，约 \(2N\) FLOPs；
2. backward 对输入求梯度，约 \(2N\) FLOPs；
3. backward 对权重求梯度，约 \(2N\) FLOPs。

故每 token 约 \(6N\) FLOPs，训练 \(D\) tokens 得

\[
\boxed{C\approx6ND}.
\]

这个推导刻意忽略：

- embedding、normalization、激活函数、optimizer update；
- attention 的 \(O(S^2d)\) 项；
- activation checkpointing 的重计算；
- padding、数据加载、通信、低利用率和 kernel launch；
- MoE 中总参数与每 token 激活参数的差别。

因此 \(6ND\) 适合**同一家族的理论核算**，不能替代 wall-clock benchmark。A3 最终预算是 B200-hours，必须从 completed runs 的实测 runtime/throughput 建立校正。

### 参数口径

标准 dense decoder 的粗略 non-embedding 参数量为

\[
N\approx 12L_{\text{layer}}d_{\text{model}}^2,
\]

因为每层 attention 投影约 \(4d^2\)，MLP 约 \(8d^2\)。A3 要求统一使用这一近似；若一处用精确总参数、一处用该近似，拟合的截距会被口径差异污染。

## 3. 经验幂律与边界

常见单变量形式是

\[
L(x)=L_\infty+A x^{-\alpha},
\]

其中 \(x\) 可以是 \(N,D,C\)，\(L_\infty\) 是不可约项。双对数图上，若暂时忽略 \(L_\infty\)，幂律近似直线：

\[
\log L\approx\log A-\alpha\log x.
\]

但 \(L_\infty+A x^{-\alpha}\) **不能整体取 log 后做线性回归**；加法项会改变曲率。有限动态范围内，指数、floor 和截距也会高度相关。

幂律通常只在一个 regime 内近似成立。小模型可能受 optimization instability 主导；大模型可能受数据重复、下游饱和或架构切换主导。残差中的系统弯曲不是“噪声”，而是函数形式失配的证据。

### 3.1 联合 loss law 与 compute-optimal 指数

Chinchilla-style 联合形式：

\[
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
\]

固定 \(C=6ND\)，代入 \(D=C/(6N)\)：

\[
L(N\mid C)=E+A N^{-\alpha}+B\left(\frac{6N}{C}\right)^\beta.
\]

对 \(N\) 求导并令零：

\[
-\alpha A N^{-\alpha-1}
+\beta B\left(\frac6C\right)^\beta N^{\beta-1}=0,
\]

\[
N_{\mathrm{opt}}^{\alpha+\beta}
=\frac{\alpha A}{\beta B}\left(\frac C6\right)^\beta.
\]

因此

\[
N_{\mathrm{opt}}\propto C^{\frac{\beta}{\alpha+\beta}},\qquad
D_{\mathrm{opt}}\propto C^{\frac{\alpha}{\alpha+\beta}}.
\]

两指数和为 1 是 compute constraint 的结果；各自数值由 model-limited 与 data-limited
loss exponents 决定。若 empirical IsoFLOP exponent 与联合 law 不一致，应检查 fit range、
boundary minima、data rounding 与 functional-form mismatch。

### 3.2 Scaling exponent 的解释

\(\alpha\) 大表示增加参数更快降低 model-limited excess loss；\(\beta\) 大表示增加数据更有效。
它们不是“模型能力常数”，会随 architecture、optimizer、data quality、tokenizer 和 scale
regime 变化。\(E\) 也不一定是真实 Bayes entropy，常只是当前 domain/evaluation 的 fitted floor。

## 4. IsoFLOP：在相同预算下比较

给定若干预算 \(C_i\)，每档选择多个模型规模 \(N_{ij}\)，令

\[
D_{ij}=\frac{C_i}{6N_{ij}}.
\]

同一档中：

- \(N\) 太小：容量受限，即使看很多 tokens 也无法继续降低 loss；
- \(N\) 太大：可用 tokens/steps 太少，模型 under-trained；
- 中间应出现近似 U 形的 loss-\(N\) profile。

每档取离散最低点：

\[
N_{\mathrm{opt}}(C_i)=
\arg\min_{N_{ij}}L_{ij},\qquad
D_{\mathrm{opt}}(C_i)=\frac{C_i}{6N_{\mathrm{opt}}(C_i)}.
\]

再拟合

\[
N_{\mathrm{opt}}=aC^b,\qquad
D_{\mathrm{opt}}=cC^d.
\]

由 \(C=6ND\) 可知理想一致情形下 \(b+d=1\)。这是一项 sanity check，而不是独立实验结论：因为 \(D\) 本来就是从 \(C,N\) 推出来的。

### 一个手算例

若 \(C=6\times10^{18}\) FLOPs、\(N=10^8\)，则

\[
D=\frac{6\times10^{18}}{6\times10^8}=10^{10}\text{ tokens}.
\]

若把模型扩大十倍而保持预算，数据就缩小十倍。IsoFLOP 比较的核心正是这条预算约束。

### 4.1 离散 minimum、profile interpolation 与边界

直接取离散最低点最透明，但 model grid 过稀时 optimum 被量化。对 log-\(N\) 邻域做二次拟合可
插值 minimum，却引入“局部曲线确实近似二次”的假设。建议同时报告：

- discrete winner；
- 若使用，interpolated optimum 与拟合窗口；
- 邻近点 loss gap；
- minimum 是否在搜索边界；
- seed/measurement uncertainty。

若最低点在最小 \(N\)，结论是 \(N_{\mathrm{opt}}\le N_{\min}\)，不是
\(N_{\mathrm{opt}}=N_{\min}\)。继续把 boundary point 当精确 optimum 会严重扭曲 exponent。

### 4.2 Winner’s curse

每档从 noisy runs 选最小 loss，会偏向负噪声。候选越多，winner bias 越强。可用：

- 对 ridge 附近配置补 seeds；
- 用独立 validation/checkpoint 确认 winner；
- hierarchical/noise-aware fit；
- bootstrap 时重做“选 minimum”步骤；
- 报告 second-best 与 uncertainty，而不只 winner。

### 4.3 Token/step rounding

\(D=C/(6N)\) 常需 round 到 `batch×sequence×steps` 的倍数。尤其大 \(N\)/小 \(D\) 时，
rounding error 占比可观。拟合必须使用实际 tokens 与实际 \(C=6ND\)，不能继续使用 target \(C\)。
不同 run 若因 timeout 未完成目标 tokens，不应伪装为同一 IsoFLOP tier 的完整点。

## 5. Shape 与复杂度账本

| 对象 | shape / 标量 | 计算或存储 |
|---|---|---|
| run matrix | \([n_C,n_N]\)；可不规则 | 约 \(n_Cn_N\) 次训练 |
| token batch | \([B,S]\) | 每 step tokens \(=BS\) |
| hidden states | \([B,S,d]\) | activation memory 随 \(BSdL\) 增长 |
| dense training | \(N,D\) 标量 | FLOPs \(\approx6ND\) |
| attention score | \([B,h,S,S]\)（概念上） | 每层 \(O(BS^2d)\) |
| power-law fit | \(n_C\) 个 optima | 拟合成本可忽略，数据成本主导 |

A3 固定 \(S=512\)，且 `total_train_tokens` 必须被 \(512\times B\) 整除。目标 \(D\) 应先做 batch-aligned rounding，再记录**实际** \(D\) 和由此得到的实际 \(C\)。

### 5.1 Unique data、repeated tokens 与 data quality

\(D\) 表示 consumed tokens，不等于 unique tokens。Data-constrained setting 中会重复 epochs；
早期重复可能仍有收益，过度重复导致 diminishing return/overfitting。可扩展 law 为 unique data
\(U\)、epochs \(e=D/U\) 的函数，而不是把所有 consumed tokens 视为独立信息
[[6]](#ref-6)。

Data quality 改变每 token 的有效信息。过滤后的 1B tokens 与随机 Web 1B tokens 不在同一
scaling surface；mixture、dedup、contamination 与 curriculum 都会改变 coefficient/exponent。
若不同规模 run 使用不同数据子集/质量，fit 混入 data effect。

### 5.2 FLOPs-optimal 与 wall-clock-optimal

实际 throughput 依 \(N\)、batch、shape、parallelism 而变。令有效速度
\(\eta(N,D)\) FLOP/s，则

\[
T_{\mathrm{wall}}=\frac{6ND}{\eta(N,D)}.
\]

FLOPs 相同不代表时间相同。小模型可能 GPU utilization 低，大模型可能通信/显存压力高。
Hosted API 或真实集群预算应拟合 completed runs 的 runtime correction，分别报告 FLOPs-optimal
与 wall-clock-optimal。

## 6. 实验设计：信息量比格点数量重要

1. 预算 \(C_i\) 在 log space 中铺开，扩大动态范围。
2. 每档至少覆盖预期 optimum 两侧；最低点落在最小/最大 \(N\) 时只能报告 bound。
3. 固定 architecture family、tokenizer、data order、评估集和 loss 定义。
4. 先校准 batch、吞吐和学习率，再铺主矩阵。
5. 对关键点做 seed 重复；不要把所有预算平均花在低价值重复上。
6. 留出确认预算，验证预测 ridge 附近的配置。

超参数公平性有两种合法但不同的口径：

- 固定 recipe：归因清楚，但可能对某些规模不公平；
- 每个规模独立调优：比较更接近“可达到的最优”，但搜索成本必须计入。

报告必须声明采用哪一种。

### 6.1 Sequential design：让下一次 run 最大化信息

不必预先铺满大网格。更高效流程：

1. 以少量 model sizes 做 throughput/LR calibration；
2. 在最低 compute tier 扫宽范围 \(N\)，找 U 形；
3. 根据当前 exponent 预测下一 tier optimum；
4. 在预测点两侧 log-spaced 补点；
5. 若 winner 落边界，优先向边界外扩；
6. 若相邻点差小于噪声，优先补 seed 而非新规模；
7. 保留预算做 held-out tier/confirmation。

这近似 active learning / optimal experimental design：每次选择最能缩小 ridge/exponent
uncertainty 的配置，而不是平均分配 GPU hours。

### 6.2 Noise 与 uncertainty

Loss noise 来源包括 initialization、data order、evaluation sample、hardware timeout 和
checkpoint selection。简单 log-linear regression 的标准误往往低估“选择 minimum”带来的不确定性。
推荐：

- 在每 tier 内对 runs/seeds 重采样；
- 每次 bootstrap 重新选择 minimum 并重拟合；
- 报 exponent 与目标规模的 95% interval；
- 做 leave-one-tier-out prediction；
- 比较 discrete/interpolated/joint-law 三种方法的 sensitivity。

外推距离可用 \(C_{\text{target}}/C_{\max,\text{observed}}\) 表示。该倍率越大，functional-form
uncertainty 通常比回归标准误更重要。

### 6.3 Run manifest 与 provenance

每个 run 至少保存 architecture、exact parameter count、tokens、target/actual FLOPs、
optimizer/schedule、batch/sequence、seed、status、runtime、final/best loss、code/data hash。
失败/OOM/timeout 不应删除：它们定义 feasible region，并防止只汇报成功点。

## 7. 与 A3 实现的映射

| 概念 | 仓库映射 | 关键行为 |
|---|---|---|
| 官方数据 | `assignment3-scaling/data/isoflops_curves.json` | 72 条、9 个 compute tiers |
| 选离散最优 | `scripts/analyze_scaling.py::select_optima` | 按 budget 分组，取 final loss 最小 run |
| token 反推 | 同函数 | `tokens = compute / (6 * parameters)` |
| 幂律拟合 | `fit_power` | 在 \((\log C,\log N)\) 上拟合 |
| 图与残差 | `plot_profiles`、`plot_power` | 同时展示观测点、外推和 residual |
| 运行证据 | `report/results/experiment_log.csv` | 配置、预算和结果可追溯 |

本仓库报告对官方 synthetic 数据得到的指数只用于验证 pipeline；RTX 6000D TinyStories proxy 的指数明显不同，恰好说明跨数据、硬件和尺度搬运指数是不可靠的。

## 8. 常见误区

- **把最低边界点当 optimum：** 它只说明真正最优点可能还在搜索区间外。
- **把 wall-clock 等同 FLOPs：** 小模型、短 run、不同 batch 的利用率不同。
- **混用 training loss 与 validation loss：** 优化目标和最终报告口径必须固定。
- **纳入 timeout 的 partial loss：** A3 正式比较只使用 completed run 的最后一个 `val_losses`。
- **不同 tokenizer 直接比较 token loss：** token 粒度不同；必要时比较 nats/byte。
- **只画拟合线，不画原始 profile：** 看不出 coverage、异常点和 winner's curse。
- **认为高 \(R^2\) 保证远距离外推：** in-domain fit 无法衡量函数形式在域外是否成立。
- **拿总参数给 MoE 算 \(6ND\)：** 训练 FLOPs 由每 token 激活参数决定（DeepSeek-V3：671B 总参 / 37B 激活、14.8T tokens），容量与成本口径必须分开。
- **把 Chinchilla 1:20 当部署指南：** 它只优化单次训练 FLOPs；Llama-3 8B 用 15T tokens（约 1875 tokens/param）换推理经济性，目标函数不同结论就不同。

## 9. Checklist

- [ ] 明确 \(N,D,C,L\) 的口径与单位。
- [ ] 记录 target 与 actual tokens/FLOPs/runtime。
- [ ] 每个 tier 的 minimum 位于扫描内部。
- [ ] 所有 run 使用可比数据、tokenizer、评估和训练 recipe。
- [ ] 失败与 timeout 保留记录但不混入正式 fit。
- [ ] 同时展示 IsoFLOP profiles、optima、拟合和 residual。
- [ ] 报告目标 compute 相对最大观测 compute 的倍数。

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| 每档 winner 都在最小模型 | grid 边界 | 缺更小 \(N\) |
| 每档 winner 都在最大模型 | grid 边界 | 缺更大 \(N\) |
| profile 不呈 U 形 | LR/steps/data | undertrained/不公平 recipe |
| \(a+b\ne1\) 很多 | actual \(C,D\) | rounding/口径/fit 错 |
| residual 有系统曲率 | regime/function | floor/幂律失配 |
| exponent 对单点敏感 | tier 数/动态范围 | leverage 太高 |
| wall-clock 与 FLOPs 排名不同 | utilization | shape/compile/communication |
| loss 异常低 | contamination/eval | train-val 泄漏 |
| 大模型 timeout | runtime estimate | max_runtime 太短 |
| bootstrap interval 过窄 | 重采样层级 | 未重做 winner selection |

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| 拟合残差系统性偏大 | log-space 与噪声加权 | 异方差未处理、未按 loss 方差加权 |
| 最优 \(N\) 落在网格边界 | 实验点布局 | 边界 optimum 不可信，须扩网格 |
| IsoFLOP 最低点不稳定 | 离散 minimum 与 seed 方差 | 单 seed 噪声、未做 bootstrap |
| 外推预测严重偏差 | 函数形式风险 | 幂律远离支撑集外推 |
| 各点之间不可比 | recipe/tokenizer 一致性 | optimizer、lr schedule、词表不一致 |
| \(C\approx6ND\) 对不上 | embedding 与短序列修正 | \(l_v{\neq}d\) 模型、序列太短时近似失效 |

## 10. 讨论：效度威胁与结论边界

### Construct validity
- \(6ND\) 是 dense training 近似，不是 wall-clock 或 energy；
- training/validation/downstream loss 不是同一指标；
- non-embedding/total/active parameters 不可混用；
- consumed tokens 不等于 unique/high-quality tokens。

### Internal validity
- 不同规模超参/tuning budget 不等造成偏差；
- winner selection、checkpoint selection 和 timeout 造成选择偏差；
- rounding 后仍用 target compute；
- eval noise/seed 未传播到 exponent。

### External validity
- 一个 architecture/data/tokenizer 的指数不普适；
- 小规模 regime 不保证延伸数百倍 compute；
- synthetic/API proxy 不代表真实集群或数据；
- FLOPs-optimal 不代表 inference/cost/carbon optimal。

推荐结论应包含 observed range、target/observed ratio、confidence/sensitivity、boundary points
和 recipe；避免把单一 exponent 写成普遍定律。

## 面试要点速记

**高频问题与答题要点**

1. **Q：Kaplan 与 Chinchilla 的关键差异是什么？** 要点：Kaplan 用固定长度的
   lr schedule，短训练的小模型被系统性欠训练，得出“参数优先”结论；Chinchilla
   让 schedule 与训练 token 匹配，并用 isoFLOP 得到 \(N\propto D\) 近似线性
   （**1:20** tokens/params）。
2. **Q：isoFLOP 方法怎么做？** 要点：固定 \(C=6ND\) 枚举 (N,D) 组合，扫 N 找
   loss 最低点；多条等预算曲线的最低点连线即 compute-optimal 轨迹。
3. **Q：为什么拟合要在 log-space 且加权？** 要点：loss 噪声随规模下降
   （异方差），log-space 才能跨数量级同量纲比较残差。
4. **Q：最优解落在网格边界怎么办？** 要点：这是实验设计缺陷的证据，必须扩
   网格重跑后才能下结论。
5. **Q：重复数据还有多少价值？** 要点：data-constrained scaling（Muennighoff
   等）显示重复训练最多约 4 epochs 价值接近新鲜数据，之后边际收益急剧下降；
   高质量 token 成为新瓶颈，过滤/去重/配比（FineWeb/DCLM 路线）重要性上升。
6. **Q：为什么 Llama-3 8B 训到约 1875 tokens/param，远超 Chinchilla 的约 20？**
   要点：Chinchilla 只优化单次训练 FLOPs；部署侧推理成本随服务量累积，
   "过训练"用一次性训练开销换长期推理经济性。
7. **Q：MoE 模型的 \(6ND\) 怎么算？** 要点：训练 FLOPs 由每 token 激活参数决定
   （DeepSeek-V3：671B 总参 / 37B 激活、14.8T tokens），容量与成本口径必须分开。
8. **Q：测试时计算算不算 scaling law？** 要点：算——o1 用 RL 优化解题过程
   （回答前完成拆解/规划/校验），R1 以纯 RL 涌现长思考；scaling 轴从预训练
   扩展到覆盖训练/推理/数据/架构/对齐/多模态的全链条资源优化。

**必背数字**

- \(C\approx6ND\)；Chinchilla 最优 N:D ≈ 1:20；联合幂律
  \(L(N,D)=E+A/N^\alpha+B/D^\beta\)（α≈0.34、β≈0.28，Chinchilla 拟合值）。

**工业界参照**

- GPT-3（2020）：175B 参数、300B tokens，证明 scale 带来 few-shot 能力。
- Chinchilla（2022）：70B 参数 + 1.4T tokens，多项评测优于 280B Gopher；计算最优点 D*/N* ≈ 20 tokens/param（教学口径）。
- Llama-3 8B（2024）：15T tokens（约 1875 tokens/param）；405B：15T tokens、128K 上下文——"过训练"路线代表。
- DeepSeek-V3：671B 总参 / 37B 激活 MoE、14.8T tokens——稀疏激活代表。
- 数据受限（Muennighoff 等，arXiv:2305.16264）：重复训练最多约 4 epochs 价值接近新鲜数据，之后边际收益急剧下降。

## 行业现状与最新进展（2024–2026）

### Scaling law 六阶段演进（2017–2026）

scaling law 不是一条一次写成的定律，而是不断被修正的研究纲领（行业深度报告整理口径，2026-08）：

| 阶段 | 时间 | 核心问题 | 代表成果 | 一句话结论 |
|---|---|---|---|---|
| 0 误差幂律前史 | 2017 | 深度模型误差如何随规模变化 | Hestness 等 | 跨领域误差随数据/算力幂律下降，为预训练时代埋下伏笔 |
| 1 预训练幂律确立 | 2020 | loss 是否可预测 | Kaplan、GPT-3、Henighan | loss 对 N/D/C 幂律；GPT-3（175B、300B tokens）证明 scale 带来 few-shot |
| 2 能力 scaling 与涌现争论 | 2021–2022 | 规模是否带来质变 | Gopher、PaLM、Wei、Schaeffer | 参数扩张推动能力上限；Schaeffer 指出部分"涌现"是指标选择造成的假象 |
| 3 计算最优与部署经济性 | 2022–2023 | 固定算力怎么分 | Chinchilla、LLaMA | 70B Chinchilla + 1.4T tokens 多项评测优于 280B Gopher；LLaMA 以部署成本为导向开启"过训练" |
| 4 数据约束与质量 | 2023–2024 | 数据不够怎么办 | Muennighoff、Chung、FineWeb、DataComp-LM | 重复约 4 epochs 后边际收益急剧下降；过滤、去重、配比、质量上升为主轴 |
| 5 新 scaling 轴 | 2024–2026 | 除预训练外还能 scale 什么 | o1、DeepSeek-R1、Snell、DiT、MoE scaling | 测试时计算、RL reasoning、扩散 Transformer 与视频；scaling law 演化为全链条资源优化框架 |

对本讲最直接相关的是阶段 3 与阶段 4：Chinchilla 回答"固定算力下 \(N:D\) 怎么分"，Muennighoff 回答"unique data 不足时 consumed tokens 的边际价值如何衰减"。

### "过训练"时代：从 Chinchilla 最优到推理经济性

Chinchilla 的最优只针对**单次训练 FLOPs**；一旦把部署后的推理成本计入 lifetime 目标，最优点就向"更小模型 + 更多 tokens"移动。工业界代表配置：

| 模型 | 参数 | 训练 tokens | tokens/param | 训练导向 |
|---|---|---|---|---|
| GPT-3（2020） | 175B | 300B | ≈1.7 | Kaplan 时代：参数优先 |
| Chinchilla（2022） | 70B | 1.4T | ≈20 | 单次训练 FLOPs 最优 |
| Llama-3 8B（2024） | 8B | 15T | ≈1875 | 推理经济性（"过训练"） |
| DeepSeek-V3（2024） | 671B 总参 / 37B 激活 | 14.8T | ≈400（按激活参数口径） | MoE 稀疏激活解耦容量与成本 |

Llama-3 8B 的约 1875 tokens/param 是 Chinchilla 口径（约 20）的近百倍：一次性多花训练算力，换服务侧每个 token 更便宜；405B 档（15T tokens、128K 上下文）则承担另一端的高能力场景。DeepSeek-V3 展示了第三条路——总参数堆容量、激活参数控成本，14.8T tokens 的 \(6ND\) 用 37B 激活参数核算。

### 阶段 5 新轴：scaling 从预训练规律到全链条资源优化

- **MoE 稀疏激活：** 容量（总参）与每 token 成本（激活参数）解耦，\(6ND\) 的 \(N\) 必须换成激活口径。
- **测试时计算：** o1 用 RL 优化解题过程，回答前完成拆解/规划/校验；DeepSeek-R1 以纯 RL 涌现长思考——推理阶段的算力成为可优化的新预算。
- **RL reasoning：** 奖励信号驱动的能力 scaling 补充预训练 loss scaling。
- **扩散 Transformer 与视频：** DiT 把 scaling law 带入视觉/视频生成域。
- 一句话总结：scaling law 已从"预训练大模型规律"演化为覆盖训练/推理/数据/架构/对齐/多模态的全链条资源优化框架。

### 对本讲学习者的启示

第一，本讲的 IsoFLOP 方法论没有过时，反而更通用——测试时计算、MoE、数据质量轴上的最优分配，本质上都是"固定预算下在多个杠杆间求最优"，与 \(C=6ND\) 下分配 \(N,D\) 同构。第二，指数是经验参数：Llama-3 与 DeepSeek-V3 的配置说明"最优"依赖目标函数（训练 FLOPs、wall-clock 还是 lifetime 成本），面试与工程决策都应先问"在优化什么成本"。第三，数据质量与 unique data 上限正在取代参数量成为第一瓶颈，阶段 4 的结论（约 4 epochs 后重复收益急剧下降）是规划数据管线时的硬约束。

## 大厂面试真题与答题框架

以下均为高频面试题（公开面经风格），不指向特定公司的特定考题。

**题目 1：Chinchilla 与 Kaplan 的结论为何不同？谁的实验设计更可信？**
- 考点：lr schedule 与训练长度的耦合、IsoFLOP 设计、fixed recipe vs per-scale tuning。
- 答题框架：1) Kaplan（2020）：loss 对 N/D/C 幂律，GPT-3（175B、300B tokens）证明 scale 带来 few-shot；但固定长度 lr schedule 使短训练的小模型被系统性欠训练，得出"参数优先"。2) Chinchilla（2022）：schedule 与训练 token 数匹配 + 密集 IsoFLOP，得到 N 与 D 近似等比增长，D*/N* ≈ 20 tokens/param。3) 实证：70B Chinchilla（1.4T tokens）多项评测优于 280B Gopher。4) 收尾：两者不矛盾，是 recipe 与实验域不同导致指数不同。
- 加分项：提到后续复现/复分析（如 Besiroglu、Porian）指出拟合与 recipe 选择会显著影响系数；强调 exponent 是指定条件下的经验参数。
- 踩坑：把 1:20 当自然常数；忽略 lr schedule 这一 confounder；简单说"Kaplan 错了"。

**题目 2：给你 1e22 FLOPs 预算，怎么定 N 和 D？**
- 考点：\(C\approx6ND\)、Chinchilla 比例、外推流程。
- 答题框架：1) Chinchilla 口径一阶估计：D=20N 代入 C=6N·20N=120N²，得 N≈9.1B、D≈180B tokens。2) 声明口径：non-embedding 参数、tokenizer、loss 定义。3) 检查数据侧：unique 高质量数据是否够 180B；不足则进入数据受限 regime。4) 若部署导向，参照 Llama-3 8B 过训练（更小 N、更大 D）。5) 用小规模 IsoFLOP 校准本家族指数后再外推，报告外推倍率。
- 加分项：区分 FLOPs-optimal 与 wall-clock-optimal；给出"目标函数不同则 N/D 不同"的敏感性讨论。
- 踩坑：直接拿总参数（含 embedding）算；不问数据够不够就报数；把估算当结论而不给验证计划。

**题目 3：重复数据的价值曲线是怎样的？高质量数据不够时怎么办？**
- 考点：data-constrained scaling（Muennighoff 等，arXiv:2305.16264）。
- 答题框架：1) 区分 unique tokens U 与 consumed tokens D，epochs e=D/U。2) 核心结论：重复训练最多约 4 epochs 价值接近新鲜数据，之后边际收益急剧下降。3) 对策：质量过滤、去重、配比（FineWeb/DCLM 路线）提升每 token 有效信息；必要时补合成数据；或把预算转投参数/新轴。4) 拟合时把 D 与 U 分开口径，不能把重复 token 当独立信息。
- 加分项：指出数据质量改变 scaling surface，不同质量的 1B tokens 不可比；提到"高质量 token 成为新瓶颈"的行业判断。
- 踩坑：把 consumed tokens 等同于信息量；在"重复没损失"与"重复完全没用"两个极端之间选边。

**题目 4：既然 Chinchilla 给出计算最优，为什么 Llama-3 8B 用约 1875 tokens/param？**
- 考点：训练最优 vs lifetime 成本最优；"过训练"的经济学。
- 答题框架：1) Chinchilla 只优化单次训练 FLOPs。2) 部署后推理成本随服务量累积，小模型每 token 便宜。3) Llama-3 8B 用 15T tokens（约 1875 tokens/param，vs Chinchilla 约 20）一次性多花训练算力换长期推理节省。4) 结论：最优 D/N 是预期服务规模的函数，不是常数。
- 加分项：定性给出"服务 token 量越大，最优点越向小模型大 tokens 移动"；提到 405B 档（15T tokens、128K 上下文）承担高能力场景。
- 踩坑：把 Chinchilla 当部署指南；忽略过训练同样有边际收益递减与数据约束。

**题目 5：MoE 模型的 \(6ND\) 怎么算？scaling law 口径要注意什么？**
- 考点：总参数 vs 每 token 激活参数。
- 答题框架：1) dense 的 N 指 non-embedding 参数。2) MoE 每 token 只激活专家子集，训练 FLOPs 由激活参数决定，容量由总参决定。3) 实例：DeepSeek-V3 671B 总参 / 37B 激活、14.8T tokens。4) 拟合与汇报要分开两个口径，或使用 routed/MoE 扩展 law。
- 加分项：链接本讲"参数口径"一节；指出混用口径会污染拟合截距。
- 踩坑：拿 671B 总参算 \(6ND\)；只报总参不报激活。

**题目 6：怎么评价"涌现能力"的证据？**
- 考点：Wei 涌现 vs Schaeffer 指标假象。
- 答题框架：1) 涌现：能力随规模非线性跃迁，是 Gopher/PaLM 时代的争论焦点。2) Schaeffer 等指出：非线性/不连续指标（如 exact-match 准确率）会在平滑的底层改进上制造"突变"假象。3) 方法论：换连续指标复检、审查指标定义、检查统计显著性。4) 与 scaling law 兼容：loss 平滑下降不排斥某些下游指标的陡峭改善。
- 加分项：能说明"同一能力、不同指标、结论相反"的具体案例逻辑。
- 踩坑：直接断言"涌现是真/假"；只用一个指标下结论。

**题目 7：如何设计实验验证一个 scaling law 猜想？**
- 考点：IsoFLOP 矩阵、sequential design、bootstrap、边界 optimum。
- 答题框架：1) 固定 architecture family、tokenizer、数据、评估口径。2) 预算在 log space 铺开，每档覆盖预期最优两侧。3) 先做 throughput/LR 校准再铺主矩阵。4) Sequential 补点：最低档扫宽找 U 形，按当前 exponent 预测下一档。5) Bootstrap 时重做 winner selection；报告 exponent 区间、held-out tier 验证与外推倍率。
- 加分项：winner's curse；leave-one-tier-out；声明 fixed recipe vs per-scale tuning 口径。
- 踩坑：winner 落网格边界仍当精确 optimum；平均撒点浪费预算；只报拟合线不报原始 profile。

## 系统设计题

**设计题 1：为一家公司规划下一代基座模型（预算约 1e23 FLOPs）**
- 需求澄清：优化目标是单次训练 FLOPs 还是 lifetime 成本？预期服务 token 量级？unique 高质量数据可用量？是否要长上下文/多模态？集群实际利用率？
- 规模估算：Chinchilla 口径 N*≈29B、D*≈580B tokens（由 N=sqrt(C/120)、D=20N）；推理导向可取约 10B 模型 + 约 1.7T tokens（约 170 tokens/param，介于 Chinchilla 的 20 与 Llama-3 8B 的 1875 之间）。
- 架构：数据管线（FineWeb/DCLM 风格过滤、去重、配比）→ 小规模 IsoFLOP 校准本家族指数 → 主训练 → 下游评测与服务成本核算。
- trade-off 表：

| 方案 | 训练成本 | 推理成本/token | 数据需求 | 主要风险 |
|---|---|---|---|---|
| Chinchilla 最优（≈29B/580B） | 基准 | 高 | 580B unique 高质量 | 小规模指数外推失真 |
| 过训练（≈10B/1.7T） | 同预算 | 显著更低 | 数据压力大；不足时重复 ≤4 epochs | 重复收益衰减、能力上限略低 |
| MoE（小激活/大总参） | 同预算 | 中 | 同上 | 路由稳定性、口径汇报复杂 |

- 评测方案：held-out loss、下游 benchmark、每百万 token 服务成本；正式训练前用 held-out tier 验证 loss 预测。
- 追问预案：数据不够 → 重复（≤4 epochs）+ 质量过滤 + 合成数据；预算砍半 → 降 tier 重拟合并重报外推倍率；被问"为什么不信 Chinchilla 20" → 答"目标函数含推理成本，参照 Llama-3 8B 约 1875 tokens/param"。

**设计题 2：设计 IsoFLOPs 实验矩阵（对齐 A3）**
- 需求澄清：目标外推倍率（C_target/C_max）？tier 数与每档点数？单 run 时数上限？可承受 seed 数？
- 规模估算：参照 A3 官方数据 72 条 runs、9 个 compute tiers（每档约 8 个规模）；S=512，total_train_tokens 须被 512×B 整除。
- 架构：tier 在 log C 等距铺开；每档 N 覆盖预期最优两侧至少半个 decade；最低档先扫宽找 U 形，再按当前 exponent 预测下一档（sequential design）。
- trade-off 表：

| 策略 | runs 数 | 信息量 | 风险 |
|---|---|---|---|
| 全网格一次铺满 | 多 | 低 | 预算浪费在低价值点 |
| sequential + 边界外扩 | 少 | 高 | 依赖预测，需保留确认预算 |
| 每档二次插值 minimum | 少 | 中 | 引入"局部二次"假设 |

- 评测方案：b+d≈1 sanity check；discrete/interpolated/joint-law 三法对比；bootstrap 重做 winner selection；residual 检查系统曲率；报告外推倍率。
- 追问预案：winner 落边界 → 只报 bound 并扩网格；相邻点差小于噪声 → 补 seed 而非新规模；timeout run → 保留记录但不入正式 fit。

**设计题 3：数据受限场景的数据策略（unique 高质量数据仅 300B tokens，目标是消费 1T+ tokens）**
- 需求澄清：300B 的质量分布与来源？允许的合成数据比例？下游任务重点？质量过滤的算力预算？
- 规模估算：按"重复约 4 epochs 内价值接近新鲜数据"（Muennighoff 等），300B×4≈1.2T consumed tokens 是接近新鲜价值的量级上限；再往上边际收益急剧下降。
- 架构：质量轴优先（过滤、去重、配比，FineWeb/DCLM 路线）→ epochs 上限约束（约 4）→ 合成数据与课程补充 → 剩余预算转投参数或测试时计算。
- trade-off 表：

| 策略 | 有效信息 | 额外成本 | 风险 |
|---|---|---|---|
| 重复至约 4 epochs | 接近新鲜 | 低 | 超过 4 epochs 收益急剧下降 |
| 更强质量过滤（牺牲数量） | 每 token 信息上升 | 过滤算力 | 可用 token 总量下降 |
| 合成数据补充 | 量级补充 | 生成 + 校验成本 | 分布偏移、模式坍缩 |

- 评测方案：不同 epochs/过滤强度下的 val loss 曲线；held-out 域评测防泄漏；下游 benchmark 对照。
- 追问预案：被问"为什么不多重复几次" → 引约 4 epochs 后边际收益急剧下降的结论；被问"合成数据可信吗" → 答需 held-out 校验与配比实验，不能无条件信任。

## 代码实现题

**代码实现题 1：幂律 loss 拟合（log-log 线性回归 + Huber）**
- 题目：给定观测 (x_i, L_i)，拟合 \(L(x)=L_\infty+A x^{-\alpha}\)，要求对离群点稳健。
- 考察点：floor 处理（不能整体取 log）、log-space 回归、Huber 加权 IRLS。

```python
import numpy as np

def fit_power_law(x, y, floor=0.0, delta=1.0, iters=100):
    """拟合 y = floor + A * x**(-alpha)：扣 floor 后 log-log 线性回归 + Huber IRLS。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.any(y <= floor):
        raise ValueError("存在低于 floor 的观测：floor 或数据有问题")
    logx, logy = np.log(x), np.log(y - floor)
    theta = np.array([np.mean(logy), 0.5])       # [logA, alpha] 初值
    for _ in range(iters):
        r = logy - (theta[0] - theta[1] * logx)  # 残差
        w = np.where(np.abs(r) <= delta, 1.0, delta / np.abs(r))
        X = np.stack([np.ones_like(logx), -logx], axis=1)
        theta_new = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * logy))
        if np.max(np.abs(theta_new - theta)) < 1e-10:
            theta = theta_new
            break
        theta = theta_new
    return float(np.exp(theta[0])), float(theta[1])  # (A, alpha)
```

- 验收标准：合成数据（已知 A、α）恢复误差 <1%；注入 5% 强噪声后 α 变化 <5%（Huber 生效）；floor>0 时不整体取 log；对非正值输入给出明确报错。

**代码实现题 2：Chinchilla 最优 N/D 求解器**
- 题目：给定预算 C，输出 Chinchilla 口径的 (N*, D*)，并支持联合 loss law 的解析最优。
- 考察点：\(C\approx6ND\)、D*=20N 的代数、联合 law \(L=E+A/N^\alpha+B/D^\beta\) 固定 C 的最优推导。

```python
import math

def chinchilla_optimal(C, ratio=20.0, coeff=6.0):
    """D = ratio*N 且 C = coeff*N*D  =>  N = sqrt(C/(coeff*ratio))。"""
    if C <= 0:
        raise ValueError("C must be positive")
    N = math.sqrt(C / (coeff * ratio))
    D = ratio * N
    assert abs(coeff * N * D - C) / C < 1e-12
    return N, D

def joint_law_optimal(C, A, B, alpha, beta, coeff=6.0):
    """L = E + A/N^alpha + B/D^beta 固定 C 的解析最优：
    N**(alpha+beta) = (alpha*A/(beta*B)) * (C/coeff)**beta。"""
    if min(alpha, beta) <= 0:
        raise ValueError("alpha/beta must be positive")
    N = ((alpha * A) / (beta * B) * (C / coeff) ** beta) ** (1.0 / (alpha + beta))
    D = C / (coeff * N)
    return N, D
```

- 验收标准：C=1e22 时 `chinchilla_optimal` 返回 N*≈9.1e9、D*≈1.8e11；`joint_law_optimal` 的隐含指数 b=β/(α+β)、d=α/(α+β)，取 α≈0.34、β≈0.28 时 b≈0.45、d≈0.55 且 b+d=1；对非法输入抛错。

**代码实现题 3：IsoFLOP 最优点选择器（含边界检测）**
- 题目：从一批 runs（参数量、实际 tokens、final loss、状态）中按 compute tier 选最优点，并标记边界 optimum。
- 考察点：用实际 tokens 反推 C、completed 过滤、边界只能报 bound。

```python
import math

def select_isoflop_optima(runs, coeff=6.0):
    """runs: [{parameters, tokens, final_loss, status}, ...]
    按 compute tier 分组取 completed 最低 loss；winner 在扫描边界时置 boundary=True。"""
    tiers = {}
    for r in runs:
        if r["status"] != "completed":
            continue                                # timeout/OOM 不入正式 fit
        C = coeff * r["parameters"] * r["tokens"]   # 用实际 tokens，而非 target
        tiers.setdefault(round(math.log10(C), 1), []).append(r)
    optima = []
    for _, group in sorted(tiers.items()):
        best = min(group, key=lambda r: r["final_loss"])
        sizes = sorted(g["parameters"] for g in group)
        optima.append({
            "N_opt": best["parameters"],
            "D_opt": best["tokens"],
            "loss": best["final_loss"],
            "boundary": best["parameters"] in (sizes[0], sizes[-1]),
        })
    return optima
```

- 验收标准：boundary=True 时调用方只能报 \(N_{\mathrm{opt}}\) 的 bound 而非精确值；C 由实际 tokens 计算（目标与实际不一致时以实际为准）；未完成 run 不参与选择但保留在实验日志中。

## 11. 结论与本讲小结

\(C\approx6ND\) 把模型规模和数据预算连接起来，IsoFLOP 则把“固定算力如何分配”变成可实验的问题。可靠结论依赖内部最优点、统一口径和足够动态范围，而不只是一条双对数直线。Lecture 11 将把离散 envelope 扩展为联合 loss law，并重点处理 bootstrap、诊断与外推不确定性。

## 参考文献

<a id="ref-1"></a>[1] J. Kaplan et al. “Scaling Laws for Neural Language
Models.” arXiv:2001.08361, 2020. https://arxiv.org/abs/2001.08361

<a id="ref-2"></a>[2] J. Hoffmann et al. “Training Compute-Optimal Large
Language Models.” arXiv:2203.15556, 2022. https://arxiv.org/abs/2203.15556

<a id="ref-3"></a>[3] T. Henighan et al. “Scaling Laws for Autoregressive
Generative Modeling.” arXiv:2010.14701, 2020.
https://arxiv.org/abs/2010.14701

<a id="ref-4"></a>[4] Y. Bahri et al. “Explaining Neural Scaling Laws.”
arXiv:2102.06701, 2021. https://arxiv.org/abs/2102.06701

<a id="ref-5"></a>[5] I. Alabdulmohsin, B. Neyshabur, X. Zhai.
“Revisiting Neural Scaling Laws in Language and Vision.” *NeurIPS*, 2022.
https://arxiv.org/abs/2209.06640

<a id="ref-6"></a>[6] N. Muennighoff et al. “Scaling Data-Constrained Language
Models.” *NeurIPS*, 2023. https://arxiv.org/abs/2305.16264

<a id="ref-7"></a>[7] T. Porian et al. “Resolving Discrepancies in
Compute-Optimal Scaling of Language Models.” arXiv:2406.19146, 2024.
https://arxiv.org/abs/2406.19146

<a id="ref-8"></a>[8] T. Besiroglu et al. “Chinchilla Scaling:
A Replication Attempt.” arXiv:2404.10102, 2024.
https://arxiv.org/abs/2404.10102

<a id="ref-9"></a>[9] A. Clark et al. “Unified Scaling Laws for Routed
Language Models.” *ICML*, 2022. https://arxiv.org/abs/2202.01169

<a id="ref-10"></a>[10] D. Hernandez et al. “Scaling Laws for Transfer.”
arXiv:2102.01293, 2021. https://arxiv.org/abs/2102.01293

## 延伸阅读与复现材料

- Stanford CS336, [Lecture 9 — Scaling Laws](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf).
- [A3 官方导读与本地实现](../experiments/official/a3-scaling.md).
- [Scaling Laws 主题导航](../experiments/topics/scaling-laws.md).
- 本仓库：[A3 IsoFLOP 报告](../assignments/spring2026/assignment3-scaling/report/writeup.pdf).
- [Muennighoff et al., Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)（访问日期 2026-10-04）.
- [Llama 3: The Herd of Models](https://arxiv.org/abs/2407.21783)（访问日期 2026-10-04）.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)（访问日期 2026-10-04）.
- [Stanford CS336 课程主页](https://cs336.stanford.edu)（访问日期 2026-10-04）.
