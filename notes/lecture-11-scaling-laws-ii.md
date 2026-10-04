---
title: "Lecture 11 — Scaling Laws II"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-05-04"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_11.pdf"
  - "../experiments/topics/scaling-laws.md"
  - "../assignments/spring2026/assignment3-scaling/report/main.tex"
---

# Lecture 11 — Scaling Laws II：联合拟合、不确定性与外推

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
- 适用对象：自学者、ML Systems 工程师、Scaling Laws 研究者

## 摘要

Scaling law 的价值不在于把若干点连成直线，而在于用有限 pilot runs 支持高成本训练决策。
本文从 Kaplan-style parameter-dominant scaling 与 Chinchilla compute-optimal scaling 的差异
出发，系统推导联合 loss surface、固定 compute 下的最优模型/数据配置及其 loss exponent；
继而讨论 IsoFLOP envelope、非线性拟合、参数不可识别、cluster bootstrap、profile
likelihood、leave-one-tier-out、模型错设和远距离外推。工程部分将理论 FLOPs 连接到
wall-clock、MFU、数据质量、训练失败和预算配置；研究部分给出可证伪假设、顺序实验设计、
不确定性分层和可复现报告规范。核心结论是：scaling law 是条件于 architecture、data、
tokenizer、optimizer 与实验范围的决策模型，而不是跨系统不变的自然定律。

**关键词：** Neural Scaling Laws；Compute-optimal Training；Chinchilla；
IsoFLOP；Nonlinear Regression；Uncertainty Quantification；Extrapolation；
Experimental Design

## 本文贡献

1. 从联合 law 完整推导 \(N_{\mathrm{opt}},D_{\mathrm{opt}},L_{\mathrm{opt}}\)；
2. 区分统计噪声、参数不确定性、方法敏感性与分布外模型风险；
3. 比较 Kaplan、Chinchilla、broken/saturating laws 的假设和适用范围；
4. 给出从 pilot grid 到大规模训练决策的工程闭环；
5. 提供论文级 diagnostics、ablation、复现实验和负结果报告模板。

## 学习目标

- 从联合 loss law 推导 compute-optimal \(N(C)\) 与 \(D(C)\)。
- 比较 IsoFLOP envelope 与联合拟合的假设、信息利用率和失败模式。
- 用 bootstrap、leave-one-tier-out 和 residual 诊断传播不确定性。
- 把统计区间与远距离外推的模型风险分开陈述。
- 判断某个 scaling 结论是否被实验覆盖，而非只看 \(R^2\)。
- 将理论最优点修正为受 wall-clock、memory、data quality 约束的可执行配置。

## 先修知识

- Lecture 09：IsoFLOP 曲线、幂律拟合与实验设计（本讲的直接前置）。
- 统计基础：非线性最小二乘、参数可辨识性、profile likelihood 与 cluster bootstrap。

## 相关工作与问题演进

神经网络经验 scaling 研究可追溯到学习曲线和幂律泛化误差。Hestness et al. 系统展示多个
领域中性能随数据、模型和 compute 的 power-law 趋势 [[1]](#ref-1)；Kaplan et al. 在语言
模型上提出参数、数据、compute 的可分 scaling 描述，并得到相对 parameter-heavy 的训练
建议 [[2]](#ref-2)。Hoffmann et al. 通过更密集的 IsoFLOP 实验指出，固定 compute 下模型
与 tokens 应更均衡扩张，即 Chinchilla scaling [[3]](#ref-3)。后续工作研究了函数形式：
Alabdulmohsin et al. 讨论饱和与可迁移参数化 [[4]](#ref-4)，Caballero et al. 用 broken
power laws 刻画 regime transition [[5]](#ref-5)。

Scaling 还被用于 downstream transfer [[6]](#ref-6)、数据重复与有限数据
[[7]](#ref-7)、数据质量/过滤 [[8]](#ref-8)，以及预测 emergent abilities 的争论
[[9]](#ref-9)[[10]](#ref-10)。这些结果共同表明：loss 往往比离散 downstream metric
平滑，但任一指数都依赖观测区间、训练 recipe 和测量定义。

## 1. 联合 scaling law

Chinchilla 常用的可解释形式为

\[
\boxed{L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}},
\]

其中所有 \(A,B,\alpha,\beta>0\)：

- \(E\)：在当前数据分布/tokenizer/loss 下的渐近 floor；
- \(A/N^\alpha\)：有限模型容量项；
- \(B/D^\beta\)：有限数据项。

它用全部 runs，而不是只用每档 winner。好处是数据效率高、可直接预测 loss surface；代价是函数形式假设更强，\(E,A,\alpha,B,\beta\) 在窄范围内可能不可辨识。

## 2. 关键推导：固定 compute 的闭式最优解

在约束

\[
C=6ND
\]

下令 \(D=C/(6N)\)，则

\[
L(N\mid C)
=E+A N^{-\alpha}
+B\left(\frac{6N}{C}\right)^\beta.
\]

对 \(N\) 求导并令其为零：

\[
-\alpha A N^{-\alpha-1}
+\beta B6^\beta C^{-\beta}N^{\beta-1}=0.
\]

整理：

\[
N^{\alpha+\beta}
=\frac{\alpha A}{\beta B6^\beta}C^\beta.
\]

因此

\[
\boxed{
N_{\mathrm{opt}}(C)
=
\left(\frac{\alpha A}{\beta B6^\beta}\right)^{1/(\alpha+\beta)}
C^{\beta/(\alpha+\beta)}
}
\]

以及

\[
\boxed{
D_{\mathrm{opt}}(C)
=
\left(\frac{\beta B}{\alpha A6^\alpha}\right)^{1/(\alpha+\beta)}
C^{\alpha/(\alpha+\beta)}
}.
\]

两个 compute 指数之和为 1。若 \(\alpha\approx\beta\)，模型与数据近似按 \(C^{1/2}\) 同速扩张；这不是预先假定的固定 tokens-per-parameter，而是联合 law 与成本约束的结果。

### 2.1 最优 loss 的 compute exponent

一阶条件还给出最优点两项的比例：

\[
\alpha A N_{\mathrm{opt}}^{-\alpha}
=\beta B D_{\mathrm{opt}}^{-\beta}.
\]

两项不必数值相等；其比值由 \(\alpha/\beta\) 决定。代回可得

\[
L_{\mathrm{opt}}(C)-E
=K C^{-\gamma},\qquad
\boxed{\gamma=\frac{\alpha\beta}{\alpha+\beta}},
\]

其中 \(K\) 吸收 \(A,B,6\) 和指数常数。这连接了 loss surface 与 IsoFLOP envelope：在理想
可加 law 和连续最优假设下，envelope 也应是带 floor 的 power law。若观测 envelope 的
斜率与 joint fit 推导的 \(\gamma\) 明显不一致，需检查 finite grid、优化失败或模型错设。

### 2.2 一般化 compute 模型

\(C\approx6ND\) 假设 dense Transformer、训练 token 数 \(D\) 和近似稳定的 forward/backward
比例。若实际成本是

\[
C=kN^pD^q,
\]

则最优扩张指数将改变。MoE 中 total parameters、active parameters 和 routing/communication
成本不同；长上下文还增加 attention 的 \(S^2\) 项。因此报告必须说明 \(N\) 是 total 还是
active parameters，\(D\) 是否包含重复 tokens，\(C\) 是理论 FLOPs 还是 profiler 计数。

### 2.3 Tokens-per-parameter 不是普适常数

\[
\frac{D_{\mathrm{opt}}}{N_{\mathrm{opt}}}
\propto C^{(\alpha-\beta)/(\alpha+\beta)}.
\]

只有 \(\alpha=\beta\) 时，该比例才与 compute 无关。实践中的“20 tokens/parameter”是特定
实验范围和 recipe 下的经验近似，不应机械迁移到新 tokenizer、data mixture、context
length、optimizer 或 continual pretraining。

## 3. IsoFLOP 与 joint law 是两种估计器

| 维度 | IsoFLOP envelope | Joint loss law |
|---|---|---|
| 使用数据 | 每档一个 minimum | 全部 \((N,D,L)\) |
| 主要假设 | optimum 随 \(C\) 为 power law | loss 可加分解为容量项+数据项 |
| 优点 | 直观、容易看 coverage | 样本效率高、直接预测 loss |
| 弱点 | 丢弃大量点、离散 winner bias | 非线性、参数耦合、模型错设 |
| 诊断 | profile 是否 U 形/内部最优 | residual surface、参数稳定性 |

两者不必被“强行拟合到一致”。差异本身可揭示有限网格、超参数 confound 或函数形式失配。实务上应把二者作为 sensitivity analysis。

### 3.1 Kaplan 与 Chinchilla 为何不同

差异不能简化为“旧论文错、新论文对”。潜在原因包括实验覆盖范围、固定/调优 learning
rate schedule、训练是否充分、数据重复、tokenizer、模型定义和 optimum 提取方法。
Kaplan-style 结论偏向更大模型、更少 tokens；Chinchilla 的 IsoFLOP sweep 对固定 compute
直接寻找 minimum。复现时应在同一代码、数据和 optimizer 条件下比较，而非跨论文抄指数。

### 3.2 替代函数形式

可比较以下 nested/non-nested candidates：

\[
\begin{aligned}
&\text{floor power: } L(x)=E+Ax^{-\alpha},\\
&\text{shifted power: } L(x)=E+A(x+x_0)^{-\alpha},\\
&\text{broken power: } \log L=f_{\text{piecewise}}(\log x),\\
&\text{interaction: }L(N,D)=E+A N^{-\alpha}+B D^{-\beta}
 +G N^{-\alpha}D^{-\beta}.
\end{aligned}
\]

增加自由度总能改善 in-sample error；应以 held-out tier prediction、AIC/BIC（仅在误差假设
合理时）、parameter stability 和 extrapolation divergence 比较。若候选模型在观测域内
近似等价、域外分叉，就应报告 model ensemble/range，而不是宣称已识别唯一 law。

### 3.3 Emergence 与测量尺度

连续 cross-entropy 改善映射到 exact-match、pass@k 或阈值能力时，可出现视觉上的骤变。
Schaeffer et al. 指出部分 emergence 源于非线性/离散 metric [[9]](#ref-9)，Wei et al.
则强调跨任务的大模型能力跃迁 [[10]](#ref-10)。严谨分析应同时绘制 loss、连续概率型指标
与离散任务指标，并区分“底层能力突变”和“measurement threshold crossing”。

## 4. 稳健拟合

### 参数化

直接优化正参数容易跑到负值或退化解。可令

\[
A=e^a,\ B=e^b,\ \alpha=e^u,\ \beta=e^v
\]

并对 \(E\) 设置合理 bounds；或使用 bounded nonlinear least squares。多初值优化比相信一次局部最优更可靠。

### 目标函数

普通平方误差默认各 run 同方差。若不同规模的 loss 噪声不同，可用 seed 方差做权重；存在失败 run 或 outlier 时比较 Huber loss。不要在没有依据时同时变换 target、删点和改权重以追求漂亮曲线。

### 识别性

若观测远离 floor，很多 \(E,\alpha\) 组合都能给出相似预测。应查看：

- Hessian/profile likelihood 是否平坦；
- 多初值是否得到不同参数但相似 RMSE；
- 去掉一个 compute tier 后参数是否剧烈变化；
- 参数区间是否碰 bounds。

预测可能比单个参数更稳定，也可能相反；两者都要检查。

### 数值实现与可辨识性诊断

对 \(N,D\) 先按几何均值归一化，避免量级导致 Jacobian 病态。保存每次 initial point、
termination reason、gradient norm 与 Hessian condition number。建议同时执行：

- **profile likelihood**：固定一个参数，重拟合其余参数；
- **parameter correlation**：检查 \(E\)-\(\alpha\)、\(A\)-\(\alpha\) 耦合；
- **synthetic recovery**：用已知参数生成与真实 grid 相同的数据，验证 estimator；
- **perturbation test**：轻微改变 loss/compute 后预测不应灾难性跳变；
- **boundary audit**：若最优点或参数频繁碰 bounds，区间不应按普通渐近正态解释。

### 误差模型

Run loss 并非同方差 iid Gaussian：共享数据顺序、初始化族或同一 compute tier 会产生相关；
大模型的 seed 方差可能更小，但训练失败形成重尾。可比较 weighted least squares、Huber/
Student-\(t\) likelihood 与 hierarchical model。权重必须来自重复实验或预先规则，不能用
拟合后的 residual 反复调权制造自洽。

## 5. Bootstrap 必须重跑完整 pipeline

目标不是只估计最后一条回归线的误差，而是传播：

1. run/seed 噪声；
2. 每档 winner 选择；
3. scaling fit；
4. 目标 compute 外推。

推荐以 compute tier 为 cluster 重采样；若每 tier 有独立 seeds，可先在 tier 内按 seed 重采样。对每个 bootstrap replicate：

1. 重采样 cluster；
2. 重新选择每档 optimum；
3. 重新拟合参数；
4. 重新计算目标 \(N,D,L\)；
5. 保存预测与边界/失败标记。

分位数给出 bootstrap interval。但它只覆盖观测生成机制与拟合模型内的波动，不覆盖“未来架构/数据/优化器不同”或 law 形状错误。

本仓库 `bootstrap_power` 只对 optima pairs 重采样，适合演示末端回归不确定性；严格报告应将 `select_optima` 放进每个 replicate，并按 tier/seed 保留相关结构。

### 5.1 四层不确定性

1. **Aleatoric/run noise**：seed、minibatch order、measurement；
2. **Estimator uncertainty**：有限 grid 下拟合参数和 optimum selection；
3. **Specification uncertainty**：joint/IsoFLOP、floor/broken law、loss function；
4. **Distribution shift**：新规模下 optimizer、data、hardware/architecture 改变。

Bootstrap 主要覆盖前两层；跨方法 spread 可部分呈现第三层；第四层无法由同分布重采样解决。
报告不应把 bootstrap 95% interval 命名为“总不确定性”。

### 5.2 Bootstrap、Bayesian 与 conformal 的边界

Bayesian posterior 能编码 prior 并联合传播参数，但 posterior interval 仍条件于 likelihood 和
函数形式；弱识别时 prior 会显著影响远端预测。Conformal prediction 依赖 exchangeability，
远距离 extrapolation 通常不满足。任何方法都不能凭统计技巧创造观测范围外的信息。

## 6. 外推诊断

### Residual

\[
r_i=L_i-\hat L(N_i,D_i)
\]

分别对 \(\log N,\log D,\log C\)、seed、batch、learning rate 作图。结构性曲线、扇形异方差、某一规模全偏正都不是 iid noise。

### Leave-one-tier-out

每次留出整个 \(C_i\)，在其余 tiers 拟合，再预测被留出的 profile/optimum。它比随机留出单个 run 更接近目标任务，因为同 tier runs 高度相关。

### Extrapolation distance

报告

\[
\rho_C=\frac{C_{\text{target}}}{C_{\max,\text{observed}}}.
\]

本仓库官方 synthetic 分析外推到 \(10^{24}\) FLOPs 时距离最大观测预算约 333 倍。即使 in-sample RMSE 很小，函数形式风险仍可能主导。

### Sensitivity

至少比较：

- 离散 winner 与 profile interpolation；
- IsoFLOP 与 joint law；
- 含/不含边界 optimum；
- 不同 loss/bounds/初值；
- 最大一档有/无；
- 理论 FLOPs 与 runtime-corrected compute。

如果点估计跨方法变化远大于 bootstrap interval，应报告方法差异，而不是挑最顺眼的结果。

## 7. Shape 与复杂度

设 \(R\) 个 runs、\(P\) 个参数、\(K\) 次初值、\(B_s\) 次 bootstrap：

| 阶段 | 数据 shape | 复杂度 |
|---|---|---:|
| run manifest | \([R,\text{fields}]\) | \(O(R)\) 清洗 |
| joint design/evaluation | \(N,D,L:[R]\) | 每 objective \(O(R)\) |
| nonlinear fit | \(P=5\) | 约 \(O(KI RP^2)\) |
| cluster bootstrap | \(B_s\) 份数据 | 约 \(B_s\) 倍完整 fit |
| target prediction | \([B_s,n_{\text{target}}]\) | \(O(B_sn_{\text{target}})\) |

拟合计算几乎总比训练便宜，因此没有理由省略多初值、bootstrap 和 held-out diagnostics。真正稀缺的是实验预算和覆盖范围。

## 8. A3 实现映射

| 方法 | 本仓库实现 | 说明 |
|---|---|---|
| power fit | `scripts/analyze_scaling.py::fit_power` | log-log `polyfit` |
| tier winner | `select_optima` | 官方题面建议取离散 minimum |
| bootstrap | `bootstrap_power` | 固定 RNG seed，预测 \(10^{23},10^{24}\) |
| joint law | `fit_joint` | 扫描 \(\alpha,\beta\)，线性求 \(E,A,B\) |
| residual | `official_power_residuals` | 检查 log-parameter residual |
| provenance | `manifest.sha256`、raw JSON | 输入和资产可追溯 |
| proxy 验证 | `scripts/analyze_proxy.py` | 展示边界 winner 与 finite-grid bias |

`fit_joint` 的网格扫描实现简单、可复现，但线性 least squares 未强制 \(E,A,B>0\)，也没有系统不确定性估计。它适合教学基线，不应被误认为生产级统计拟合。

## 9. 从 FLOPs law 到 wall-clock 决策

A3 的探索预算和最终预算以 B200-hours 计，而模型 law 以 FLOPs 计。可建立

\[
t_{\text{wall}}(N,D,\theta)
\approx\frac{6ND}{\operatorname{effective\ FLOP/s}(N,B,S,\theta)}
+t_{\text{overhead}},
\]

其中 \(\theta\) 包含 dtype、batch、shape、编译和系统配置。小 run 的启动/评估开销占比高；不同宽深比的 MFU 也不同。最终选择应在 wall-clock 可行域上优化预测 loss，而不是先假设所有配置 FLOP/s 相同。

若模型部署后会被大量调用，训练 compute-optimal 也未必是生命周期成本最优；更小、训练
更多 tokens 的模型可能增加训练成本却降低长期 inference 成本 [[11]](#ref-11)。

### 9.1 数据质量、重复与有效 tokens

原始 token count 不等于信息量。重复数据在早期可能有用，重复次数过高会降低 marginal
return，并改变 \(D^{-\beta}\) 假设 [[7]](#ref-7)。过滤改善平均质量也可能损害领域覆盖，
改变 evaluation distribution；大规模 data benchmark 也显示数据 recipe 是独立研究变量
[[12]](#ref-12)。应把 data mixture/filtering 作为条件变量，至少记录：

- corpus/version、dedup level、quality classifier 与阈值；
- domain/language/source mixture 和 sampling weights；
- unique/total tokens、epoch count 与 contamination audit；
- tokenizer revision 及每字节 tokens。

若不同规模使用不同数据 recipe，观测到的是“规模+数据”的复合效应，不能归因于 scale。

### 9.2 可执行配置不是连续闭式解

理论 \(N_{\mathrm{opt}},D_{\mathrm{opt}}\) 需投影到 width/head/FFN/depth 整除约束、global batch、
parallel topology、memory 与 deadline 的离散可行集。工程上应在邻域枚举候选，以预测 loss、
uncertainty、wall-clock 和 failure risk 做多目标选择，而不是将闭式实数四舍五入一次。

## 10. 实验设计与研究方法

### 10.1 Pilot grid

使用 log-spaced compute tiers；每档配置应覆盖 U-shaped profile 两侧，并在预计 minimum
附近加密。边界 winner 说明 coverage 不足，不是有效 optimum。至少部分点做多 seed，
用来估计 noise scale；所有 runs 使用同一训练/评估 protocol。

### 10.2 Sequential design

第一轮宽覆盖用于识别 curvature；第二轮根据 profile uncertainty 在最有信息的区域增补。
选择新点可最大化 parameter information、最小化 target prediction variance，或比较候选
laws 的分歧。Adaptive design 必须保存选择规则，否则事后分析会有 selection bias。

### 10.3 预注册式假设与停止规则

实验前定义 primary law、备选 laws、outlier/failed-run 标准、目标 compute、评价指标和
停止规则。示例可证伪假设：“leave-one-largest-tier-out 对 \(N_{\mathrm{opt}}\) 的预测误差
低于 25%，且 joint 与 IsoFLOP 在目标预算处相差不超过 2×。”未通过应报告负结果并扩 grid，
而不是只更换函数直到结论稳定。

### 10.4 报告最小集合

公开 run-level manifest、原始 loss curves、checkpoint selection、theoretical/measured
compute、失败 runs、完整拟合代码与 seeds。主表至少包括 point estimate、bootstrap interval、
method spread、extrapolation ratio、boundary rate、held-out error 和目标预算的离散候选。

## 11. 常见误区

- **bootstrap 单个相关 run：** 会假装同 tier 点相互独立，区间过窄。
- **只 bootstrap 最终直线：** 丢失 winner selection 与清洗不确定性。
- **CI 包含真实外推值：** 区间通常条件于所选函数形式，不含模型错设。
- **小 RMSE 等于可靠外推：** 插值拟合和 300× 外推是不同任务。
- **指数是模型常数：** 它依赖数据、架构、优化 recipe、范围与口径。
- **删掉“不服从 law”的点：** 除非有预先定义的失败标准，否则可能正是在删反证。
- **把 proxy exponent 搬到 B200：** 本仓库 proxy 是方法验证，不是官方环境替代。
- **把 Chinchilla 1:20 当普适配方：** tokens/param 是部署经济性决策
  （Llama-3 8B ≈1875 tok/param 的过训练是理性偏离），不是自然常数。
- **混淆 test-time compute 与预训练算力：** 二者是并列的 scaling 轴，
  推理期算力（更长思考/搜索/验证）不能直接折算进 \(C=6ND\)。

## 12. Checklist

- [ ] 明确联合 law 的参数约束、初始化与 objective。
- [ ] 使用多初值并记录最优 objective/失败次数。
- [ ] 对 residual 的 \(N,D,C\) 结构作图。
- [ ] cluster bootstrap 重跑清洗、选 optimum 和拟合。
- [ ] 做 leave-one-tier-out，而非只随机留 run。
- [ ] 比较 IsoFLOP 与 joint law 的预测。
- [ ] 报告 extrapolation ratio 与 interval。
- [ ] 单独陈述统计不确定性、方法敏感性和分布外风险。
- [ ] 用实测 runtime 检查理论 \(6ND\) 的 wall-clock 偏差。

### 诊断速查表

| 现象 | 最可能问题 | 后续动作 |
|---|---|---|
| optimum 总在 grid 边界 | coverage 不足 | 扩展该 tier 的 \(N,D\) |
| RMSE 小但参数跨初值跳变 | 不可识别 | profile likelihood/扩大范围 |
| residual 随 \(\log C\) 弯曲 | law 错设 | floor/broken law 对比 |
| 大 tier 全部预测偏乐观 | regime/recipe shift | leave-largest-out、查 MFU/data |
| bootstrap 区间异常窄 | 重采样层级错误 | tier/seed cluster bootstrap |
| joint 与 IsoFLOP 相差数倍 | grid/assumption conflict | 不平均，报告 method spread |
| 理论最优无法按时训练 | FLOPs≠wall-clock | runtime surrogate/离散搜索 |
| 指数换 tokenizer 后变化 | token 口径变化 | byte/character-normalized audit |
| downstream 跳变而 loss 平滑 | metric threshold | 连续指标与 calibration |

## 13. 讨论：效度威胁、伦理与决策边界

### Construct validity

Training loss 不等于 downstream utility、安全性或人类偏好；tokenizer 改变后 per-token loss
不可直接比较。Theoretical FLOPs 不等于能源、时间或成本。应使用 bits-per-byte、统一
evaluation 与 measured runtime 作为互补指标。

### Internal validity

规模往往与 batch、learning rate、regularization、data order、hardware 和 training stability
共变；任何一个未控制变量都可能伪装成 scaling。Checkpoint cherry-picking、静默删除失败
runs 和事后改变 fit range 会产生研究者自由度偏差。

### External validity

同一 family 内的插值结果不保证外推到新 architecture、MoE、长上下文、合成数据、RL
post-training 或新硬件。数百倍 compute extrapolation 应表达为 scenario，而非承诺。
公开复现还表明，原论文细节和数据不足会使精确指数难以独立恢复 [[13]](#ref-13)。

### 资源与责任

Scaling 预测会驱动高成本决策。报告应包含预计 GPU-hours、能源/碳、数据治理、潜在重复
训练成本及较小 baseline。统计上“最优”不自动意味着社会或经济上值得训练。

## 面试要点速记

**高频问题与答题要点**

1. **Q：emergent abilities 是真的吗？** 要点：Wei 等报告跨任务不连续跃迁；
   Schaeffer 等证明选用非线性/不连续指标（如 exact match）可制造“涌现”
   表象，换连续指标往往平滑——结论依赖指标构造，这是面试的标准辩证题。
2. **Q：数据受限下的 scaling？** 要点：Muennighoff 等——重复 ~4 epochs 内
   近似等效新数据，之后边际收益骤降；数据受限时最优模型更小。
3. **Q：推理算力如何改变最优配置？** 要点：Sardana & Frankle——部署期推理
   占大头时，compute-optimal 模型应显著小于 Chinchilla（训练更小模型、
   更长服务期摊薄）。
4. **Q：远距离外推的风险控制？** 要点：profile likelihood、leave-one-tier-out
   报告参数不确定性；不超出拟合支撑集一个数量级以上。
5. **Q：test-time compute 会不会取代预训练 scaling？** 要点：不会——二者是
   并列算力轴。o1 用 RL 优化解题过程（拆解/规划/校验后作答），R1 纯 RL 下
   AIME 2024 准确率 15.6%→86.7%；预训练给基座能力，推理算力换推理深度。
6. **Q：为什么 Llama-3 8B 敢训 15T tokens？** 要点：过训练
   （≈1875 tok/param）偏离 Chinchilla 1:20 是理性决策——推理经济性成为
   与训练最优并列的选型轴；生命周期成本 = 训练 + 推理总量。
7. **Q：MoE 怎么改写 scaling law 口径？** 要点：\(C=kN^pD^q\) 中的 \(N\)
   必须区分 total/active parameters，routing/通信成本另计；DeepSeek-V3 =
   671B MoE / 14.8T tokens；跨论文抄指数前先对齐 \(N\) 口径。
8. **Q：数据质量与规模哪个更重要？** 要点：不是二选一——重复 ~4 epochs
   后边际收益骤降，高质量 token 成新瓶颈；FineWeb/DataComp-LM 把 data
   recipe 当独立变量；跨规模换 recipe 观测的是复合效应，不能归因 scale。

**必背数字**

- 4 epochs 重复上限；Chinchilla 1:20 是“训练算力最优”而非“生命周期最优”；
  inference-aware 最优 N:D 随部署强度下移。

**工业界参照（2024–2026）**

- Llama-3 8B = 15T tokens（≈1875 tok/param）：过训练换推理经济性。
- DeepSeek-V3 = 671B MoE / 14.8T tokens：稀疏激活改写 \(N\) 口径。
- DeepSeek-R1（2025-01）纯 RL（规则奖励、无冷启动）：AIME 2024 准确率
  15.6%→86.7%，自发涌现 aha moment（重新评估、分配更多思考时间）。
- o1（2024）：RL 优化解题过程——回答前完成问题拆解、步骤规划与推理校验；
  GPT-5 将快速模型/深度推理/实时路由整合为统一系统。
- 重复数据 ~4 epochs 临界（Muennighoff）；高质量 token 成新瓶颈。
- 行业判断：能力提升重心从参数扩张转向“后训练对齐 + 测试时计算 +
  工具调用 + 外部数据 + 自动化工作流”的系统化扩展。

## 行业现状与最新进展（2024–2026）

### 涌现能力的再审视

- Wei 等提出 emergent abilities（跨任务不连续跃迁）→ Schaeffer 等
  “指标幻觉”质疑：非线性/不连续指标（exact match、阈值判定）可从
  平滑 loss 中制造骤变表象。
- 业界转向：连续指标（loss、平滑 pass@k、校准概率）+ 严格误差棒
  （多 seed、bootstrap CI）；报告能力变化时同时画 loss 与离散 metric。
- 与本讲呼应：3.3 节的 measurement threshold crossing 正是争论的技术内核；
  换指标不是否定能力，而是把“结论条件于指标构造”写进报告规范。

### 新 scaling 轴全景（2024–2026）

- **MoE 稀疏激活 scaling：** total/active parameters 分离，\(C=kN^pD^q\)
  的 \(N\) 口径必须写明；DeepSeek-V3 = 671B MoE / 14.8T tokens。
- **测试时计算（test-time compute）：** o1（2024）通过 RL 优化解题过程——
  回答前完成问题拆解、步骤规划与推理校验；o3/o4-mini 延伸到工具使用与
  多模态；GPT-5 将快速模型/深度推理/实时路由整合为统一系统。
- **纯 RL 长思考：** DeepSeek-R1（2025-01）规则奖励、无冷启动，AIME 2024
  准确率 15.6%→86.7%，自发涌现 aha moment（重新评估、分配更多思考时间）。
- **更长思考/搜索/验证/动态算力分配：** RL reasoning（Snell 等）把
  “每题投入多少推理算力”变成新决策变量。
- **扩散 DiT 与视频生成：** scaling 轴从语言 loss 扩展到视觉/视频生成
  质量曲线。

### 数据质量与重复

- Muennighoff：重复数据最多 ~4 epochs 价值接近新鲜数据，之后急剧衰减；
  高质量 token 成新瓶颈。
- FineWeb、DataComp-LM（DCLM）开启“过滤配比时代”：data recipe 是独立
  研究变量，不再是默认恒定条件。
- 对 scaling law 的冲击：\(D^{-\beta}\) 假设条件于 dedup/mixture；跨规模
  换 recipe 观测到的是复合效应（见 9.1 节）。

### 六阶段演进对照（行业深度报告口径 2026-08）

| 阶段 | 时间 | 代表工作 | 核心认知 |
|---|---|---|---|
| 0 | 2017–2019 | Hestness 等 | 误差幂律前史 |
| 1 | 2020 | Kaplan、GPT-3、Henighan | 预训练幂律确立、参数主导 |
| 2 | 2021–2022 | Gopher、PaLM、Wei 涌现 | 大模型跃迁叙事 + Schaeffer 指标幻觉质疑 |
| 3 | 2022–2023 | Chinchilla、LLaMA | 计算最优（70B/1.4T 优于 280B Gopher）+ 部署经济性 |
| 4 | 2023–2024 | Muennighoff、Chung 等、FineWeb/DataComp-LM | 数据质量与重复瓶颈 |
| 5 | 2024–2026 | MoE、o1/R1、Snell、DiT、视频生成 | 多变量时代：测试时算力、RL reasoning |

**对本讲学习者的启示：** scaling law 从“单变量（参数）幂律”演进为
“多变量决策模型”——预训练参数扩张不再是唯一轴；当前行业判断，能力提升
重心转向“后训练对齐 + 测试时计算 + 工具调用 + 外部数据 + 自动化工作流”
的系统化扩展。本讲的联合拟合、不确定性分层与外推诊断，正是把每个新轴
（test-time compute、MoE 激活口径、数据 epochs）纳入同一严谨框架的方法论。

## 大厂面试真题与答题框架

以下为高频面试题（公开面经风格），非任何公司真题。

**题目 1：如何拟合并外推一条 loss scaling 曲线？**
- 考点：非线性拟合流程、数值稳定性、外推纪律
- 答题框架：① 参量化（\(A=e^a\) 等保证正性）+ 几何均值归一化；
  ② 多初值 bounded least squares，失败 run 多时换 Huber loss；
  ③ residual 对 \(\log N/\log D/\log C\) 作图查结构；
  ④ leave-one-tier-out 验证 held-out tier 预测；
  ⑤ 报告 extrapolation ratio \(\rho_C\) 与 bootstrap CI，明确区间不含模型错设
- 加分项：区分四层不确定性；提 profile likelihood 判不可识别
- 踩坑：log-log polyfit 一把梭；in-sample RMSE 小就承诺 300× 外推

**题目 2：外推会在哪里失败？**
- 考点：失败模式枚举
- 答题框架：① coverage 不足——optimum 落 grid 边界；② floor 不可识别——
  远离渐近线时 \(E,\alpha\) 耦合；③ regime shift——大 tier 换 recipe/hardware；
  ④ 模型错设——真实 law 带 bend/floor，加性三参数撑不住；
  ⑤ 噪声结构——tier 内相关使 CI 过窄
- 加分项：每类失败配一个诊断动作（boundary audit、leave-largest-out、
  broken law 对比）
- 踩坑：把外推失败归咎于“数据点不够多”，忽视函数形式风险

**题目 3：test-time compute 与预训练 scaling 是什么关系？**
- 考点：新 scaling 轴的理解
- 答题框架：① 定义：推理期投入更多算力（更长思考、搜索、验证、动态分配）；
  ② 证据：o1 用 RL 优化解题过程（拆解/规划/校验后作答），R1 纯 RL 下
  AIME 2024 准确率 15.6%→86.7%；③ 关系：与预训练 \(C=6ND\) 并列的独立
  算力轴，不是替代——预训练给基座能力，测试时算力换推理深度；
  ④ 选型：小模型+大推理算力 vs 大模型一次作答，按部署强度与延迟预算权衡
- 加分项：GPT-5 将快速模型/深度推理/实时路由整合为统一系统；Snell 等
  的动态算力分配
- 踩坑：把测试时算力折算进预训练 FLOPs 混为一谈

**题目 4：Kaplan 与 Chinchilla 结论为何不同？**
- 考点：方法论差异溯源
- 答题框架：① 估计器不同：Kaplan 联合参数化外推 vs Chinchilla IsoFLOP
  envelope 直接找 minimum；② 混杂：LR schedule 是否随 token 数调、训练
  是否充分、数据重复、tokenizer；③ 覆盖范围与 winner 提取方法；
  ④ 结论：不是对错之分，是“条件于 recipe 的最优配置”不同
- 加分项：Besiroglu 等复现尝试表明精确指数难独立恢复
- 踩坑：只答“Kaplan 错了”

**题目 5：数据受限时 scaling 决策怎么改？**
- 考点：Muennighoff 结论 + 配置修正
- 答题框架：① 重复 ~4 epochs 内近似等效新数据，之后边际收益骤降；
  ② 有效 tokens 封顶 → 最优模型更小（\(D\) 受限时 \(N_{\mathrm{opt}}\) 下移）；
  ③ 换轴：数据质量（过滤、去重、配比）成为新杠杆；
  ④ 报告口径：unique/total tokens 分开记
- 加分项：FineWeb/DataComp-LM 把 data recipe 当独立变量
- 踩坑：拿 total token 数当信息量

**题目 6：给一个高 QPS 问答产品选模型规模，怎么选？**
- 考点：inference-aware scaling
- 答题框架：① 生命周期成本 = 训练 + 推理总量；部署重时小模型+过训练
  （Llama-3 8B = 15T tokens ≈1875 tok/param）胜过 Chinchilla 大模型；
  ② 部署强度 → inference-aware N:D 下移（Sardana & Frankle）；
  ③ 若瓶颈是单题难度而非吞吐 → test-time compute 小模型方案（o1/R1 路线）；
  ④ 用 wall-clock 模型把理论 FLOPs 折算成 B200-hours
- 加分项：把“训练最优”与“生命周期最优”分开陈述
- 踩坑：只看 Chinchilla 1:20

**题目 7：emergence 之辩你怎么收尾？**
- 考点：指标构造意识
- 答题框架：① Wei：跨任务跃迁；Schaeffer：非线性/不连续指标可造涌现表象；
  ② 正解：结论条件于指标构造——同时报告 loss、连续概率指标、离散指标
  + 误差棒；③ 工程含义：评测选连续指标可提前预判能力，而非等“涌现”
- 加分项：联系本讲诊断速查表的 threshold crossing 条目
- 踩坑：站队“涌现是假/是真”而不给适用条件

## 系统设计题

**设计题 1：设计一个外推到 10× 算力的预测实验**
- 需求澄清：10× 相对哪个 tier？预测 loss 还是 \(N_{\mathrm{opt}}/D_{\mathrm{opt}}\)？
  容忍误差多少？wall-clock 与 GPU 预算？
- 规模估算：小模型代理 grid——3–4 个 log-spaced compute tiers，
  每 tier 3–5 个 \((N,D)\) 点覆盖 U 形两侧；多初值 K≈10；
  cluster bootstrap B≈200；外推目标点 \(\rho_C=10\)
- 架构：① run manifest（理论 FLOPs + measured runtime）；
  ② joint fit + IsoFLOP envelope 双估计器；③ cluster bootstrap 重跑
  select_optima + fit；④ leave-one-tier-out 验证；
  ⑤ 目标预算预测 + 区间 + 方法 spread
- Trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍 |
|---|---|---|---|
| 估计器 | joint law | IsoFLOP | 样本效率 vs 假设强度；两者都跑报 spread |
| 外推验证 | leave-one-tier-out | synthetic recovery | 真实 held-out vs estimator 校验 |
| 报告口径 | 点估计 | 多方法 ensemble | 简洁 vs 诚实 |

- 评测方案：held-out tier 预测误差 <25%；joint 与 IsoFLOP 在目标预算处
  相差 <2×；boundary rate = 0
- 追问预案：optimum 落 grid 边界 → 扩 tier；两估计器分叉 → 报 method
  spread 不平均；被问“为何信 10× 外推” → 引 \(\rho_C\) + 四层不确定性分层

**设计题 2：推理产品选型——小模型过训练 vs 大模型 Chinchilla**
- 需求澄清：QPS 与部署时长？单题难度分布？延迟预算？蒸馏/量化是否可用？
- 规模估算：过训练档参照 Llama-3 8B = 15T tokens（≈1875 tok/param）vs
  Chinchilla 1:20；DeepSeek-V3 = 671B MoE / 14.8T tokens 展示稀疏激活
  第三条路；总推理 tokens = QPS × 时长 → 生命周期成本曲线
- 架构：① 成本模型：训练一次 + 推理 per-token × 量；② 小模型曲线族
  （不同过训倍数）+ 大模型 Chinchilla 点；③ 难度主导 → 加 test-time
  compute 档（更长思考/验证）；④ 交叉点分析（部署强度阈值）
- Trade-off 表：

| 方案 | 训练成本 | 单位推理成本 | 能力上限 | 适用 |
|---|---|---|---|---|
| 小模型过训练（≈1875 tok/param） | 低 | 低 | 中 | 高 QPS、易任务 |
| Chinchilla 大模型 | 高 | 高 | 高 | 低 QPS、难任务 |
| 小模型 + test-time compute | 中 | 中（可变） | 中高 | 中 QPS、难题为主 |

- 评测方案：按任务难度分层报准确率/延迟/成本 Pareto；难度分桶看
  小模型+推理算力是否补齐差距
- 追问预案：延迟预算紧 → 推理算力上限受制，回小模型过训练；被问
  “Chinchilla 还适用吗” → 答它是训练算力最优，推理经济性是并列选型轴

**设计题 3：数据受限下的 scaling 实验（unique tokens 上限固定）**
- 需求澄清：unique tokens 上限？允许 epochs 数？质量过滤预算？
- 规模估算：epochs 取 0.5/1/2/4/8 档位扫（4 为 Muennighoff 临界），
  模型 2–3 档；固定 compute 下比较“更多 epochs vs 更小模型”
- 架构：有效 tokens 曲线拟合（epoch 增益衰减项）+ 过训练配置搜索；
  unique/total tokens 分开入 manifest
- Trade-off 表：

| 杠杆 | 收益来源 | 衰减点 |
|---|---|---|
| 更多 epochs | 重复曝光 | ~4 epochs 后边际收益骤降 |
| 更小模型 | 与有效 tokens 匹配 | 容量下限 |
| 更强过滤/配比 | 质量/覆盖平衡 | 过滤过狠损害领域覆盖 |

- 评测方案：held-out loss + 下游分桶；跨规模保持同一 data recipe
  （否则是复合效应）
- 追问预案：被问“4 epochs 后还能涨吗” → 边际收益骤降，转向数据质量/
  合成数据（口径：据 Muennighoff）

## 代码实现题

**实现题 1：三参数幂律 \(L(N)\) 拟合 + bootstrap 置信区间**

题目：给定 runs \((N_i,L_i)\)，拟合 \(L(N)=E+A/N^{\alpha}\)，输出参数 CI
与目标 \(N\) 处预测区间。
考察点：正性参量化、多初值、cluster bootstrap、区间解读。

```python
import numpy as np
from scipy.optimize import least_squares

def loss_law(theta, N):
    E, logA, alpha = theta
    return E + np.exp(logA) * N ** (-alpha)

def fit_power_law(N, L, n_starts=10):
    N0 = np.exp(np.mean(np.log(N)))      # 几何均值归一化，稳定 Jacobian
    rng = np.random.default_rng(42)
    best = None
    for _ in range(n_starts):
        x0 = np.array([
            max(np.min(L), 0.0) * rng.uniform(0.5, 0.9),
            np.log(max(np.max(L) - np.min(L), 1e-8)),
            rng.uniform(0.2, 1.0),
        ])
        res = least_squares(
            lambda t: loss_law(t, N / N0) - L, x0,
            bounds=([0.0, -np.inf, 1e-3], [np.inf, np.inf, 10.0]),
        )
        if best is None or res.cost < best.cost:
            best = res
    return best.x, N0

def bootstrap_ci(N, L, n_boot=200, target_N=None):
    tiers = np.unique(N)                  # 按 N 分层重采样，保留相关结构
    params, preds = [], []
    for _ in range(n_boot):
        idx = np.concatenate([
            np.random.default_rng(int(t)).choice(
                np.where(N == t)[0], size=int((N == t).sum()), replace=True)
            for t in tiers
        ])
        theta, N0 = fit_power_law(N[idx], L[idx])
        params.append(theta)
        if target_N is not None:
            preds.append(loss_law(theta, target_N / N0))
    params = np.asarray(params)
    lo, hi = np.percentile(preds, [2.5, 97.5])
    return params, lo, hi                 # 区间不含函数形式错设
```

验收标准：多初值收敛到同一 cost；bootstrap CI 随 B 收敛；参数碰 bounds
时不报普通渐近 CI；输出注明区间条件于所选函数形式。

**实现题 2：外推残差诊断脚本（外推崩坏检测）**

题目：给定拟合参数与 runs，输出残差对 \(\log C\) 的结构化诊断与告警。
考察点：残差结构识别、外推距离、告警阈值设计。

```python
import numpy as np

def extrapolation_diagnostics(N, D, L, pred_fn, C_targets, C_max_observed):
    C = 6.0 * N * D
    resid = L - pred_fn(N, D)
    report = {
        "rho_C": [ct / C_max_observed for ct in C_targets],
        "resid_logC_corr": float(np.corrcoef(np.log(C), resid)[0, 1]),
        "tier_bias": {},
        "alerts": [],
    }
    # 按 compute tier 看偏差符号：某一档全偏正/偏负 = 结构性失配
    std = float(np.std(resid))
    for t in np.unique(C):
        mask = C == t
        bias = float(np.mean(resid[mask]))
        report["tier_bias"][f"{t:.2e}"] = bias
        if abs(bias) > 3 * std:
            report["alerts"].append(f"tier {t:.2e} 系统性偏差 {bias:+.4f}")
    # 最大档预测偏乐观 + residual 随 log C 增大 → 外推崩坏预警
    if report["resid_logC_corr"] > 0.5:
        report["alerts"].append("residual 随 log C 增大：law 在大 compute 端错设")
    return report
```

验收标准：在合成数据（带 floor 的真实 law 用无 floor 模型拟合）上触发
alert；对 iid 噪声数据不误报；输出可直接贴进实验报告（含 \(\rho_C\)）。

## 14. 本讲小结

联合 law 把容量受限与数据受限写成一个 loss surface，并在 \(C=6ND\) 下给出 compute-optimal 闭式比例。真正困难的不是求导，而是确认有限实验是否支持该函数形式，以及如何把 winner selection、seed、拟合和外推的不确定性一起传播。远距离预测应以多方法敏感性和外推距离为中心，而不是用狭窄 bootstrap 区间制造确定感。

## 参考文献

<a id="ref-1"></a>[1] J. Hestness et al. “Deep Learning Scaling is Predictable,
Empirically.” arXiv:1712.00409, 2017.

<a id="ref-2"></a>[2] J. Kaplan et al. “Scaling Laws for Neural Language Models.”
arXiv:2001.08361, 2020.

<a id="ref-3"></a>[3] J. Hoffmann et al. “Training Compute-Optimal Large Language
Models.” *NeurIPS*, 2022. [arXiv](https://arxiv.org/abs/2203.15556)

<a id="ref-4"></a>[4] I. Alabdulmohsin, B. Neyshabur, and X. Zhai. “Revisiting Neural
Scaling Laws in Language and Vision.” *NeurIPS*, 2022.

<a id="ref-5"></a>[5] E. Caballero et al. “Broken Neural Scaling Laws.”
*ICLR*, 2023. [arXiv](https://arxiv.org/abs/2210.14891)

<a id="ref-6"></a>[6] D. Hernandez et al. “Scaling Laws for Transfer.”
arXiv:2102.01293, 2021.

<a id="ref-7"></a>[7] D. Muennighoff et al. “Scaling Data-Constrained Language
Models.” *NeurIPS*, 2023.

<a id="ref-8"></a>[8] P. Marion et al. “When Less is More: Investigating Data Pruning
for Pretraining LLMs at Scale.” arXiv:2309.04564, 2023.

<a id="ref-9"></a>[9] R. Schaeffer, B. Miranda, and S. Koyejo. “Are Emergent Abilities
of Large Language Models a Mirage?” *NeurIPS*, 2023.

<a id="ref-10"></a>[10] J. Wei et al. “Emergent Abilities of Large Language Models.”
*TMLR*, 2022.

<a id="ref-11"></a>[11] M. Sardana and J. Frankle. “Beyond Chinchilla-Optimal:
Accounting for Inference in Language Model Scaling Laws.” *ICML*, 2023.

<a id="ref-12"></a>[12] S. Gadre et al. “DataComp-LM: In Search of the Next Generation
of Language Model Pretraining Datasets.” *NeurIPS Datasets and Benchmarks*, 2024.

<a id="ref-13"></a>[13] A. Besiroglu et al. “Chinchilla Scaling: A Replication Attempt.”
arXiv:2404.10102, 2024.

## 延伸阅读

- Stanford CS336, [Lecture 11 — Scaling Laws](https://github.com/stanford-cs336/lectures/blob/main/lecture_11.pdf).
- [A3 报告源码](../assignments/spring2026/assignment3-scaling/report/main.tex)。
- [Scaling Laws 主题导航](../experiments/topics/scaling-laws.md)。
- [Muennighoff et al., Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)（访问日期 2026-10-04）。
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)（访问日期 2026-10-04）。
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)（访问日期 2026-10-04）。
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)（访问日期 2026-10-04）。
