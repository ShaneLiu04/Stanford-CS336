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
- 文档性质：原创中文自学综述，非课程提交
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

**必背数字**

- 4 epochs 重复上限；Chinchilla 1:20 是“训练算力最优”而非“生命周期最优”；
  inference-aware 最优 N:D 随部署强度下移。

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
