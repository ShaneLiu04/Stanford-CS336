# Scaling Laws：用小规模实验做有边界的外推

Scaling law 研究的不是“模型越大越好”这句常识，而是在有限 compute 下如何分配 model parameters 与 training tokens，并量化外推不确定性。可靠流程是：先定义成本和可比 loss，设计 IsoFLOP experiments，再拟合、诊断残差，最后才做预测。

## 推荐阅读顺序

### 1. 从经验幂律与 compute accounting 开始

- **官方 lecture**：[Lecture 9 — Scaling laws](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf)。
- **论文**：[Kaplan et al., 2020, Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)。
- **解决的问题**：模型规模 \(N\)、数据量 \(D\) 和训练 compute \(C\) 增长时，test loss 如何规律性下降；固定 compute 时如何选 \(N,D\)。
- **关键结论**：在特定数据、架构和未饱和区间内，loss 常近似 power law。dense Transformer 训练成本常用
  \[
  C\approx 6ND
  \]
  作一阶估算，其中 \(N\) 是非 embedding 或近似有效参数量、\(D\) 是训练 tokens。常数会随 forward/backward、sequence length、attention 和实现细节变化，必须明确口径。
- **对应作业**：[Assignment 3 — Scaling](https://github.com/stanford-cs336/assignment3-scaling) 的 resource allocation、training API 与 IsoFLOP analysis；本仓库快照位于 `assignments/spring2026/assignment3-scaling/`。
- **局限**：Kaplan 结论来自特定模型、数据和训练设置；幂律是经验模型，不是跨架构定律。参数口径或 tokenization 改变会破坏直接比较。

### 2. 理解 compute-optimal training

- **论文**：[Hoffmann et al., 2022, Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)（Chinchilla）。
- **解决的问题**：许多大模型在固定 compute 下参数过多、数据过少。Chinchilla 重新估计 \(N\) 与 \(D\) 的平衡，强调 model size 与 training tokens 应共同扩张。
- **关键公式/结论**：常见联合模型为
  \[
  L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta},
  \]
  \(E\) 是不可约或渐近 loss。结合 \(C\approx6ND\) 可求 compute-optimal \(N_\text{opt}(C)\) 与 \(D_\text{opt}(C)\)。实践中也可先在每个 compute budget 上拟合 loss 对 \(N\) 的曲线，取最优点，再回归
  \[
  N_\text{opt}=aC^b,\qquad D_\text{opt}=cC^d.
  \]
- **对应作业**：A3 的多档 compute budget、每档多个 model size、optimal point 提取和最终预算预测。
- **局限**：最优点若落在 sweep 边界，说明搜索区间不足，不应把边界当稳定 optimum。early stopping、learning-rate schedule、batch size 或训练失败会造成伪 scaling signal。

### 3. 学会设计 IsoFLOP matrix

- **官方 lecture**：[Lecture 11 — Scaling laws](https://github.com/stanford-cs336/lectures/blob/main/lecture_11.pdf)。
- **解决的问题**：若每个模型同时改变 compute、tokens 和 optimization recipe，loss 差异无法归因。IsoFLOP 在同一预算 \(C_i\) 下选择若干 \(N_{ij}\)，并用
  \[
  D_{ij}=\frac{C_i}{6N_{ij}}
  \]
  配置 token budget。
- **实践要点**：
  - 每个 budget 至少覆盖 optimum 两侧，而不是只测单调区间。
  - 训练步数应由准确 token count 推导，处理 batch rounding，并记录实际 tokens/FLOPs。
  - architecture family、tokenizer、data distribution 与 evaluation set 保持固定。
  - 不要让不同 run 的 warmup 比例、learning-rate tuning 努力或数据重复程度成为隐藏变量。
- **对应作业**：A3 的 hosted run querying、run selection 与本地 proxy matrix。
- **局限**：固定超参数对小模型和大模型未必都公平，逐模型调参又会改变总搜索 compute；报告必须声明采用哪种比较原则。

### 4. 进行稳健拟合，而非只画一条直线

- **论文**：[Alabdulmohsin et al., 2022, Revisiting Neural Scaling Laws in Language and Vision](https://arxiv.org/abs/2209.06640) 讨论函数形式与有限数据拟合；[Caballero et al., 2022, Broken Neural Scaling Laws](https://arxiv.org/abs/2210.14891) 展示单一幂律在 regime transition 附近的失效。
- **工具文档**：[SciPy optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) 与 [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。
- **解决的问题**：非线性模型、测量噪声、失败 run 和窄动态范围会让指数估计不稳定。
- **关键结论**：
  - 在 log space 的线性回归方便，但含 \(E\) 的加法模型不能简单整体取 log。
  - 应对正参数使用 bounds 或 log-parameterization，多初值优化并报告 objective。
  - 用 bootstrap 重采样 run/seed 或 compute group，重新执行完整“选 optimum + 拟合”流程，得到 prediction interval；只对最终回归线 bootstrap 会低估上游不确定性。
  - 同时检查 residual 对 \(N,D,C\) 的结构、leave-one-budget-out 稳定性与 extrapolation distance。
- **对应作业**：A3 的 bootstrap uncertainty、joint scaling law、residual diagnostics 与 final prediction。
- **局限**：样本很少时 bootstrap 分布也不可靠；置信区间只覆盖设定模型和观测噪声，不覆盖 architecture/data/optimizer 的分布外变化。

### 5. 将 prediction 与 evaluation 分开

- **官方 lecture**：[Lecture 12 — Evaluation](https://github.com/stanford-cs336/lectures/blob/main/lecture_12.py)。
- validation cross-entropy 适合平滑拟合，但 downstream capability、calibration、robustness 和 safety 可能出现不同曲线或阈值行为。
- tokenizer 不同时应考虑 nats/byte；数据污染会让 evaluation loss 虚低；同一个 benchmark 的多次调参会形成隐性 overfitting。
- **对应作业**：A3 的 held-out loss 分析；若扩展到 benchmark，应把 benchmark selection 与 scaling fit 预先固定。
- **局限**：低 loss 不自动等价于更可靠的生成系统，也不能推出部署成本最优；inference compute、latency 与 memory 是另一个优化目标。

## 建议分析流水线

1. 建立 run manifest：模型 shape、\(N\)、目标/实际 \(D,C\)、seed、状态、loss、环境。
2. 画每个 compute budget 内的 loss-\(N\) 曲线，先人工检查边界 optimum、失败点和异常残差。
3. 分别拟合 IsoFLOP optimum law 与联合 \(L(N,D)\) law，比较结论而非强行一致。
4. bootstrap 完整 pipeline，报告 median、区间与 out-of-range 比例。
5. 对最终预算给 point estimate，也给可行配置区间；注明它离观测 compute 最大值有多少倍。

## 阅读边界

Scaling law 最容易被误用在远距离外推。若目标 compute 比观测范围大几个数量级，统计误差通常小于模型错设误差。结论必须附带数据分布、架构族、tokenizer、训练 recipe、预算范围和失败 run 处理规则；缺少这些上下文的单一指数没有可迁移意义。
