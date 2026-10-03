---
title: "Lecture 06 — Kernels, Triton & FlashAttention"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-04-15"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py"
  - "../experiments/topics/systems.md"
  - "../assignments/assignment2-systems/"
---

# Lecture 06 — Kernels、Triton 与 FlashAttention：IO 感知的融合算子设计

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述，非课程提交
- 适用对象：自学者、GPU kernel 工程师与系统研究者

## 摘要

高性能深度学习 kernel 的核心任务是把张量代数映射为适合 memory hierarchy 与并行执行单元的
tile program。本文以 Triton 为实现语言，以 softmax/attention 为核心案例，系统讨论 program
instance、launch grid、mask、stride、coalescing、register/shared-memory pressure、warp/stage
配置与 autotuning。随后从 online softmax 推导 FlashAttention forward，并完整推导
\(dQ,dK,dV\) 与 delta preprocessing，解释为何 backward 常拆成 query-major 与 key-major
两遍以避免 atomics。本文还讨论 numerical stability、causal/ragged boundary、compiler
specialization、correctness oracle、gradient checking、benchmark/profiler protocol 与
IO complexity。目标是让读者既能写出正确 kernel，也能解释其为何更快、在哪些 shape 失效，
并将性能结论组织为可复现的系统研究。

**关键词：** GPU Kernel；Triton；Tiling；Online Softmax；FlashAttention；
IO Complexity；Autotuning；Numerical Stability；Fused Backward

## 本文贡献

1. 建立 tensor expression → tile → program instance → hardware resource 的映射；
2. 以 elementwise/reduction/GEMM 总结 Triton kernel 的三类基础模板；
3. 给出 online softmax 与 FlashAttention forward/backward 的逐步推导；
4. 解释两遍 backward、race/atomics、causal boundary 与 mixed precision；
5. 提供 correctness、profiling、autotuning、效度威胁和论文式 benchmark 规范。

## 学习目标

1. 把 PyTorch 张量程序还原成 kernel、HBM 读写和 launch；
2. 掌握 Triton 的 program/block、pointer、mask、stride、reduction 与 tiling；
3. 从 online softmax 推导 FlashAttention forward；
4. 深入推导不保存 \(N^2\) 概率矩阵的 fused backward；
5. 建立 correctness—memory—latency 三条实验线。

## 先修知识

- Lecture 02：arithmetic intensity、Roofline 与显存账本。
- Lecture 05：GPU 的 SM/寄存器/shared memory 层次与占用率模型。
- Lecture 04 §6：MQA/GQA 与 KV cache，为 fused attention kernel 提供动机。

## 相关工作与系统背景

经典 I/O complexity 研究用 red-blue pebble game 形式化计算与不同存储层之间的数据移动
下界 [[1]](#ref-1)。Roofline model 将 arithmetic intensity 与带宽/算力上界连接
[[2]](#ref-2)，为 kernel 优化提供可证伪的性能假设。Triton 进一步以 blocked program
抽象和编译器 layout/optimization，把许多手写 CUDA 优化表达为接近 Python 的 tile 代码
[[3]](#ref-3)。

FlashAttention 将 attention 重新组织为 IO-aware tiled algorithm，在不近似 softmax 的前提下
避免 \(N^2\) 中间矩阵写回 HBM [[4]](#ref-4)；FlashAttention-2 通过更好的 work
partitioning 与减少 non-matmul FLOPs 提高硬件利用率 [[5]](#ref-5)，FlashAttention-3
进一步利用 Hopper 的 asynchrony、warp specialization 与 FP8 低精度 [[6]](#ref-6)；
memory-efficient attention 则独立给出不需要 \(O(N^2)\) 显存的 tiling 与精确 recompute
方案 [[7]](#ref-7)。PyTorch 2 的
TorchDynamo/AOTAutograd/Inductor 展示另一条路径：从动态图捕获并生成 fused kernels
[[8]](#ref-8)；FlexAttention 以 score-mod/mask 抽象加 codegen，让用户在 Python 层表达
稀疏 attention 变体并生成高效 kernel [[9]](#ref-9)。compiler fusion 适合规则图，
手写 Triton 适合需要改变 tile 算法、online statistics 和 backward dataflow 的热点；
FlexAttention 一类的编程模型正在持续移动这条边界。

数值层面，softmax/log-sum-exp 的稳定计算需要考虑 overflow、underflow 与 rounding；
相关误差分析表明“减最大值”稳定但不是所有低精度误差的终点
[[10]](#ref-10)。因此 kernel correctness 应同时比较 output、LSE、gradient 与多步训练 drift。

## 1. 从张量表达式到 kernel

PyTorch 表达式写得短，不代表只启动一个 kernel。以朴素 row-wise softmax 为例，max、减法、exp、sum、除法可能各自读写 HBM。对 \(X\in\mathbb{R}^{M\times N}\)，官方讲义中的朴素账本约为：

\[
\text{reads}=5MN+M,\qquad \text{writes}=3MN+2M.
\]

若一行能放入片上存储，fused kernel 理想上只需读 \(MN\)、写 \(MN\)。fusion 的收益主要来自减少中间量和 launch，而不是减少 softmax 的数学 FLOPs。

### Triton 编程模型

Triton 让一个 program instance 描述一个数据 block：

```python
pid = tl.program_id(0)
offsets = pid * BLOCK + tl.arange(0, BLOCK)
x = tl.load(x_ptr + offsets, mask=offsets < n, other=0.0)
# 在寄存器/片上张量上计算
tl.store(y_ptr + offsets, y, mask=offsets < n)
```

必须显式思考：

- grid 中每个 program 负责哪个输出 tile；
- shape 与 stride 如何映射到地址；
- 尾块 mask 的 `other` 应为 0、\(-\infty\) 还是别的单位元；
- accumulator 是否 FP32；
- `BLOCK_*`、`num_warps` 对寄存器、occupancy 和尾波的影响。

### 1.1 Program instance、lane 与 tile

Triton 不是“一条 Python 语句对应一个 GPU thread”。`tl.arange` 构造 compile-time block
坐标，compiler 决定它们如何分布到 warps/lanes。一个 program instance 通常对应一个 output
tile/row/block，而 grid 决定 instance 数。设计时先写 ownership：

```text
program_id -> output tile -> required input tiles -> reduction axis
```

若两个 programs 写同一 output，需要 atomics 或重新划分 ownership。避免 race 的最好方式通常
不是添加 atomic，而是让每个 program 独占输出并在片上完成 reduction。

### 1.2 Pointer arithmetic 与 layout

二维 tensor 地址：

\[
\mathrm{ptr}(i,j)=\mathrm{base}+i\,s_0+j\,s_1.
\]

不能假设最后一维 contiguous；transposed/non-contiguous input 的 strides 不同。Block pointer
能显式描述 shape/strides/order，并让 compiler 推理 boundary 与 coalescing。Mask 的 `other`
必须是代数单位元：sum 用 0，max 用 \(-\infty\)，乘法/概率需结合公式选择。

### 1.3 Specialization 与 compilation cache

`tl.constexpr` 让 shape/flags（如 causal、block sizes）在编译期 specialization，消除分支并
展开循环；代价是更多 kernel variants 与 compile/cache cost。动态 shape workload 可能频繁
cache miss。报告 steady-state latency 时，应同时记录第一次 compile 时间和 variant 数。

## 2. 三类 kernel

### Elementwise：GeLU

每元素独立，天然并行，通常 bandwidth/launch-bound。把多个逐元素操作融合成一次 load/store 是主要收益。

### Reduction：softmax/row sum

若整行适合一个 program，可在片上完成 max/sum。若行太长，需沿列分 tile、局部累加，再做最终 reduction。为了稳定：

\[
\operatorname{softmax}(x_i)=
\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)},\quad m=\max_jx_j.
\]

### GEMM：二维 tile

对 \(C=A B\)，一个 program 计算 \(C\) 的 \(B_M\times B_N\) tile，循环遍历 \(K\) 维的 \(B_K\) tile。输入复用使 arithmetic intensity 从常数提升到与 tile 尺寸同阶。epilogue 中融合 bias/activation 可避免再次写回再读。

### 2.1 三类 kernel 的优化目标不同

| 类型 | 主要资源 | 常见优化 | 常见失败 |
|---|---|---|---|
| elementwise | HBM/launch | fusion、vectorized/coalesced IO | 过度 fusion 导致 register spill |
| reduction | synchronization/片上容量 | tree reduction、两阶段 reduction | mask 单位元错误、长行溢出 |
| GEMM | tensor cores/register/shared | 3D tiling、double buffer、epilogue fusion | tile underfill、layout/cast |

Reduction 若无法单 program 完成，常采用 first-stage partials + second-stage final reduction。
这增加一次 global IO/launch，但比跨 programs atomic contention 更稳定。GEMM 的
`num_stages` 允许 load/compute pipeline；tile 过大或 stages 过多会因 shared/register
压力降低 occupancy。

### 2.2 Fusion 的边界

Fusion 节省 intermediates 和 launch，但不是越多越好。把 reduction、large epilogue 和多个
outputs 强塞一个 kernel 可能：

- 延长 live ranges、增加 registers；
- spill 到 local memory；
- 降低 occupancy；
- 阻止 library GEMM 使用最佳 kernel；
- 增加 compile time 和 variant explosion。

工程上比较 `unfused baseline → minimal useful fusion → aggressive fusion`，同时记录 HBM bytes、
kernel count、registers/thread 和 end-to-end time。

## 3. Online softmax

FlashAttention 的关键不是“近似 attention”，而是 softmax 可以分块精确合并。

对一行 scores \(x\)，已处理前缀的最大值、指数和为

\[
m=\max_i x_i,\qquad \ell=\sum_i e^{x_i-m}.
\]

新块统计量为

\[
m_b=\max_{j\in b}x_j,\qquad
\ell_b=\sum_{j\in b}e^{x_j-m_b}.
\]

合并：

\[
m'=\max(m,m_b),\qquad
\ell'=e^{m-m'}\ell+e^{m_b-m'}\ell_b.
\]

若还维护未归一化输出 \(o=\sum_i e^{x_i-m}v_i\)，则

\[
o'=e^{m-m'}o+\sum_{j\in b}e^{x_j-m'}v_j,\qquad O=o'/\ell'.
\]

这里旧 accumulator 必须与旧 \(\ell\) 同比例 rescale。漏掉这一项是最常见且不易在小随机输入上发现的错误。

### 3.1 Associative summary 与并行化

一个 block 可总结为 \((m,\ell,o)\)。合并操作满足数学上的结合性，因此不仅能顺序 streaming，
也能做 tree reduction/parallel scan。浮点实现不严格结合，合并顺序会改变 rounding error；
测试应覆盖不同 tile sizes，不能只与同一分块 reference 对比。

### 3.2 LSE 是 backward 的充分统计量

\[
L=\log\sum_i e^{x_i}=m+\log \ell,\qquad
p_i=e^{x_i-L}.
\]

保存每行 LSE 后，backward 可重算 tile scores 与 probabilities，不需要保存 \(P\)。
LSE 通常 FP32；若用 BF16/FP16 保存，长序列与尖锐 logits 会放大 gradient error。

### 3.3 数值边界

- 初始 \(m=-\infty,\ell=0,o=0\)；
- fully masked tile 不应改变 state；
- denominator 为 0 说明整行无合法 key，应显式定义行为；
- exp 输入在减 row max 后不大于 0，避免 overflow；
- output accumulator 和 row sums 使用 FP32；
- 最终 cast 回输入 dtype，而非全过程低精度。

## 4. FlashAttention forward

令

\[
Q\in\mathbb{R}^{B\times N_q\times d},\quad
K,V\in\mathbb{R}^{B\times N_k\times d}.
\]

标准 attention：

\[
S=QK^\top/\sqrt d,\quad P=\operatorname{softmax}(S),\quad O=PV.
\]

FLOPs 仍为 \(\Theta(BN_qN_kd)\)。标准实现若 materialize \(S/P\)，中间空间和 HBM 流量包含 \(\Theta(BN_qN_k)\)。FlashAttention 对 query tile 驻留 \(Q_i,m_i,\ell_i,O_i\)，流式扫描 \(K_j,V_j\)，不把完整 \(S,P\) 写入 HBM；保存的主要 backward 统计量是

\[
L_i=\log\sum_j e^{S_{ij}}=m_i+\log\ell_i,
\]

shape 为 \([B,N_q]\)。

### Causal mask

对 causal self-attention，合法条件是 \(j\le i\)。mask 必须在 row max 和指数和之前生效。全 mask tile 会出现 \(-\infty-(-\infty)\)；实现需让旧缩放为 1、无效概率为 0，不能让 NaN 污染 accumulator。

### Shape 与片上资源

一个 forward program 常处理 \([B_M,d]\) query tile，循环 \([B_N,d]\) key/value tile。片上量级近似：

- \(Q\)：\(B_Md\)
- \(K,V\)：各 \(B_Nd\)
- scores：\(B_MB_N\)
- output accumulator：\(B_Md\)

增大 tile 提高复用，也增加 register/shared-memory 压力。head dimension 常 pad 到 2 的幂；mask 可保证正确，但 padding 仍消耗计算资源。

### 4.1 IO complexity 与 tiling

FlashAttention 的收益来自让 Q/K/V tiles 在片上复用，而不是减少 \(QK^\top\) 与 \(PV\)
的主 FLOPs。用 red-blue pebble / IO complexity 视角，片上容量 \(M\) 限制 tile 尺寸，
合理分块可把 HBM traffic 从显式 \(N^2\) intermediate 降到与输入/输出和分块重读相关的量级
[[1]](#ref-1)[[4]](#ref-4)。

Forward loop order 影响重读：

- query-major：Q/O statistics 驻留，循环 K/V；
- key-major：K/V 驻留，循环 Q；
- grid 是否跨 batch/head/query tiles 并行；
- causal 场景可跳过完全未来的 K/V tiles。

Causal block pruning 只有在 compile-time/tile boundary 直接不执行时才减少 FLOPs；先计算 score
再 mask 不会获得该收益。

### 4.2 Ragged、cross-attention 与 grouped-query 扩展

实际系统可能有 \(N_q\ne N_k\)、variable lengths、padding、GQA/MQA。Kernel interface 需记录
每序列 lengths/offsets，而不能让 padding 参与 softmax denominator。Paged KV cache——如
vLLM 的 PagedAttention，将 KV 分页管理以降低碎片并提升 serving 吞吐
[[11]](#ref-11)——又改变地址连续性；training FlashAttention kernel 不应未经验证直接当
serving paged-attention kernel。

## 5. 深入推导 fused backward

设上游梯度 \(G=dO\)。若保存完整 \(P\)，标准公式为：

\[
dV=P^\top G,\qquad dP=GV^\top.
\]

softmax Jacobian 对每行满足

\[
dS_{ij}=P_{ij}\left(dP_{ij}-\sum_k P_{ik}dP_{ik}\right).
\]

关键恒等式：

\[
D_i=\sum_kP_{ik}dP_{ik}
=\sum_c O_{ic}G_{ic}
=\langle O_i,G_i\rangle.
\]

证明最后一步只需代入 \(O_i=\sum_kP_{ik}V_k\)。因此可先用 shape \([B,N_q]\) 的 delta kernel 计算 \(D\)，然后按 tile 重算：

\[
P_{ij}=e^{S_{ij}-L_i},
\]

\[
dS_{ij}=P_{ij}\left((G_iV_j^\top)-D_i\right),
\]

\[
dQ=dS K/\sqrt d,\quad
dK=dS^\top Q/\sqrt d,\quad
dV=P^\top G.
\]

forward 只需保存 \(Q,K,V,O,L\)，不用保存 \(P\)。这是 compute-for-memory：backward 重算 scores/probabilities，避免读取 \(N^2\) 张量。

### 为什么拆成 delta、dQ、dK/dV

仓库实现采用三个阶段：

1. **delta**：每个 query row 计算 \(\langle O_i,G_i\rangle\)；
2. **dQ kernel**：固定 query tile，扫描全部 key tile；每个 program 独占一个 dQ tile，无需跨 program 原子累加；
3. **dK/dV kernel**：固定 key tile，扫描全部 query tile；每个 program 独占 dK/dV tile。

若用 query-centric kernel 同时写 dK/dV，不同 query blocks 会写同一 key gradient，需要 atomic 或额外归并；若用 key-centric kernel 同时写 dQ，同理。双视角分解以重算换无冲突写入，通常比大量原子操作更可控。

### 反向 shape 检查

对 tile：

- \(Q_i,G_i,O_i:[B_M,d]\)
- \(K_j,V_j:[B_N,d]\)
- \(S_{ij},P_{ij},dP_{ij},dS_{ij}:[B_M,B_N]\)
- \(dQ_i=dS_{ij}K_j:[B_M,d]\)
- \(dK_j=dS_{ij}^\top Q_i:[B_N,d]\)
- \(dV_j=P_{ij}^\top G_i:[B_N,d]\)

\(1/\sqrt d\) 只乘 \(dQ,dK\) 路径，因为 \(S=QK^\top/\sqrt d\)；不要乘到 \(dV\)。

### 5.1 Autograd contract 与 saved tensors

自定义 `torch.autograd.Function` 的 backward 返回项必须与 forward inputs 一一对应；
非 tensor 参数返回 `None`。`ctx.save_for_backward` 应只保存必要 tensors，其他 metadata
保存为普通属性。保存 \(O,L,Q,K,V\) 的内存为线性量级；保存 \(P\) 会重新引入 \(N^2\)。

Backward 需处理 `grad_output` stride/contiguity，不应假设上游梯度连续。若 kernel 要求 contiguous，
显式 copy 的时间和内存必须计入 benchmark，而不能在 wrapper 中隐藏。

### 5.2 Race-free ownership 与 parallel reduction

两遍 backward 的本质是 output ownership：

- query-major program 独占 \(dQ_i\)；
- key-major program 独占 \(dK_j,dV_j\)；
- program 内沿另一轴顺序/分块 reduction。

替代方案是产生 partial gradients 后第二阶段 reduce，或 atomic add。Partials 增加 memory/launch；
atomics 在高 contention 下吞吐差且浮点顺序不确定。最佳方案取决于 \(N,d,tile\) 与 hardware，
应实测而非教条地“永远两遍”。

### 5.3 Gradient correctness

验证层级：

1. 与高精度 PyTorch reference 比 output/LSE；
2. 比 \(dQ,dK,dV\) max/mean/quantile error；
3. 小 shape 用 finite difference：
   \[
   \frac{\partial L}{\partial q_i}\approx
   \frac{L(q_i+\epsilon)-L(q_i-\epsilon)}{2\epsilon};
   \]
4. causal/non-causal、rectangular、tail tiles、non-contiguous；
5. 把 kernel 放入多 step optimizer，检查 drift。

Tolerance 应按 dtype 与 reduction length 分层。只用一个宽松 `allclose` 可能掩盖 dK/dV 累加错误。

### 5.4 Autotuning 搜索空间

典型 meta-parameters：`BLOCK_M/BLOCK_N/BLOCK_D`、`num_warps`、`num_stages`。
搜索应：

- 按 dtype/head dimension/causal 分组；
- 限制 compile 与 benchmark 总预算；
- 验证每个 candidate 正确；
- 保存 winning config 和 Triton/GPU version；
- 用 held-out shapes 检查是否过拟合。

最优 tile 可能随 forward/dQ/dK-dV 不同；强行共享配置简化代码，却不一定最优。

## 6. 代码与实验映射

- PyTorch tiled + Triton 实现：`assignments/assignment2-systems/cs336_systems/flash_attention.py`
- forward kernel：`_flash_attention_forward_kernel`
- backward：`_flash_attention_delta_kernel`、`_flash_attention_dq_kernel`、`_flash_attention_dkdv_kernel`
- correctness：`scripts/benchmark_correctness.py`
- performance：`scripts/benchmark_systems.py attention`
- 报告结果：`report/results/raw/benchmark_fused_backward.csv`

仓库实验说明了“算法正确”与“融合后才快”的区别：

- Python tiled online-softmax 不物化 \(N^2\)，但双重 Python 循环启动大量小 kernel；
- BF16、\(N=4096,d=64\)，旧 backward fallback 的端到端 597.05 ms，fused 为 0.320 ms；
- BF16、\(N=32768,d=64\)，fused 10.06 ms/48.5 MiB，SDPA 88.56 ms/16444 MiB；
- 256 个 correctness cases 的最大绝对误差为 0.015625；
- 环境是单张 RTX 6000D，不能外推到 B200 或不同 Triton 版本。

## 7. Benchmark 设计

至少扫描：

- \(N\)：短到 OOM 边界；
- \(d\)：64、128，以及非 2 的幂边界；
- dtype：FP32、BF16/FP16；
- causal 与 non-causal；
- forward 与 forward+backward；
- PyTorch reference/SDPA、PyTorch tiled、Triton。

分别记录 output/dQ/dK/dV 误差、mean/std latency、peak allocated、OOM。首次 Triton compile 不应混入 steady-state，但真实短任务可另报 cold-start。

### 7.1 计时与统计协议

- warm-up 足以触发 compile/autotune/allocator；
- 用 CUDA events 或边界 synchronize；
- 至少多次 repetitions，报告 mean/std 与 p50/p95；
- forward 后必须在相同条件运行 backward，避免缓存/graph 差异；
- 每个 shape 记录 status（ok/OOM/compile error），不丢弃失败点；
- cold-start/steady-state 分开；
- 锁定 clocks/power 或至少记录共享云环境。

### 7.2 Profiler 证据

除 wall-clock 外，检查：

- kernel count 与 launch gaps；
- achieved HBM/L2/shared throughput；
- tensor-core instructions / achieved FLOP/s；
- registers/thread、shared memory/program、occupancy；
- warp stalls、bank conflicts、spill；
- hidden copies/casts/contiguous；
- causal tile pruning 是否真的少执行 programs。

如果 Triton 快于 reference，应能用更少 HBM bytes、更少 launches 或更高 compute utilization
解释；若解释不了，结论可能依赖测量噪声或不同数学路径。

### 7.3 研究实验示例

> 在 BF16、\(d=64\) 下，fused backward 相比 PyTorch tiled fallback 的 speedup 随
> sequence 增长；相对 fused SDPA 的优势主要来自更低 HBM traffic，而非更少主 FLOPs。

预注册 \(N\)、causal、tile、warm-up、tolerance 和 primary metric。报告 speedup 同时展示
absolute latency/memory，避免小 baseline 上夸大的倍数。

## 8. 易错点

- online 更新只 rescale \(\ell\)，没有 rescale output accumulator；
- mask 在 softmax 之后才乘零，导致无效位置参与 max/sum；
- 全 mask tile 产生 NaN；
- LSE 用低精度保存，长序列误差积累；
- backward 忘记 softmax correction \(D_i\)；
- \(1/\sqrt d\) 多乘、少乘或误乘到 dV；
- dK/dV 被多个 program 非原子写；
- 非连续 stride 被当作 contiguous；
- 只测 forward，就声称 FlashAttention 训练端到端更快；
- 用 `torch.allclose` 单一 tolerance 掩盖某个 gradient 分支错误。

## 9. Checklist

- [ ] 为每个 load/store 写清 shape、stride、mask
- [ ] reduction accumulator/LSE/delta 使用 FP32
- [ ] causal/non-causal 和尾块均有测试
- [ ] 对照 output、dQ、dK、dV
- [ ] 覆盖 \(N\) 非 tile 整数倍和 \(d\) padding
- [ ] 检查无 NaN/Inf，尤其是全 mask tile
- [ ] warmup 后测 forward 与完整 backward
- [ ] 同时报 memory，不只报 speedup
- [ ] profile 确认没有 Python 小 kernel 洪泛或隐式 contiguous copy

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| output 全 NaN | max/LSE/mask | fully-masked tile、低精度 accumulator |
| forward 对、backward 错 | delta/scale/transpose | correction 或 \(1/\sqrt d\) 错 |
| 只在 tail shape 错 / kernel 与参考不符 | load/store mask | `other` 单位元或越界 store、边界 tile 未 mask |
| causal 小 shape 通过、大 shape 失败 | tile offsets | local/global index 混用 |
| Triton 比 PyTorch 慢 / 提速不达标 | launch/register/shape/autotune | tile 太小、spill、underfill、grid 过小 |
| register spill | compiler metadata | tile/stages/live range 过大 |
| occupancy 低 | warps/shared/register | config 资源超限 |
| dK/dV 不确定漂移 / 随机 NaN | write ownership | atomics/race/reduction order、未同步 shared memory |
| fused backward 不收敛 | softmax 重算与 LSE 复用 | dK/dV 分块累计错误、LSE 传播错位 |
| compile 时间爆炸 / 编译慢失败 | specialization count / block 尺寸 | 动态 shape、过多 autotune configs、非编译期常量 |
| benchmark 忽快忽慢 / 波动大 | sync/clocks/autotune | 测到异步 launch、冷启动、未锁 clock、未取中位数 |

## 10. 讨论：效度威胁与结论边界

### 10.1 Construct validity
- memory-efficient 不等于 fast；FLOPs 相同也不等于 runtime 相同；
- peak allocated 不等于 HBM traffic；
- synthetic random correctness 不代表训练稳定；
- speedup ratio 不能替代 absolute latency。

### 10.2 Internal validity
- compile/autotune/caching 混入会扭曲计时；
- reference 可能走不同 dtype/TF32/math mode；
- wrapper hidden copy 会把成本错归 kernel；
- 只保留最佳 autotune run 会引入 selection bias。

### 10.3 External validity
- 单 GPU/Triton 版本的 config 不泛化；
- \(d=64/128\) 结果不代表 irregular head dimensions；
- self-attention kernel 不直接适用 paged KV/cross-attention；
- microbenchmark 不等于 full model time-to-quality。

推荐表述应限定 GPU、Triton/PyTorch version、dtype、shape、causal flag、warm-up、
repetitions 与 tolerance，并公开 OOM/compile failures。

## 11. 面试备考（Interview Prep）

> kernel/Triton/FlashAttention 是 ML Systems 面试的进阶考点：面试官常从「FlashAttention 为什么快」
> 一路追到「online softmax 怎么合并」「backward 为什么不存概率矩阵」「为什么拆 dQ 和 dK/dV」。
> 核心是把「张量表达式 → tile → 硬件资源」这条链讲清楚，并区分「IO 优化」与「数学近似」。
> 下面按「一页速览 → 高频题 → 手撕 → 追问」四层组织。

### 11.1 一页速览卡（面试前 1 分钟）

**核心主张**：kernel 性能来自「把数据放在正确存储层 + 足够大的融合 program 摊销 HBM 与 launch」；
FlashAttention 的数学核心是**可精确合并的 online softmax**，不是近似。

**必背数字与公式**

- online softmax 统计量 \(m=\max x,\ \ell=\sum e^{x-m}\)；新块合并 \(m'=\max(m,m_b),\ \ell'=e^{m-m'}\ell+e^{m_b-m'}\ell_b\)。
- FlashAttention：FLOPs 仍 \(O(T^2d)\)，但显存与 HBM 读写从 \(O(T^2)\) 降到 \(O(T)\)。
- backward 只存 \(Q,K,V,O,L\)（线性），不存 \(P\)（\(T^2\)）；用 LSE \(L_i=m_i+\log\ell_i\) 重算概率。
- 仓库实测：BF16 `N=4096,d=64` 旧 fallback 597.05 ms → fused 0.320 ms；`N=32768` fused 10.06 ms vs SDPA 88.56 ms。

**三句话答高频**

1. online softmax 维护 running max 与归一化指数和，用 rescale 把分块统计量精确合并。
2. FA 不减少 FLOPs，只把 HBM 读写从 \(O(N^2)\) 降到 \(O(N)\)，属于 exact 优化。
3. backward 存 LSE 而非 \(P\)，用 delta \(D=\langle O,G\rangle\) 重算概率，双视角分解避免 atomic。

### 11.2 高频面试题与答题框架

**Q1：online softmax 维护什么？分块怎么精确合并？**

- 维护 running max \(m\) 与归一化指数和 \(\ell=\sum e^{x-m}\)（输出累加器还维护 \(o\)）。
- 新块 \((m_b,\ell_b)\) 合并：\(m'=\max(m,m_b)\)，\(\ell'=e^{m-m'}\ell+e^{m_b-m'}\ell_b\)，\(o'=e^{m-m'}o+\sum_{j\in b}e^{x_j-m'}v_j\)。
- **关键坑**：旧 accumulator \(o\) 必须与旧 \(\ell\) 同比例 rescale，漏掉这一项在小随机输入上不易暴露。

**Q2：FlashAttention 为什么快？减少 FLOPs 吗？**

- **不减少 FLOPs**，仍是 \(O(T^2d)\)。它是 exact IO 优化：tiling + online softmax 让 Q/K/V tile 片上复用，不把 \(T^2\) 的 score/probability 矩阵写回 HBM。
- 显存与 HBM 读写从 \(O(T^2)\) 降到 \(O(T)\)，因此长序列下不再受带宽瓶颈限制。
- **对比**：linear attention 改公式/状态（近似）；FA 数学结果与 full softmax 完全一致。

**Q3：FlashAttention 的 backward 为什么不存概率矩阵 \(P\)？**

- 若存 \(P\)，backward 要读 \(T^2\) 张量，又引入二次显存。
- FA 只存 \(Q,K,V,O,L\)（线性量级）；backward 时用 LSE \(L_i\) 重算 \(P_{ij}=e^{S_{ij}-L_i}\)。
- 这是 **compute-for-memory**：用重算换显存，避免读取 \(N^2\) 中间量。

**Q4：为什么 backward 拆成 delta、dQ、dK/dV 三个阶段？**

- softmax correction 需要 \(D_i=\sum_k P_{ik}dP_{ik}=\langle O_i,G_i\rangle\)，先用 delta kernel 算。
- **dQ kernel**（query-major）：每个 program 独占一个 dQ tile，扫全部 K/V，无需跨 program atomic。
- **dK/dV kernel**（key-major）：每个 program 独占 dK/dV tile，扫全部 Q。
- 若单 query-centric 同时写 dK/dV，不同 query block 会写同一 key gradient → 需要 atomic；双视角分解以重算换无冲突写入，通常比大量 atomic 更可控。

**Q5：Triton 的编程模型？相比 CUDA 的取舍？**

- 一个 program instance 操作一个 data block（tile），用 `tl.arange`/pointer/mask 表达，编译器决定如何分布到 warps/lanes。
- `tl.constexpr` 编译期 specialization 消除分支，代价是更多 kernel variants 与 compile/cache cost。
- **取舍**：牺牲部分底层控制（如显式 shared memory、warp 调度）换接近 Python 的开发效率；适合需要改 tile 算法/online statistics/backward dataflow 的热点。

**Q6：什么是 compute-for-memory（重算换显存）？**

- 不保存某些中间量，backward 时重算，以额外 FLOPs 换更少显存。
- FA backward 重算 \(P\)；activation checkpointing 重算 forward；本质是同一时空权衡主线。
- 判断标准：比较「保存 bytes」与「重算 FLOPs」，只丢弃保存贵、重算便宜的中间量（selective recomputation）。

**Q7：怎么验证一个 fused kernel 的正确性？**

- 分层：先与高精度 PyTorch reference 比 output/LSE；再比 \(dQ,dK,dV\) 的 max/mean/quantile error；小 shape 用 finite difference。
- 覆盖 causal/non-causal、rectangular、tail tile、non-contiguous stride；最后放进多步 optimizer 查 drift。
- **陷阱**：单一 `allclose` 宽容差会掩盖 dK/dV 的累加错误；tolerance 应按 dtype 与 reduction length 分层。

**Q8：FlashAttention-2 / -3 改进了什么？**

- **FA-2**：更好的 work partitioning 与减少 non-matmul FLOPs，提升硬件利用率。
- **FA-3**：利用 Hopper 的 asynchrony、warp specialization 与 FP8 低精度。
- 共同点：都保持 exact softmax，优化的是「如何把 tile 计算映射到硬件」，而非近似注意力。

**Q9：什么时候手写 Triton，什么时候用 `torch.compile`？**

- compiler fusion 适合规则图、固定 shape；手写 Triton 适合需要改变 tile 算法、online statistics、backward dataflow 的热点。
- FlexAttention 一类编程模型（score-mod/mask 抽象 + codegen）正在移动这条边界。
- 决策依据：先 profile 确认真是热点，再比 `unfused → minimal fusion → aggressive fusion`，而非拍脑袋手写。

**Q10：为什么 backward 里 \(1/\sqrt d\) 只乘 dQ/dK 不乘 dV？**

- 因为 \(S=QK^\top/\sqrt d\)，缩放只进入 score 路径；dV 由 \(P^\top G\) 得到，与缩放无关。
- 多乘/少乘/误乘到 dV 是最常见且难在小 shape 上发现的错误之一。

### 11.3 手撕要点（online softmax 合并）

面试让「推导 online softmax 的分块合并」时，按统计量一步步写：

```text
对一行 scores，已处理前缀维护:
  m = max(x_已见),   l = sum(exp(x - m)),   o = sum(exp(x - m) * v)

新 block b 的局部统计量:
  m_b = max(x_b),   l_b = sum(exp(x_b - m_b)),   o_b = sum(exp(x_b - m_b) * v_b)

合并:
  m' = max(m, m_b)
  l' = exp(m - m') * l + exp(m_b - m') * l_b
  o' = exp(m - m') * o + exp(m_b - m') * o_b

最终输出:  O = o' / l'
```

**三个必踩坑**

1. **旧 accumulator 必须 rescale**：`o` 要与旧 `l` 同乘 \(e^{m-m'}\)，漏掉会导致输出错。
2. **fully-masked tile**：`-inf - (-inf)` 产生 NaN，需让旧缩放为 1、无效概率为 0。
3. **LSE 用 FP32**：低精度保存 LSE 在长序列/尖锐 logits 下放大 gradient error。

### 11.4 高频追问与陷阱

| 追问 | 正确方向 |
| --- | --- |
| FA 是近似注意力吗？ | 否，exact；linear attention 才是近似 |
| 存 LSE 为什么就够 backward？ | \(P=e^{S-L}\)，重算概率无需存 \(P\) |
| 两遍 backward 一定最优吗？ | 否，取决于 N/d/tile/硬件；partitals 或 atomic 也可能更优，需实测 |
| causal 能省 FLOPs 吗？ | 只有 tile 边界直接跳过未来块才算省；先算 score 再 mask 不省 |
| 融合越多越好吗？ | 否，过度 fusion 导致 register spill、降低 occupancy |
| FA 能直接用于 serving paged KV 吗？ | 否，训练 FA kernel ≠ paged-attention kernel |

## 12. 小结

Triton 性能来自把正确的数据放在正确的存储层，并用足够大的融合 program 摊销 HBM 与 launch。FlashAttention 的数学核心是可合并的 online softmax；训练性能的关键则是用 LSE 和 delta 重算概率，把 dQ 与 dK/dV 的写所有权设计成无冲突的 fused kernels。从 memory-efficient attention 到 FlashAttention 系列再到 FlexAttention 类编程模型，IO 感知的 attention 算子仍在随硬件代际（asynchrony、FP8）与编译器能力继续演化；“手写还是编译”也应像性能判断一样，以可复现实验为依据。

## 参考文献

<a id="ref-1"></a>[1] J. W. Hong, H. T. Kung. “I/O Complexity: The Red-Blue
Pebble Game.” *STOC*, 1981. https://doi.org/10.1145/800076.802486

<a id="ref-2"></a>[2] S. Williams, A. Waterman, D. Patterson. “Roofline:
An Insightful Visual Performance Model for Multicore Architectures.”
*Communications of the ACM*, 2009. https://doi.org/10.1145/1498765.1498785

<a id="ref-3"></a>[3] P. Tillet, H.-T. Kung, D. Cox. “Triton: An Intermediate
Language and Compiler for Tiled Neural Network Computations.” *MAPL*, 2019.
https://doi.org/10.1145/3315508.3329973

<a id="ref-4"></a>[4] T. Dao et al. “FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness.” *NeurIPS*, 2022.
https://arxiv.org/abs/2205.14135

<a id="ref-5"></a>[5] T. Dao. “FlashAttention-2: Faster Attention with Better
Parallelism and Work Partitioning.” *ICLR*, 2024.
https://arxiv.org/abs/2307.08691

<a id="ref-6"></a>[6] J. Shah, G. Bikshandi, Y. Zhang, et al. “FlashAttention-3:
Fast and Accurate Attention with Asynchrony and Low-precision.”
arXiv:2407.08608, 2024. [link](https://arxiv.org/abs/2407.08608)

<a id="ref-7"></a>[7] M. N. Rabe, C. Staats. “Self-attention Does Not Need
O(n^2) Memory.” arXiv:2112.05682, 2021. [link](https://arxiv.org/abs/2112.05682)

<a id="ref-8"></a>[8] J. Ansel et al. “PyTorch 2: Faster Machine Learning
Through Dynamic Python Bytecode Transformation and Graph Compilation.”
*ASPLOS*, 2024. https://doi.org/10.1145/3620665.3640366

<a id="ref-9"></a>[9] J. Dong, B. Feng, D. Guessous, et al. “Flex Attention:
A Programming Model for Generating Optimized Attention Kernels.”
arXiv:2412.05496, 2024. [link](https://arxiv.org/abs/2412.05496)

<a id="ref-10"></a>[10] P. Blanchard, D. J. Higham, N. J. Higham. “Accurately
Computing the Log-Sum-Exp and Softmax Functions.” *IMA Journal of Numerical
Analysis*, 2021. https://doi.org/10.1093/imanum/draa038

<a id="ref-11"></a>[11] W. Kwon, Z. Li, S. Zhuang, et al. “Efficient Memory
Management for Large Language Model Serving with PagedAttention.” *SOSP*,
2023. [link](https://arxiv.org/abs/2309.06180)

## 延伸阅读与复现材料

- [CS336 Lecture 6 可执行讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [PyTorch scaled dot product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [A2 Systems 官方题面](../assignments/assignment2-systems/cs336_assignment2_systems.pdf)
- 本仓库：[A2 fused backward 报告](../assignments/assignment2-systems/report/main.pdf)
