---
title: "Lecture 02 — PyTorch & Resource Accounting"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-04-01"
lecturer: "Percy Liang"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py"
  - "../assignments/spring2026/assignment1-basics/"
---

# Lecture 02 — PyTorch 与 Resource Accounting：从 tensor shape 到资源账本

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
- 适用对象：自学者、ML Systems 工程师与性能研究者

## 摘要

深度学习系统优化的前提不是“使用更快的 API”，而是建立可验证的资源模型：给定 tensor
shape、dtype、operator graph 与硬件层次，能够预测参数量、FLOPs、activation、optimizer
state、memory traffic 与理论吞吐上界。本文以 PyTorch tensor/autograd 为可执行语义，
系统推导 decoder-only Transformer 的计算与显存成本，并用 arithmetic intensity 与
Roofline model 区分 compute-bound、memory-bound 和 launch-bound workloads。在工程层面，
本文讨论 stride/view/contiguity、broadcast、in-place operation、CUDA asynchronous timing、
allocator fragmentation、mixed precision、gradient accumulation、activation checkpointing
与 selective recomputation；
在研究层面，给出可复现 profiling protocol、controlled ablation、MFU 解释边界与效度威胁。
目标是使读者能够从 shape 推出资源账本，从 profiler 证据定位瓶颈，并把性能结论写成可复核实验。

**关键词：** PyTorch；Automatic Differentiation；FLOPs；Memory Accounting；
Arithmetic Intensity；Roofline Model；Mixed Precision；Profiling；Reproducibility

## 本文贡献

1. 建立 `shape → FLOPs → bytes → bottleneck → experiment` 的统一分析链；
2. 给出 Transformer parameter/state/activation 的逐项核算方法与计数约定；
3. 区分 theoretical FLOPs、hardware FLOPs、MFU、GPU utilization 与 wall-clock；
4. 总结 PyTorch storage/autograd/allocator 的工程陷阱与诊断流程；
5. 将 selective recomputation 纳入 activation memory 设计空间，给出按算子权衡
   保存与重算的决策方法；
6. 提供面向研究的 benchmark protocol、uncertainty、ablation 和 validity checklist。

> Resource accounting 的目的不是背一个“训练约等于 \(6ND\) FLOPs”的口号，而是在运行前回答：
> tensor 形状是否正确、模型能否放入显存、瓶颈是算力还是带宽、一次实验需要多久，以及优化究竟减少了什么。

## 学习目标

1. 熟练追踪 PyTorch tensor 的 shape、dtype、device 与 autograd 生命周期；
2. 用 `einsum` / `rearrange` 把公式直接映射为带名字的维度；
3. 推导 linear、attention、FFN、LM head 的参数量与 FLOPs；
4. 分解训练显存为 parameters、gradients、optimizer states、activations；
5. 用 arithmetic intensity 与 roofline 判断 compute-bound / memory-bound；
6. 正确解释 mixed precision、gradient accumulation、activation checkpointing 与 MFU。

## 先修知识

- 线性代数：矩阵乘法、内积、矩阵形状；
- 微积分：chain rule；
- Lecture 1 的 \([B,T]\) token batch；
- Python 与基本 PyTorch tensor 操作。

## 核心概念与公式推导路线

本讲把 tensor shape、dtype、device、lifetime 作为统一账本，再依次推导 Linear 与 Transformer
FLOPs、训练状态/activation memory、arithmetic intensity、roofline 与 MFU。公式不是孤立估算：
每一项都应能映射到一个 PyTorch tensor 或 kernel，并由 profiler/实测边界校验。

## 相关工作与方法脉络

PyTorch 以 imperative tensor program 与 reverse-mode automatic differentiation 提供研究友好的
动态图语义，并通过 dispatcher、ATen 与 accelerator backends 连接高性能 kernels
[[1]](#ref-1)。NumPy 建立的 n-dimensional array、broadcasting 与 stride 语义是其重要基础
[[2]](#ref-2)。然而，框架抽象不会消除硬件约束：Roofline model 用 arithmetic intensity
把应用需求与峰值带宽/算力联系起来 [[3]](#ref-3)，至今仍是判断 operator bottleneck 的核心工具。

大模型训练进一步把单卡账本扩展到系统账本。Mixed Precision 利用低精度 tensor cores 与
FP32 master state 提高吞吐 [[4]](#ref-4)；activation checkpointing 用 recomputation
交换显存 [[5]](#ref-5)；Megatron-LM、ZeRO 与 FSDP 分别从 tensor parallel 和 state
sharding 扩展模型规模 [[6]](#ref-6)[[7]](#ref-7)[[8]](#ref-8)。FlashAttention
说明 FLOP 数相近的算法可因 IO complexity 不同而产生显著性能差异 [[9]](#ref-9)。
在 activation memory 一侧，Korthikanti 等人将 selective recomputation 与 sequence
parallelism 结合，在不显著增加计算的前提下大幅压缩需要保存的激活 [[10]](#ref-10)；
PaLM 等大型训练报告则以 MFU 作为跨模型效率比较的标准口径 [[11]](#ref-11)。
因此资源核算必须同时包含计算量、memory traffic、并行通信和 kernel realization。

## 1. Tensor 是资源核算的共同语言

训练过程中的主要对象都可视为 tensors：

- parameters：模型的持久状态；
- activations：forward 中为 backward 保留的中间结果；
- gradients：与可训练参数同 shape；
- optimizer states：如 AdamW 的一、二阶矩；
- data / logits / masks：输入输出与临时张量。

对每个 tensor，始终问四件事：

1. **shape**：每个轴代表什么？
2. **dtype**：每个元素多少 bytes、动态范围如何？
3. **device**：CPU RAM 还是 accelerator HBM？
4. **lifetime**：何时创建、何时 backward 后释放？

仅看 `numel()` 不足以解释 peak memory；峰值由多个 tensor 生命周期重叠决定。

## 2. PyTorch shape 语义

### 2.1 Linear 与 batch 维

令输入 \(x\in\mathbb R^{B\times T\times d_{\text{in}}}\)，权重
\(W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}\)，则

\[
y_{bto}=\sum_{i=1}^{d_{\text{in}}}x_{bti}W_{oi},
\qquad y\in\mathbb R^{B\times T\times d_{\text{out}}}.
\]

仓库代码写成：

```python
einsum(x, weight, "... d_in, d_out d_in -> ... d_out")
```

`...` 保留任意 leading dimensions，比手写 `view` 更不容易把 batch/sequence 混在一起。
但 `einsum` 只表达语义，不保证自动选择最快 kernel；性能仍取决于是否能 lower 成高效 GEMM。

### 2.2 Rearrange 不是免费魔法

Multi-head attention 常见变换：

\[
[B,T,d]\rightarrow[B,T,h,d_h]\rightarrow[B,h,T,d_h],\qquad d=hd_h.
\]

`rearrange(x, "b t (h dh) -> b h t dh", h=h)` 让拆轴意图清晰。若操作仅改变 stride，
它可能是 view；若之后要求 contiguous layout，就可能触发真实拷贝。核算 runtime 时要区分
“逻辑 reshape”和“数据搬运”。

### 2.3 Autograd 保存什么

对 \(Y=XW\)，backward 需要：

\[
\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Y}W^\top,\qquad
\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Y}.
\]

因此 forward 中的 \(X\) 或等价信息必须留到 backward。`torch.no_grad()` 能避免构建图；
in-place 修改被 autograd 保存的 tensor 可能触发 version error，或更糟地破坏梯度语义。

### 2.4 Storage、stride、view 与 aliasing

Tensor 是对 storage 的带 shape/stride 视图。地址近似为

\[
\mathrm{addr}(i_1,\dots,i_k)=\mathrm{base}+\sum_j i_j\,\mathrm{stride}_j.
\]

`transpose` 通常只改 stride；`contiguous()` 才按目标布局复制。两个 tensors 可能共享 storage：
对一个 view 做 in-place update 会影响另一个 alias。工程排查时同时查看：

```python
x.shape, x.stride(), x.storage_offset(), x.is_contiguous()
```

常见性能陷阱是：逻辑上无成本的 transpose 进入只接受 contiguous input 的 kernel，框架悄悄插入
copy。Profiler 中应把 `aten::contiguous` / `copy_` 与真正 matmul 分开归因。

### 2.5 Broadcasting 与 materialization

Broadcasting 通过 stride 0 复用数据，逻辑 shape 变大不一定分配。例如
`mask[None,None,:,:]` 可共享 storage；但后续 `clone`、某些 fused kernel 或写操作会 materialize。
因此 memory accounting 需要区分：

- logical elements；
- physical storage bytes；
- temporary expanded output；
- allocator reserved memory。

### 2.6 Autograd graph、hooks 与 compilation

Reverse-mode AD 的成本不仅是数学 gradient，还包括 graph nodes、saved tensors、Python dispatch
和 synchronization。`torch.compile` 尝试把稳定 graph 捕获并融合，但 graph break、动态 shape、
side effect 和数据依赖 control flow 会降低收益。正确比较应分开：

1. compile time；
2. warm-up/autotuning；
3. steady-state iteration；
4. graph cache miss（新 shape）；
5. eager fallback 区域。

`register_hook` / `register_post_accumulate_grad_hook` 可在 gradient ready 时触发通信或统计；
错误 hook 顺序会造成重复 all-reduce、梯度覆盖或 host synchronization。

## 3. FLOPs 推导

### 3.1 先统一计数约定

本笔记沿用课程常见约定：一次乘法和一次加法各算 1 FLOP，因此矩阵乘
\([m,k]@[k,n]\) 约为

\[
2mkn\ \text{FLOPs}.
\]

不同 profiler 可能把 fused multiply-add 算 1 或 2 FLOPs；比较前必须写明约定。

### 3.2 Linear 层

对 \([B,T,d_{\rm in}]\to[B,T,d_{\rm out}]\)：

\[
F_{\rm linear,fwd}=2BTd_{\rm in}d_{\rm out}.
\]

若忽略小型逐元素操作，两个 backward GEMMs 各与 forward 同阶：

\[
F_{\rm linear,bwd}\approx4BTd_{\rm in}d_{\rm out},\quad
F_{\rm train}\approx3F_{\rm fwd}.
\]

这产生 dense model 的粗略规则：对 \(D\) 个训练 tokens、\(N\) 个参与 matmul 的参数，

\[
F_{\rm train}\approx6ND.
\]

该口径在 Kaplan 等人的 scaling laws 分析中被用作 compute 的定义量
[[12]](#ref-12)，后续 Chinchilla 沿用并因此得到 \(C=6ND\) 约束下的最优配置
（见 Lecture 9/11）。它忽略 attention 的 \(T^2\) 项、embedding lookup、softmax、
normalization、optimizer、recomputation 和通信；只能用于量级估算，
报告时应说明 \(N\) 的计数口径（是否含 embedding/head、是否 tied）。

### 3.3 Decoder Transformer 一层

记 \(d=d_{\text{model}}\)、\(f=d_{\text{ff}}\)、序列长度 \(T\)，暂略 batch：

1. Q/K/V/O 四个 \(d\times d\) projection：
   \[
   4(2Td^2)=8Td^2.
   \]
2. score \(QK^\top\) 与 weighted sum \(AV\)：
   \[
   2T^2d+2T^2d=4T^2d.
   \]
3. SwiGLU 三个矩阵乘：
   \[
   2Tdf+2Tdf+2Tfd=6Tdf.
   \]

所以 \(L\) 层加 untied LM head：

\[
F_{\rm fwd}
\approx L(8Td^2+4T^2d+6Tdf)+2TdV.
\]

Batch 为 \(B\) 时整体乘 \(B\)。当 \(T\) 较小时 dense projections / FFN 占主导；当
\(T\) 变长，\(T^2\) attention 会改变占比。不能把 \(6ND\) 当作所有 context length 都精确。

### 3.4 参数量

不计 bias，单层约：

\[
P_{\rm layer}=4d^2+3df+2d,
\]

其中 \(2d\) 是两组 RMSNorm gain，量级很小。完整 untied LM：

\[
P\approx Vd+L(4d^2+3df+2d)+d+Vd.
\]

首尾两个 \(Vd\) 分别是 embedding 与 LM head；weight tying 后只保留一份。

### 3.5 FLOPs 少不等于耗时少

RMSNorm、softmax、dropout、mask、activation、optimizer update 的 FLOPs 远少于 GEMM，
但可能分别启动 kernel、读写整张量。对 \(n\) elements 的逐元素算子，arithmetic intensity
常为 \(O(1)\) FLOPs / 若干 bytes，容易 memory-bound。许多“小算子”还会产生：

- kernel launch latency；
- intermediate tensor materialization；
- dtype cast；
- reduction synchronization；
- host-side dispatch。

因此 theoretical FLOP breakdown 适合解释主项，却不能直接预测 wall-clock。Kernel fusion
的价值常是减少 HBM round trips 和 launches，而不是减少数学 FLOPs。

### 3.6 有效训练计算量与 recomputation

若 activation checkpointing 使某些 forward block 在 backward 重算，则实际计算量变为

\[
F_{\mathrm{effective}}
=F_{\mathrm{forward}}+F_{\mathrm{backward}}+F_{\mathrm{recompute}}.
\]

同理，padding、capacity overflow、MoE routing、failed/overflowed step 也会消耗硬件时间，
但不一定计入“有效 tokens FLOPs”。报告 MFU 时应说明 numerator 使用 dense theoretical FLOPs、
实际 executed FLOPs，还是 useful non-padding FLOPs。

## 4. Memory Accounting

### 4.1 静态训练状态

若全部使用 fp32，AdamW 的常见粗估：

| 项目 | 每参数 bytes |
|---|---:|
| parameter | 4 |
| gradient | 4 |
| first moment \(m\) | 4 |
| second moment \(v\) | 4 |
| 合计 | 16 |

Mixed precision recipe 并不唯一。课程中的简化 bf16 情形可按 parameter 2 + gradient 2 +
fp32 moments 8 = 12 bytes/parameter；某些实现还保留 fp32 master weights，变为 16 bytes
或更多。必须根据实际 optimizer / AMP 实现核对，不能套固定数字。

两个 moment 是对梯度一、二阶矩的指数移动平均，其定义来自 Adam
[[13]](#ref-13)；AdamW 将 weight decay 从梯度自适应更新中解耦、直接作用于参数
[[14]](#ref-14)。解耦不改变 state 计账（\(m,v\) 仍与参数同 shape），但改变更新
语义：decay 项的实现位置（乘性衰减 vs. 加到梯度）是常见的正确性差异来源，
differential test 应覆盖多步更新后的参数漂移。

### 4.2 Activations

Activation memory 随 \(B,T,L,d,f,h\) 增长，并依赖 kernel 保存哪些中间量。一个教学级上界可写成

\[
A/B\approx L(7Td+4Tf+2hT^2)+Td+2TV
\]

个元素；其中 \(hT^2\) attention probabilities 与 \(TV\) logits 在大序列/大词表下很昂贵。
Fused cross-entropy、FlashAttention、重计算会改变实际保存集合，所以公式必须与实现配套。

总显存不是简单的“模型文件大小”：

\[
M_{\rm peak}\approx M_{\rm params}+M_{\rm grads}+M_{\rm opt}
+M_{\rm saved\ activations}+M_{\rm temp}+M_{\rm allocator}.
\]

`reserved`、`allocated` 与 profiler 的 peak 定义也不同。

### 4.3 Dtype 的取舍

- fp32：8-bit exponent、23-bit fraction，稳但占 4 bytes；
- fp16：5-bit exponent、10-bit fraction，范围小，梯度 underflow 常需 loss scaling；
- bf16：8-bit exponent、7-bit fraction，2 bytes，动态范围接近 fp32，精度较粗；
- fp8/fp4：格式和 scaling recipe 更复杂，不能只按 byte 数推断训练质量。

bf16 保持 fp32 的 8-bit exponent，使其在大规模训练中可免除 loss scaling；这一
取舍在 Intel 的系统研究中得到验证 [[15]](#ref-15)，PaLM 亦以 bf16 训练并按
fp32 口径报告 MFU [[11]](#ref-11)。fp8 训练进一步引入 E4M3/E5M2 等格式与
per-tensor/per-channel scaling [[16]](#ref-16)，数值边界比 bf16 更依赖实现与
scale 管理，报告 fp8 结果时必须写明格式、scaling 策略与失败处理。

本仓库 A1 的 `RMSNorm.forward` 在平方前 upcast fp32，再 cast 回输入 dtype；这是因为 reduction
和平方比普通 matmul 更易受低精度影响。报告中的无 `GradScaler` fp16 实验 loss 退化，不能被解释为
“fp16 更快且同样可用”。

### 4.4 Allocator、fragmentation 与 OOM 诊断

PyTorch CUDA caching allocator 会保留释放的 blocks 供后续复用，因此：

- `memory_allocated`：live tensors 占用；
- `memory_reserved`：allocator 向 CUDA 申请但未必正在使用；
- `max_memory_allocated`：自上次 reset 后 live peak；
- `nvidia-smi`：整个 process/context 级占用，通常更高。

OOM 不一定表示总 free bytes 小于请求；fragmentation 可能没有足够连续 block。诊断步骤：

1. 在逻辑阶段前后 `torch.cuda.synchronize()`；
2. 记录 allocated/reserved/peak；
3. 使用 memory snapshot 查 allocation stack 和 lifetime；
4. 区分 persistent state、saved tensor、temporary workspace；
5. 检查是否意外保留带 graph 的 loss/output list；
6. 最后才考虑 `empty_cache()`；它不释放 live tensors，也不是常规优化。

训练中常见“内存泄漏”是把未 `detach()` 的 tensor 放进 Python container，使整张 autograd
graph 跨 step 存活。真正验证方式是观察多个相同步后 allocated baseline 是否持续增长。

## 5. Arithmetic Intensity 与 Roofline

Arithmetic intensity：

\[
I=\frac{\text{FLOPs}}{\text{bytes transferred}}.
\]

设硬件峰值算力 \(C_{\max}\) FLOP/s、内存带宽 \(BW\) bytes/s，则可达性能上界：

\[
\operatorname{Perf}\le\min(C_{\max}, BW\cdot I).
\]

转折点

\[
I^\*=\frac{C_{\max}}{BW}
\]

称 accelerator intensity。若 \(I<I^\*\)，memory-bound；若 \(I>I^\*\)，compute-bound。

### 5.1 为什么 GEMM 通常更“值钱”

方阵 matmul \([n,n]@[n,n]\) 有约 \(2n^3\) FLOPs，而至少读取两个输入、写一个输出，
数据量 \(O(n^2)\)，所以 \(I=O(n)\)：矩阵越大，复用越强。

逐元素 ReLU 对每个元素约 1 FLOP，却至少读写一次，\(I=O(1)\)，通常 memory-bound。
AdamW 也是多个逐元素 kernel；虽然 FLOPs 相对少，仍可能占可见 wall-clock。

矩阵向量乘是推理 decode 常见形态，权重几乎只为一个 token 读取一次，复用低，所以更容易
memory-bound。训练中更大 batch 能提高 GEMM 利用率，但 activation memory 也近似随 batch 增长。

### 5.2 MFU 不等于 GPU utilization

\[
\text{MFU}=\frac{\text{模型理论 FLOPs/step}\times\text{steps/s}}
{\text{硬件峰值 FLOP/s}\times\text{设备数}}.
\]

MFU 依赖模型 FLOPs 公式、dtype 对应峰值、是否计算重计算 FLOPs等假设；该口径由
PaLM 等大型训练报告普及，用于跨模型/跨硬件的效率比较 [[11]](#ref-11)。任务管理器显示的
“GPU 100%”不代表接近 tensor-core 峰值；GPU 可能忙于低强度 kernel、通信或数据搬运。

### 5.3 从理论 Roofline 到实测证据

理论分类需要用 profiler 校正。建议记录：

- kernel duration、launch count；
- DRAM bytes / achieved bandwidth；
- tensor-core utilization / achieved FLOP/s；
- occupancy、register/shared-memory pressure；
- CPU launch gaps、CUDA synchronization；
- communication overlap。

若 profiler 无法直接给 FLOPs，可用 operator shapes 估算；若无法给 bytes，至少区分
读写张量尺寸和 cache reuse 假设。结论应写成：

> 在给定 shape/dtype/hardware 下，该 kernel 达到 X GB/s（峰值的 Y%），而估算 arithmetic
> intensity 低于 ridge point，因此证据支持 memory-bound。

不要只凭“逐元素算子通常 memory-bound”替代测量。

## 6. 用空间换时间，或用时间换空间

### 6.1 Gradient accumulation

把 global batch \(B_g\) 分为 \(K\) 个 microbatches \(B_\mu\)：

\[
B_g=KB_\mu,\qquad
g=\frac1K\sum_{k=1}^{K}g_k.
\]

每个 microbatch loss 要除以 \(K\)，累计完再 clip 和 `optimizer.step()`。它降低 activation
peak，却不减少总 forward/backward FLOPs；小 GEMM 还可能降低 throughput。若每个 microbatch
都 step、clip 或重新采样不同语义的数据，就不再等价于 global batch。

### 6.2 Activation checkpointing

普通反向保留每层 activation，memory 约 \(O(L)\)。Checkpointing 只保存若干边界，backward
时重算中间 forward：

- 保存所有层：低重算、高内存；
- 几乎不保存：可能造成大量重复；
- 每约 \(\sqrt L\) 层设 checkpoint：教学上可得到 \(O(\sqrt L)\) 保存与 \(O(L)\) 级额外重算的折中。

它减少 saved activations，不减少 parameters、gradients 或 optimizer states。

**选择性重计算（selective recomputation）** 是更细粒度的替代：不按层边界一刀切，
而是对每个算子比较"保存成本（bytes）"与"重算成本（FLOPs）"，只丢弃那些保存昂贵、
重算便宜的中间量——典型如 attention 中为 backward 保留的 \(T^2\) 级辅助矩阵
（softmax 输入、dropout mask），而继续保存 \(O(Td)\) 级的线性投影输出。
Korthikanti 等人将这一策略与 sequence parallelism 结合，在不显著增加计算的前提下
大幅压缩 activation memory，支撑了百亿–千亿参数模型的训练 [[10]](#ref-10)。
工程流程应该是：先列出 per-op 的 saved bytes / recompute FLOPs 表，按
"每节省 1 byte 的重算代价"排序，再决定 checkpoint 边界；而不是机械地每
\(\sqrt L\) 层切一刀。该思路在 Lecture 7/8 的 sequence parallelism 中会再次出现。

### 6.3 `zero_grad(set_to_none=True)`

把 `.grad` 设为 `None` 可避免无意义清零写带宽，并允许下一次 backward 直接分配/覆盖；但任何读取
`.grad` 的代码必须处理 `None`。仓库训练循环正使用这一模式。

## 7. 面向工程与研究的 Benchmark 方法

### 7.1 正确计时 GPU

CUDA 默认异步；仅用 `time.perf_counter()` 包围 `model(x)` 往往只测到 kernel launch。
可靠做法是 warm-up 后在计时边界同步，或使用 CUDA events：

```python
for _ in range(warmup):
    step()
torch.cuda.synchronize()
start.record()
for _ in range(repetitions):
    step()
end.record()
torch.cuda.synchronize()
```

首次运行可能包含 context initialization、library loading、kernel compilation、autotuning
和 allocator growth。`torch.compile` 必须把 compile latency、warm-up 与 steady-state 分开。

### 7.2 Controlled experiment

一次只改变一个主要因素，并固定 model/config、input shape、seed、dtype、TF32、autocast、
warm-up/repetitions、software/driver/commit 与 synchronization。至少报告 mean/std；
latency-sensitive 服务还应报告 p50/p95。长训练同时报告 tokens/s 与 time-to-quality，
因为更快的 step 不保证更快达到目标 loss。

### 7.3 把“优化问题”写成研究问题

弱问题：“`torch.compile` 快不快？”
强问题：

> 在固定 GPU、BF16、模型与 batch 下，steady-state speedup 如何随 sequence/model size
> 变化？compile cost 需要多少 steps 才能 amortize？

预先定义 primary metric、secondary metrics、expected mechanism、OOM/数值 failure criterion
与 generalization boundary。若只在一个 shape、一次运行上测量，结论只能是 case study。

### 7.4 Differential correctness

优化实现需与 reference 比较 outputs、loss、gradients、optimizer update 和多 step drift，
并覆盖 edge shapes、mask、mixed precision 与 RNG state。`allclose` tolerance 应由 dtype
与误差传播解释，不能为通过测试任意放宽。

## 8. Shape / 复杂度速查

| 操作 | 输入 → 输出 | Forward FLOPs（约） | 主要内存风险 |
|---|---|---:|---|
| Embedding | `[B,T] → [B,T,d]` | lookup，非 dense FLOPs | \(Vd\) weights |
| Linear | `[B,T,d_i] → [B,T,d_o]` | \(2BTd_id_o\) | input 保存、weights |
| Q/K/V | `[B,T,d] → 3×[B,h,T,d_h]` | \(6BTd^2\) | QKV activations |
| Scores | `QKᵀ → [B,h,T,T]` | \(2BT^2d\) | \(BT^2h\) matrix |
| Attention × V | `[B,h,T,T]×V` | \(2BT^2d\) | probabilities |
| SwiGLU | `[B,T,d] → [B,T,d]` | \(6BTdf\) | two width-\(f\) branches |
| LM head | `[B,T,d] → [B,T,V]` | \(2BTdV\) | logits \(BTV\) |
| AdamW | parameter-shaped | \(O(P)\) | moments \(8P\) bytes in fp32 |

## 9. 实现映射（本仓库）

| 概念 | 路径 / 符号 | 阅读问题 |
|---|---|---|
| `einsum` Linear | `assignments/spring2026/assignment1-basics/cs336_basics/model.py`：`Linear.forward` | weight 为何是 `[d_out,d_in]`？ |
| Head reshape | 同文件：`CausalMultiHeadSelfAttention.forward` | `[B,T,d]` 如何变 `[B,h,T,d_h]`？ |
| FLOPs 组件 | 同文件：Q/K/V/O、`scaled_dot_product_attention`、`SwiGLU` | 对照本讲逐项计数 |
| Mixed precision | `.../cs336_basics/training.py`：`torch.autocast` | 哪些状态仍是 fp32？ |
| Gradient accumulation | 同文件：`micro_loss / gradient_accumulation_steps` | step/clip 在循环内还是外？ |
| Peak memory / throughput | 同文件：metrics record | 指标是 allocated 还是 reserved？ |
| AdamW states | `.../cs336_basics/optimizer.py`：`AdamW.step` | `m`、`v` 何时创建、什么 dtype？ |
| A1 accounting 结果 | `.../report/main.tex` 第 3.3、4.3 节 | 理论假设与实测如何分开？ |
| Shape 契约 | `.../tests/adapters.py`、`test_model.py` | 任意 leading dims、state keys |

## 10. 易错点与反例

1. **FLOPs 与 FLOP/s 混用。** 前者是工作量，后者是速率。
2. **把参数量当训练显存。** Adam states 与 activations 往往更大。
3. **所有 matmul 都套 \(2mnk\)，却忘记 batch/sequence。**
4. **只算 attention 的 \(T^2\)，漏掉 QKVO、FFN 和 LM head。** 短 context 下后者可能主导。
5. **假设 backward 等于 forward。** 对 matmul 粗略是 2 倍 forward；特殊算子需单独看。
6. **把 view 当拷贝，或把拷贝当 view。** `permute` 后 `.contiguous()` 可能搬运数据。
7. **认为 bf16 与 fp16 数值范围相同。** 两者都是 2 bytes，但 exponent 不同。
8. **按公式算出“可放下”就断言能运行。** 临时 workspace、allocator fragmentation、CUDA context
   和 kernel 选择都会抬高峰值。
9. **用更大 batch 的相同步数比较 loss。** 它看过更多 tokens；应固定 token budget。
10. **把 gradient accumulation 当免费大 batch。** 它节省显存，但可能降低 kernel efficiency。
11. **把 activation checkpointing 当总内存除以二。** 它只作用于可重算 activations。
12. **把高 GPU utilization 当高 MFU。** memory-bound kernel 也能让设备一直忙。
13. **把 ZeRO/FSDP 当无损省显存。** 分片引入参数 all-gather，小卡数/低带宽下可能
    把瓶颈从显存搬到通信。
14. **按 byte 数直推 fp8 的显存与质量。** fp8 依赖 E4M3/E5M2 格式与
    per-tensor/per-channel scaling 实现，账本要按实际保存的 scale tensor 重算。

## 11. 实践 Checklist

- [ ] 为每个 forward 写出带语义轴名的 shape。
- [ ] 用 `numel() * element_size()` 验证 tensor bytes。
- [ ] 明确 FLOP 约定：FMA 算 1 还是 2。
- [ ] 手算一个小 Transformer 的参数量并与 `sum(p.numel())` 对齐。
- [ ] 分别核算 QKVO、attention、FFN、LM head FLOPs。
- [ ] 列出 parameter / gradient / optimizer / activation dtype。
- [ ] 测量 peak allocated memory，而非只看 checkpoint 文件。
- [ ] 预热后计时，并在 CUDA 计时点同步。
- [ ] 固定 token budget 扫 physical batch；同时记录 tokens/s、VRAM、loss。
- [ ] 比较 accumulation 前后时，把 global batch、clip 时机和数据顺序写清。
- [ ] 对照硬件 dtype 峰值计算 MFU，并记录假设。
- [ ] 对 memory-bound 猜测，用 arithmetic intensity 或 profiler 证据验证。

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| 第一次 step 极慢 | 分离 compile/warm-up | CUDA context、JIT、autotune |
| 计时异常快 | 是否同步 | 只测到 asynchronous launch |
| 显存逐 step 增长 | live graph / Python list | 未 `detach()`、retain graph |
| reserved 很高、allocated 较低 | allocator snapshot | fragmentation/caching |
| matmul 很慢 | shape、layout、dtype | 小 GEMM、非 contiguous、未用 tensor core |
| GPU 100% 但 MFU 低 | kernel mix、bandwidth | memory/launch bound |
| compile 无加速 | graph breaks / shape cache | 动态控制流、频繁新 shape |
| accumulation loss 不一致 | loss scaling、clip/step 时机 | 未除 \(K\)、每 microbatch step |
| mixed precision NaN | reduction dtype / scaling | fp16 overflow/underflow |
| 理论能放下却 OOM | temporary/workspace/reserved | 漏算 activation 或 fragmentation |

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| 训练 OOM | 参数+梯度+优化器+激活账本 | 未计 optimizer state；未用 checkpointing |
| 推理也 OOM | 参数量×dtype 与引用释放 | 张量被计算图/全局变量持有 |
| MFU 远低于预期 | arithmetic intensity 与 Roofline | 小 batch GEMM、launch-bound 循环 |
| 实测比理论慢数倍 | 通信/数据加载 overlap | CPU 预处理瓶颈、未异步预取 |
| matmul 结果错乱 | dtype 与 broadcast 审计 | fp16 溢出、维度隐式广播 |
| loss 出现 NaN | loss scaling 与梯度范数 | 混合精度未用 GradScaler、上溢 |
| 长时间运行显存缓涨 | allocator 碎片化 | 变长 shape 反复分配；固定 shape 或设分配器策略 |
| kernel 计时不可信 | CUDA 异步语义 | 未 `synchronize` 就取时间戳 |

## 12. 作业关联

- A1 `linear` / `embedding`：tensor shape 与参数布局；
- `scaled_dot_product_attention` / `multihead_self_attention`：\(T^2\) shape 与 FLOPs；
- `transformer_accounting`：参数、forward FLOPs、加载内存；
- `adamw_accounting`：optimizer states、activation memory、训练时间；
- `batch_size_experiment`：occupancy、显存与固定 token budget；
- `training_together`：autocast、throughput、peak VRAM、experiment log；
- A2 的 profiling、FlashAttention、DDP/FSDP 都建立在本讲的“算什么、搬多少、存多久”上。

## 13. 讨论：效度威胁与结论边界

### 14.1 Construct validity

- FLOPs 是算法工作量近似，不包含所有 memory、launch、communication 和 control overhead；
- peak allocated memory 不等于 process VRAM，也不等于可用 batch；
- MFU 依赖 FLOP 公式和峰值规格，跨论文比较前必须统一 numerator/denominator；
- GPU utilization 只表示设备忙，不表示有效模型计算。

### 14.2 Internal validity

- 未同步 CUDA、未 warm-up 或把 compile time 混入，会系统性扭曲计时；
- 不同 batch/sequence 同 step 数看过的 tokens 不同，不能归因于优化；
- dtype、TF32、autocast、determinism 或 kernel version 未固定，会成为 confounder；
- profiler 自身有 overhead，尤其是 stack trace、shape record 和 memory history。

### 14.3 External validity

- 单 GPU、单 shape、单软件版本的 speedup 不自动迁移到其他硬件/规模；
- microbenchmark kernel speedup 不等于 end-to-end time-to-quality；
- synthetic tensors 可能缺少真实 padding、sparsity、data pipeline 和 communication；
- 峰值规格是理论上限，power/thermal/clock 与共享环境会改变可达性能。

### 14.4 研究报告最低信息

硬件型号/数量、driver/CUDA/framework/commit、模型与 shapes、dtype、warm-up/repetitions、
synchronization、统计量、OOM/error policy、correctness tolerance、完整运行命令。
缺少这些信息的“X× 加速”不可可靠复现。

## 面试要点速记

**高频问题与答题要点**

1. **Q：训练 1B 模型最少要多少显存（不含激活）？** 要点：bf16 参数 2N + 梯度
   2N + Adam fp32 状态（m、v、master weights）12N ≈ **16N bytes** → 1B ≈ 16GB。
2. **Q：\(C\approx6ND\) 的 6 从哪来？** 要点：前向 ~2ND，反向 ~4ND（对权重梯度
   需重放两次矩阵乘）。
3. **Q：什么时候是 memory-bound？** 要点：arithmetic intensity（FLOPs/字节）
   低于机器峰值算力/带宽之比：elementwise、小 batch decode、optimizer step。
4. **Q：activation checkpointing 的代价？** 要点：只存层边界激活、反向时重算
   → 显存大降、时间 +约 33%（多一次前向）。
5. **Q：MFU 怎么算、多少算健康？** 要点：实测 FLOPs /（峰值 FLOPs × 时间）；
   大模型密集训练 40–60% 正常，低于 ~30% 先查通信、数据加载与小 kernel。
6. **Q：ZeRO 三个 Stage 各省多少显存？** 要点（DeepSpeed 业界口径）：
   Stage-1 优化器分片 ≈ 12Φ/P + 4Φ；Stage-2 再分梯度 ≈ 14Φ/P + 2Φ；
   Stage-3 全分片 ≈ 16Φ/P（Φ=参数量、P=卡数）。
7. **Q：为什么 activation checkpointing 有时反而"省时间"？** 要点：省下的激活
   显存换更大 batch → GEMM 复用更强、kernel 效率提升，可能抵消约 +33% 的重算开销。
8. **Q：bf16 与 fp16 的本质区别？** 要点：同为 2 bytes；fp16 5-bit exponent 易
   underflow 需 loss scaling；bf16 8-bit exponent 动态范围接近 fp32、可免 scaling。
9. **Q：GPU 利用率 100% 但训练慢，先查什么？** 要点：utilization ≠ MFU；
   查 kernel mix（memory-bound 小算子）、通信是否重叠、数据加载空泡。

**必背数字**

- 16N bytes 规则（Adam+bf16 训练态）；6ND；checkpointing 时间 ~1.33×；
  roofline 拐点 = 峰值算力 / 峰值带宽。

**工业界参照（2024–2026 口径）**

- 硬件峰值（SXM 口径）：A100 BF16 312 TFLOPS / 80GB HBM2e / 2TB/s；
  H100 BF16 dense 989 TFLOPS / 80GB HBM3 / 3.35TB/s / NVLink4 900GB/s；
  H200 141GB HBM3e / 4.8TB/s；B200 BF16 dense 2.25 PFLOPS、FP8 dense 4.5 PFLOPS /
  192GB HBM3e / 8TB/s / NVLink5 1.8TB/s。
- ZeRO 分片公式：Stage-1 ≈ 12Φ/P + 4Φ；Stage-2 ≈ 14Φ/P + 2Φ；Stage-3 ≈ 16Φ/P。
- MFU 实测参照：业界教程案例 LLaMA1-65B @ 昇腾 910B2 约 0.46；
  第一梯队集群一般 40%–55%；训练 FLOPs 简化口径 ≈ 6·N_active·D。
- H100 FP8（配合 DeepSpeed）显存约 -42%；1-bit Adam 梯度通信压缩 5 倍。
- activation checkpointing：全量重计算约 +33% 计算；选择性重计算显存降 30%–70%。
- 通信兜底：NCCL ring all-reduce；GPU 通信时延约 1µs（NVLink/IB）。

## 行业现状与最新进展（2024–2026）

### 混合精度训练的显存账本（AdamW + bf16）

业界通用口径把训练状态拆成"fp32 训练态 + bf16 计算态"两部分：

| 项目 | dtype | 每参数 bytes | 说明 |
|---|---|---:|---|
| master weights | fp32 | 4 | 权重主副本，更新语义的正确性来源 |
| AdamW 一阶矩 m | fp32 | 4 | 与参数同 shape |
| AdamW 二阶矩 v | fp32 | 4 | 与参数同 shape |
| fp32 训练态小计 | | 12 | 即 ZeRO 论文口径的 12Φ |
| 参数 | bf16 | 2 | 前向/反向实际计算精度 |
| 梯度 | bf16 | 2 | backward 产物 |
| 合计（不含激活） | | ~16 | 即 16N bytes 规则 |

7B 模型：16 × 7G ≈ 112GB——单张 80GB 的 H100 放不下训练态本身，这正是
分片并行（ZeRO / FSDP）的出发点。

### ZeRO / FSDP：分片对显存公式的改写

本讲概念与工业界实践/数字的对照：

| 本讲概念 | 工业界实践/数字 |
|---|---|
| 16N bytes 规则 | ZeRO 口径的 16Φ（bf16 参数/梯度 + fp32 master/m/v） |
| optimizer states | Stage-1 分片：≈ 12Φ/P + 4Φ |
| gradients | Stage-2 再分片：≈ 14Φ/P + 2Φ |
| parameters | Stage-3 全分片：≈ 16Φ/P |
| activation checkpointing | 全量重算约 +33% 计算；selective 重算显存降 30%–70% |
| MFU | 第一梯队集群 40%–55%；LLaMA1-65B 案例约 0.46 |

各方案的显存与通信代价：

| 方案 | 每卡训练态显存（Φ=参数量，P=卡数） | 通信模式变化 |
|---|---|---|
| 不分片（DDP） | ~16Φ | 梯度 all-reduce |
| ZeRO-1 | 12Φ/P + 4Φ | 优化器状态 all-gather |
| ZeRO-2 | 14Φ/P + 2Φ | 梯度 reduce-scatter |
| ZeRO-3 / FSDP | 16Φ/P | 参数 all-gather（前后向各一次） |

FSDP 是 PyTorch 原生的 ZeRO-3 等价物：自动 bucket 划分、支持异构分片。
FSDP2（PyTorch 2.4 起）升级为 per-parameter sharding、支持参数重分片，易用性
大幅提升，兼得 ZeRO-3 级显存优化。ZeRO 论文见
[arXiv:1910.02054](https://arxiv.org/abs/1910.02054)，FSDP 论文见
[arXiv:2304.11277](https://arxiv.org/abs/2304.11277)。

### FP8 与新一代硬件的显存变化

- H100 上 FP8（配合 DeepSpeed）可减少约 42% 显存占用；1-bit Adam 把梯度通信
  压缩 5 倍——显存与通信可以一起省。
- B200：BF16 dense 2.25 PFLOPS、FP8 dense 4.5 PFLOPS、192GB HBM3e、8TB/s、
  NVLink5 1.8TB/s；算力、带宽、互联同步上台阶，通信/计算重叠窗口更宽。
- 实测案例（LLaMA2-70B，TP=8，单节点）：H200 454.6 TFLOPS/GPU vs
  B200 815.2 TFLOPS/GPU（+79%）；MLPerf 口径 B200 对 H200：GPT-3 175B
  预训练约 2.0×、Llama-70B LoRA 约 2.2×。
- 但 fp8 的 E4M3/E5M2 格式与 per-tensor/per-channel scaling 强依赖实现：
  显存账本要按实际保存的 scale tensor 重算，不能只按 byte 数减半推总账。

### 对本讲学习者的启示

本讲的 `shape → bytes → bottleneck` 链路正是工业界训练前可行性评估（feasibility
check）的核心：16B/参数规则决定卡数下限；ZeRO 公式决定分片档位；6·N_active·D
除以峰值与 MFU（40%–55% 是第一梯队水平）决定训练时长；activation
checkpointing（全量重算 +33%，或 selective 重算显存降 30%–70%）决定 batch
上限。把这套账本练成手算能力，面试与工程排障都直接受益。

## 大厂面试真题与答题框架

高频面试题（公开面经风格），以下为建议答题框架。

**题目 1：手算 7B 模型全参训练的显存构成**
- 考点：16B/参数规则、混合精度状态拆分、从账本推出并行方案。
- 答题框架：
  1. 先分状态：fp32 master 4B + m 4B + v 4B = 12B/参数；bf16 参数 2B + bf16 梯度 2B；
  2. 合计 ~16B/参数 → 7B ≈ 112GB（不含激活）；
  3. 单张 80GB 放不下 → 引出分片：ZeRO-3 下 16Φ/P，P=8 时每卡训练态 14GB；
  4. 再补激活：随 B、T、L 增长，说明 activation checkpointing 的必要性；
  5. 声明口径：业界通用 AdamW+bf16，不含临时 workspace 与 allocator 预留。
- 加分项：区分 allocated 与 reserved；指出 fp32 master 是否保留依实现而异。
- 踩坑：把 16N bytes 说成"模型文件大小"；漏掉激活和通信 buffer。

**题目 2：为什么 activation checkpointing 反而可能"省时间"？**
- 考点：时空交换、arithmetic intensity 与 batch 的联动。
- 答题框架：
  1. 直接代价：全量重计算约 +33% 计算（多一次前向）；
  2. 换来的显存可投给更大 batch → GEMM 更大、复用更强、kernel 效率更高；
  3. 大 batch 的吞吐增益可能抵消重算开销，净效应为正；
  4. 更进一步用 selective recomputation：只重算 attention 等 T² 项，
     显存降 30%–70%、计算增量更小；
  5. 结论绑定具体 shape/hardware——这是权衡，不是免费优化。
- 加分项：给出 per-op 的"每节省 1 byte 的重算代价"排序思路。
- 踩坑：说"checkpointing 把显存减半"；忽略它不减少参数/梯度/优化器状态。

**题目 3：MFU 为什么上不去？**
- 考点：MFU 定义、瓶颈归因、roofline 证据链。
- 答题框架：
  1. 先写定义：MFU = 模型理论 FLOPs/step × steps/s ÷（峰值 FLOP/s × 卡数）；
  2. 给参照系：第一梯队集群一般 40%–55%（业界教程案例 LLaMA1-65B 约 0.46）；
  3. 逐项归因：通信未重叠、小 batch GEMM、elementwise/optimizer kernel 是
     memory-bound、数据加载空泡、recompute 的额外 FLOPs 口径不一致；
  4. 用 profiler 证据区分 compute/memory/launch/communication bound；
  5. 对应优化：增大 batch、kernel fusion、通信 overlap、selective recomputation。
- 加分项：指出 MFU 依赖 FLOPs 公式口径（dense vs executed、是否含重算）。
- 踩坑：把 GPU utilization 100% 当 MFU 高；把 wall-clock 差异全归因于 MFU。

**题目 4：ZeRO 三个 Stage 各省什么、代价是什么？**
- 考点：ZeRO 分片公式、通信换显存。
- 答题框架：
  1. 不分片 ~16Φ；Stage-1 优化器分片 ≈ 12Φ/P + 4Φ；
  2. Stage-2 再分梯度 ≈ 14Φ/P + 2Φ；Stage-3 全分片 ≈ 16Φ/P；
  3. 通信从梯度 all-reduce 变为参数 all-gather（每层前后向各一次）；
  4. 选型逻辑：单卡放得下参数与梯度、只是优化器态大 → Stage-1/2；
     参数本身放不下 → Stage-3；
  5. FSDP 是 PyTorch 原生等价物，FSDP2（PyTorch 2.4 起）per-parameter
     sharding、支持参数重分片，易用性大幅提升。
- 加分项：把 12Φ/P 中的 12 对应到 m/v/master 各 4B 的拆分。
- 踩坑：只背公式说不出通信代价；忽略 Stage-3 下 bucket/all-gather 的额外 buffer。

**题目 5：bf16 / fp16 / fp8 怎么选？**
- 考点：dtype 取舍、数值边界、显存账本联动。
- 答题框架：
  1. fp16：5-bit exponent，范围小，梯度 underflow 需 loss scaling；
  2. bf16：8-bit exponent 接近 fp32 动态范围，大规模训练可免 loss scaling；
  3. fp8（E4M3/E5M2）：需 per-tensor/per-channel scaling，H100 上配合
     DeepSpeed 显存约 -42%；
  4. 显存差异：2B vs 2B vs 1B，但优化器 master 通常仍是 fp32；
  5. 报告 fp8 结果必须写明格式、scaling 策略与失败处理。
- 加分项：举 RMSNorm 前先 upcast fp32 的例子说明 reduction 对精度敏感。
- 踩坑：认为 bf16 与 fp16 范围相同；按 byte 数直接减半推 fp8 总账。

**题目 6：训练前怎么估一次实验要多久？**
- 考点：6·N_active·D、MFU、tokens/s 交叉校验。
- 答题框架：
  1. 训练 FLOPs ≈ 6·N_active·D（声明 N 口径：是否含 embedding/head、是否 tied）；
  2. 按硬件峰值与目标 MFU（第一梯队 40%–55%）估有效 FLOP/s；
  3. 时间 ≈ FLOPs ÷（卡数 × 峰值 × MFU）；
  4. 交叉校验：用 tokens/s × step 数做 sanity check；
  5. 声明误差来源：长 context 下 6ND 比例失真、通信、数据管线。
- 加分项：提 Chinchilla 配比下 D 与 N 的联动（引 Lecture 9/11）。
- 踩坑：把 6ND 当精确值；忽略 attention 的 T² 项在长 context 下的占比。

**题目 7：训练 OOM，排查顺序是什么？**
- 考点：memory accounting 体系化、allocator 行为。
- 答题框架：
  1. 先核对账本：参数+梯度+优化器+激活是否算全（16B/参数 + 激活公式）；
  2. 分离 reserved 与 allocated：是 fragmentation 还是容量真不够；
  3. 查泄漏：未 detach 的 loss/output 跨 step 存活，观察 allocated baseline 是否持续增长；
  4. 降峰值手段排序：gradient accumulation → activation checkpointing /
     selective recomputation → ZeRO 分片 → offload；
  5. 最后才 empty_cache()，并说明它不释放 live tensors。
- 加分项：用 memory snapshot 定位 allocation stack 与 lifetime。
- 踩坑：一上来就减 batch；把 fragmentation 误判为容量不足。

## 系统设计题

**设计题 1：在 8×H100（80GB）上做 70B 全参训练——先给显存预算表，再判可行性**

- 需求澄清：全参 vs LoRA；序列长度与 batch；是否允许 CPU offload；MFU 目标。
- 规模估算：
  - 训练态：70B × 16B ≈ 1120GB（bf16 参数/梯度 + fp32 master/m/v）；
  - 单节点总显存：8 × 80GB = 640GB < 1120GB；
  - ZeRO-3 全分片：16Φ/P = 1120/8 = 140GB/卡 > 80GB——单节点不可行；
  - 反推卡数：留约 40% 余量给激活/通信/临时量，每卡训练态 ≤ ~48GB →
    P ≥ 1120/48 ≈ 24 卡（3 节点）起步；
- 架构：3 节点及以上 ZeRO-3 / FSDP2 全分片 + activation checkpointing
  （全量重算 +33% 计算，或 selective 重算显存降 30%–70%）+
  gradient accumulation 控激活峰值；节点内 NVLink4 900GB/s，节点间走
  NCCL ring all-reduce（IB 时延 ~1µs 量级）。
- Trade-off 表：

| 手段 | 显存收益 | 时间/通信代价 | 适用前提 |
|---|---|---|---|
| 加节点（增大 P） | 训练态 16Φ/P 线性降 | 通信总量随 P 增长 | 集群与成本允许 |
| ZeRO-3 / FSDP2 | 训练态全分片 | 参数 all-gather 每层两次 | 带宽充足 |
| CPU offload | 训练态移出 HBM | PCIe 换步时间 | 最后手段 |
| activation checkpointing | 激活近线性下降 | 计算 +约 33% | 激活是主要矛盾时 |
| gradient accumulation | 激活峰值 ÷ K | kernel 效率可能下降 | 需大 global batch 时 |

- 评测方案：报告 tokens/s、peak allocated、MFU（对照 40%–55% 健康区间）、
  前几步 loss 与 fp32 参考实现的数值一致性。
- 追问预案：为什么不以 TP 为主（TP=8 也压不下 140GB/卡的训练态）；
  FP8 是否值得（H100 FP8 + DeepSpeed 显存约 -42%，但需评估数值稳定性）。

**设计题 2：训练前"卡数-显存-时间"可行性估算器**

- 需求澄清：输入（参数量 N、tokens D、序列长度、目标 global batch）；输出
  （最少卡数、每卡显存预算、预计天数）；允许的并行策略集合。
- 规模估算（公式库，均来自本讲口径）：
  - 训练 FLOPs ≈ 6·N_active·D；
  - 训练态显存：不分片 ~16Φ；ZeRO-1/2/3 分别 12Φ/P+4Φ、14Φ/P+2Φ、16Φ/P；
  - 时间 ≈ 6ND ÷（P × 峰值 FLOP/s × MFU），MFU 取 40%–55% 区间做敏感性分析；
  - 激活：全量重算 +33% 计算，或 selective 重算显存降 30%–70%。
- 架构：CLI 读 config → 账本计算（训练态 + 激活 + 通信 buffer）→ 策略枚举
  （P、ZeRO 档位、checkpointing 档位）→ 过滤显存可行解 → 按（卡数, 天数）
  输出 Pareto 前沿；对峰值另加安全余量。
- Trade-off 表：

| 维度 | 保守假设 | 激进假设 | 风险 |
|---|---|---|---|
| 每参数字节 | 16B（含 fp32 master） | 12B（无 master） | 数值语义变化 |
| MFU | 40% | 55% | 时间低估 |
| 激活策略 | 不重算 | 全量重算 | OOM / 时间 +33% |
| 通信 | 串行计入 | 完全重叠 | 时间低估 |

- 评测方案：用已公开实测案例回测（LLaMA1-65B @ 昇腾 910B2 的 MFU ≈ 0.46；
  LLaMA2-70B @ H200 454.6 TFLOPS/GPU 的量级），误差在可接受区间内为可用。
- 追问预案：长 context 下 6ND 失真怎么办——换逐项 FLOPs 公式
  F = 3E + 4B·s²·l·h + 6B·s·l·h·V（业界教程即按此口径估出 MFU ≈ 0.46）。

**设计题 3：推理服务的"卡数-显存"估算（连接本讲的 decode memory-bound）**

- 需求澄清：并发数、序列长度、是否量化、目标时延（TTFT / p95）。
- 规模估算：decode 是矩阵向量乘，权重每 token 读取一次 → arithmetic
  intensity O(1) → memory-bound；吞吐上界 ≈ 显存带宽 ÷ 每 token 读取 bytes
  （bf16 约 2B/参数）；带宽参照 H100 3.35TB/s、H200 4.8TB/s、B200 8TB/s。
- 架构：batch 调度提升权重复用；KV cache（随并发 × 序列长度增长）与权重
  显存二分预算；continuous batching 提高利用率。
- Trade-off 表：

| 手段 | 显存/吞吐收益 | 代价 |
|---|---|---|
| 增大 batch | 权重读取摊薄、吞吐升 | KV cache 线性涨 |
| 量化（fp8 等） | 权重与 KV 同降 | 精度损失需评测 |
| PagedAttention | KV 碎片减少 | 实现复杂度 |

- 评测方案：tokens/s/GPU、TTFT、p50/p95 时延；口径对照 vLLM 文档（docs.vllm.ai）。
- 追问预案：为什么训练 GEMM 偏 compute-bound 而推理 decode 偏 memory-bound
  ——复用率差异（矩阵×矩阵 vs 矩阵×向量）。

## 代码实现题

**实现题 1：param_bytes 显存会计函数**

- 题目：给定模型，按"参数/梯度/优化器状态"三类、按 dtype 统计训练显存，
  输出可与 16B/参数 手算口径对账。
- 考察点：dtype × element_size 核算、AdamW 状态生命周期、口径声明。
- Python 骨架：

```python
import torch

def param_bytes_accounting(model: torch.nn.Module,
                           fp32_master: bool = True) -> dict:
    """按 dtype 统计参数/梯度/优化器状态字节数（AdamW+bf16 业界口径）。"""
    params, grads = {}, {}
    n_trainable = 0
    for p in model.parameters():
        key = str(p.dtype).replace("torch.", "")
        params[key] = params.get(key, 0) + p.numel() * p.element_size()
        n_trainable += p.numel() if p.requires_grad else 0
        if p.grad is not None:
            gkey = str(p.grad.dtype).replace("torch.", "")
            grads[gkey] = grads.get(gkey, 0) \
                + p.grad.numel() * p.grad.element_size()
    opt_m_v = 8 * n_trainable        # fp32 的 m、v 各 4B/参数
    master = 4 * n_trainable if fp32_master else 0
    total = sum(params.values()) + sum(grads.values()) + opt_m_v + master
    return {
        "params": params,              # bf16: 2B/参数
        "grads": grads,                # bf16: 2B/参数（backward 后非空）
        "optimizer_m_v_fp32": opt_m_v, # 8B/参数
        "fp32_master_weights": master, # 4B/参数，视实现
        "total_bytes": total,
        "bytes_per_param": total / max(n_trainable, 1),
    }
```

- 验收标准：对 bf16 的 7B 模型，backward 后输出
  参数 14GB + 梯度 14GB + m/v 56GB + master 28GB = 112GB，
  `bytes_per_param ≈ 16`；`fp32_master=False` 时 ≈ 12；说明口径不含激活。

**实现题 2：模拟 ZeRO 分片的显存计算器**

- 题目：输入参数量 Φ、卡数 P、卡上 HBM 容量与激活预算，输出各 ZeRO Stage
  的每卡训练态显存与可行性判定。
- 考察点：ZeRO 公式落地、边界条件（P=1、显存余量）、单位换算。
- Python 骨架：

```python
def zero_stage_memory(phi: int, world_size: int,
                      hbm_bytes_per_gpu: int,
                      activation_bytes: int = 0,
                      safety_margin: float = 0.2) -> dict:
    """按 DeepSpeed ZeRO 业界口径估算各 Stage 每卡训练态显存。"""
    P = max(1, world_size)
    ddp    = 16 * phi
    stage1 = 12 * phi // P + 4 * phi
    stage2 = 14 * phi // P + 2 * phi
    stage3 = 16 * phi // P
    budget = int(hbm_bytes_per_gpu * (1 - safety_margin)) - activation_bytes
    stages = {"ddp": ddp, "stage1": stage1,
              "stage2": stage2, "stage3": stage3}
    return {
        **{f"{k}_per_gpu": v for k, v in stages.items()},
        "feasible": {k: v <= budget for k, v in stages.items()},
        "per_gpu_budget_after_activation": budget,
    }

# 例：phi=7e9、P=8、80GB H100、激活预算 20GB、余量 20%
# stage3 ≈ 16*7e9/8 = 14GB（feasible），ddp = 112GB（infeasible）
```

- 验收标准：Φ=7e9、P=8、80GB 卡时 `stage3_per_gpu ≈ 14GB` 且 feasible、
  `ddp ≈ 112GB` 不可行；P=1 时各 Stage 均退化为 16Φ，与公式一致；
  `safety_margin` 只作用于容量、不改变公式项。

**实现题 3：用 CUDA events 的可信 step 计时器**

- 题目：实现 warm-up + repetitions 的训练 step 计时，输出 sec/step、
  tokens/s 与估算 MFU。
- 考察点：CUDA 异步语义、同步边界、MFU 口径。
- Python 骨架：

```python
import torch

def timed_train_step(step_fn, model_flops_per_step: int,
                     peak_flops: int, num_gpus: int,
                     tokens_per_step: int,
                     warmup: int = 3, reps: int = 10) -> dict:
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        step_fn()
    end.record()
    torch.cuda.synchronize()
    sec_per_step = start.elapsed_time(end) / 1e3 / reps
    mfu = model_flops_per_step / sec_per_step / (peak_flops * num_gpus)
    return {
        "sec_per_step": sec_per_step,
        "tokens_per_sec": tokens_per_step / sec_per_step,
        "mfu": mfu,
    }
```

- 验收标准：去掉 synchronize 时测出的时间显著偏短（识别 launch-only 假象）；
  已知峰值与 FLOPs 公式下 MFU 输出落在 0–1 且与手算一致；健康的大模型密集
  训练应在 0.40–0.55 量级（第一梯队口径）。

## 14. 结论与本讲小结

PyTorch 编程的核心不是记 API，而是保持 tensor 轴、dtype、device 与生命周期可审计。
FLOPs 决定理论工作量，memory accounting 决定能否运行，arithmetic intensity 决定硬件能否接近峰值。
梯度累积和 activation checkpointing 都是资源交换，不会凭空消除工作；selective
recomputation 把这一交换细化到算子粒度，使"换什么"成为可优化的设计变量。
任何性能结论都应同时给出数学成本、硬件上界和实测证据。

## 参考文献

<a id="ref-1"></a>[1] A. Paszke et al. “PyTorch: An Imperative Style,
High-Performance Deep Learning Library.” *NeurIPS*, 2019.
https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library

<a id="ref-2"></a>[2] C. R. Harris et al. “Array Programming with NumPy.”
*Nature*, 2020. https://doi.org/10.1038/s41586-020-2649-2

<a id="ref-3"></a>[3] S. Williams, A. Waterman, D. Patterson. “Roofline:
An Insightful Visual Performance Model for Multicore Architectures.”
*Communications of the ACM*, 2009. https://doi.org/10.1145/1498765.1498785

<a id="ref-4"></a>[4] P. Micikevicius et al. “Mixed Precision Training.”
*ICLR*, 2018. https://openreview.net/forum?id=r1gs9JgRZ

<a id="ref-5"></a>[5] T. Chen et al. “Training Deep Nets with Sublinear Memory Cost.”
arXiv:1604.06174, 2016. https://arxiv.org/abs/1604.06174

<a id="ref-6"></a>[6] M. Shoeybi et al. “Megatron-LM: Training Multi-Billion
Parameter Language Models Using Model Parallelism.” arXiv:1909.08053, 2019.
https://arxiv.org/abs/1909.08053

<a id="ref-7"></a>[7] S. Rajbhandari et al. “ZeRO: Memory Optimizations Toward
Training Trillion Parameter Models.” *SC*, 2020.
https://arxiv.org/abs/1910.02054

<a id="ref-8"></a>[8] Y. Zhao et al. “PyTorch FSDP: Experiences on Scaling
Fully Sharded Data Parallel.” *VLDB*, 2023.
https://arxiv.org/abs/2304.11277

<a id="ref-9"></a>[9] T. Dao et al. “FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness.” *NeurIPS*, 2022.
https://arxiv.org/abs/2205.14135

<a id="ref-10"></a>[10] V. Korthikanti, J. Casper, S. Lym, L. McAfee, et al.
“Reducing Activation Recomputation in Large Transformer Models.” *MLSys*,
2023. [arXiv](https://arxiv.org/abs/2205.05198)

<a id="ref-11"></a>[11] A. Chowdhery et al. “PaLM: Scaling Language Modeling
with the Pathways Language Model.” arXiv:2204.02311, 2022.
[link](https://arxiv.org/abs/2204.02311)

<a id="ref-12"></a>[12] J. Kaplan et al. “Scaling Laws for Neural Language
Models.” arXiv:2001.08361, 2020. [link](https://arxiv.org/abs/2001.08361)

<a id="ref-13"></a>[13] D. P. Kingma and J. Ba. “Adam: A Method for Stochastic
Optimization.” *ICLR*, 2015. [link](https://arxiv.org/abs/1412.6980)

<a id="ref-14"></a>[14] I. Loshchilov and F. Hutter. “Decoupled Weight Decay
Regularization.” *ICLR*, 2019. [link](https://arxiv.org/abs/1711.05101)

<a id="ref-15"></a>[15] D. Kalamkar, D. Mudigere, N. Mellempudi, et al.
“A Study of BFLOAT16 for Deep Learning Training.” arXiv:1905.12322, 2019.
[link](https://arxiv.org/abs/1905.12322)

<a id="ref-16"></a>[16] P. Micikevicius, D. Stosic, N. Burgess, et al.
“FP8 Formats for Deep Learning.” arXiv:2209.05433, 2022.
[link](https://arxiv.org/abs/2209.05433)

## 延伸阅读与实践材料

- Stanford CS336, [Spring 2026 Lecture 2](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)
- PyTorch, [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- einops, [Einstein notation and rearrange](https://einops.rocks/)
- [A1 实验报告](../assignments/spring2026/assignment1-basics/report/main.tex)
- [A2 实验报告](../assignments/spring2026/assignment2-systems/report/writeup.pdf)
- [Systems 主题导航](../experiments/topics/systems.md)
- [Megatron-LM（NVIDIA GitHub）](https://github.com/NVIDIA/Megatron-LM)（访问日期 2026-10-04）
- [DeepSpeed（Microsoft GitHub）](https://github.com/microsoft/DeepSpeed)（访问日期 2026-10-04）
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)（访问日期 2026-10-04）
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)（访问日期 2026-10-04）
