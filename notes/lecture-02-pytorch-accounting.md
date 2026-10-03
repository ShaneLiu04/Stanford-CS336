---
title: "Lecture 02 — PyTorch & Resource Accounting"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-04-01"
lecturer: "Percy Liang"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py"
  - "../assignments/assignment1-basics/"
---

# Lecture 02 — PyTorch 与 Resource Accounting：从 tensor shape 到资源账本

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述，非课程提交
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
| `einsum` Linear | `assignments/assignment1-basics/cs336_basics/model.py`：`Linear.forward` | weight 为何是 `[d_out,d_in]`？ |
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
| 计时异常快 / kernel 计时不可信 | 是否同步 | 只测到 asynchronous launch、未 `synchronize` |
| 显存逐 step 增长 / 长时间运行缓涨 | live graph / Python list / allocator 碎片 | 未 `detach()`、retain graph、变长 shape 反复分配 |
| reserved 很高、allocated 较低 | allocator snapshot | fragmentation/caching |
| 训练/推理 OOM | 参数+梯度+优化器+激活账本 | 未计 optimizer state、未用 checkpointing、被计算图持有 |
| 理论能放下却 OOM | temporary/workspace/reserved | 漏算 activation 或 fragmentation |
| matmul 很慢 | shape、layout、dtype | 小 GEMM、非 contiguous、未用 tensor core |
| matmul 结果错乱 | dtype 与 broadcast 审计 | fp16 溢出、维度隐式广播 |
| GPU 100% 但 MFU 低 | kernel mix、bandwidth | memory/launch bound、小 batch GEMM |
| MFU 远低于预期 | arithmetic intensity 与 Roofline | 小 batch GEMM、launch-bound 循环 |
| 实测比理论慢数倍 | 通信/数据加载 overlap | CPU 预处理瓶颈、未异步预取 |
| compile 无加速 | graph breaks / shape cache | 动态控制流、频繁新 shape |
| accumulation loss 不一致 | loss scaling、clip/step 时机 | 未除 \(K\)、每 microbatch step |
| mixed precision NaN / loss 出现 NaN | reduction dtype / scaling | fp16 overflow/underflow、未用 GradScaler |

## 12. 作业关联

- A1 `linear` / `embedding`：tensor shape 与参数布局；
- `scaled_dot_product_attention` / `multihead_self_attention`：\(T^2\) shape 与 FLOPs；
- `transformer_accounting`：参数、forward FLOPs、加载内存；
- `adamw_accounting`：optimizer states、activation memory、训练时间；
- `batch_size_experiment`：occupancy、显存与固定 token budget；
- `training_together`：autocast、throughput、peak VRAM、experiment log；
- A2 的 profiling、FlashAttention、DDP/FSDP 都建立在本讲的“算什么、搬多少、存多久”上。

## 13. 讨论：效度威胁与结论边界

### 13.1 Construct validity

- FLOPs 是算法工作量近似，不包含所有 memory、launch、communication 和 control overhead；
- peak allocated memory 不等于 process VRAM，也不等于可用 batch；
- MFU 依赖 FLOP 公式和峰值规格，跨论文比较前必须统一 numerator/denominator；
- GPU utilization 只表示设备忙，不表示有效模型计算。

### 13.2 Internal validity

- 未同步 CUDA、未 warm-up 或把 compile time 混入，会系统性扭曲计时；
- 不同 batch/sequence 同 step 数看过的 tokens 不同，不能归因于优化；
- dtype、TF32、autocast、determinism 或 kernel version 未固定，会成为 confounder；
- profiler 自身有 overhead，尤其是 stack trace、shape record 和 memory history。

### 13.3 External validity

- 单 GPU、单 shape、单软件版本的 speedup 不自动迁移到其他硬件/规模；
- microbenchmark kernel speedup 不等于 end-to-end time-to-quality；
- synthetic tensors 可能缺少真实 padding、sparsity、data pipeline 和 communication；
- 峰值规格是理论上限，power/thermal/clock 与共享环境会改变可达性能。

### 13.4 研究报告最低信息

硬件型号/数量、driver/CUDA/framework/commit、模型与 shapes、dtype、warm-up/repetitions、
synchronization、统计量、OOM/error policy、correctness tolerance、完整运行命令。
缺少这些信息的“X× 加速”不可可靠复现。

## 14. 面试备考（Interview Prep）

> 本讲是 ML Systems / LLM 面试最硬核的「算账」考点：面试官会让你当场手算参数量、FLOPs、
> 训练显存、KV cache、MFU。核心不是背数字，而是掌握 `shape → FLOPs → bytes → bottleneck`
> 这条推导链。下面按「一页速览 → 高频题 → 手撕核算 → 追问」四层组织，每题仍用
> **定义 → 为什么/原理 → 公式 → 工程落地 → 边界**的框架。

### 14.1 一页速览卡（面试前 1 分钟）

**核心主张**：resource accounting 是用 `shape → FLOPs → bytes → bottleneck → experiment`
回答「模型能否放下、瓶颈是算力还是带宽、一次实验要多久」。

**必背数字与公式**

- 训练 FLOPs \(C\approx6ND\)：forward \(2ND\) + backward \(4ND\)（\(N\) 为参与 matmul 的参数量）。
- 参数量 \(P\approx12Ld^2\)（`d_ff=4d`，单层 QKV+out `4d²` + FFN `8d²`）。
- **16N 规则**：Adam+bf16 训练态 ≈ parameter 2 + gradient 2 + fp32 moments(m,v) 8 + master 4 = **16 bytes/参数**。
- 一次 matmul 的 backward ≈ 2× forward（对 X 和 W 各做一次同阶 GEMM）。
- activation checkpointing 时间 ≈ **1.33×**（多一次 forward），显存从 \(O(L)\) 降到 \(O(\sqrt L)\)。
- roofline 拐点 \(I^\*=C_{\max}/BW\)；\(I<I^\*\) 是 memory-bound，否则 compute-bound。

**三句话答高频**

1. FLOPs 决定「理论算多少」，memory 决定「能不能放」，arithmetic intensity 决定「能不能跑满」。
2. Adam 显存大头是 fp32 的 m/v（8 bytes/参数），不是模型权重。
3. backward 不是 forward 的 1 倍，对 matmul 约为 2 倍，所以总训练 ≈ 3× forward。

### 14.2 高频面试题与答题框架

**Q1：\(C\approx6ND\) 的 6 是怎么来的？什么时候失效？**

- **推导**：forward 一次 matmul 约 \(2ND\)；backward 需要算对输入的梯度和对权重的梯度，各是同阶的 GEMM，共约 \(4ND\)；合计 \(6ND\)。
- **为什么**：反向的两次 GEMM 正是「激活转置 × 输出梯度」与「输入转置 × 输出梯度」。
- **失效边界**：忽略 attention 的 \(T^2\) 项、embedding lookup、softmax/norm、optimizer、recomputation、通信；只在长训练、dense matmul 主导时量级准确。

**Q2：训练一个 1B 参数模型最少要多少显存（不含激活）？**

- **bf16+Adam 的 16N 规则**：parameter 2 + gradient 2 + Adam 两个 fp32 moment 8 + fp32 master weight 4 = 16 bytes/参数。
- 1B × 16 = **16 GB**；这还没算 activation、临时 workspace、CUDA context 与 allocator 碎片。
- **为什么 Adam 是 3 倍权重**：m、v 各 4 bytes 与参数同 shape，master weight 再 4 bytes，共 12 bytes 纯优化器状态。
- **注意**：纯 bf16 无 master weight 可降到 12N；fp32 全量是 16N 但来源不同（param 4 + grad 4 + m 4 + v 4）。

**Q3：模型参数量怎么手算？**

- 单层：QKV+out 四个 \(d\times d\) 矩阵 `4d²` + SwiGLU 三个矩阵（`d_ff=4d` 时）`3·4d²=12d²`，其中两个门 `2·4d²=8d²` 记入 FFN → 单层 `12d²`。
- 总 `12Ld²` + embedding/head `2Vd`（tied 时 1 份）。
- 例：`d=4096, L=32, V=32000` → `12×32×4096² ≈ 6.4B` 非嵌入参数，与 LLaMA-7B 量级一致。

**Q4：训练显存由哪些部分组成？激活显存怎么估？**

- 总 \(M_{\rm peak} = M_{\rm params}+M_{\rm grads}+M_{\rm opt}+M_{\rm activations}+M_{\rm temp}+M_{\rm allocator}\)。
- activation 上界形如 \(A/B \approx L(7Td+4Tf+2hT^2)+Td+2TV\)，其中 `hT²` attention 概率与 `TV` logits 在大序列/大词表下最贵。
- **工程**：用 `memory_allocated`（live）区分 `memory_reserved`（caching allocator 预留），OOM 常是碎片而非总 free 不足。

**Q5：什么是 memory-bound？arithmetic intensity 与 roofline 怎么用？**

- arithmetic intensity \(I=\text{FLOPs}/\text{bytes}\)；可达性能 \(\le\min(C_{\max}, BW·I)\)。
- 拐点 \(I^\*=C_{\max}/BW\)：低于它受带宽限制（memory-bound），高于它受算力限制（compute-bound）。
- **例子**：GEMM 的 \(I=O(n)\)（越大的矩阵复用越强，compute-bound）；逐元素 ReLU、softmax 的 \(I=O(1)\)（memory-bound）；decode 的 matvec 复用低、偏 memory-bound。

**Q6：MFU 是什么？怎么算？多少算健康？**

- \(\text{MFU}=\frac{\text{模型理论 FLOPs/step}×\text{steps/s}}{\text{峰值 FLOP/s}×\text{设备数}}\)。
- 大模型密集训练 **40–60%** 算正常；低于 ~30% 先查通信、数据加载、小 kernel 与 launch gap。
- **陷阱**：GPU utilization 100% ≠ MFU 高；memory-bound kernel 也能让设备一直「忙」但没算有效 FLOPs。

**Q7：fp32 / fp16 / bf16 区别？为什么 bf16 通常不用 loss scaling？**

- fp32：8-bit exponent + 23-bit fraction（稳、4 bytes）；fp16：5+10（范围小、梯度易 underflow，需 loss scaling）；bf16：8+7（2 bytes，动态范围≈fp32，精度粗）。
- bf16 保留 8-bit exponent，梯度范围与 fp32 相近，因此大模型常可免 loss scaling（Kalamkar et al. 2019；PaLM）。
- fp8 需 E4M3/E5M2 + per-tensor/channel scaling，数值边界更依赖实现。

**Q8：gradient accumulation 为什么省显存？代价是什么？**

- 把 global batch \(B_g\) 分成 \(K\) 个 microbatch \(B_\mu\)，loss 除以 \(K\) 累计后再 `clip + optimizer.step()`。
- 它只降低 activation peak（每个 microbatch 的激活用完即释放），**不减少总 FLOPs**，且小 GEMM 可能降低 kernel 效率。
- **语义等价前提**：累计完才 step/clip，且各 microbatch 数据分布一致；否则不等于 global batch。

**Q9：activation checkpointing 的原理与代价？为什么约 1.33×？**

- 普通反向保存每层激活（\(O(L)\)）；checkpointing 只存边界，backward 时重算中间 forward，把激活降到 \(O(\sqrt L)\)。
- 代价：多跑一次 forward，理论时间 ≈ \((2+1)/(2) \approx 1.33×\) 的前向/反向总量；只作用于 activation，不动参数/梯度/优化器。
- **进阶**：selective recomputation 按「省 1 byte 的重算代价」排序，只丢弃保存贵、重算便宜的中间量（如 `T²` attention 辅助矩阵）。

**Q10：为什么 attention 是 memory-bound，而 GEMM 是 compute-bound？**

- GEMM 的 FLOPs 是 \(O(n^3)\)、搬运 \(O(n^2)\)，arithmetic intensity \(O(n)\)，越大越 compute-bound。
- 标准 attention 的 score 矩阵是 \(T^2\) 中间量，算术强度随序列长度**下降**，因此长序列下受带宽限制；这正是 FlashAttention 通过减少 HBM 往返加速的动机。

### 14.3 手撕核算要点（显存 + FLOPs）

面试常让「算一个给定 config 的模型显存 / FLOPs」，按固定步骤走，避免漏项：

```text
显存核算（训练，bf16+Adam）
  1. 参数量 N = 12 L d² + 2 V d
  2. params   = 2N bytes（bf16）
  3. grads    = 2N bytes（bf16）
  4. optimizer = 8N bytes（m,v 各 4N，fp32）
  5. master   = 4N bytes（fp32，可选）
  6. 小计     ≈ 16N bytes
  7. activations（按公式估算，或用 torch.autograd 实测）
  8. 加 workspace / context / 碎片余量

FLOPs 核算（每 token，d_ff=4d）
  1. QKV+O: 4 × 2T d²  = 8 T d²
  2. scores + AV:       = 4 T² d
  3. SwiGLU: 3 × 2T d·4d = 24 T d²
  4. LM head:             = 2 T d V
  → 每 token ≈ (8+24)T d² + 4 T² d + 2 T d V
```

**三个必踩坑**

1. **别把参数量当显存**：Adam 状态与 activation 往往更大，16N 才是训练态小计。
2. **backward ≠ forward**：对 matmul 约 2×，训练总计约 3× forward，才有 `6ND`。
3. **`numel()` 不是峰值**：多个 tensor 生命周期重叠、allocator 碎片、临时 workspace 都会抬高真实 peak。

### 14.4 高频追问与陷阱

| 追问 | 正确方向 |
| --- | --- |
| view 和 contiguous 有什么区别？ | view 只改 stride 不搬数据；transpose 后要求 contiguous 会触发真实拷贝，profiler 里看 `aten::copy_` |
| einsum 一定快吗？ | 否，只表达语义；能否 lower 成高效 GEMM 才决定性能 |
| 为什么 Adam 显存是权重的 3 倍？ | m、v 各 4 bytes + master weight 4 bytes = 12 bytes 优化器状态 |
| 大 batch GEMM 为什么效率高？ | 更大矩阵复用强、arithmetic intensity 高，但激活显存也随 batch 涨 |
| MFU 和 GPU utilization 一样吗？ | 否，utilization 只表示设备忙，memory/launch-bound 也能 100% |
| loss scaling 解决什么？ | fp16 梯度 underflow；bf16 因 8-bit exponent 通常不需要 |
| 梯度累积等于免费大 batch 吗？ | 否，省显存但总 FLOPs 不变、小 GEMM 可能更慢 |
| checkpointing 是总内存减半吗？ | 否，只作用于可重算的 activation，不动参数/梯度/优化器 |

## 15. 结论与本讲小结

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
- [A1 实验报告](../assignments/assignment1-basics/report/main.tex)
- [A2 实验报告](../assignments/assignment2-systems/report/main.pdf)
- [Systems 主题导航](../experiments/topics/systems.md)
