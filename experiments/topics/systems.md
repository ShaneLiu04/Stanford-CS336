# Systems：把正确的 Transformer 变成高效训练系统

Systems 的核心不是“用了某个快库”，而是建立可解释的 performance model：时间花在 compute、memory movement、kernel launch 还是 communication；优化是否保持数值正确；速度提升是否来自公平配置。阅读时应从资源核算与 profiling 开始，再进入 kernels、并行和 sharding。

## 推荐阅读顺序

### 1. 先学会计算资源账本

- **官方 lecture**：[Lecture 2 — PyTorch、FLOPs、memory、arithmetic intensity](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py) 与 [Lecture 5 — GPUs, TPUs](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)。
- **工具文档**：[PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)、[NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。
- **解决的问题**：在修改代码前区分 bottleneck。参数、gradient、optimizer state、activation、temporary buffer 都占显存；wall-clock 还包括 host-device transfer、编译和同步。
- **关键结论**：arithmetic intensity
  \[
  I=\frac{\text{FLOPs}}{\text{bytes moved}}
  \]
  与硬件的 peak FLOPs / memory bandwidth 共同决定 Roofline 上限。低 \(I\) 往往 memory-bound，高 \(I\) 才可能 compute-bound。benchmark 必须 warmup、显式同步、重复采样，并分开报告 latency、throughput 与 peak memory。
- **对应作业**：[Assignment 2 — Systems](https://github.com/stanford-cs336/assignment2-systems) 的 model/layer benchmarking、profiling 与 resource accounting；本仓库快照位于 `assignments/spring2026/assignment2-systems/`。
- **局限**：profiler 本身有开销；单个 kernel 的 microbenchmark 不能代表 end-to-end step，尤其在 shape、layout 或 compile graph 不同的情况下。

### 2. 从 online softmax 推导 FlashAttention

- **官方 lecture**：[Lecture 6 — Kernels, Triton](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)。
- **论文**：[FlashAttention](https://arxiv.org/abs/2205.14135) 与 [FlashAttention-2](https://arxiv.org/abs/2307.08691)。
- **工具文档**：[Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) 与 [PyTorch scaled dot product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。
- **解决的问题**：标准 attention 会将 \(N\times N\) scores/softmax materialize 到高带宽显存；长序列时 memory traffic 和空间成为瓶颈。FlashAttention 以 tiling 在 SRAM 中重算局部块，避免保存完整 score matrix。
- **关键公式/结论**：对新 block 的行最大值 \(m_b\) 和指数和 \(\ell_b\)，online softmax 可更新为
  \[
  m'=\max(m,m_b),\qquad
  \ell'=e^{m-m'}\ell+e^{m_b-m'}\ell_b.
  \]
  output accumulator 用同样缩放合并，因此无需一次看到整行。数学复杂度仍是 \(O(N^2d)\)，但额外 memory 从 \(O(N^2)\) 降到近似 \(O(Nd)\)，主要收益来自 IO-aware algorithm，而非减少 attention FLOPs。
- **对应作业**：A2 的 PyTorch tiled implementation、Triton forward/backward、causal mask、不同 dtype/head dimension/sequence length 的 correctness 与 benchmark。
- **局限**：block size、occupancy、register pressure 和 layout 决定实际性能；小序列上 launch/compile overhead 可能让 naive 或 vendor kernel 更快。误差容限要按 dtype、长度和累加顺序解释。

### 3. 理解 activation memory 与重计算

- **论文**：[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)。
- **工具文档**：[torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html)。
- **解决的问题**：训练显存常被随 batch/sequence/depth 增长的 saved activations 主导。activation checkpointing 只保存分段边界，backward 时重算内部 forward。
- **关键结论**：checkpointing 是 compute-for-memory trade-off，不会免费省显存；segment 数过少节省有限，过多则重算和调度开销上升。应比较同一 global batch 与同一数值配置下的 peak allocated memory 和 step time。
- **对应作业**：A2 的 Transformer memory accounting、checkpoint segment sweep 与 OOM boundary。
- **局限**：带 dropout、随机 op 或 stateful layer 时必须正确处理 RNG/state；框架 memory allocator 的 reserved memory 不等于 tensor 实际 allocated memory。

### 4. 再读 data parallel 与 communication overlap

- **官方 lecture**：[Lecture 7 — Parallelism](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py) 与 [Lecture 8 — Parallelism](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)。
- **论文**：[Megatron-LM](https://arxiv.org/abs/1909.08053) 提供 tensor/model parallel 视角；[ZeRO](https://arxiv.org/abs/1910.02054) 系统化分析 data parallel 的冗余 state。
- **工具文档**：[PyTorch DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)、[FSDP](https://pytorch.org/docs/stable/fsdp.html) 与 [distributed collectives](https://pytorch.org/docs/stable/distributed.html)。
- **解决的问题**：DDP 在每个 rank 复制模型和 optimizer，通过 all-reduce 聚合 gradients；当单卡放不下时，optimizer/gradient/parameter sharding 进一步消除复制。
- **关键结论**：
  - \(P\) 个 data-parallel ranks 的 global batch 通常为 \(B_{\text{global}}=P B_{\text{local}} A\)，其中 \(A\) 是 gradient accumulation steps。
  - ring all-reduce 每个 rank 的通信量量级为 \(2(P-1)S/P\)，\(S\) 为待聚合字节数。
  - per-parameter hook 可在该 gradient ready 后异步发起 collective，使后层 backward compute 与前层 communication overlap；最后必须等待 handle 并确保平均/缩放语义正确。
  - FSDP 在计算某层前 all-gather 参数，之后 reduce-scatter gradients，并按策略 reshard；省显存的代价是更多通信与复杂生命周期。
- **对应作业**：A2 的 naive DDP、asynchronous overlap、optimizer state sharding 和 FSDP。
- **局限**：CPU/Gloo correctness 不能证明 GPU/NCCL performance；单机互联结果也不能外推到跨节点。小 bucket 或频繁 all-gather 会被 latency 支配。

### 5. 最后组合 mixed precision、compile 与端到端验证

- **工具文档**：[PyTorch AMP](https://pytorch.org/docs/stable/amp.html)、[torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html) 与 [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)。
- BF16 常减少 bandwidth 与 Tensor Core 计算成本；FP16 的动态范围更小，通常需要 `GradScaler`。部分 reduction、normalization 或 optimizer state 保持 FP32 可改善稳定性。
- `torch.compile` 能融合 operations、消除 Python overhead，但首次编译时间、graph breaks 与 dynamic shapes 必须单独记录。
- **对应作业**：A2 的 dtype/compile sweep、fused kernels 与完整 training-step benchmark。
- **局限**：TF32、determinism、kernel autotuning 和 driver/library 版本都会改变结果。性能报告必须记录 GPU、软件版本、shape、dtype、warmup、同步方式和 commit。

## 建议实验顺序

1. 用 FP32 reference 做 output/gradient correctness，覆盖 causal/non-causal 与极端 shape。
2. 建立 eager baseline，区分 forward、backward、optimizer 与 communication。
3. 单独 benchmark kernel，再测完整 model；不把 compile time 混入 steady-state。
4. 做 sequence length、head dimension、dtype、batch sweep，同时记录 error/OOM。
5. 多卡实验报告 scaling efficiency
   \[
   E_P=\frac{\text{throughput}_P}{P\cdot\text{throughput}_1},
   \]
   并用 profiler 解释未达到线性加速的原因。

## 阅读边界

Systems 结论高度依赖硬件和 workload。某张 GPU 上的最优 tile、某个版本的编译器收益或某种并行策略都不是普遍规律；可靠结论应来自完整环境元数据、原始样本分布和与同语义 baseline 的比较。
