# A2 Systems：把单卡训练优化扩展到多卡

> 中文导读；不替代[官方 handout](../../assignments/spring2026/assignment2-systems/cs336_assignment2_systems.pdf)，不提供实现答案。

## 目标

建立“测量—定位—优化—扩展”的系统训练能力：正确 benchmark CUDA；用 Nsight 和 memory snapshot 定位瓶颈；理解 mixed precision 与 activation checkpointing；实现 PyTorch/Triton FlashAttention-2；再实现 DDP overlap、optimizer state sharding 和教学型 FSDP，并分析 DP/FSDP/TP/2D parallelism 的通信边界。

## Problem、deliverable 与分值

### 单卡 profiling 与 kernel

| Problem | 分值 | 核心交付 |
| --- | ---: | --- |
| `benchmarking_script` | 4 | 可配置模型/上下文/forward-backward 的正确计时脚本与结果 |
| `nsys_profile` | 5 | Nsight Systems trace、最耗时 kernel 与分析 |
| `mixed_precision_accumulation` | 1 | mixed-precision accumulation 数值分析 |
| `benchmarking_mixed_precision` | 2 | FP32/BF16 计时、dtype 路径与评论 |
| `memory_profiling` | 4 | memory snapshot、peak table 与归因 |
| `gradient_checkpointing` | 4 | 理论最优与受限重计算策略、实测 peak memory |
| `pytorch_attention` | 2 | attention forward/backward 时间、显存与 OOM 边界 |
| `torch_compile` | 2 | compiled attention 与端到端 Transformer 对照 |
| `flash_forward` | 15 | FlashAttention-2 forward：先 PyTorch autograd，再 Triton 路径 |
| `flash_backward` | 5 | backward 正确性 |
| `flash_benchmarking` | 5 | 单 B200，batch 1、causal，按题面笛卡尔积比较实现 |

### 分布式与分片

| Problem | 分值 | 核心交付 |
| --- | ---: | --- |
| `distributed_communication_single_node` | 5 | 1 MB–1 GB × 2/4/6 GPU all-reduce 测量；单次 <5 分钟 |
| `naive_ddp` / `_benchmarking` | 5 / 3 | 逐参数 all-reduce DDP、2 GPU xl step/通信时间 |
| `minimal_ddp_flat_benchmarking` | 2 | flat-gradient 单次 all-reduce 对照 |
| `ddp_overlap_individual_parameters` | 5 | backward hook + async all-reduce 的 overlap DDP |
| 对应 `_benchmarking` | 1 | 2 GPU 性能与两张 Nsight 对照图 |
| `optimizer_state_sharding` | 15 | optimizer state 按 rank 分片并同步参数 |
| `optimizer_state_sharding_accounting` | 5 | 2 GPU peak memory、step 时间、与 ZeRO-1 差异 |
| `fsdp` | 15 | Linear/Embedding 分片、all-gather/reduce-scatter、可选低精通信 |
| `fsdp_accounting` | 5 | 内存节省估算与 2 GPU Nsight 验证 |

### 并行策略分析与 leaderboard

| Problem | 分值 | 核心交付 |
| --- | ---: | --- |
| `alternate_ring_all_reduce` | 1 | 环形 collective 的替代构造/代价分析 |
| `data_parallel_calcs` | 3 | DP 通信与 compute scaling |
| `fsdp_calcs` | 3 | FSDP 通信/内存边界 |
| `tp_calcs` | 4 | tensor parallel 通信核算 |
| `fsdp_tp_calcs` | 6 | 二维 FSDP×TP 组合分析 |
| `leaderboard` | 10 | 2×B200、batch 2 完整训练 step 的最佳 wall time |

榜单从空 PyTorch/Triton cache 起算，完整 benchmark 必须在 10 分钟内结束；目标包含 forward、loss、backward、AdamW，题面期望快于 10 秒 naïve baseline。

## 测试接口

本快照 `tests/adapters.py` 暴露：

- `get_flashattention_autograd_function_pytorch()`：只用标准 PyTorch；
- `get_flashattention_autograd_function_triton()`：Triton forward/backward；
- `get_ddp(module)` 与 `ddp_on_after_backward(...)`；
- `get_sharded_optimizer(params, optimizer_cls, **kwargs)`；
- `get_fsdp(module, compute_dtype)`、`fsdp_on_after_backward(...)`；
- 本地扩展还有 `fsdp_gather_full_params(...)`，应视当前测试契约核对是否为官方必需。

FlashAttention 的输入/输出、log-sum-exp 保存和 causal mask 语义，以原始 tests 为准。分布式测试应多次运行，排查 race、未 wait 的异步通信和 rank 间状态漂移。

## 推荐实验矩阵

1. **基础模型：** handout 给定模型尺寸 × context length × `{forward, forward+backward}` × `{FP32, BF16}`。
2. **Attention：** sequence length × head dimension × dtype × `{SDPA, PyTorch tiled, Triton}`；分别测 forward/backward、误差、peak allocated、OOM。
3. **Checkpoint：** segment 粒度/递归深度 × context；报告 memory-latency Pareto，而非只给“省了多少”。
4. **Collective：** `{1MB,10MB,100MB,1GB}` × `{2,4,6 ranks}`；warmup、同步并聚合各 rank。
5. **DDP：** `{逐参数同步, flat, overlap}`，固定 1 node × 2 GPU × xl；用 profiler 证明确有 overlap。
6. **Sharding：** `{DDP, state-sharded, FSDP FP32, FSDP low-precision communication}`；测初始化后、step 前、step 后内存和 step time。
7. **Leaderboard 扩展：** tile/autotune、fused CE/LM head、fused AdamW、compile；每次只改变一个因素并保留冷启动总耗时。

CUDA benchmark 必须 warmup、`torch.cuda.synchronize()`，并同时记录均值、方差、峰值显存和失败状态。

## 硬件

- Triton 需要 Linux + CUDA + NVIDIA GPU；原生 Windows 不能作为 Triton/NCCL 最终验证环境。
- 单卡部分以 B200 为官方性能参照；通信 sweep 最多 6 GPU。
- DDP/FSDP 核心对照通常为 1 node × 2 GPU；Gloo/CPU 只适合逻辑调试，不能替代 NCCL 性能。
- leaderboard 固定 2×B200、batch 2。不同 GPU、互联拓扑、driver/CUDA 版本的绝对时间不可直接排名。

## 提交物

- 运行 `test_and_make_submission.sh` 生成 `code.zip`；
- Gradescope 提交 typeset `writeup.pdf` 与 `code.zip`；
- trace、memory snapshot、表格和截图应在报告中可定位，脚本和参数足以复现；
- leaderboard 为独立 GitHub 提交流程。

## 资源受限替代

- 无 GPU：用 tiny model + Gloo 验证 DDP/sharding 数学正确性和 state 同步。
- 单 GPU：完成 profiling、attention correctness、compile 和 checkpointing；不能声称 multi-GPU overlap。
- 2 张消费卡：可做 NCCL/DDP/FSDP 对照，但必须报告 PCIe/NVLink 拓扑，不能换算成 B200 成绩。
- 无 Triton 环境：保留 PyTorch tiled reference，使用 SDPA 作正确性 oracle；Triton deliverable 仍未完成，应明确标注。
- 显存不足：缩小 context/model 验证曲线趋势，保留题面矩阵中的 OOM 记录，不选择性删除失败点。

## 版本风险与本仓库报告

- 固定上游 commit：`ca8bc81a59b70516f7ebb2da4808daade877c736`，见 [`UPSTREAM.md`](../../UPSTREAM.md)。
- `torch.compile`、Triton、private memory profiler API 和 GPU 架构高度版本敏感；复现必须固定 PyTorch/Triton/CUDA/driver。
- 本地实现含额外 fused backward 与 FSDP 验证接口，不代表原始题面最低要求。
- 报告：[源码](../../assignments/spring2026/assignment2-systems/report/main.tex) · [PDF](../../assignments/spring2026/assignment2-systems/report/writeup.pdf) · [资产说明](../../assignments/spring2026/assignment2-systems/report/README.md) · [验证记录](../../assignments/spring2026/assignment2-systems/VERIFICATION.md)。
