---
title: "Lecture 07 — Collectives & Data Parallelism"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-04-20"
lecturer: "Percy Liang"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py"
  - "../experiments/topics/systems.md"
  - "../assignments/spring2026/assignment2-systems/"
---

# Lecture 07 — Collectives 与 Data Parallelism：通信语义、同步正确性与重叠调度

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
- 适用对象：自学者、分布式训练工程师与系统研究者

## 摘要

Data Parallelism 通过复制模型并切分数据扩展训练吞吐，其数学目标简单，但系统性能取决于
gradient readiness、collective algorithm、network topology、bucket scheduling 与计算通信重叠。
本文从 collective semantics 出发，用 \(\alpha\)-\(\beta\) 模型推导 ring/tree all-reduce、
reduce-scatter/all-gather 的通信量与 latency；随后证明 DDP 与大 batch gradient 的等价条件，
分析 loss reduction、gradient accumulation、RNG、BatchNorm、tied/unused parameters 等语义陷阱。
工程部分讨论 hook 生命周期、CUDA stream/event、bucket ordering、NCCL topology、
deadlock/timeout 与性能诊断；研究部分覆盖 strong/weak scaling、exposed communication、
gradient compression 与效度威胁。目标是让读者同时回答“结果是否正确”“通信为何暴露”
和“扩展效率能否迁移”三个问题。

**关键词：** Data Parallelism；Collectives；All-Reduce；DDP；NCCL；
Gradient Bucketing；Communication Overlap；Strong Scaling；Gradient Compression

## 本文贡献

1. 统一 collective 的 tensor 语义、通信量、latency 与 topology；
2. 严格说明 DDP 等价于 global batch 的条件和 loss/gradient scaling；
3. 解释 per-parameter、flat、bucketed overlap 的依赖与生命周期；
4. 给出 process group、async work、CUDA streams、unused/tied parameters 的工程规范；
5. 建立 scaling benchmark、profiler 证据、故障排查与论文式结论边界；
6. 指出复制式 DP 的容量边界，并给出 ZeRO-1/2/3 的显存-通信权衡（§4.4）。

## 学习目标

1. 精确定义 broadcast、all-gather、reduce-scatter、all-reduce、all-to-all；
2. 用 \(\alpha\)-\(\beta\) 模型核算通信量与延迟；
3. 从单卡训练推导 DDP 的数值语义；
4. 理解 gradient bucket、异步 hook 和 compute/communication overlap；
5. 设计可信的 NCCL benchmark，而不把 Gloo 正确性当性能证据。

## 先修知识

- Lecture 02：梯度与 optimizer state 的显存账本（DDP 冗余的来源）。
- Lecture 05：带宽/延迟层次与集合通信的硬件约束。
- 建议：对照 A2 systems 的 DDP 实现阅读 §5–§6。

## 相关工作与系统脉络

高性能 collective 的目标是在 latency 与 bandwidth 两个 regime 中接近下界。Ring all-reduce
可达到大消息 bandwidth-optimal [[1]](#ref-1)，recursive doubling/tree algorithms 则在小消息
中减少 steps。Horovod 将 ring all-reduce 与 tensor fusion 引入易用的数据并行训练
[[2]](#ref-2)；MPI collective 优化工作系统讨论了算法选择与消息大小
[[3]](#ref-3)。PyTorch DDP 进一步用 autograd hooks、gradient bucketing 与异步 collectives
实现框架级 overlap [[4]](#ref-4)。

通信调度研究表明，按固定 tensor 顺序发起 collectives 未必最优。Poseidon、ByteScheduler
分别从 hybrid communication 与分块优先级调度减少 exposed communication
[[5]](#ref-5)[[6]](#ref-6)。在带宽受限网络上，Deep Gradient Compression 与 PowerSGD
通过稀疏/低秩压缩降低 bytes，但引入误差、额外计算和新的收敛假设
[[7]](#ref-7)[[8]](#ref-8)。Local SGD 则减少同步频率，以 optimization drift 换通信
[[9]](#ref-9)。ZeRO 进一步证明，DP 复制的 optimizer state/gradient/parameter 可以按
rank 切分并按需重构，在不显著增加通信量的前提下突破显存容量墙
[[10]](#ref-10)。

因此 DDP 不是单一算法，而是数值同步语义、collective implementation、bucket scheduler
和网络拓扑的组合。

## 1. 为什么多卡问题首先是数据移动问题

层级通常从快到慢为：寄存器/片上存储 → HBM → NVLink/NVSwitch → 跨节点 InfiniBand/RoCE/Ethernet。多 GPU 的目标有两个：

- **容量**：单卡放不下参数、optimizer state、gradient 或 activation；
- **速度**：用更多 FLOPs 缩短同一训练任务。

若增加设备后通信暴露时间大于新增计算收益，吞吐不会线性增长。应区分理论总 bytes、fabric 可用带宽、collective 算法和未被计算隐藏的 exposed communication。

## 2. Collective 速查

设 world size 为 \(p\)，每 rank 的输入 shard 大小为 \(S/p\) 或完整张量大小为 \(S\)。

- **broadcast**：一个源 rank 把 \(S\) 复制给所有 rank；常用于初始化参数。
- **scatter**：源 rank 把不同 shard 发给不同 rank。
- **gather**：所有 shard 收集到一个目的 rank。
- **all-gather**：每 rank 最初持有 \(S/p\)，最后每 rank 都持有完整 \(S\)。
- **reduce**：将各 rank 同 shape 输入按 sum/min/max 规约到一个 rank。
- **reduce-scatter**：规约后，每 rank 只保留结果的一个 shard。
- **all-reduce**：规约结果复制到所有 rank；可分解为 reduce-scatter + all-gather。
- **all-to-all**：每 rank 给每个 rank 发送不同分片；MoE token dispatch 常用。

名称里的 **reduce** 表示有结合/交换规约，**all** 表示每个 rank 都得到目标结果。

### 2.1 Shape contract 与内存布局

以 \([P,d]\) 参数矩阵为例：

- all-reduce：每 rank 输入/输出均 `[P,d]`；
- reduce-scatter：每 rank 输入完整或等价 flat buffer，输出约 `[P/p,d]` shard；
- all-gather：输入 `[P/p,d]`，输出 `[P,d]`；
- all-to-all：输入常 reshape 为 `[p, tokens_per_dest, d]`，各 rank 的 split sizes
  可能不等。

Collective 通常要求 ranks 在 dtype、numel、顺序上匹配。某 rank 少调用一次或 shape 不同，
常表现为 hang 而非 Python exception。Non-contiguous tensor 可能触发隐式 pack/copy；
benchmark 需把 pack 成本计入或显式分离。

### 2.2 In-place、async 与 ordering

`async_op=True` 返回 work handle，只表示 collective 已入队，不表示结果可读。Tensor storage
在 work 完成前不能复用/释放；consumer stream 必须等待 event/work。所有 ranks 对同一
process group 的 collective ordering 必须一致，否则可能 deadlock。

多 process groups/streams 能增加 overlap，也增加 ordering complexity。正确做法是定义
collective sequence、ownership 和 lifetime，而不是依赖偶然调度。

## 3. 通信量与时间模型

令 \(\alpha\) 为一次通信步骤的固定延迟，\(\beta\) 为有效 bytes/s。

### Ring all-reduce

对每 rank 的 \(S\)-byte tensor，ring 分 reduce-scatter 和 all-gather 两阶段，各 \(p-1\) 步，每步传 \(S/p\)：

\[
V_{\text{rank}}=2\frac{p-1}{p}S,
\]

\[
T_{\text{ring}}\approx 2(p-1)\alpha+
2\frac{p-1}{p}\frac{S}{\beta}.
\]

这是“每 rank 发送量”，不是把所有链路流量相加。大消息趋向 bandwidth-bound；小消息的 \(2(p-1)\alpha\) 主导，因此逐参数 collective 往往很差。

### Tree all-reduce

树算法延迟步数约 \(O(\log p)\)，小消息更有优势；但每步消息和链路利用方式不同。不能只用 ring 公式解释 NCCL 自动选择的所有算法。

### 有效带宽

原始 \(S/T\) 不是 all-reduce 的算法带宽。常见 bus-bandwidth 归一化会乘 \(2(p-1)/p\)。报告必须写清公式，否则不同工具的 GB/s 不能直接比较。

### Hierarchical all-reduce

多节点常先 node 内 NVLink/NVSwitch reduce，再跨节点 InfiniBand/RoCE，再 node 内 broadcast/
all-gather。设 node 数 \(n\)、每 node GPU 数 \(g\)，不同层有
\((\alpha_{\text{intra}},\beta_{\text{intra}})\) 与
\((\alpha_{\text{inter}},\beta_{\text{inter}})\)。如果直接把所有 GPU 当同质 ring，会让最慢跨节点
链路重复承载过多数据。

NCCL 会根据 topology、消息大小、channels 与环境选择 ring/tree/CollNet 等路径。性能报告应保存
topology dump、NCCL algorithm/protocol 与网络接口，不只写“8 GPUs”。

### \(\alpha\)-\(\beta\) 模型的局限

线性模型忽略 congestion、serialization overlap、PCIe/NIC sharing、GPU copy engines、
protocol threshold 与 straggler。它适合预估 regime 和 lower bound；真实结论必须通过
message-size sweep 与 timeline 验证。对 all-to-all，uneven split 的 maximum destination load
可能比平均 bytes 更能决定时间。

## 4. DDP 的数学语义

第 \(r\) 个 rank 在本地 batch \(B_r\) 上计算 mean loss \(\mathcal L_r\) 和 gradient \(g_r\)。若各 rank batch 大小相同：

\[
g=\frac1p\sum_{r=0}^{p-1}g_r
\]

等价于拼接后的 global mean loss gradient。若只 all-reduce SUM 而不除 \(p\)，有效学习率会放大 \(p\) 倍。

global batch：

\[
B_{\text{global}}=p\,B_{\text{local}}\,A,
\]

其中 \(A\) 是 gradient accumulation steps。扩大 \(p\) 但不调整 local batch、accumulation 或学习率，会同时改变优化问题，不能只归因于系统加速。

### 一个完整 DDP step

1. rank 0 参数与 buffer broadcast，确保初始化相同；
2. 每 rank 读取不同数据；
3. 独立 forward/loss/backward；
4. 对 gradient 做 all-reduce average；
5. 每 rank 执行相同 optimizer step，参数继续一致。

模型参数、gradient 和 optimizer state 都复制，因此 DDP 提供计算扩展，不解决参数状态容量。

### 4.1 等价性的隐藏假设

Global-batch 等价要求：

- 每 rank local batch 大小相等，或按样本数加权 gradient；
- loss reduction 口径一致；
- data shards 无重复且覆盖目标 batch；
- 参数/optimizer state 初始一致；
- stochastic layers 的 RNG 语义符合目标；
- regularization 按 global objective 定义；
- 所有 requires-grad parameters 得到相同同步规则。

若 local batch 大小 \(B_r\) 不同，正确 gradient 是

\[
g=\frac{\sum_r B_rg_r}{\sum_r B_r},
\]

不是 rank mean。最后一个不完整 batch、dynamic batching 或 sequence token normalization
都会破坏“简单除 world size”。

### 4.2 BatchNorm、Dropout 与数据语义

BatchNorm 使用 local statistics 时不等价于 global batch；可用 SyncBatchNorm，但增加通信。
LayerNorm/RMSNorm 按 token/features，不依赖跨样本统计，通常更适合 DDP。

Dropout 若各 rank 使用相同 RNG stream，可能对不同样本生成相同 mask；通常希望 seed 中包含 rank，
但模型初始化前又需相同参数。实践中分离 initialization seed、data seed 和 stochastic-op seed。

### 4.3 Gradient accumulation 与 `no_sync`

Accumulation 的前 \(A-1\) 个 microbatches 若每次 all-reduce，会产生无意义通信。
DDP `no_sync()` 可只在最后一次 backward 同步；但 loss 必须除 \(A\)，clip 与 optimizer step
只执行一次。遗漏最后同步会让 ranks 从此分叉，错误可能到数步后才显现。

### 4.4 容量上限与 ZeRO 的显存-通信权衡

DDP 复制完整的参数、gradient 与 optimizer state，因此提供计算扩展而不解决容量问题。
ZeRO 按冗余程度依次切分三类状态 [[10]](#ref-10)（沿用 §4 的 \(16P\) 口径，
\(S=4P\) 为 gradient bytes）：

| 方案 | 切分对象 | 每 rank 常驻下界 | 每 rank 每步通信量 |
|---|---|---|---|
| DDP | 无 | \(16P\) bytes | all-reduce \(\approx 2S\) |
| ZeRO-1 | optimizer states | \(8P+8P/p\) | \(\approx 2S\)（reduce-scatter + all-gather） |
| ZeRO-2 | + gradients | \(4P+12P/p\) | \(\approx 2S\) |
| ZeRO-3/FSDP | + parameters | \(16P/p\) | \(\approx 3S\)（fwd/bwd 各一次 all-gather + reduce-scatter） |

关键洞察是：切分把容量问题转化为 collective 调度问题。ZeRO-1/2 的通信与 DDP 同级，
因此显存收益近乎“免费”；ZeRO-3 因参数需要在 forward 与 backward 前各 all-gather
一次，通信增至约 \(1.5\times\)，换取显存随 \(p\) 线性下降。FSDP 是该思想在 PyTorch
中的系统化实现；完整的 TP/PP/FSDP 多维并行谱系在 Lecture 08 展开。

## 5. Naive、flat、bucketed overlap

### Naive：backward 后逐参数同步

正确但关键路径是

\[
T_{\text{backward}}+T_{\text{all-reduce}}.
\]

大量小 collective 被 latency 支配。

### Flat gradient

把全部 gradient 拼成一个连续 buffer，只做一次 all-reduce。优点是减少 launch/latency；缺点是必须等所有 gradient ready，无法与 backward 重叠，还可能有 flatten/copy 成本。

### Bucketed overlap

autograd 从后层向前层产生 gradient。某个 bucket 全部 ready 后，立即在独立 stream 发起异步 all-reduce，让它与更早层的 backward overlap：

\[
T_{\text{step}}\approx
T_{\text{forward}}+
T_{\text{backward}}+
T_{\text{comm, exposed}}+
T_{\text{optim}}.
\]

理想情况下，除了最后无法隐藏的 bucket，大部分通信被 backward 覆盖。

bucket 太小：\(\alpha\) 与 launch 主导；太大：ready 太晚，重叠减少。生产 DDP 常按逆向参数顺序组 bucket，并使用 contiguous gradient views。

### 5.1 Ready order 与 bucket rebuild

Backward gradient readiness 由 autograd graph 决定，通常与 module registration order 不完全一致。
若 bucket 按 forward order 排列，某 bucket 可能被一个很晚 ready 的 parameter 阻塞。生产框架会在
首轮观察 ready order 并 rebuild buckets。Dynamic control flow 会让不同 iteration ready/unused
集合变化，增加同步风险。

### 5.2 Overlap 的资源竞争

Communication 与 compute overlap 不保证“免费”：NCCL kernel 占用 SM、copy engine、HBM bandwidth
与 network injection。若 backward 本身 memory-bound，通信可能争抢 HBM 并拖慢 compute。
应比较：

- backward-only；
- communication-only；
- overlap timeline；
- overlap 后 compute kernel duration 是否变长。

### 5.3 Gradient compression

量化/稀疏/低秩 compression 减少 bytes，但 naive compression 可能产生有偏 gradient。
Error feedback 累积未发送 residual，可改善收敛 [[7]](#ref-7)；PowerSGD 用低秩近似矩阵
gradients [[8]](#ref-8)。公平比较需把 compression/decompression FLOPs、额外 state、
通信 bytes 和 time-to-quality 全部计入。

## 6. Hook 与异步生命周期

仓库教学实现位于：

- `assignments/spring2026/assignment2-systems/cs336_systems/ddp.py`
- `cs336_systems/sharded_optimizer.py`

其 DDP 核心语义是：

1. 对唯一参数注册 gradient/post-accumulate hooks；
2. gradient ready 后先除 world size；
3. `dist.all_reduce(..., async_op=True)`；
4. 保存 `Work` handle 与对应 buffer；
5. optimizer 前 `work.wait()`，再把结果写回/累加到 `.grad`。

异步并不等于自动 overlap。必须满足：

- collective 使用的 gradient buffer 在完成前不被覆写或释放；
- 所有 rank 以兼容顺序调用 collective，否则可能死锁；
- optimizer 读取前等待全部 handle；
- tied/shared parameter 按对象 identity 去重；
- repeated backward/gradient accumulation 不把已规约结果再次混入。

仓库实现为 per-parameter overlap，便于教学；生产实现通常 bucketize，否则小 collective latency 过高。

### 6.1 Tied、unused 与 dynamic graph

Tied parameters 必须按 object identity 去重，不能按 name 去重；否则同一 storage 注册两次 hook，
gradient 被重复规约。Unused parameters 若某 rank 未产生 gradient，而其他 rank 发起 collective，
会造成 ordering mismatch/hang。`find_unused_parameters` 通过 graph traversal 协调，但增加 overhead；
static graph 可关闭以提速。

### 6.2 Deadlock 诊断

典型症状是所有 rank 无 Python exception 地等待。排查顺序：

1. 为每个 collective 编号并在 rank-local log 记录 enter/exit；
2. 检查各 rank tensor shape/dtype/device/sequence；
3. 设置 process-group timeout 与 NCCL debug；
4. 检查异常 rank 是否提前 OOM/退出；
5. 检查 CUDA stream/event 与 async work lifetime；
6. 最小化到两 rank/单 collective。

分布式错误常由“较早的真正异常 + 其他 rank 后续 hang”组成；只看 hang rank 会误诊。

### 6.3 Fault tolerance 与 checkpoint

传统 synchronous DDP 失去任一 rank 即无法继续。Elastic restart 需要恢复 model、optimizer、
scheduler、data sampler epoch/offset、RNG states 与 global step。只保存 rank 0 model weights
不足以 bitwise resume；sharded optimizer/checkpoint 还需 world-size migration 策略。

## 7. Shape、内存与复杂度

设模型参数元素数为 \(P\)，gradient dtype 每元素 \(s_g\) bytes：

- DDP 每 rank gradient 通信 tensor 大小 \(S=P s_g\)；
- ring all-reduce 每 rank 网络量约 \(2(p-1)S/p\)；
- 每 rank 仍保存 \(P\) 参数、\(P\) gradient 和完整 optimizer states；
- 本地计算约为单卡 global batch 计算的 \(1/p\)，但通信量不会随 local batch 同比例下降。

因此弱扩展（每 rank local batch 固定）通常比强扩展（global batch 固定）更容易获得高效率。强扩展下每 rank compute 变少，而参数 gradient 通信大小不变。

扩展效率：

\[
E_p=\frac{\text{throughput}_p}{p\cdot\text{throughput}_1}.
\]

还应报告 time-to-train；更大 global batch 可能改变收敛步数。

### 7.1 Strong scaling、weak scaling 与优化统计

- **Strong scaling**：global problem/batch 固定，设备增加；每 rank compute 下降，通信不变，
  最容易受 Amdahl 串行项限制。
- **Weak scaling**：每 rank local workload 固定，global batch 随 \(p\) 增大；系统效率更高，
  但 optimization problem 改变。

吞吐扩展之外，报告：

\[
\text{speedup}_p=\frac{T_1}{T_p},\quad
E_p=\frac{\text{speedup}_p}{p},\quad
\text{time-to-quality}.
\]

若 global batch 增大，可用 LR scaling/warmup 等 recipe，但这属于 optimization intervention；
不能把其收益全部记为 systems scaling。

### 7.2 Compute/communication crossover

单 rank backward 时间粗略随 local tokens 下降，而 gradient bytes 主要由参数量决定。
当

\[
T_{\text{backward,remaining after overlap}}
\lesssim T_{\text{collective}},
\]

继续强扩展会进入 communication-bound。扩大 bucket/压缩/更快网络或改变 parallelism 才能移动
crossover；单纯增加 GPUs 不会。

## 8. 代码与实验映射

A2 对应实验：

- 1 MB–1 GB、2/4/6 GPU all-reduce sweep；
- naive per-parameter DDP；
- flat-gradient 单 collective；
- backward hook + async all-reduce overlap；
- state-sharded optimizer（近似 ZeRO-1 [[10]](#ref-10)）；
- Nsight 对照 compute 与 NCCL timeline。

本仓库只有单张 RTX 6000D：

- CPU/Gloo 多进程测试验证了 DDP/FSDP/sharding 数值语义；
- 没有 NCCL multi-GPU bandwidth、overlap speedup 或 2×B200 leaderboard 证据；
- `report/main.tex` 中的通信结论应视为理论推导，而非多卡实测。

### NCCL benchmark 最小规范

1. 每个 rank 绑定独立 GPU，初始化 NCCL process group；
2. 对每个 size warmup 多次；
3. collective 前后用 event/同步与 barrier 明确边界；
4. 汇总最慢 rank 时间，而不是只看 rank 0；
5. 记录拓扑、NCCL/CUDA/driver、dtype、算法环境变量；
6. 报 latency、algorithm bandwidth、bus bandwidth 和分布；
7. 避免 barrier 被误算进 collective，或明确报告端到端同步时间。

## 9. 面向研究的实验设计

### 9.1 Collective microbenchmark matrix

扫描 message size（bytes 到 GiB）、world size、单/多节点、dtype、collective、algorithm/protocol。
每点 warm-up，多次重复，汇总 slowest rank p50/p95。画 latency-size curve 与 bandwidth plateau，
识别 latency/bandwidth/crossover。

### 9.2 End-to-end DDP matrix

固定 model/data/optimizer，扫描 local batch、bucket size、accumulation、`no_sync`、compile。
记录 forward/backward/collective/optimizer、exposed communication、tokens/s、peak memory、
gradient correctness 与 time-to-loss。Microbenchmark 带宽高不保证 overlap 好。

### 9.3 预注册假设示例

> 在单节点 NVLink 上，小 bucket 受 launch/latency 主导；增大 bucket 会提高 algorithm
> bandwidth，但超过某阈值后 ready time 延迟使 exposed communication 增加，因此
> end-to-end step time 呈 U 形。

预先定义 bucket range、primary metric、timeline 判据、OOM/failure policy 与硬件拓扑，
避免事后只展示最佳 bucket。

## 10. 易错点

- SUM 后忘记除 world size，或 loss 已按 global sum 又重复缩放；
- DistributedSampler 未 `set_epoch`，各 rank 数据重复或每 epoch 顺序不变；
- rank 间 collective 次序不同导致 hang；
- 异步 handle 未保存/未 wait，optimizer 读到未完成 gradient；
- 在通信进行中原地修改同一 buffer；
- tied weight 注册两次 hook；
- `no_sync` accumulation 阶段仍发 collective；
- 只测 rank 0，忽略最慢 rank；
- 把 Gloo/CPU 曲线当 NCCL/GPU 曲线；
- 只算 bytes，不考虑小 bucket 的 latency 和拓扑拥塞；
- 在 ZeRO-3/FSDP 下只比状态显存，忽略前向 all-gather 参数缓冲的峰值，误判 OOM 位置；
- 误以为增大 local batch 能摊薄梯度通信——梯度 bytes 只由参数量决定（§7）。

## 11. Checklist

- [ ] 初始化后参数和 buffers 在各 rank 一致
- [ ] 每 rank 数据不重叠，global batch 计算正确
- [ ] 明确 all-reduce 是 SUM 还是 AVG
- [ ] 所有 collective 调用顺序一致
- [ ] async buffer 生命周期覆盖 `wait()`
- [ ] optimizer 前等待全部通信
- [ ] tied/shared parameters 去重
- [ ] 与单进程 global-batch reference 比 output/gradient/step
- [ ] benchmark 记录最慢 rank、拓扑与 backend
- [ ] profiler 中确实看到 NCCL 与 backward overlap

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| 全部 ranks hang | collective sequence | shape/order 不一致 |
| 仅某 rank OOM 后 hang | 最早异常 log | 其他 ranks 等 collective |
| loss 比单卡大 \(p\) 倍 | reduction scaling | SUM 未除 world size |
| ranks 参数逐步分叉 | wait/no_sync | optimizer 前未完成同步 |
| overlap 图上没有重叠 | bucket ready order | bucket 太大/顺序错误 |
| 重叠后 backward 变慢 | HBM/SM contention | NCCL 与 compute 争资源 |
| 小 tensor 带宽极低 | latency/message count | 过多 per-param collectives |
| 多节点骤降 | topology/NIC | 错网卡、跨节点算法 |
| tied weight 梯度翻倍 | hooks | 同一 Parameter 重复注册 |
| epoch 数据重复 | sampler | 未 shard 或未 `set_epoch` |

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| DDP 挂起 | bucket/hook 生命周期 | 梯度未完成 all-reduce 即被使用 |
| 扩展比远低于线性 | 通信量 \(2S\) 与 overlap | bucket 过小、计算通信未重叠 |
| 各 rank 梯度不一致 | 未参与计算的参数 | 未正确处理 unused parameters |
| 显存随 bucket 增长 | bucket_cap_mb 与 flatten | 分桶上限过大 |
| ZeRO-1 显存未下降 | optimizer 分片是否生效 | 分片只作用于 optimizer state，参数/梯度仍在 |
| 节点间扩展骤降 | 跨机带宽与拓扑 | 未做拓扑感知的分桶或分级归约 |

## 12. 讨论：效度威胁与结论边界

### Construct validity
- bus bandwidth、algorithm bandwidth、application throughput 不是同一指标；
- overlap percentage 不等于 step-time reduction；
- weak scaling efficiency 不等于固定任务加速；
- gradient equality 不保证相同 stochastic training trajectory。

### Internal validity
- barrier/同步是否计时会改变结果；
- 只测 rank 0 隐藏 straggler；
- network contention、topology、NCCL env 是 confounders；
- 增 world size 同时改变 global batch/LR 会混合 systems 与 optimization。

### External validity
- Gloo/CPU 不代表 NCCL/GPU；
- 单节点 NVLink 不外推跨节点 Ethernet/IB；
- 小模型 bucket optimum 不外推大模型；
- synthetic all-reduce 不含 backward readiness 与 HBM contention。

推荐结论限定 model、batch、topology、backend、message range 与统计协议，并公开 hang/OOM/failed
runs，而不是只写“DDP scales linearly”。

## 面试要点速记

**高频问题与答题要点**

1. **Q：DDP 每步通信量是多少？** 要点：all-reduce 梯度 2S（ring 实现约
   2(P−1)/P·S）；bucket 聚合小梯度、计算-通信 overlap 隐藏延迟。
2. **Q：ZeRO-1/2/3 各分片什么、代价几何？** 要点：1=optimizer state
   （16N → ~4N + 12N/P）；2=再分梯度；3=再分参数（~16N/P + 缓冲）；
   ZeRO-3 通信由 2S 升为 3S（前/反向 all-gather + 梯度 reduce-scatter）。
3. **Q：bucket 大小的权衡？** 要点：大桶通信效率高（摊薄延迟），但首个桶
   就绪晚、峰值显存上升；`bucket_cap_mb` 需按模型调。
4. **Q：梯度裁剪为何必须在 all-reduce 之后？** 要点：全局范数依赖完整梯度；
   本地裁剪再归约会改变数学语义。
5. **Q：ZeRO-3/FSDP 通信量为何是 3S？** 要点：前向 all-gather 参数、反向
   再 all-gather 一次、梯度 reduce-scatter；通信约 1.5× DDP，换显存随 P 线性下降。
6. **Q：异步 all-reduce 为什么不等于自动 overlap？** 要点：还需 buffer 生命周期
   覆盖 `wait()`、跨 rank collective 顺序一致、optimizer 前等待全部 handle（§6）。
7. **Q：FSDP2 与 FSDP 的区别？** 要点：FSDP2（torch 2.4）做 per-parameter
   sharding + 参数重分片（动态调整分片粒度），易用性大幅提升；语义仍是 ZeRO-3。
8. **Q：strong vs weak scaling 哪个更"好看"？** 要点：weak scaling 效率更高但
   改变优化问题；strong scaling 受通信与串行项限制（§7.1），报告须区分二者。

**必背数字**

- 16N 账本（L02）是 ZeRO 各阶段收益的分母；DDP 2S / ZeRO-3 3S 是
  扩展性估算的基本量。

**工业界参照**

- ZeRO 显存（DeepSpeed 口径）：Stage-1 ≈ 12Φ/P + 4Φ；Stage-2 ≈ 14Φ/P + 2Φ；
  Stage-3 ≈ 16Φ/P（Φ=参数量，P=DP 卡数）。
- 混合精度账本分解：fp32 master 权重 4B + m/v 各 4B = 12B/参数训练状态；
  bf16 参数+梯度 2+2B——合计 16B/参数。
- 通信压缩与精度：1-bit Adam 梯度通信压缩约 5×；H100 FP8（配合 DeepSpeed）
  显存省约 42%（量级）。
- 激活优化：激活检查点显存降 30%–70%（选择性重计算）；SP+TP 联合下长序列
  激活显存降约 5×；FlashAttention 使 1M tokens 长序列激活 O(n) 级。
- 通信时延：NVLink/IB 单次通信时延约 ~1µs 量级；nccl-tests
  （`all_reduce_perf`）是 ring vs tree 带宽比测的标准工具。
- 框架版本锚点：FSDP2 随 torch 2.4 引入（per-parameter sharding）；
  torchtitan 为 PyTorch 原生 4D 并行框架（持续完善阶段，未正式集成进 PyTorch）。

## 行业现状与最新进展（2024–2026）

### DDP → ZeRO → FSDP/FSDP2 的演进脉络

业界调研口径下的框架定位对照（本讲概念 ↔ 工业实践）：

| 框架 | 定位 | 核心机制 | 与本讲概念的对应 |
|---|---|---|---|
| DeepSpeed | ZeRO 分阶段分片 + 多级卸载 | ZeRO-Offload/Infinity（CPU/NVMe），单卡可训千亿参数，以时间换空间 | §4.4 ZeRO 三阶段的工程化 |
| Megatron-LM | 3D 并行训练框架 | TP+PP+DP + 激活检查点 + 序列并行（SP） | DP 是其中一维（Lecture 08 展开） |
| PyTorch FSDP | 全参数分片（类 ZeRO-3） | 自动 bucket + 异构分片支持 | §5 bucketed overlap 的分片版 |
| FSDP2（torch 2.4） | FSDP 的重写 | per-parameter sharding + 参数重分片（动态调整分片粒度）+ 易用性大幅提升 | 解决 flat buffer 与重分片痛点 |
| torchtitan | PyTorch 原生 4D 并行全能框架 | DP/TP/PP/SP 组合，持续完善阶段，未正式集成进 PyTorch | 4D 并行参考实现 |

易用性生态：Accelerate 提供分布式抽象层；LLaMA-Factory/MS-SWIFT 等微调封装
底层依赖 PyTorch DDP/FSDP/Megatron/DeepSpeed。趋势是"分片默认化"——
ZeRO-3/FSDP 从极限优化手段变成大模型训练的默认起点。

### ZeRO 三阶段显存公式与取舍（DeepSpeed 口径）

| 阶段 | 切分对象 | 每 rank 显存 | 通信量 | 取舍 |
|---|---|---|---|---|
| Stage-1 | optimizer states | ≈ 12Φ/P + 4Φ | ≈ 2S | 近乎免费的显存收益 |
| Stage-2 | + gradients | ≈ 14Φ/P + 2Φ | ≈ 2S | 通信不变，仍非最终解 |
| Stage-3 | + parameters | ≈ 16Φ/P | ≈ 3S | 显存随 P 线性下降，通信约 1.5× |

公式与 §4.4 的 16P 账本一致：fp32 master 权重 4B + m/v 各 4B = 12B/参数
优化器状态，加 bf16 参数+梯度 2+2B。进一步的压缩手段：1-bit Adam 梯度通信
压缩约 5×；H100 FP8（配合 DeepSpeed）显存省约 42%。激活侧：激活检查点显存
降 30%–70%（选择性重计算）；SP+TP 联合下长序列激活显存降约 5×；
FlashAttention 使 1M tokens 长序列激活 O(n) 级。

### 通信原语实测工具链：nccl-tests

业界标准做法是用 [nccl-tests](https://github.com/NVIDIA/nccl-tests) 对 fabric
做基准：`all_reduce_perf -b 512M -e 4G -f 2 -g 8` 扫消息大小，比较 ring vs
tree 算法带宽。NCCL 集合通信覆盖 broadcast/all-gather/reduce-scatter/
all-reduce 等；单次通信时延约 ~1µs（NVLink/IB 量级）。这正对应 §3 的
α-β 模型与 §8 的 benchmark 规范：小消息看 tree（O(log p) 步），大消息看
ring（bandwidth-optimal）。

**对本讲学习者的启示**：本讲的 collective 语义、2S/3S 通信量与 bucket
overlap 并未过时——FSDP2、torchtitan 的调度内核仍是同一套 reduce-scatter/
all-gather 原语，差异只在分片粒度（per-parameter）与自动调优程度。掌握
§2–§5 的推导，即可读懂任意新框架的设计文档与 release notes。

## 大厂面试真题与答题框架

以下为高频面试题（公开面经风格）。

**题目 1：ZeRO-1/2/3 各解决什么问题、代价是什么？**
- 考点：§4.4 容量上限、显存-通信权衡。
- 答题框架：
  1. 先摆 16B/参数账本：优化器状态 12B（fp32 master 4B + m/v 各 4B）+ bf16 参数/梯度 2+2B；
  2. Stage-1 切优化器状态（≈12Φ/P+4Φ），Stage-2 再切梯度（≈14Φ/P+2Φ），Stage-3 再切参数（≈16Φ/P）；
  3. 通信量：Stage-1/2 与 DDP 同级（2S），Stage-3 因前/反向 all-gather 升至约 3S；
  4. 结论：切分把容量问题转化为 collective 调度问题。
- 加分项：指出 FSDP 即 ZeRO-3 的系统化实现；FSDP2 进一步做 per-parameter sharding。
- 踩坑：把 ZeRO-3 说成"通信量不变"；忽略 Stage-3 前向 all-gather 参数缓冲的峰值显存。

**题目 2：DDP 的 all-reduce 通信量为什么是 2S？**
- 考点：ring all-reduce 的两阶段分解。
- 答题框架：
  1. all-reduce = reduce-scatter + all-gather；
  2. ring 两阶段各 p−1 步、每步 S/p，每 rank 发送量 2(p−1)/p·S ≈ 2S；
  3. 大消息 bandwidth-bound，小消息被 2(p−1)α 主导——所以要 bucket；
  4. 区分"每 rank 发送量"与全网链路流量两个口径。
- 加分项：写出 T_ring ≈ 2(p−1)α + 2(p−1)/p·S/β；说明 bus bandwidth 归一化乘 2(p−1)/p。
- 踩坑：把 2S 说成"每张卡收 2S"；忽略 S 是梯度 bytes（bf16 下即 2Φ）。

**题目 3：bucket 与 compute/communication overlap 为什么有效？**
- 考点：§5 bucketed overlap、autograd 反向就绪顺序。
- 答题框架：
  1. naive 逐参数同步被 latency 主导（2(p−1)α × 参数个数）；
  2. flat 单次 collective 消除 latency 但必须等全部梯度 ready，无法与 backward 重叠；
  3. bucket 折中：某桶梯度全部 ready 即在独立 stream 发起异步 all-reduce，与更早层 backward 重叠；
  4. 桶太小 → launch/latency 主导；太大 → ready 太晚、重叠减少、峰值显存上升。
- 加分项：生产框架按首轮 ready order rebuild buckets；NCCL kernel 与 compute 争 SM/HBM，overlap 非免费（§5.2）。
- 踩坑：认为 `async_op=True` 就自动 overlap——buffer 生命周期与顺序一致才是前提。

**题目 4：FSDP2 相比 FSDP 改进了什么？**
- 考点：框架演进、per-parameter sharding。
- 答题框架：
  1. FSDP（类 ZeRO-3）用 flat parameter 块分片，需 flatten/reshard，与 `torch.compile` 等组合受限；
  2. FSDP2（torch 2.4）改为 per-parameter sharding + 参数重分片（动态调整分片粒度）；
  3. 好处：无需重写模块结构、组合性更好、易用性大幅提升。
- 加分项：对照 torchtitan（PyTorch 原生 4D 并行，未正式集成进 PyTorch）说明官方路线。
- 踩坑：把 FSDP2 说成新算法——它是同一 ZeRO-3 语义的实现重构。

**题目 5：梯度裁剪为什么必须在 all-reduce 之后？accumulation 时怎么处理？**
- 考点：全局语义、`no_sync`。
- 答题框架：
  1. clip 依赖全局梯度范数，本地裁剪再归约会改变数学语义；
  2. accumulation 前 A−1 个 microbatch 用 `no_sync()` 跳过同步；
  3. loss 除 A，clip 与 optimizer step 只在最后一次 backward 后执行一次；
  4. 遗漏最后同步会让 ranks 分叉，错误可能数步后才显现。
- 加分项：不同 local batch 应按 ΣB_r 加权而非简单除 world size（§4.1）。
- 踩坑：把 no_sync 说成"异步训练"——它只是延迟同步，语义仍是全局 batch。

**题目 6：如何设计一次可信的 NCCL all-reduce benchmark？**
- 考点：§8 benchmark 规范、效度意识。
- 答题框架：
  1. 用 nccl-tests（如 `all_reduce_perf -b 512M -e 4G -f 2 -g 8`）扫消息大小；
  2. warmup、event 划界、汇总最慢 rank p50/p95 而非只看 rank 0；
  3. 记录拓扑、NCCL 算法/协议、dtype、CUDA/driver 版本；
  4. 报 algorithm bandwidth 与 bus bandwidth 两种口径并写清公式。
- 加分项：比较 ring vs tree 的小/大消息 regime；指出 Gloo/CPU 结果不能外推 NCCL/GPU。
- 踩坑：只报 rank 0 平均时间；把 barrier 时间误算进 collective。

**题目 7：什么场景选 DeepSpeed、什么场景选 FSDP2/Megatron-LM？**
- 考点：框架定位（业界调研口径）。
- 答题框架：
  1. 显存极限/单卡训超大模型 → DeepSpeed ZeRO-Offload/Infinity（CPU/NVMe 多级卸载，以时间换空间）；
  2. PyTorch 生态内常规大模型训练 → FSDP2（per-parameter sharding、易用性大幅提升）；
  3. 超大规模预训练追求极致吞吐 → Megatron-LM 3D 并行（TP+PP+DP+SP）；
  4. 顶层可用 Accelerate 做抽象、LLaMA-Factory/MS-SWIFT 做微调封装。
- 加分项：指出 torchtitan 是 PyTorch 原生 4D 并行参考实现（未正式集成进 PyTorch）。
- 踩坑：无脑选"最新框架"而不做容量估算——应先算 16Φ/P 再选阶段。

## 系统设计题

**设计题 1：为 70B/150B 模型在 64 卡集群（8 节点 × 8 卡，NVLink + IB）设计 ZeRO/FSDP 配置（含 offload 决策）**

- 需求澄清：预训练还是微调？序列长度与 global batch 目标？显存优先还是吞吐优先？
- 规模估算（混合精度，每参数训练状态 16B：fp32 master 4B + m/v 各 4B = 12B，加 bf16 参数+梯度 2+2B）：
  - 70B：总状态 70G×16B = 1120 GB，DDP 复制完全不可行；ZeRO-3 @P=64 → 16Φ/P ≈ 17.5 GB/rank 状态；
  - 150B：16Φ/P ≈ 37.5 GB/rank 状态，加激活与 all-gather 缓冲后逼近单卡 HBM 上限。
- 架构：
  - 70B：FSDP2/ZeRO-3，P=64，bf16 混合精度，选择性激活检查点（显存降 30%–70%），offload 关闭；
  - 150B：方案 A = ZeRO-3 + CPU offload（以时间换空间）；方案 B = 引入 TP/SP
    （Megatron-LM 3D，长序列激活显存降约 5×）；带宽充足时优先 B。
- Trade-off 表：

| 方案 | 状态显存/卡 | 通信量 | 风险 |
|---|---|---|---|
| ZeRO-3（70B） | ≈17.5 GB + 激活 | 3S ≈ 1.5× DDP | 激活峰值 OOM |
| ZeRO-3 + offload（150B） | 更低 | 3S + PCIe 搬运 | step 时间显著上升 |
| TP+PP+DP（150B） | TP 内再分片 | TP 通信多但留在 NVLink 域 | 工程复杂度高 |

- 评测方案：tokens/s、扩展效率 E_p、exposed communication 占比（Nsight + NCCL
  timeline）、peak memory、最慢 rank p50/p95（§9 规范）、time-to-loss。
- 追问预案：IB 带宽减半 → 降 DP 增 accumulation 或梯度压缩（1-bit Adam 约 5×，
  需验证收敛）；CPU offload 成瓶颈 → 只卸载优化器状态、或改 NVMe offload
  （ZeRO-Infinity 思路）；H100 平台可评估 FP8（配合 DeepSpeed 显存省约 42%）。

**设计题 2：为多租户训练平台选型并行框架**

- 需求澄清：租户模型规模分布（7B–150B+）？任务类型（SFT/LoRA/全参预训练）？
  硬件是否异构？平台运维团队规模？
- 规模估算：7B 全参状态 7G×16B = 112 GB（DDP @8 卡每卡 14 GB 可行）；
  70B 需 ZeRO-2/3；150B+ 需 ZeRO-3 + offload 或 3D 并行。
- 架构（分层）：
  1. 后端层：可插拔 PyTorch DDP/FSDP2、DeepSpeed、Megatron-LM；
  2. 抽象层：Accelerate 式分布式抽象，用户声明模型规模与资源，平台映射到并行策略与 ZeRO 阶段；
  3. 应用层：LLaMA-Factory/MS-SWIFT 式微调封装，屏蔽配置细节。
- Trade-off 表：

| 场景 | 推荐 | 理由 |
|---|---|---|
| 单机多卡微调 7B–13B | DDP/LoRA | 状态复制可承受，运维最简单 |
| 全参微调 30B–70B | FSDP2（类 ZeRO-3） | per-parameter sharding、易用性大幅提升 |
| 显存极限/单卡大模型 | DeepSpeed ZeRO-Offload/Infinity | CPU/NVMe 多级卸载，单卡可训千亿参数，以时间换空间 |
| 千亿级预训练 | Megatron-LM 3D / torchtitan 参考实现 | TP+PP+DP+SP 组合 |

- 评测方案：租户隔离性（显存配额、NCCL 通信争抢）、多任务并发吞吐、
  fabric 巡检（nccl-tests 定期 `all_reduce_perf` 基线，检测 ring vs tree
  带宽退化）、框架升级回归（梯度与单进程 reference 对比）。
- 追问预案：租户要求可复现 → 固定 seed/NCCL 算法与版本；FSDP2 需 torch 2.4+ →
  容器镜像锁定版本；故障恢复 → sharded checkpoint + world-size 迁移策略（§6.3）。

## 代码实现题

**实现题 1：ZeRO 分片显存计算器**

- 题目：给定参数量 Φ、DP 卡数 P（bf16 混合精度），输出 ZeRO 三阶段每 rank 显存估算。
- 考察点：16B/参数账本、DeepSpeed 口径公式（12Φ/P+4Φ / 14Φ/P+2Φ / 16Φ/P）。
- 骨架：

```python
def zero_stage_memory_bytes(phi: int, P: int) -> dict:
    """DeepSpeed 口径 ZeRO 三阶段每 rank 显存（bytes，bf16 混合精度）。

    phi: 参数元素数 Φ；P: DP 卡数。
    账本：fp32 master 4B + m/v 各 4B = 12B 优化器状态；bf16 参数+梯度 2+2B。
    """
    OPT = 12 * phi
    GRAD = 2 * phi
    PARAM = 2 * phi
    return {
        "stage1": OPT / P + GRAD + PARAM,        # ≈ 12Φ/P + 4Φ
        "stage2": (OPT + GRAD) / P + PARAM,      # ≈ 14Φ/P + 2Φ
        "stage3": (OPT + GRAD + PARAM) / P,      # ≈ 16Φ/P
        "ddp": OPT + GRAD + PARAM,               # 复制基线 16Φ
    }
```

- 验收标准：P=1 时四项相等（≈16Φ）；各阶段随 P 单调下降；
  Φ=70e9、P=64 时 stage3 ≈ 17.5 GB；预留激活显存参数的扩展接口。

**实现题 2：带 bucket 与异步 all-reduce 的伪 DDP 训练步**

- 题目：实现按反向就绪顺序组桶、桶内梯度齐后立刻异步 all-reduce 的 DDP 训练步。
- 考察点：§5 bucketed overlap、§6 hook 生命周期、buffer 所有权与 wait 边界。
- 骨架：

```python
import torch
import torch.distributed as dist

class BucketedDDP:
    """桶内梯度齐后立即异步 all-reduce，与更早层 backward 重叠。"""

    def __init__(self, model, bucket_cap_bytes=int(25e6)):
        self.world = dist.get_world_size()
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.buckets = self._build_buckets(self.params, bucket_cap_bytes)
        self.param_to_bucket = {id(p): b for b in self.buckets for p in b}
        self.inflight = set()      # 已发起规约的桶 id
        self.pending = []          # [(work, flat, bucket)]，持有 buffer 生命周期
        for p in self.params:
            p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _build_buckets(self, params, cap):
        buckets, cur, cur_bytes = [], [], 0
        for p in reversed(params):                 # 近似反向就绪顺序
            nbytes = p.numel() * p.element_size()
            if cur and cur_bytes + nbytes > cap:
                buckets.append(cur)
                cur, cur_bytes = [], 0
            cur.append(p)
            cur_bytes += nbytes
        if cur:
            buckets.append(cur)
        return buckets

    def _make_hook(self, param):
        def hook(_unused):
            bucket = self.param_to_bucket[id(param)]
            if id(bucket) in self.inflight:
                return
            if any(p.grad is None for p in bucket):  # 桶未齐，等后续 hook
                return
            self.inflight.add(id(bucket))
            flat = torch._utils.flatten_dense_tensors(
                [p.grad / self.world for p in bucket])  # 先除 world size
            work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
            self.pending.append((work, flat, bucket))
        return hook

    def finish_step(self):
        """optimizer step 前调用：等待全部 handle 并写回均值梯度。"""
        for work, flat, bucket in self.pending:
            work.wait()                              # wait 之后 flat 才可读
            shards = torch._utils.unflatten_dense_tensors(
                flat, [p.grad for p in bucket])
            for p, g in zip(bucket, shards):
                p.grad = g
        self.pending.clear()
        self.inflight.clear()
```

- 验收标准：与单进程 global-batch reference 梯度 allclose；profiler 中 NCCL
  kernel 与 backward kernel 时间重叠（§5.2）；人为打乱某 rank 桶顺序 → hang，
  验证顺序一致性要求（§6.2）；去掉 `/self.world` 且用 SUM → 梯度放大 P 倍（§4）。

**实现题 3：ring all-reduce α-β 计算器并用 nccl-tests 校验**

- 题目：实现 ring all-reduce 的通信量/时间估算，并输出两种带宽口径。
- 考察点：ring 两阶段推导、latency/bandwidth 两个 regime、algorithm vs bus bandwidth。
- 骨架：

```python
def ring_allreduce(S_bytes: float, p: int,
                   alpha_s: float = 1e-6, beta_Bps: float = 2e11):
    """α-β 模型：返回 (每 rank 发送量, 时间, algbw, busbw)。

    alpha_s: 单步时延（NVLink/IB 量级 ~1µs）；beta_Bps: 有效带宽 bytes/s。
    """
    per_rank = 2 * (p - 1) / p * S_bytes             # V_rank = 2(p-1)/p·S
    t = 2 * (p - 1) * alpha_s + per_rank / beta_Bps  # T_ring
    algbw = S_bytes / t
    busbw = algbw * 2 * (p - 1) / p                  # bus bandwidth 归一化
    return per_rank, t, algbw, busbw
```

- 验收标准：p=2 时 V_rank = S；小消息时间由 2(p−1)α 主导、大消息由带宽项主导；
  与 nccl-tests `all_reduce_perf -b 512M -e 4G -f 2 -g 8` 实测对比，并注明
  线性模型忽略 congestion/协议切换的偏差边界（§3 局限）。

## 13. 小结

DDP 的本质是复制模型、切分数据、平均 gradient。collective 的总 bytes 只是第一层分析；真正性能取决于消息粒度、\(\alpha/\beta\)、拓扑和能否与 backward 重叠。异步 API 只有在 buffer 生命周期、collective 顺序和等待边界正确时才既安全又有效。当复制本身成为容量瓶颈时，ZeRO 表明“数据并行”与“状态切分”可以在同一 collective 框架内统一——这正是 Lecture 08 多维并行的入口。

## 参考文献

<a id="ref-1"></a>[1] P. Patarasuk, X. Yuan. “Bandwidth Optimal All-Reduce
Algorithms for Clusters of Workstations.” *Journal of Parallel and Distributed
Computing*, 2009. https://doi.org/10.1016/j.jpdc.2009.05.002

<a id="ref-2"></a>[2] A. Sergeev, M. Del Balso. “Horovod: Fast and Easy
Distributed Deep Learning in TensorFlow.” arXiv:1802.05799, 2018.
https://arxiv.org/abs/1802.05799

<a id="ref-3"></a>[3] R. Thakur, R. Rabenseifner, W. Gropp. “Optimization of
Collective Communication Operations in MPICH.” *IJHPCA*, 2005.
https://doi.org/10.1177/1094342005051521

<a id="ref-4"></a>[4] S. Li et al. “PyTorch Distributed: Experiences on
Accelerating Data Parallel Training.” *VLDB*, 2020.
https://arxiv.org/abs/2006.15704

<a id="ref-5"></a>[5] H. Zhang et al. “Poseidon: An Efficient Communication
Architecture for Distributed Deep Learning on GPU Clusters.” *USENIX ATC*, 2017.
https://www.usenix.org/conference/atc17/technical-sessions/presentation/zhang

<a id="ref-6"></a>[6] Y. Peng et al. “A Generic Communication Scheduler for
Distributed DNN Training Acceleration.” *SOSP*, 2019.
https://doi.org/10.1145/3341301.3359642

<a id="ref-7"></a>[7] Y. Lin et al. “Deep Gradient Compression: Reducing the
Communication Bandwidth for Distributed Training.” *ICLR*, 2018.
https://arxiv.org/abs/1712.01887

<a id="ref-8"></a>[8] T. Vogels, S. P. Karimireddy, M. Jaggi. “PowerSGD:
Practical Low-Rank Gradient Compression for Distributed Optimization.”
*NeurIPS*, 2019. https://arxiv.org/abs/1905.13727

<a id="ref-9"></a>[9] S. U. Stich. “Local SGD Converges Fast and Communicates
Little.” *ICLR*, 2019. https://arxiv.org/abs/1805.09767

<a id="ref-10"></a>[10] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He.
“ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.”
*SC*, 2020. [link](https://arxiv.org/abs/1910.02054)

## 延伸阅读与复现材料

- [CS336 Lecture 7 可执行讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py)
- [PyTorch DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch distributed collectives](https://pytorch.org/docs/stable/distributed.html)
- [NCCL tests performance notes](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [A2 Systems 官方题面](../assignments/spring2026/assignment2-systems/cs336_assignment2_systems.pdf)
- 本仓库：[A2 DDP/FSDP 报告](../assignments/spring2026/assignment2-systems/report/writeup.pdf)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)（访问日期 2026-10-04）
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)（访问日期 2026-10-04）
- [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)（访问日期 2026-10-04）
- [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)（访问日期 2026-10-04）
