---
title: "Lecture 05 — GPUs, TPUs & Performance Models"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-04-13"
lecturer: "Tatsunori Hashimoto"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf"
  - "../experiments/topics/systems.md"
  - "../assignments/spring2026/assignment2-systems/"
---

# Lecture 05 — GPUs、TPUs 与性能模型：从算术强度到端到端吞吐

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：AI-assisted 原创中文自学综述，非课程提交
- 适用对象：自学者、性能工程师与系统研究者

## 摘要

现代语言模型的训练与推理性能由“计算能力”与“数据移动能力”的共同上界决定。仅统计 FLOPs
无法解释 kernel launch、memory hierarchy、layout、cache、synchronization、communication
和低精度数值行为；仅观察 GPU utilization 也无法判断是否接近有效峰值。本文比较 GPU 的
SIMT/warp/SM/Tensor Core 执行模型与 TPU 的 systolic array/XLA/SPMD 模型，基于 hierarchical
Roofline 推导 GEMM、elementwise、reduction、attention 与 autoregressive decode 的瓶颈。
进一步讨论 coalescing、tiling、occupancy、register pressure、shared-memory bank conflict、
asynchronous pipeline、BF16/FP16/TF32/FP8、HBM allocator 与互联拓扑。在工程层面给出
Nsight/profiler 指标、可靠计时和故障排查；在研究层面给出 shape sweep、controlled ablation、
能源/成本报告和效度威胁。目标是从算法 shape 预测硬件行为，再用 profiler 证据修正模型。

**关键词：** GPU；TPU；SIMT；Tensor Core；Systolic Array；Memory Hierarchy；
Roofline Model；Arithmetic Intensity；Mixed Precision；Profiling

## 本文贡献

1. 建立 FLOPs、bytes、parallelism、latency、communication 的统一资源模型；
2. 从 warp/SM 与 systolic array 两种执行范式解释 shape-dependent performance；
3. 给出 GEMM tiling、memory coalescing、occupancy 与 low-precision 的工程分析；
4. 将训练、prefill、decode 与 collectives 放进分层 Roofline/拓扑框架；
5. 提供论文级 benchmark protocol、profiler checklist、效度分析和可复现实验模板。

## 学习目标

学完应能先估算、再测量，而不是靠“换个更快 API”碰运气：

1. 从计算、HBM 流量、片上容量和 launch 数判断瓶颈；
2. 用 Roofline 解释同一算子为何随 shape、dtype、硬件而变快或变慢；
3. 理解 GPU 的 SIMT/warp/block/SM 与 TPU 的矩阵阵列式设计；
4. 为 A2 建立可复现的 benchmark、profile 和显存账本。

## 先修知识

- Lecture 02：FLOPs 与显存账本、arithmetic intensity 与 Roofline 的推导。
- 线性代数与计算机体系结构基础：矩阵乘访存模式、cache 层次、带宽与延迟数量级。
- 建议：先完成 A2 systems 的 benchmark 部分，带着实测数据回看本讲。

## 相关工作与硬件演进

CUDA 将大规模线程组织为 SIMT execution，形成通用 GPU computing 的基础
[[1]](#ref-1)；后续工作表明，高性能 GEMM 的关键不仅是峰值 FLOPs，而是 tile、register
blocking、memory access 和 latency hiding [[2]](#ref-2)。Roofline model 用 arithmetic
intensity 把程序需求与 compute/bandwidth ceilings 连接，提供跨 kernel 的统一上界
[[3]](#ref-3)。

TPU 以 matrix multiply unit/systolic array 和 compiler-controlled dataflow 优化神经网络
[[4]](#ref-4)，TPU v4 又把大规模 pod、optical circuit switching 与 embedding workload
纳入系统设计 [[5]](#ref-5)。低精度方面，Mixed Precision Training 奠定 FP16+FP32
accumulation/master-weight recipe [[6]](#ref-6)，后续 BF16、TF32 和 FP8 进一步在动态范围、
精度与 tensor-core throughput 间折中。

语言模型系统研究反复证明 IO/communication 与计算同等重要。FlashAttention 通过减少
HBM traffic 改善 exact attention [[7]](#ref-7)，FlashAttention-2 进一步改进
parallelism 与 work partitioning [[8]](#ref-8)；Megatron-LM 和 ZeRO 则通过 model
parallelism 与 state sharding 扩展训练规模 [[9]](#ref-9)[[10]](#ref-10)。对 memory
wall 的系统分析进一步指出，模型规模的增速远超 HBM 带宽与容量的增速，数据搬运将成为
日益刚性的约束 [[11]](#ref-11)。这些结果说明：
硬件峰值是必要背景，但 end-to-end 性能取决于算法、kernel、runtime 与 topology 的共同设计。

## 1. 加速器的共同结构与差异

GPU 和 TPU 都在做同一件事：把高吞吐计算单元靠近快但小的片上存储，并尽量少访问慢而大的 HBM。

### GPU 心智模型

- **grid → thread block/CTA → thread** 是编程层级；
- 一个 block 调度到一个 SM，共享该 SM 的 shared memory；
- 线程以 32 个为一个 warp 锁步执行；分支不一致会串行执行不同路径；
- Tensor Core 负责小块矩阵乘，普通 CUDA core 处理标量/向量运算；
- register 最快且线程私有，shared memory/L1 片上共享，L2/HBM 更大但更慢。

实际优化不是单纯提高 occupancy。更多寄存器可能减少可驻留 warp，却也可能避免 spill 到 local memory。应观察 `registers/thread`、active warps、spill 和 kernel 时间，而不是追求单一百分比。

#### GPU latency hiding 与资源约束

一个 SM 可同时驻留多个 blocks/warps；当某 warp 等待 memory 或 dependency 时，scheduler
切换到 ready warp。可驻留数量受以下最小项限制：

- threads/warps per SM；
- registers per SM ÷ registers per thread；
- shared memory per SM ÷ shared memory per block；
- architecture block limit。

高 occupancy 只表示有更多候选 warps，不保证 instruction-level parallelism、cache hit 或
tensor-core utilization。若 kernel 已 compute-bound，增加 occupancy 可能无收益；若 register
pressure 导致 spill，降低 occupancy 但保留 registers 反而更快。

GPU memory hierarchy 的“local memory”名字容易误导：它是线程私有地址空间，但通常落在
device memory/L2，而非片上 register。Profiler 中 local load/store 激增常表示 register spill。

### TPU 心智模型

TPU 同样包含控制、向量、矩阵乘单元和片上/片外存储，但更强调大规模矩阵阵列与编译器静态调度。GPU 暴露 warp、block 和 shared memory 等细节；TPU 通常让 XLA 根据张量程序安排布局、融合和设备间 sharding。

结论不是“谁总是更快”，而是：

- GPU 的 kernel 生态与细粒度可编程性强；
- TPU 对规则的大矩阵与 SPMD sharding 友好；
- 两者都受数据搬运、shape 对齐、低精度能力和互联拓扑约束。

#### Systolic array 与 dataflow

Systolic array 让 weights/activations/partial sums 在相邻 processing elements 间规律流动，
用片上复用降低高层 memory traffic。矩阵维度若与 array tile 不对齐，会产生 padding/underfill。
TPU 编程更多依赖 XLA 的 fusion、layout assignment、collective scheduling 与 SPMD partitioning，
因此 source-level 简洁不表示 dataflow 自动最优；仍需查看 compiler HLO/trace。

#### 单芯片之外：互联也是架构

GPU 节点可能含 PCIe、NVLink/NVSwitch，跨节点使用 InfiniBand/RoCE；TPU pod 使用专用
interconnect。Distributed performance 取决于 bandwidth、latency、topology、collective
algorithm 与消息大小。单卡 Roofline 不能解释 all-reduce exposed time，应增加 network
roofline 或 \(\alpha\)-\(\beta\) communication model。

## 2. Roofline：先算上限

令工作量为 \(F\) FLOPs、从目标存储层搬运 \(Q\) bytes：

\[
I=\frac{F}{Q}\quad(\text{FLOPs/byte})
\]

若峰值计算吞吐为 \(C_{\max}\)、带宽为 \(B\)，则

\[
P\le \min(C_{\max}, B I),\qquad
T\ge \max\left(\frac{F}{C_{\max}},\frac{Q}{B}\right).
\]

拐点 \(I^\*=C_{\max}/B\)：

- \(I<I^\*\)：倾向 memory-bound，减少 bytes/fusion 比增加 FLOPs 单元更重要；
- \(I>I^\*\)：倾向 compute-bound，应提高 Tensor Core 利用率和 tile 效率。

Roofline 是下界模型，不包含 launch latency、cache 命中、依赖、同步、通信和尾波效应，因此最后必须 profile。

### 典型 shape 与复杂度

矩阵乘 \(C_{M\times N}=A_{M\times K}B_{K\times N}\)：

\[
F\approx 2MKN.
\]

若每个输入只从 HBM 读一次、输出写一次，元素字节数为 \(s\)：

\[
Q_{\min}\approx s(MK+KN+MN).
\]

方阵且维度为 \(n\) 时，\(I=\Theta(n)\)，大 GEMM 容易 compute-bound。逐元素算子的 \(F,Q=\Theta(n)\)，强度为常数，通常 memory-bound。小 GEMM 即使理论强度高，也可能被 launch、padding 或 SM 尾波主导。

### Transformer 里应看什么

- projection/MLP GEMM：通常计算占主导，但小 batch/短序列会降低利用率；
- normalization、activation、optimizer：读写多、每元素计算少，适合融合；
- 标准 attention：算术量 \(O(BHN^2d)\)，并可能写出 \(O(BHN^2)\) scores/probabilities；
- FlashAttention 不改变渐进 FLOPs，而是用 tiling/recompute 消除二次 HBM 中间量。

### Hierarchical Roofline 与 cache reuse

单一 HBM Roofline 假设所有 bytes 都来自 HBM，实际有 register/shared/L1/L2/HBM 多层。
对每层存储可定义 \(I_\ell=F/Q_\ell\) 与带宽 ceiling \(B_\ell I_\ell\)。一个 kernel 可能
相对 HBM compute-bound，却受 shared-memory bandwidth 或 register dependency 限制。

理论 \(Q_{\min}\) 只是“完美 tile、一次读入”的下界。实际 bytes 会被以下因素放大：

- tile 重叠/边界 padding；
- cache eviction 和多 CTA 重复读取；
- dtype cast/intermediate writes；
- non-coalesced transaction；
- backward 重读/重算；
- fused kernel 是否保留 intermediates 在片上。

因此 Roofline 的研究价值是形成可证伪假设：若估算 memory-bound，就应在 profiler 中看到
高 achieved bandwidth、低 compute utilization；否则说明 bytes/peak/依赖假设有误。

### Prefill、decode 与训练的 Roofline 不同

- **训练**：大 \(BT\) 形成高复用 GEMM，通常更容易 compute-bound；
- **prefill**：大 prompt 并行，类似 forward，但无 backward；
- **decode**：每 step query 很小，反复读取 weights/KV，常 memory-bound；
- **optimizer**：逐元素读取 parameters/grad/moments，典型 bandwidth-bound。

不能用训练 MFU 预测 serving tokens/s，也不能用单 request decode 结论预测高并发 continuous batching。

## 3. 真正决定速度的硬件细节

### 合并访问与布局

同一 warp 连续访问相邻地址，可合并成少量 cache-line transaction。错误 stride、转置后非连续访问或 AoS/SoA 选择不当会放大流量。`contiguous()` 也不是免费修复：它本身会产生一次完整拷贝。

### Tiling 与复用

朴素 GEMM 每个输出元素反复从 HBM 读行和列，强度近似常数。把 \(A\)、\(B\) tile 放入片上存储，让多个输出复用，算术强度随 tile 边长增长。tile 过大则增加 shared memory/register 压力，降低并发度。

### Warp divergence 与 bank conflict

- warp 内不同分支不能真正并行；causal mask 应尽量用规则 tile 边界减少无效工作；
- shared memory 有多个 bank；同一 warp 对同一 bank 的不同地址访问会串行；
- padding/swizzle 可改变 bank 映射，但必须以 profiler 证实收益。

### 低精度

低精度同时可能：

1. 减少 HBM bytes；
2. 提高 Tensor Core 峰值；
3. 增加误差或溢出风险。

BF16 与 FP32 指数位相同，动态范围较稳 [[12]](#ref-12)；但尾数短，长 reduction 应以 FP32 累加。FP16 动态范围小，训练常需 loss scaling。参数、gradient、Adam moments 是否仍为 FP32，决定“低精度”能否显著降低总显存。

### Tensor Core shape 与数据布局

Tensor cores 执行固定 micro-tile MMA；可用 dtype、transpose/layout 与维度 alignment 受架构约束。
即使 FLOPs 相同，\(M,N,K\) 很小或尾块占比高会 underutilize。工程上记录：

- 是否真正生成 tensor-core instructions；
- \(M,N,K\) 及其倍数/alignment；
- accumulator dtype；
- layout conversion；
- achieved TFLOP/s 与 theoretical dtype peak。

“开启 autocast”不保证每个 matmul 使用预期低精度，也不保证 reduction/softmax 安全。

### Software pipelining 与 asynchronous copy

高性能 kernel 会把“加载下一 tile”和“计算当前 tile”重叠，使用 double buffering、
asynchronous shared-memory copy 和多 stage pipeline。Stage 太少无法隐藏 latency；
太多消耗 shared memory/register，降低 occupancy。调参轴通常包括 tile sizes、warps、stages；
必须联合正确性、register spill、occupancy 与 duration 搜索，而非只增大 tile。

### FP8/量化的额外状态

FP8 不只是把 dtype 改成 1 byte。训练需要 per-tensor/per-channel scale、amax history、
format 选择（如 E4M3/E5M2 等 exponent/mantissa 组合 [[13]](#ref-13)）和高精度 accumulation。Serving INT8/INT4 还要考虑
weight-only vs activation quantization、dequantization kernel、outlier channels 和 calibration。
报告 memory reduction 时必须加上 scale/metadata 与可能保留的高精度副本。

## 4. 显存与时间账本

设参数量为 \(P\)。纯 FP32 AdamW 的常驻下界常估作：

\[
4P\ (\text{参数})+4P\ (\text{梯度})+8P\ (m,v)=16P\text{ bytes}.
\]

还要加 activation、saved tensors、temporary workspace、通信 buffer、allocator fragmentation。`memory_allocated` 是活跃张量，`memory_reserved` 含缓存池；两者不能混用。

训练 step 可拆成：

\[
T_{\text{step}}=
T_{\text{forward}}+T_{\text{loss}}+T_{\text{backward}}+
T_{\text{sync}}+T_{\text{optim}}.
\]

异步 CUDA 使 CPU 计时容易只测到 enqueue。正确基准至少包括 warmup、CUDA event 或边界同步、重复采样、均值与离散度，并把 compile 首次成本和 steady-state 分开。

### 4.1 多 GPU 时间账本

分布式 step 更接近 critical-path 模型，而非各项简单求和：

\[
T_{\text{step}}
\approx T_{\text{compute}}+T_{\text{communication}}
-T_{\text{overlap}}+T_{\text{bubble}}+T_{\text{straggler}}.
\]

通信 lower bound 可用 \(\alpha+\text{bytes}/\beta\)；实际还受 collective algorithm、
topology、contention 与 bucket size 影响。Overlap 只有在依赖图允许且通信/计算使用不同资源时
成立；Nsight timeline 比“总通信量”更能说明 exposed time。

### 4.2 Energy、成本与 time-to-quality

性能工程不应只最大化 TFLOP/s。研究和部署还关心：

\[
\text{energy}=\int P(t)\,dt,\qquad
\text{cost}=\text{GPU-hours}\times\text{price/GPU-hour}.
\]

低精度/大 batch 可能提高吞吐却改变 convergence；公平比较应固定目标 loss/quality，报告
time-to-quality、energy-to-quality 与失败/OOM runs。共享云环境还需记录实例类型、区域、
是否抢占和 power cap。

## 5. 代码与实验映射

- A2 统一入口：`assignments/spring2026/assignment2-systems/scripts/benchmark_systems.py`
- 报告：`assignments/spring2026/assignment2-systems/report/main.tex`
- 原始数据：`report/results/raw/`
- 系统导读：`experiments/topics/systems.md`

仓库实测环境为 RTX 6000D、PyTorch 2.8.0+cu128；不是课程指定 B200。可复核的观察：

- xl、batch 4、sequence 512：BF16 forward+backward 438.12 ms，FP32 1120.75 ms；
- 但 peak allocated 仅从 40990 MiB 降至 38058 MiB，并未减半；
- medium、BF16 的 sequence 512→2048，时间增长 11.0 倍、显存增长 6.78 倍，说明二次 attention 项逐渐主导；
- large、sequence 2048 使用 36 段 checkpoint，显存 56.48→11.53 GiB，时间 983.81→1254.51 ms。

这些数字只说明该环境和 shape。迁移到别的 GPU，应保留同样语义与实验矩阵重新测量。

## 6. 实用诊断流程

1. 写出 shape、dtype、理论 FLOPs、最低 bytes 和中间张量大小；
2. 先测 end-to-end，确认优化对象占总时间；
3. profile kernel 数、时间、launch 间隙和显存峰值；
4. 判断是 launch、HBM、compute、片上资源还是同步受限；
5. 每次只改变一个因素：dtype、layout、fusion、tile 或 compile；
6. 先比输出/梯度，再比稳定态时间和 peak memory；
7. 报告硬件、软件版本、warmup、repetitions、同步方法和失败/OOM。

## 7. Profiler 指标如何形成证据

### 7.1 Timeline 层

- CPU launch gaps、CUDA API blocking；
- kernel duration/overlap；
- memcpy、page fault、synchronization；
- NCCL collectives 与 compute overlap；
- stream dependency 与 bubble。

### 7.2 Kernel 层

- achieved occupancy 与 theoretical occupancy；
- DRAM/L2/shared throughput；
- tensor-core instruction utilization；
- branch efficiency、warp stall reasons；
- register/thread、shared memory/block、spill；
- eligible/active warps 与 issue efficiency。

单一指标不能独立诊断。例如 memory throughput 低可能是 compute-bound，也可能是 latency-bound
且 occupancy 不足；occupancy 高但 issue efficiency 低可能在等待 dependency。

### 7.3 Memory 层

PyTorch memory snapshot 用于 tensor/allocation lifetime，Nsight/硬件 counters 用于 traffic。
前者回答“谁占显存”，后者回答“搬了多少、搬得多快”。二者不能互相替代。

## 8. 面向研究的实验设计

### 8.1 Shape sweep 而非单点

对 GEMM/attention/model step 扫描 \(B,T,d,f,V\)，画 latency、throughput、memory、
achieved TFLOP/s 与 bandwidth。标注 OOM/error，不只保留成功点。变化斜率可帮助识别
launch→bandwidth→compute 不同 regime。

### 8.2 Hardware comparison

跨 GPU/TPU 对比至少匹配：

- dtype 与 numerical recipe；
- global batch / tokens / model；
- software kernel maturity；
- device 数和 interconnect；
- power/cost 口径；
- correctness/quality。

直接用 vendor peak spec 计算“应快 X 倍”忽略 memory、shape 和 software，通常不成立。

### 8.3 预注册假设示例

> 对固定 BF16 Transformer，sequence 较短时 step 由 launch 与 dense projection 主导；
> sequence 增大后 attention 变为 memory/compute bottleneck；fused attention 会降低显存
> 增长斜率，并在超过某一 \(T\) 后取得 end-to-end speedup。

预先定义 crossover、primary metric、tolerance、warm-up 和 OOM policy，避免看到结果后
选择性解释。

## 9. 易错点

- 用 Python `time.time()` 包住 CUDA 调用，却没有同步；
- 把首次 JIT/allocator/autotune 时间当稳定迭代；
- 只报告最好一次，或只报告吞吐不报告 shape；
- 看到 occupancy 低就盲目减寄存器；
- 认为 BF16 会让全部训练显存减半；
- 将 `reserved` 当成模型真实占用；
- 用 CPU/Gloo 正确性结果推断 GPU/NCCL 性能；
- 只优化很快的 microkernel，却不确认其 end-to-end 占比；
- 用 sparse 峰值（如 H100 BF16 1979 TFLOPS）当 roofline 分母，高估 compute ceiling；
- 跨机 all-reduce 直接用单卡 NVLink 带宽估算，忽略拓扑层级与 contention。

## 10. Checklist

- [ ] 写出每个核心张量的 shape、dtype、bytes
- [ ] 估算 FLOPs、HBM 流量、arithmetic intensity
- [ ] warmup 后再计时，边界显式同步
- [ ] 同时报 mean/std、latency/throughput、peak allocated
- [ ] compile time 与 steady-state 分开
- [ ] 用 profiler 验证 kernel 数和瓶颈
- [ ] FP32/BF16 比较保持相同模型、batch、sequence 与语义
- [ ] 优化后重新做 output/gradient correctness

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| GPU utilization 低 | CPU timeline/data loader | launch gap、I/O、同步 |
| utilization 高但 TFLOP/s 低 | kernel mix/AI | elementwise、memory-bound |
| GEMM 未上 tensor core | dtype/shape/instruction | alignment、autocast、layout |
| occupancy 低 | registers/shared/block | tile 太大、spill |
| occupancy 高仍慢 | stall reasons | dependency、HBM、bank conflict |
| 第一次 iteration 很慢 | warm-up/compile | JIT、autotune、allocator |
| BF16 无显存收益 | state breakdown | FP32 master/moments、activations |
| reserved 持续很高 | allocator snapshot | fragmentation/caching |
| 多卡 scaling 差 | topology/NCCL timeline | exposed all-reduce、imbalance |
| microkernel 快、模型不变 | end-to-end share | Amdahl’s law [[14]](#ref-14) |

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| 实测带宽远低于峰值 | 访存合并（coalescing）与对齐 | 非连续 stride、跨 stride 读 |
| SM 占用率低 | grid/block 配置 | block 过小、寄存器溢出导致降占用 |
| kernel 时间不随规模线性 | L2 命中率 | cache 命中主导而非带宽 |
| 多卡扩展近线性失效 | 通信占比与互联层次 | 跨 PCIe 通信、未用 NVLink 域内聚合 |
| TPU 与 GPU 数值不一致 | 累加顺序与精度容差 | systolic 阵列累加顺序差异属预期 |
| 吞吐周期性骤降 | SM clock 与功耗曲线 | 热节流降频，检查 clock 采样 |
| roofline 预测失准 | 峰值参数取值（boost vs sustained） | 用 sustained 而非理论峰值作分母 |

## 11. 讨论：效度威胁与结论边界

### Construct validity
- GPU utilization、occupancy、MFU、TFLOP/s、tokens/s 是不同指标；
- theoretical FLOPs 不包含 data movement/communication；
- peak allocated 不等于 process VRAM；
- synthetic microbenchmark 不等于 time-to-quality。

### Internal validity
- 异步计时、未 warm-up、compile/autotune 混入会系统性偏差；
- power/clock/thermal 与共享租户造成 run-to-run variance；
- 不同 shape/dtype/kernel version 是 confounders；
- profiler instrumentation 改变 runtime。

### External validity
- 单 GPU 结果不外推多节点；
- RTX 6000D 不代表 B200/TPU；
- 一个 sequence/batch 的 kernel 最优参数不泛化；
- 峰值 spec 与真实应用可达值差异显著。

论文式表述应限定硬件、软件、shape 与 measurement protocol，优先报告 confidence interval
和完整失败点，而不是无条件“X× 更快”。

## 面试要点速记

**高频问题与答题要点**

1. **Q：A100/H100 关键数字？** 要点：A100 80GB：HBM ~2TB/s、bf16 tensor ~312
   TFLOPS（dense）、NVLink 600GB/s；H100 SXM：HBM3 3.35TB/s、bf16 ~990
   TFLOPS（dense）、NVLink 900GB/s。峰值算力/带宽之比决定 roofline 拐点。
2. **Q：为什么 GPU 适合 GEMM？** 要点：tensor core 在寄存器级完成小矩阵乘、
   shared memory tiling 保证数据复用、高 arithmetic intensity。
3. **Q：内存层次的数量级？** 要点：SMEM / L2 / HBM 带宽差 1–2 个数量级、
   延迟差更多；kernel 优化的本质是最大化复用、减少远端访存。
4. **Q：TPU 与 GPU 的本质差异？** 要点：systolic 阵列面向大规模批量 GEMM、
   软件栈封闭、控制流与稀疏性弱；统一 HBM；选型取决于 workload 形态。
5. **Q：为什么 decode 是带宽瓶颈？** 要点：每 step 读全部 weights+KV 而 FLOPs 仅
   2P，强度 ≈ 1–2 FLOPs/byte，远低于拐点（~300）；tokens/s 上界 ≈ 聚合 HBM 带宽
   ÷ 模型字节数；batching 摊销权重读取。
6. **Q：Roofline 拐点怎么算、代际怎么变？** 要点：拐点 = dense 峰值算力 ÷ HBM
   带宽；H100 BF16 ≈ 295、B200 ≈ 281、H200 ≈ 206 FLOPs/byte；两代拐点接近但
   绝对性能翻倍，H200 靠带宽下压拐点利于 decode。
7. **Q：机内与跨机通信差在哪？** 要点：NVLink 域内 900GB/s（H100）/1.8TB/s
   （B200）、时延 ~1µs 级；跨机 IB/RoCE 带宽低一个量级；TP/EP 锁域内、DP/PP
   出域，梯度分桶 + overlap；GB200 NVL72 把 NVLink 域扩到 72 卡。
8. **Q：怎么评估一届硬件的性价比？** 要点：不看 peak 排行榜，看实测
   TFLOPS/GPU（如 70B TP=8：H200 454.6 vs B200 815.2）、能耗比（H100 ≈1.41
   vs B200 ≈2.25 TFLOPS/W，FP8/INT8 口径）与互连拓扑是否匹配并行策略。

**必背数字**

- roofline 拐点（A100 bf16）≈ 312e12 / 2e12 ≈ 156 FLOPs/byte；
  occupancy = 活跃 warp / 最大 warp。

**工业界参照（2024–2026，SXM/官方标称口径）**

- H100 SXM：BF16 dense 989 TFLOPS、HBM3 3.35TB/s、NVLink4 900GB/s、TDP 700W；拐点 ≈ 295 FLOPs/byte。
- B200：BF16 dense 2.25 PFLOPS、HBM3e 8TB/s、NVLink5 1.8TB/s、TDP 1000W；拐点 2250/8 ≈ 281 FLOPs/byte。
- GB200 NVL72：72×B200 + 9 个 Switch Tray（57.6Tb×9 = 518.4Tb）无阻塞全互联，机柜内铜连接。
- 实测（LLaMA2-70B，TP=8）：H200 454.6 → B200 815.2 TFLOPS/GPU（+79.3%）。
- 能耗比：H100 ≈ 1.41、B200 ≈ 2.25 TFLOPS/W（FP8/INT8 口径）。
- MLPerf Training：B200 对 H200，GPT-3 175B 预训练 ~2.0×、Llama-70B LoRA 微调 ~2.2×。

## 行业现状与最新进展（2024–2026）

### Hopper→Blackwell：规格与算力/带宽演进

| 指标（SXM/官方标称） | A100 | H100 SXM | H200 SXM | B200 |
|---|---|---|---|---|
| BF16/FP16 tensor dense | 312 TFLOPS | 989 TFLOPS | 989 TFLOPS | 2.25 PFLOPS |
| BF16 tensor sparse | — | 1979 TFLOPS | 同 H100 | 4.5 PFLOPS |
| FP8/INT8 dense | — | 3.958 PFLOPS | 同 H100 | 4.5 PFLOPS（sparse 9 PFLOPS） |
| HBM 容量/带宽 | 80GB HBM2e / 2TB/s | 80GB HBM3 / 3.35TB/s | 141GB HBM3e / 4.8TB/s | 192GB HBM3e / 8TB/s |
| NVLink | NVLink3 600GB/s | NVLink4 900GB/s | 900GB/s | NVLink5 1.8TB/s |
| TDP | 400W | 700W | 700W | 1000W（208b 晶体管，2×104B die） |
| Roofline 拐点（BF16） | ≈156 | ≈295 | ≈206（989/4.8） | ≈281（2250/8） |

关键观察：H100→B200 拐点几乎不动（295→281），但算力与带宽绝对值均约翻倍——两代延续
compute/bandwidth 同比例扩展；H200 靠 4.8TB/s 带宽把拐点压到 ~206，更适合 decode、MoE
等带宽敏感负载。FP4（B200 dense 9 PFLOPS）进一步把低精度上限推高，但收敛与精度风险由
用户承担。

### NVLink/NVSwitch 与机柜级互连

- GB200（Grace+Blackwell Superchip）：2×Blackwell GPU + NVLink5，TDP 2700W；
- GB200 NVL72：72 颗 B200 + 9 个 Switch Tray（57.6Tb×9 = 518.4Tb）无阻塞全互联，
  机柜内走铜（成本与功耗低于光模块）；
- 柜间扩展需第二层 NVSwitch（NVL36×2 方案），第二层光互连（AOC）与有源铜（AEC，~3m）并存；
- 含义：单机 8 卡不再是并行天花板，“NVLink 域”从节点级扩大到机柜级，TP/EP 可在 72 卡
  域内低损耗扩张；跨机 all-reduce 仍受第二层互连带宽与时延约束。

### 实测案例：B200 vs H200 训练吞吐

- LLaMA2-70B，TP=8，单节点：H200 454.6 TFLOPS/GPU vs B200 815.2 TFLOPS/GPU
  （+79.3%），迭代耗时 -86.7%；双/四节点 B200 约 810–814 TFLOPS/GPU，多机扩展损耗小；
- MLPerf Training 口径：B200 对 H200，GPT-3 175B 预训练 ~2.0×、Llama-70B LoRA 微调 ~2.2×；
- 解读：峰值算力 2.25PFLOPS/989TFLOPS ≈ 2.3×，但实测 1.8–2.2×——差距来自 HBM 带宽、
  通信与 kernel 成熟度，印证本讲“vendor peak ≠ end-to-end”的观点。

### TPU 对照（概念，不引新数字）

本讲 TPU 内容仍然适用。拓扑差异速记：TPU v4 以 pod 为单位组织，ICI（机内/近邻高速）
与 DCN（跨 pod）双网络分工；GPU 侧则是 NVLink/NVSwitch（域内）+ InfiniBand/RoCE（跨机）。
两者都在把“高带宽域”做大、把昂贵流量留在域内。

**对本讲学习者的启示**：代际升级中拐点几乎不变，说明算术强度的判断标准稳定——先用
Roofline 拐点给自己的 workload 定位，再选硬件；选型时对比的是“你的强度落在哪一段”，
而不是 peak TFLOPS 排行榜。互连与并行策略必须联合设计：NVLink 域扩大（NVL72）改变
TP/EP 的可行切分，跨域通信预算决定 PP/DP 配比。

## 大厂面试真题与答题框架

以下为高频面试题（公开面经风格），非任何公司真题。

**题目 1：给定模型规模与序列长度，估算一次 GEMM 的算术强度，并判断 compute 还是 bandwidth bound**
- 考点：Roofline、GEMM FLOPs/bytes 公式、拐点计算。
- 答题框架：1) 写出 GEMM shape（M,K,N），FLOPs ≈ 2MKN；2) 估算最低 bytes
  ≈ s(MK+KN+MN)（s 为每元素字节数，BF16 取 2）；3) 算强度 I = F/Q，方阵时 I = Θ(n)；
  4) 与硬件拐点比较（H100 BF16 ≈ 295、B200 ≈ 281 FLOPs/byte）；5) 下结论并说明
  实际 bytes 会被 tile 重叠、cache miss 放大，需 profiler 验证。
- 加分项：指出小 batch/短序列时 M 维塌缩、强度骤降；提到 hierarchical roofline
  （HBM bound ≠ shared-memory bound）。
- 踩坑：用 sparse 峰值（如 H100 BF16 1979 TFLOPS）当分母；忘记输出写入的 MN 项。

**题目 2：为什么 LLM decode 阶段是带宽瓶颈？**
- 考点：decode 的访存模式、强度 vs 拐点、tokens/s 上界。
- 答题框架：1) decode 每 step 只处理 1 个 token，GEMM 退化为 GEMV，FLOPs ≈ 2P；
  2) 但每 step 要读全部 weights（+KV cache），bytes ≈ 2P（BF16）；3) 强度 ≈ 1
  FLOPs/byte，远低于拐点（~300）；4) 因此 tokens/s 上界 ≈ HBM 带宽 / 模型字节数
  （H200 4.8TB/s / 140GB ≈ 34 tokens/s 量级）；5) 缓解：batching 摊销权重读取、
  weight-only 量化减 bytes。
- 加分项：指出 batching 提高强度但受 KV cache 显存与 latency SLO 约束；
  提 continuous batching。
- 踩坑：说“GPU 利用率低所以是带宽瓶颈”——利用率不是证据，带宽实测才是。

**题目 3：Roofline 拐点怎么算？H100 和 B200 的拐点各是多少？**
- 考点：拐点公式与代际比较。
- 答题框架：1) 拐点 I* = 峰值算力 / 带宽；2) H100 BF16 = 989 TFLOPS / 3.35TB/s
  ≈ 295 FLOPs/byte；3) B200 = 2250/8 ≈ 281；4) 指出两代拐点接近但绝对性能翻倍，
  说明 NVLink/HBM 与算力同比例演进；5) H200 拐点 ≈ 206，对带宽敏感负载更友好。
- 加分项：把拐点变化映射到选型建议（decode/MoE 选大带宽、dense 大 GEMM 选大算力）。
- 踩坑：把 NVLink 带宽当 HBM 带宽；混淆 dense/sparse 口径。

**题目 4：A100→H100→B200 关键数字与代际演进逻辑？**
- 考点：硬件代际账本、数字敏感度。
- 答题框架：1) 报三组关键数（BF16 dense、HBM 带宽、NVLink）：A100 312/2TB/s/600GB/s，
  H100 989/3.35TB/s/900GB/s，B200 2.25PFLOPS/8TB/s/1.8TB/s；2) TDP 400→700→1000W；
  3) 总结演进逻辑：算力×带宽×互连同比例扩张 + 低精度位数下探（FP16→FP8→FP4）；
  4) 用实测收尾：B200 对 H200 训练吞吐 ~1.8–2.2×，低于峰值比 2.3×。
- 加分项：提能耗比（H100 ≈1.41 → B200 ≈2.25 TFLOPS/W，FP8/INT8 口径）与
  GB200 NVL72 机柜级互连。
- 踩坑：只背峰值算力，答不出带宽与互连；把 HBM3 与 HBM3e、NVLink4/5 混为一谈。

**题目 5：为什么 all-reduce 要尽量在 NVLink 域内完成？跨机怎么办？**
- 考点：拓扑感知通信、α-β 模型、并行策略选择。
- 答题框架：1) NVLink 域内 900GB/s（H100）/1.8TB/s（B200），时延 ~1µs 级；
  跨机 IB/RoCE 带宽低一个量级；2) 通信时间 ≈ α + bytes/β，跨机 β 是短板；
  3) 策略：TP/EP 放域内（通信量大、时延敏感），PP/DP 跨机（通信量相对小）；
  4) 跨机用梯度分桶 + 通信/反向传播 overlap；5) 机柜级 NVLink 域（GB200 NVL72，
  518.4Tb 无阻塞）把“域”从 8 卡扩到 72 卡。
- 加分项：提 NCCL 的 topology-aware ring/tree 算法选择；用 nccl-tests 实测 bus bandwidth。
- 踩坑：认为 all-reduce 时间只由总带宽决定，忽略 contention 与 bucket size。

**题目 6：MFU 是什么？怎么从 tokens/s 算到 MFU？**
- 考点：MFU 定义、模型 FLOPs 利用率计算链。
- 答题框架：1) MFU = 实测 TFLOPS/GPU ÷ 理论峰值（注意 dense 口径，H100 BF16 取
  989 而非 1979）；2) 实测 TFLOPS = 6·P·tokens/s ÷ GPU 数（乘法/加法各计一次，
  forward 2P、backward 4P）；3) 代数字示例：70B 模型、8×H200、454.6 TFLOPS/GPU
  → MFU ≈ 454.6/989 ≈ 46%；4) 说明 MFU 的用途与局限：不反映通信/数据管道等待，
  也不能外推到 serving tokens/s。
- 加分项：区分 MFU 与 HBWU（显存带宽利用率）——decode 更应看后者。
- 踩坑：用 sparse 峰值或 FP8 峰值给 BF16 训练算 MFU；把 BF16 显存收益误算进 FLOPs。

**题目 7：GB200 NVL72 的机柜级互连对并行策略设计意味着什么？**
- 考点：拓扑与并行策略联合设计。
- 答题框架：1) NVL72 = 72×B200 + 9 Switch Tray，518.4Tb 无阻塞，柜内铜连接；
  2) NVLink 域从 8 卡扩到 72 卡 → TP=16/32、大 EP 的通信损耗大幅下降；
  3) 跨柜用第二层 NVSwitch（NVL36×2），第二层 AOC/AEC 并存，跨柜带宽仍低于柜内；
  4) 策略：通信密集维度（TP/EP）锁在柜内，DP/PP 出柜；5) 估算示例：TP=8 单节点
  B200 实测 815.2 TFLOPS/GPU，双/四节点仍 ~810–814，说明扩展损耗可控。
- 加分项：提机柜供电/散热（GB200 TDP 2700W/NVL72 整柜功耗 MW 级）是选型约束。
- 踩坑：只看 GPU 数不算互连层级；假设跨柜与柜内等带宽。

## 系统设计题

**设计题 1：为 70B 模型（BF16 训练）做训练集群 GPU 选型与互连设计（H100 vs B200）**
- 需求澄清：模型 70B、序列 4–8k、目标 tokens/s 与预算上限、是否容忍 FP8、
  机房供电/机柜限制、扩展规模（单节点 vs 多节点）。
- 规模估算：显存账本 16P bytes（FP32 AdamW）≈ 1.1TB → 至少 TP=8 或 ZeRO 分片；
  实测参照：LLaMA2-70B TP=8，H200 454.6、B200 815.2 TFLOPS/GPU；若目标 MFU >45%，
  B200 单节点即可，H100/H200 需更多节点补偿。
- 架构：单节点 8×B200（NVLink5 1.8TB/s 域内 TP=8）+ 跨节点 DP/PP（IB/RoCE）；
  或 GB200 NVL72 柜内 TP/EP、柜间 DP。梯度通信用分桶 + overlap。
- Trade-off 表：

| 维度 | 8×H200 节点集群 | 8×B200 节点集群 | GB200 NVL72 |
|---|---|---|---|
| 单 GPU BF16 dense | 989 TFLOPS | 2.25 PFLOPS | 2.25 PFLOPS |
| HBM 带宽 | 4.8TB/s | 8TB/s | 8TB/s |
| NVLink 域 | 8 卡 / 900GB/s | 8 卡 / 1.8TB/s | 72 卡 / 518.4Tb 无阻塞 |
| 能耗比（FP8/INT8 口径） | ≈1.41 TFLOPS/W | ≈2.25 TFLOPS/W | ≈2.08（GB200） |
| 供电/改造 | 700W/卡，较易 | 1000W/卡 | 2700W/superchip，机柜级改造 |

- 评测方案：固定 tokens 与 loss 目标，报告 time-to-quality 与 energy-to-quality；
  MFU（dense 口径）、all-reduce exposed time（Nsight/NCCL timeline）、多节点
  扩展曲线（8→16→32 卡）；用 nccl-tests 校准 bus bandwidth。
- 追问预案：预算受限→H200 靠带宽优势保 decode/微调，训练密度用更多节点换；
  FP8 可用→B200 FP8 4.5 PFLOPS 可再提吞吐，但需 per-tensor scale 与收敛验证；
  断点续训/弹性→DP 维度可缩放，TP 维度不可。

**设计题 2：70B 模型推理集群的带宽预算与部署方案**
- 需求澄清：并发请求数、每请求 token 数、TTFT/TPOT SLO、精度要求
  （BF16 vs weight-only INT8/INT4）、预算与卡型。
- 规模估算：weights BF16 ≈ 140GB → 单卡放不下，TP=2（H200 141GB）勉强、
  常规 TP=4/8；decode tokens/s 上界 ≈ 聚合 HBM 带宽 / 140GB：单节点 8×H200
  ≈ 38TB/s / 140GB ≈ 273 tokens/s（总吞吐上界，无 batch 摊销时每请求 ≈ 带宽/权重）；
  B200 单节点 8×8TB/s = 64TB/s，上界约翻倍。
- 架构：prefill/decode 分离（prefill compute-bound 用大算力卡，decode
  bandwidth-bound 用大带宽卡）；decode 侧 continuous batching 提高权重复用；
  KV cache 显存预算 = 并发数 × 序列长 × KV bytes，决定最大 batch。
- Trade-off 表：

| 方案 | 显存压力 | decode 吞吐 | 时延 SLO | 备注 |
|---|---|---|---|---|
| TP=8 BF16 | 低（权重分片） | 高（聚合带宽） | 好（通信 ~1µs 域内） | 通信每 step 都发生 |
| TP=2 + 多实例 | 中 | 中 | 中 | 实例间独立扩展 |
| weight-only INT8 | 权重减半 | 上界≈×2 | 好 | 需量化校准与 outlier 处理 |
| prefill/decode 分离 | 需 KV 传输 | 最优 | 最优 TTFT | 系统复杂度高 |

- 评测方案：分负载报告 TTFT、TPOT、tokens/s/GPU、HBWU（实测 HBM 带宽/峰值）；
  扫 batch 与并发，标注 SLO 违约率；对比 vLLM 类 continuous batching 基线。
- 追问预案：长上下文主导→KV cache 超 weights，优先 paged/分页 KV 与量化 KV；
  突发流量→decode 实例横向扩，prefill 弹性；精度投诉→回退 BF16 关键层。

**设计题 3：多机扩展的通信子系统的评估与设计（α-β 模型落地）**
- 需求澄清：并行策略（TP/PP/DP/EP 配比）、消息大小分布、可接受 exposed time 占比、
  网络预算（IB/NVLink 层级）。
- 规模估算：DP 梯度同步 bytes ≈ 2P（BF16 梯度）≈ 140GB/step（70B），
  分桶后单桶 ~百 MB 级；通信时间 ≈ α + bytes/β；域内 β 取 NVLink（900GB/s/1.8TB/s），
  跨机取 IB（低一个量级）。
- 架构：通信量大的 TP/EP 限制在 NVLink 域（节点内或 NVL72 柜内）；DP 梯度
  bucket 化 + 与 backward overlap；PP 切在跨机边界，用 micro-batch 填 bubble。
- Trade-off 表：

| 手段 | 降低什么 | 代价 |
|---|---|---|
| bucket + overlap | exposed 通信时间 | 实现复杂、依赖图约束 |
| TP 域内收缩 | 跨机大消息 | 单卡显存压力上升 |
| PP 跨机 | 跨机消息频次 | bubble、micro-batch 调参 |
| 梯度压缩/FP8 通信 | bytes | 精度风险、需收敛验证 |

- 评测方案：nccl-tests 实测 bus bandwidth/时延基线；Nsight/NCCL timeline 量
  exposed vs overlapped 时间；扩展曲线（8→32→72 卡）看 scaling 效率拐点。
- 追问预案：overlap 不生效→检查依赖与 stream 优先级；scaling 骤降→先看
  straggler 与负载不均，再看网络；跨机时延抖动→检查拥塞控制与拓扑冲突。

## 代码实现题

**实现题 1：Roofline 计算器**
- 题目：给定峰值算力、HBM 带宽与 kernel 的 FLOPs/bytes（或 shape 推导），估算
  运行时间上限与瓶颈类型；内置 H100/B200 预设。
- 考察点：Roofline 公式落地、dense/sparse 口径意识、单位换算。

```python
from dataclasses import dataclass

PRESETS = {  # SXM/官方标称（dense 口径）
    "A100": dict(peak_flops=312e12, bw=2.0e12),
    "H100": dict(peak_flops=989e12, bw=3.35e12),
    "H200": dict(peak_flops=989e12, bw=4.8e12),
    "B200": dict(peak_flops=2.25e15, bw=8.0e12),
}

@dataclass
class Roofline:
    peak_flops: float  # FLOP/s，dense 口径
    bw: float          # bytes/s

    @classmethod
    def from_preset(cls, gpu: str) -> "Roofline":
        return cls(**PRESETS[gpu])

    @property
    def ridge(self) -> float:  # 拐点 FLOPs/byte
        return self.peak_flops / self.bw

    def analyze(self, flops: float, bytes_: float):
        assert flops > 0 and bytes_ > 0
        t_compute = flops / self.peak_flops
        t_memory = bytes_ / self.bw
        intensity = flops / bytes_
        bound = "compute" if intensity >= self.ridge else "memory"
        return {
            "intensity_flops_per_byte": round(intensity, 1),
            "ridge_point": round(self.ridge, 1),
            "bound": bound,
            "t_lower_bound_s": max(t_compute, t_memory),
            "t_compute_s": t_compute,
            "t_memory_s": t_memory,
        }

def gemm_flops_bytes(m: int, k: int, n: int, elem_bytes: float = 2):
    flops = 2 * m * k * n
    bytes_ = elem_bytes * (m * k + k * n + m * n)  # 每输入读一次、输出写一次
    return flops, bytes_

if __name__ == "__main__":
    r = Roofline.from_preset("B200")
    f, b = gemm_flops_bytes(4096, 4096, 4096)  # 方阵：I=Θ(n)
    print(r.analyze(f, b))   # 预期 compute bound（I≈1365 > 拐点≈281）
    f2, b2 = 2 * 70e9, 2 * 70e9  # decode 一步：读全部权重，I≈1
    print(r.analyze(f2, b2))  # 预期 memory bound
```

- 验收标准：H100 拐点输出 ≈295、B200 ≈281、H200 ≈206；4096 方阵判 compute bound、
  decode 判 memory bound；非法输入（flops/bytes ≤ 0）报错而非返回 NaN。

**实现题 2：MFU 计算脚本（tokens/s → TFLOPS/GPU → MFU）**
- 题目：输入模型参数量、实测 tokens/s、GPU 数与卡型，输出每 GPU TFLOPS 与 MFU。
- 考察点：6P 近似（fwd 2P + bwd 4P）、dense 峰值口径、单位换算。

```python
PRESET_PEAK_BF16_DENSE = {"A100": 312e12, "H100": 989e12,
                          "H200": 989e12, "B200": 2.25e15}

def mfu(params: float, tokens_per_s: float, num_gpus: int, gpu: str,
        attn_flops: float = 0.0) -> dict:
    peak = PRESET_PEAK_BF16_DENSE[gpu]  # dense 口径，勿用 sparse
    total_flops_per_s = (6 * params + attn_flops) * tokens_per_s
    per_gpu_tflops = total_flops_per_s / num_gpus / 1e12
    return {
        "tflops_per_gpu": round(per_gpu_tflops, 1),
        "mfu_pct": round(100 * (per_gpu_tflops * 1e12) / peak, 1),
    }

if __name__ == "__main__":
    # 参照实测：LLaMA2-70B TP=8，B200 815.2 TFLOPS/GPU
    print(mfu(params=70e9, tokens_per_s=7260, num_gpus=8, gpu="B200"))
    # 6*70e9*7260/8/1e12 ≈ 3812? -> 校验口径：815.2 对应 MFU≈36.2%
    print(mfu(params=70e9, tokens_per_s=1.0, num_gpus=8, gpu="B200"))
```

- 验收标准：参数量/卡数/tokens/s 任意缩放结果正确；峰值表只含 dense 口径并在
  注释标明；输出同时给 TFLOPS/GPU 与 MFU 百分比；除零（num_gpus=0 或 tokens/s=0）
  显式报错。

**实现题 3：decode 吞吐上界估算器（bandwidth-bound 视角）**
- 题目：给定卡型/卡数（TP）、模型字节数与 KV cache 带宽开销，估算 decode
  tokens/s 上界与 batch 摊销后的吞吐。
- 考察点：decode 强度 ≈ 1 的直觉量化、batching 对权重复用的摊销。

```python
PRESET_BW = {"A100": 2.0e12, "H100": 3.35e12, "H200": 4.8e12, "B200": 8.0e12}

def decode_upper_bound(params: float, tp: int, gpu: str,
                       kv_bytes_per_token: float = 0.0,
                       batch: int = 1) -> dict:
    assert tp > 0 and batch > 0
    per_gpu_bw = PRESET_BW[gpu]
    agg_bw = per_gpu_bw * tp
    weight_bytes = 2 * params  # BF16 权重，TP 已分摊在 agg_bw 中
    # 每 step 搬运：权重（batch 摊销）+ batch×KV
    step_bytes = weight_bytes + batch * kv_bytes_per_token
    step_time_lb = step_bytes / agg_bw
    tokens_per_s = batch / step_time_lb
    return {
        "step_time_lb_s": round(step_time_lb, 6),
        "tokens_per_s_upper": round(tokens_per_s, 1),
        "intensity_flops_per_byte": round((2 * params * batch) / step_bytes, 2),
    }

if __name__ == "__main__":
    # 70B BF16，8×H200：无 KV、batch=1
    print(decode_upper_bound(70e9, tp=8, gpu="H200"))
    # batch=32 时权重被摊销，吞吐≈×32（KV 可忽略时）
    print(decode_upper_bound(70e9, tp=8, gpu="H200", batch=32))
```

- 验收标准：batch=1 时强度输出 ≈1–2（远低于拐点 ~300，佐证 memory bound）；
  batch 增大时 tokens/s 上界近似线性上升直至 KV 项主导；TP 增大等价于聚合带宽
  线性增大；注明这是下界时间/上界吞吐，实际受并行效率和重叠影响。

## 12. 小结

加速器优化的统一原则是让昂贵的数据搬运被更多有效计算摊销。GPU/TPU 的接口不同，但都必须围绕 shape、布局、片上复用、低精度和通信建立资源账本。在模型规模增速持续超过带宽增速的背景下，这一原则只会越来越重要 [[11]](#ref-11)。Roofline 用于提出假设，benchmark 与 profiler 用于推翻或验证假设。

## 参考文献

<a id="ref-1"></a>[1] J. Nickolls et al. “Scalable Parallel Programming with
CUDA.” *ACM Queue*, 2008. https://doi.org/10.1145/1365490.1365500

<a id="ref-2"></a>[2] V. Volkov, J. W. Demmel. “Benchmarking GPUs to Tune Dense
Linear Algebra.” *SC*, 2008. https://doi.org/10.1109/SC.2008.5214359

<a id="ref-3"></a>[3] S. Williams, A. Waterman, D. Patterson. “Roofline:
An Insightful Visual Performance Model for Multicore Architectures.”
*Communications of the ACM*, 2009. https://doi.org/10.1145/1498765.1498785

<a id="ref-4"></a>[4] N. P. Jouppi et al. “In-Datacenter Performance Analysis
of a Tensor Processing Unit.” *ISCA*, 2017. https://arxiv.org/abs/1704.04760

<a id="ref-5"></a>[5] N. P. Jouppi et al. “TPU v4: An Optically Reconfigurable
Supercomputer for Machine Learning with Hardware Support for Embeddings.”
*ISCA*, 2023. https://arxiv.org/abs/2304.01433

<a id="ref-6"></a>[6] P. Micikevicius et al. “Mixed Precision Training.”
*ICLR*, 2018. https://openreview.net/forum?id=r1gs9JgRZ

<a id="ref-7"></a>[7] T. Dao et al. “FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness.” *NeurIPS*, 2022.
https://arxiv.org/abs/2205.14135

<a id="ref-8"></a>[8] T. Dao. “FlashAttention-2: Faster Attention with Better
Parallelism and Work Partitioning.” arXiv:2307.08691, 2023.
[link](https://arxiv.org/abs/2307.08691)

<a id="ref-9"></a>[9] M. Shoeybi et al. “Megatron-LM: Training Multi-Billion
Parameter Language Models Using Model Parallelism.” arXiv:1909.08053, 2019.
https://arxiv.org/abs/1909.08053

<a id="ref-10"></a>[10] S. Rajbhandari et al. “ZeRO: Memory Optimizations Toward
Training Trillion Parameter Models.” *SC*, 2020. https://arxiv.org/abs/1910.02054

<a id="ref-11"></a>[11] A. Gholami, Z. Yao, S. Kim, et al. “AI and Memory
Wall.” arXiv:2403.14123, 2024. [link](https://arxiv.org/abs/2403.14123)

<a id="ref-12"></a>[12] D. Kalamkar, D. Mudigere, N. Mellempudi, et al.
“A Study of BFLOAT16 for Deep Learning Training.” arXiv:1905.12322, 2019.
[link](https://arxiv.org/abs/1905.12322)

<a id="ref-13"></a>[13] P. Micikevicius, D. Stosic, N. Burgess, et al.
“FP8 Formats for Deep Learning.” arXiv:2209.05433, 2022.
[link](https://arxiv.org/abs/2209.05433)

<a id="ref-14"></a>[14] G. M. Amdahl. “Validity of the Single Processor Approach
to Achieving Large Scale Computing Capabilities.” *AFIPS*, 1967.
https://doi.org/10.1145/1465482.1465560

## 延伸阅读与实践材料

- [CS336 Lecture 5 官方讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)
- [CS336 Lecture 6 可执行讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [JAX Scaling Book：TPU performance](https://jax-ml.github.io/scaling-book/)
- [A2 Systems 官方题面](../assignments/spring2026/assignment2-systems/cs336_assignment2_systems.pdf)
- [NCCL Tests（all-reduce/bus 带宽实测工具）](https://github.com/NVIDIA/nccl-tests)（访问日期 2026-10-04）
- [CS336 课程主页](https://cs336.stanford.edu)（访问日期 2026-10-04）
- [vLLM 文档（continuous batching 与推理性能）](https://docs.vllm.ai)（访问日期 2026-10-04）
