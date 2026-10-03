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
  - "../assignments/assignment2-systems/"
---

# Lecture 05 — GPUs、TPUs 与性能模型：从算术强度到端到端吞吐

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述，非课程提交
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

- A2 统一入口：`assignments/assignment2-systems/scripts/benchmark_systems.py`
- 报告：`assignments/assignment2-systems/report/main.tex`
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
- 只优化很快的 microkernel，却不确认其 end-to-end 占比。

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
| occupancy 低 / SM 占用率低 | registers/shared/block | tile 太大、spill、block 过小 |
| occupancy 高仍慢 | stall reasons | dependency、HBM、bank conflict |
| 第一次 iteration 很慢 | warm-up/compile | JIT、autotune、allocator |
| BF16 无显存收益 | state breakdown | FP32 master/moments、activations |
| reserved 持续很高 | allocator snapshot | fragmentation/caching |
| 多卡 scaling 差 / 扩展近线性失效 | topology/NCCL timeline | exposed all-reduce、跨 PCIe 通信、未用 NVLink |
| microkernel 快、模型不变 | end-to-end share | Amdahl's law |
| 实测带宽远低于峰值 | 访存合并（coalescing）与对齐 | 非连续 stride、跨 stride 读 |
| kernel 时间不随规模线性 | L2 命中率 | cache 命中主导而非带宽 |
| TPU 与 GPU 数值不一致 | 累加顺序与精度容差 | systolic 阵列累加顺序差异属预期 |
| 吞吐周期性骤降 | SM clock 与功耗曲线 | 热节流降频，检查 clock 采样 |
| roofline 预测失准 | 峰值参数取值（boost vs sustained） | 用 sustained 而非理论峰值作分母 |

## 11. 讨论：效度威胁与结论边界

### 11.1 Construct validity
- GPU utilization、occupancy、MFU、TFLOP/s、tokens/s 是不同指标；
- theoretical FLOPs 不包含 data movement/communication；
- peak allocated 不等于 process VRAM；
- synthetic microbenchmark 不等于 time-to-quality。

### 11.2 Internal validity
- 异步计时、未 warm-up、compile/autotune 混入会系统性偏差；
- power/clock/thermal 与共享租户造成 run-to-run variance；
- 不同 shape/dtype/kernel version 是 confounders；
- profiler instrumentation 改变 runtime。

### 11.3 External validity
- 单 GPU 结果不外推多节点；
- RTX 6000D 不代表 B200/TPU；
- 一个 sequence/batch 的 kernel 最优参数不泛化；
- 峰值 spec 与真实应用可达值差异显著。

论文式表述应限定硬件、软件、shape 与 measurement protocol，优先报告 confidence interval
和完整失败点，而不是无条件“X× 更快”。

## 12. 面试备考（Interview Prep）

> GPU/性能模型是 ML Systems 面试的必考项：面试官常从「A100 带宽多少」切入，追到
> 「为什么 GEMM 适合 GPU」「occupancy 高为什么不一定快」「怎么用 Roofline 判断瓶颈」。
> 核心是建立 `FLOPs / bytes / parallelism / latency` 的统一账本，用硬件数量级支撑判断。
> 下面按「一页速览 → 高频题 → 手撕 → 追问」四层组织。

### 12.1 一页速览卡（面试前 1 分钟）

**核心主张**：加速器优化的统一原则是**让昂贵的数据搬运被更多有效计算摊销**——性能上界是
`min(峰值算力, 带宽 × 算术强度)`，从算法 shape 预测，再用 profiler 证据修正。

**必背数字**

- **A100 80GB**：HBM 约 2 TB/s、bf16 tensor core 约 312 TFLOPS（dense）、NVLink 600 GB/s。
- **H100 SXM**：HBM3 约 3.35 TB/s、bf16 约 990 TFLOPS、NVLink 900 GB/s。
- **roofline 拐点**（A100 bf16）：\(I^\*=312\text{e}12 / 2\text{e}12 \approx 156\) FLOPs/byte。
- 内存层次带宽：register > shared/L1 > L2 > HBM，相邻差约 1–2 个数量级。
- GEMM 的 arithmetic intensity \(O(n)\)；逐元素算子 \(O(1)\)。

**三句话答高频**

1. GEMM 越大越 compute-bound（复用强）；elementwise、decode 偏 memory-bound（复用低）。
2. occupancy 高只代表候选 warp 多，不等于算满；compute-bound 时加 occupancy 无收益。
3. tensor core 有固定 micro-tile shape，`M,N,K` 不对齐、尾块占比高就会 underutilize。

### 12.2 高频面试题与答题框架

**Q1：A100 / H100 的关键数字？roofline 拐点怎么算？**

- A100 80GB：HBM 约 2 TB/s、bf16 tensor core 约 312 TFLOPS、NVLink 600 GB/s；H100 SXM：HBM3 3.35 TB/s、bf16 约 990 TFLOPS、NVLink 900 GB/s。
- 拐点 \(I^\*=C_{\max}/BW\)：A100 bf16 ≈ `312e12 / 2e12 ≈ 156` FLOPs/byte。
- \(I<I^\*\) memory-bound，\(I>I^\*\) compute-bound；低于拐点的算子优化方向是减 bytes/fusion。

**Q2：为什么 GPU 适合 GEMM？tensor core 和 tiling 怎么配合？**

- GEMM 的 FLOPs \(O(n^3)\)、搬运 \(O(n^2)\)，arithmetic intensity \(O(n)\)，越大越 compute-bound。
- **tiling**：把 A/B 切成 tile 放入片上（register/shared memory），让多个输出复用，强度随 tile 边长增长。
- **tensor core**：在寄存器级完成固定 micro-tile MMA，配合正确的 dtype/layout/alignment 才能达到峰值；`M,N,K` 很小或尾块占比高会 underutilize。

**Q3：内存层次的数量级？为什么重要？**

- register（最快、线程私有）> shared memory/L1 > L2 > HBM（最慢最大），相邻层级带宽差约 1–2 个数量级、延迟差更多。
- kernel 优化的本质是**最大化片上复用、减少远端访存**；FlashAttention 的价值就是减少 HBM 往返。
- 层级化 roofline：一个 kernel 可能相对 HBM compute-bound，却受 shared-memory 带宽或 register 依赖限制。

**Q4：TPU 与 GPU 的本质差异？**

- **GPU**：SIMT + warp/block/SM，暴露细粒度细节，kernel 生态成熟、控制流/稀疏灵活。
- **TPU**：systolic array + XLA 编译器静态调度，对规则大矩阵与 SPMD sharding 友好；控制流、稀疏与不规则 gather 较弱。
- **结论**：不是谁更快，而是 workload 形态决定——规则大 batch GEMM 偏 TPU，灵活稀疏/细粒度优化偏 GPU。

**Q5：occupancy 是什么？为什么高 occupancy 不一定快？**

- occupancy = 活跃 warp / SM 最大 warp，受 registers、shared memory、block 数共同限制。
- 高 occupancy 只表示有更多候选 warp 可供 latency hiding，不保证 ILP、cache hit 或 tensor-core utilization。
- 若已 compute-bound，加 occupancy 无收益；register spill 时**降低** occupancy 反而更快。要看 stall reason 而非单一百分比。

**Q6：memory coalescing 是什么？为什么重要？**

- 同一 warp 连续访问相邻地址，可合并成少量 cache-line transaction；错误 stride、转置后非连续访问会放大 HBM 流量。
- 布局（AoS/SoA）、`contiguous()`（本身会触发拷贝）都要在 profiler 里确认，而非默认正确。

**Q7：为什么训练偏 compute-bound、decode 偏 memory-bound？**

- 训练/大 batch：大 GEMM 复用强、arithmetic intensity 高 → compute-bound。
- decode：每步 query 极少（matrix-vector/small GEMM），却要反复读取整段 weights + KV cache → bandwidth-bound。
- 所以不能用训练 MFU 预测 serving tokens/s；decode 优化方向是压 KV cache/权重读取，而非提算力。

**Q8：latency hiding 怎么工作？什么限制它？**

- 某 warp 等待 memory/dependency 时，scheduler 切到 ready warp，用并发隐藏延迟。
- 可驻留 warp 数受 threads/registers/shared memory/block limit 的最小项限制；software pipelining 把「加载下一 tile」与「计算当前 tile」重叠。
- stage 太少藏不住延迟，太多耗 shared/register 降 occupancy，需联合调参。

**Q9：怎么判断一个 kernel 慢在哪里？**

- 先算 shape/dtype 的理论 FLOPs、最低 bytes 与 arithmetic intensity，猜 bound。
- 再 profile：kernel 数、duration、launch gap、achieved bandwidth、tensor-core utilization、occupancy、stall reason。
- 用证据证伪假设：若猜 memory-bound，应看到高 achieved bandwidth + 低 compute utilization；否则假设有误。

**Q10：为什么 bf16 不一定让显存减半？**

- bf16 只降低 parameter/gradient/activation 的 dtype；Adam moments、master weight 常仍为 fp32。
- 训练态 16N 规则里，fp32 的 m/v + master 占 12N，bf16 只省了参数+梯度的 2N→4N 部分。
- 报告 memory reduction 要写全 state breakdown，不能只按「dtype 减半」估计。

### 12.3 手撕要点（Roofline 计算）

面试让「算某个 kernel 是 compute-bound 还是 memory-bound」时，按固定步骤：

```text
1. FLOPs F：matmul [M,K]@[K,N] -> F ≈ 2 M K N
2. bytes Q：读 A + 读 B + 写 C ≈ s(MK + KN + MN)   （理想一次读入）
3. arithmetic intensity I = F / Q
4. 拐点 I* = C_max / BW
5. I < I* -> memory-bound；I > I* -> compute-bound

例子（A100 bf16，s=2）：
  方阵 n=1024 GEMM：F=2·1024³≈2.1e9，Q≈2·3·1024²≈6.3e6
  I ≈ 340 FLOPs/byte > 156 -> compute-bound
  elementwise ReLU（n 元素）：F≈n, Q≈2n·s -> I ≈ 0.25 -> memory-bound
```

**三个必踩坑**

1. **峰值用 sustained 不用 boost**：boost clock 只在短时，长期受热/功耗限制。
2. **Q 只是理想下界**：tile 重叠、cache eviction、非合并访问、中间写回都会放大实际 bytes。
3. **拐点随 dtype 变**：bf16 峰值与 fp32 峰值不同，拐点也不同，别混用。

### 12.4 高频追问与陷阱

| 追问 | 正确方向 |
| --- | --- |
| occupancy 低就减寄存器吗？ | 不一定，spill 才减；compute-bound 时反而无需高 occupancy |
| tensor core 为什么需要对齐？ | 固定 micro-tile 尺寸，M/N/K 或 layout 不对齐会 underfill |
| GPU utilization 100% 说明算满吗？ | 否，memory/launch-bound 也能让设备一直忙 |
| BF16 显存减半了吗？ | 否，Adam fp32 状态与 master weight 仍占大头 |
| reserved 是真实占用吗？ | 否，含 allocator 缓存池；看 allocated 才是 live |
| 跨卡直接按峰值倍数估加速吗？ | 否，通信/拓扑/软件成熟度都会改变，需实测 |
| microkernel 快模型就快吗？ | 否，Amdahl：只快非瓶颈部分收益有限 |

## 13. 小结

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
- [A2 Systems 官方题面](../assignments/assignment2-systems/cs336_assignment2_systems.pdf)
