---
title: "Lecture 10 — Inference"
course: "Stanford CS336"
term: "Spring 2026"
date: "2026-04-29"
lecturer: "Percy Liang"
status: "已复习"
sources:
  - "https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py"
  - "https://jax-ml.github.io/scaling-book/inference/"
---

# Lecture 10 — Inference：从单 token 解码到动态服务

**课后知识总结（学术综述式）**

- 作者：ShaneLiu04
- 课程：Stanford CS336, Spring 2026
- 文档性质：原创中文自学综述；面向自学者、推理系统工程师与研究者

## 摘要

大语言模型推理不是单一矩阵乘 benchmark，而是由请求到达、prompt prefill、autoregressive
decode、KV-cache 生命周期、batch scheduler、sampling 和分布式执行组成的在线系统。
Prefill 通常以大 GEMM 为主，decode 则在每步读取大量 weights/KV、执行小 batch GEMM，
二者具有不同 arithmetic intensity 与 SLO。本文从 KV-cache 计算/容量推导出发，系统讨论
continuous batching、PagedAttention、prefix caching、chunked prefill、prefill-decode
disaggregation、speculative decoding、MQA/GQA、weight/activation/KV quantization 与
多 GPU serving。工程部分覆盖 TTFT/TPOT/goodput、admission control、fragmentation、tail
latency、profiling 与故障排查；研究部分强调真实 arrival/length distribution、SLO-aware
Pareto、sampling correctness 与效度威胁。目标是把“模型能生成”提升为可容量规划、可优化、
可复现且满足服务等级的 inference system。

**关键词：** LLM Inference；KV Cache；Continuous Batching；PagedAttention；
Speculative Decoding；Quantization；Goodput；TTFT；Disaggregated Serving

## 本文贡献

1. 统一 prefill/decode 的 FLOPs、bytes、latency 与 memory 模型；
2. 推导 KV-cache 容量、batch/concurrency 与 MHA/GQA/MQA 的关系；
3. 系统解释 iteration-level scheduling、paging、prefix reuse 与 disaggregation；
4. 分析 speculative decoding 的 acceptance、正确分布与 speedup 边界；
5. 提供 SLO-aware benchmark、工程诊断、效度威胁和论文级评估规范。

## 学习目标

- 区分 prefill 与 decode 的依赖结构、瓶颈和服务指标。
- 推导 KV cache 的 shape、显存与 attention 复杂度。
- 理解 batching、continuous batching 与 paged attention 的取舍。
- 判断 speculative decoding 与 quantization 何时真的加速，并验证质量不退化。

## 先修知识

- Lecture 02：arithmetic intensity 与 Roofline（prefill/decode 瓶颈判断）。
- Lecture 04 §6：MQA/GQA 与 KV cache 的结构来源。
- Lecture 08：TP/PP 分布式执行（多 GPU serving 的直接前置）。

## 相关工作与系统演进

早期神经网络 serving 系统已关注 batching、latency SLO 与资源共享；Orca 将 scheduling
粒度从 request 降到 iteration，使不同长度请求可以 continuous batch
[[1]](#ref-1)。vLLM 的 PagedAttention 用 page/block abstraction 管理 KV cache，
降低 fragmentation 并支持共享 [[2]](#ref-2)。Sarathi-Serve 通过 chunked prefill 减少
prefill 对 decode 的 head-of-line blocking [[3]](#ref-3)，DistServe 将 prefill/decode
解耦到不同资源池以优化 goodput [[4]](#ref-4)。

算法层面，speculative decoding 用便宜 draft model 提议多个 tokens，再由 target model
并行验证并保持原 sampling distribution [[5]](#ref-5)[[6]](#ref-6)。MQA/GQA 通过共享
K/V heads 减少 decode cache/bandwidth [[7]](#ref-7)[[8]](#ref-8)。量化研究则从
LLM.int8、GPTQ、SmoothQuant 到 AWQ，分别处理 outlier、weight-only post-training
quantization、activation migration 与 hardware-aware scaling
[[9]](#ref-9)[[10]](#ref-10)[[11]](#ref-11)[[12]](#ref-12)。

这些工作优化的目标不同：memory capacity、TTFT、TPOT、throughput、goodput、成本或质量。
脱离 workload/SLO 的“更快”结论没有部署意义。

## 1. 服务目标不是一个数字

- **TTFT (time to first token)**：请求到首 token；主要受排队和 prefill 影响。
- **TPOT / inter-token latency**：后续每 token 时间；主要受 decode 影响。
- **E2E latency**：完整响应时间。
- **throughput**：每秒完成的 tokens 或 requests。
- **goodput**：满足 SLO 的有效吞吐。

加大 batch 往往提高 throughput，却增加排队和单请求 latency。没有 workload 分布、prompt/output 长度和 SLO 的“tokens/s”不可比较。

### 1.1 Queueing 与 goodput

令平均到达率 \(\lambda\)、平均系统停留时间 \(W\)、系统内平均请求数 \(L_q\)，稳定状态下
Little’s Law：

\[
L_q=\lambda W.
\]

当 arrival rate 接近 service capacity，queueing latency 非线性上升，tail SLO 会先恶化。
因此 benchmark 应扫 offered load，画 goodput-latency curve，而不是只在饱和点报告最高吞吐。

Goodput 可定义为满足 TTFT/TPOT/E2E SLO 的 requests/s 或 tokens/s：

\[
\mathrm{goodput}(\mathrm{SLO})
=\frac{\#\{\text{completed requests satisfying SLO}\}}{\text{time}}.
\]

不同论文的 SLO、长度分布和 token accounting 不同，必须明确。

### 1.2 Admission control 与优先级

系统需要根据可用 KV blocks、预计 output length、deadline 与 tenant quota 决定 admission。
无限接收请求会把过载转为 queueing/OOM。可选策略包括 FCFS、shortest-job-first、deadline/
priority、prefill/decode 分队列和 memory-aware rejection。公平性、tail latency 与吞吐存在冲突。

## 2. Prefill 与 decode

给定 prompt 长度 \(S\)，生成 \(T\) 个 token：

### Prefill

一次处理全部 prompt，hidden state shape 为 \([B,S,d]\)。序列维可并行，矩阵乘规模大，通常更接近 compute-bound。输出每层 prompt 的 K/V，并产生首个 next-token logits。

### Decode

每步只输入新 token，shape 为 \([B,1,d]\)，然后追加 K/V。时间维存在自回归依赖，无法并行生成未来 token；小矩阵乘还需反复读取模型权重，通常 memory-bandwidth-bound。

若每步都重算全部历史，生成 \(T\) tokens 时 attention 总成本近似立方增长。KV cache 保存历史投影，将第 \(t\) 步从“重算历史 K/V”降为“新 Q 与缓存 K/V 做 attention”。

### 2.1 Per-step cost decomposition

Decode 每层包含：

- Q/K/V/O projection 与 FFN：约按 active parameters 读取 weights；
- attention：新 query 与 \(S+t\) cache 做 \(O((S+t)d)\)；
- norm/activation/sampling：逐元素或 vocab reduction。

对 batch \(B\)，若 weights 每 step 从 HBM 读一次，weight-only arithmetic intensity 近似随 \(B\)
增长：continuous batching 能摊销 weight bytes。KV bytes 则随 \(B(S+t)\) 增长，长 context
后 attention/KV bandwidth 可能取代 weight bandwidth 成为瓶颈。

### 2.2 TTFT 与 TPOT 的目标冲突

大 prefill batch/GEMM 提高吞吐，却可能阻塞正在 decode 的 latency-sensitive requests。
Chunked prefill 把长 prompt 切成 token chunks，与 decode iterations 交错；chunk 太小增加
launch/scheduling overhead，太大仍造成 head-of-line blocking。最优 chunk 与 GPU、batch、
prompt length distribution 和 SLO 有关。

### 2.3 Sampling 也在关键路径

Temperature、top-k/top-p、repetition penalty、structured decoding 都可能触发 vocab-size
reduction/sort/mask。大 \(V\) 下 logits materialization 和 sampling kernel 可见；tensor-parallel
vocab 还需 distributed top-k/log-sum-exp。只 benchmark Transformer body 会低估端到端 TPOT。

## 3. 关键推导：KV cache 用显存换计算

令：

- \(L\)：层数；
- \(K\)：KV heads 数；
- \(H\)：head dimension；
- \(S\)：当前上下文长度；
- \(b_e\)：每元素 bytes。

每个请求每层保存

\[
K_{\text{cache}},V_{\text{cache}}\in\mathbb{R}^{S\times K\times H},
\]

故 batch 为 \(B\) 时

\[
\boxed{M_{\text{KV}}=2BLSKHb_e}.
\]

例：\(B=64,L=40,S=4096,K=8,H=128\)，bf16 \(b_e=2\)，约需

\[
2\cdot64\cdot40\cdot4096\cdot8\cdot128\cdot2
=40\text{ GiB}.
\]

这还不含权重、workspace 和 allocator 碎片。MHA 中 \(K=N_{\text{query heads}}\)；GQA 令 \(K<N\)，MQA 令 \(K=1\)，按 \(N/K\) 比例缩小 KV cache，但质量必须实测。

### 单步 attention

decode 时

- \(Q:[B,1,N,H]\)；
- cached \(K,V:[B,S,K,H]\)；
- GQA 中每个 KV head 服务 \(N/K\) 个 query heads。

计算量约 \(O(BS d)\)，而读取 KV 同样为 \(O(BSKH)\)。历史 cache 每个请求不同，batching 不能像共享 MLP 权重那样消除这部分带宽。

### 3.1 Capacity planning

给定可用于 KV 的显存 \(M_{\text{budget}}\)，平均上下文 \(\bar S\)，粗略最大并发：

\[
B_{\max}\lesssim
\frac{M_{\text{budget}}}{2L\bar S K H b_e}.
\]

实际需减去 weights、CUDA graph/workspace、fragmentation 与 safety margin。请求长度是随机变量，
p95/p99 context 比均值更决定 OOM risk。Admission control 可按 reserved max length 保守规划，
或按实际增长动态分配，但后者需可靠 eviction/rejection。

### 3.2 Prefix caching

System prompt、few-shot prefix 或共享文档可复用 prefill KV。Radix tree/trie 按 token prefix
索引 cache blocks；命中可减少 TTFT 与 compute。收益取决于 prefix popularity/length、
cache capacity 和 eviction policy。必须按 token IDs 与 model/config/version 做 cache key，
避免 tokenizer/template 变化后错误复用。

### 3.3 Eviction、sliding window 与 KV compression

长对话可：

- eviction 旧 KV；
- sliding-window attention；
- 保留 heavy hitters / sink tokens；
- KV quantization；
- semantic compression/summarization。

这些策略改变数学语义或引入误差，应测 perplexity、retrieval、long-generation stability。
StreamingLLM 观察 attention sinks 对稳定 streaming 的作用，H2O 按 heavy hitters 管理 cache
[[13]](#ref-13)[[14]](#ref-14)；它们不是对任意任务无损。

## 4. Arithmetic intensity 为什么决定瓶颈

矩阵乘 \(X_{B\times D}W_{D\times F}\) 约需 \(2BDF\) FLOPs。若 \(B\ll D,F\)，主要流量是读 \(W\)，arithmetic intensity 约为 \(B\) FLOPs/byte（bf16 常数略去）。

官方讲义给出的关键近似：

| 阶段 | MLP intensity | attention intensity |
|---|---:|---:|
| prefill | \(BS\) | \(S/2\) |
| decode | \(B\) | \(<1\) |

所以 prefill 容易通过长序列/批处理喂满算力；decode 尤其是 attention 天生偏 bandwidth-bound。量化、GQA 和 KV 压缩之所以有效，是因为它们减少 bytes，而不只是减少 FLOPs。

## 5. Batching 与动态流量

### Static batching

等待一批请求一起执行。共享权重读取，提高 decode MLP 的利用率；但短请求被长请求拖住，padding 浪费严重，等待成批会伤害 TTFT。

### Continuous batching

以 decode iteration 为调度单位：

1. 活跃请求各执行一步；
2. 完成的立即移出；
3. 新请求及时插入空位。

attention 可按 ragged sequences 分开，非 attention token 可拼成 \([\sum_i S_i,d]\) 的 packed tensor。这样减少 padding 和 head-of-line blocking，但调度器需处理 prefill/decode 的资源竞争。

### Paged attention

不为每个请求预留连续的最大长度 cache，而把 KV cache 切为固定 block：

- 逻辑 token block 映射到非连续物理 block；
- 减少内部/外部碎片；
- prefix 可共享；
- 分叉生成时用 copy-on-write。

代价是 block table 查找、kernel 复杂度和 block size 取舍。它解决的是**内存管理**，不改变 attention 的数学结果。

### Chunked prefill

将长 prompt 切成固定 token chunks，与 decode tokens 合并调度，可限制每 iteration prefill
工作量并保护 TPOT。Chunk size 小则 TTFT/launch 开销增加，大则 decode stalls 增加。
Sarathi-Serve 将 chunked prefill 与 stall-free scheduling 结合
[[3]](#ref-3)。

### Prefill-decode disaggregation

Prefill 需要高 compute throughput，decode 需要高 memory bandwidth/capacity；把二者放到不同
GPU pools 可独立扩缩容，并用 KV transfer 连接。DistServe 显示 disaggregation 可改善
SLO goodput [[4]](#ref-4)，但引入：

- KV network transfer latency/bytes；
- placement/routing；
- pool imbalance；
- failure/retry 与 cache ownership。

只有当 resource specialization 收益超过 KV transfer/queueing 时才值得。

### CUDA Graph 与 shape regularization

Decode 每 step launch pattern相似，CUDA Graph 可降低 CPU launch overhead；但 continuous
batching/ragged lengths 产生动态 shape。Serving engine 常用固定 batch/token buckets、
padding 或 graph pool 折中。过多 buckets 增加 capture/memory，过少造成 padding waste。

## 6. Speculative decoding：并行验证草稿

小 draft model \(p\) 连续提议 \(k\) 个 token；target model \(q\) 用一次类似 prefill 的并行前向验证。候选 \(x\) 以

\[
a(x)=\min\left(1,\frac{q(x)}{p(x)}\right)
\]

接受；若拒绝，从归一化残差

\[
r(x)\propto\max(q(x)-p(x),0)
\]

采样。这个修正使最终样本**严格服从 target distribution**，不是近似改 logits。

近似每次 target 调用接收 \(\mathbb{E}[A]\) 个 token 时，收益取决于：

- draft/target 成本比足够小；
- acceptance rate 足够高；
- 验证 \(k\) 个 token 的并行效率高；
- 额外 cache、调度与通信开销可控。

草稿更大可能提高接受率，却抵消加速。温度、领域、prompt 阶段和 batch size 都会改变最优 \(k\)。必须比较相同采样分布下的端到端 latency，而非只报 acceptance rate。

### 6.1 简化 speedup model

若每个 draft token 成本 \(c_d\)，一次 target verification 成本 \(c_t(k)\)，平均接受
\(\bar a\) 个 draft tokens（并可能额外生成 1 个 target token），粗略吞吐：

\[
\text{cost/token}\approx
\frac{k c_d+c_t(k)}{\mathbb E[\text{accepted output tokens}]}.
\]

相对标准 target step \(c_t(1)\) 的加速取决于 target 对 \(k\) tokens 的并行效率。接受率高但
draft 串行成本大，仍可能变慢；batch 大时 target 已高效，speculation 额外收益也会下降。

### 6.2 Tree/heads/self-speculation

后续方法用多分支 draft tree、模型额外 prediction heads（如 Medusa）、早退层或 target
self-draft，减少独立 draft model 成本。比较时应区分：

- 是否严格保持 target distribution；
- 额外训练/参数；
- verification tree kernel；
- KV cache duplication；
- 单请求与高并发收益。

### 6.3 Correctness test

不能只比较 greedy outputs。对 sampling，需在固定 prompts/temperature 下做 distributional
test：token frequency、sequence likelihood 或大量 samples 的统计一致性；接受/拒绝边界和
residual distribution 实现错误可能不影响少量样例，却产生系统 bias。

## 7. Quantization：降低带宽与容量压力

仿射量化可写为

\[
q=\operatorname{clip}\!\left(\operatorname{round}(x/s)+z\right),\qquad
\hat x=s(q-z),
\]

其中 \(s\) 为 scale，\(z\) 为 zero point。

- **weight-only**：减小权重读取；适合 decode memory-bound 场景。
- **weight+activation**：可能使用低精度 tensor cores，但 activation outlier 更难处理。
- **KV-cache quantization**：直接提高可容纳 batch/context，误差随长上下文和层累积。
- **PTQ**：用 calibration data 估计 scale，成本低。
- **QAT**：训练时模拟量化误差，质量通常更稳但训练昂贵。

粒度从 per-tensor 到 per-channel/per-group 越细，量化误差通常越小，但 metadata、反量化和 kernel 复杂度越高。压缩比不等于速度比：若硬件缺少对应低比特 kernel，反量化开销可能让 int4 比 bf16 更慢。

质量检查至少覆盖 perplexity、下游任务、长上下文、罕见 token 和生成稳定性；只测平均 loss 会漏掉 outlier channel 引发的灾难性退化。

### 7.1 Outlier 与方法差异

- **LLM.int8()**：把 outlier features 保留高精度，其余 int8 [[9]](#ref-9)；
- **GPTQ**：用近似二阶信息逐层做 weight-only PTQ [[10]](#ref-10)；
- **SmoothQuant**：把 activation outlier difficulty 平滑迁移到 weights，实现 W8A8
  [[11]](#ref-11)；
- **AWQ**：按 activation-aware importance 保护关键 weight channels [[12]](#ref-12)。

方法名称不能替代配置：bits、group size、symmetric/asymmetric、zero point、calibration data、
kernel/backend 都影响质量和速度。

### 7.2 Weight、activation 与 KV 分开核算

Weight-only quantization 减 model residency/weight bandwidth，但 activation/attention/KV
仍可能 BF16。W8A8 才能利用某些 integer matrix units；KV int8/int4 改善长上下文 capacity，
却在每 step 引入 dequant 和累积误差。需分别报告各 tensor dtype 与实际 bytes，而不是一句
“模型是 4-bit”。

### 7.3 Quantization benchmark

至少包含：

- model load size / peak VRAM；
- prefill/decode latency 和吞吐；
- kernel coverage（多少算子真正低比特）；
- calibration corpus 与规模；
- perplexity/downstream/long-context/rare-token；
- outlier fallback 与 dequant overhead。

若低比特 kernel 不成熟，压缩 4× 不保证速度 4×。

## 8. Shape / 复杂度总表

| 操作 | 输入/状态 shape | 时间 | 主要内存 |
|---|---|---:|---:|
| prefill attention | \(Q,K,V:[B,S,h,H]\) | \(O(BS^2d)\) | FlashAttention 可避免显式 \(S^2\) score |
| decode attention/step | \(Q:[B,1,h,H]\), KV:\([B,S,K,H]\) | \(O(BSd)\) | \(O(BLSKH)\) KV |
| MLP/step | \([B,1,d]\) × weights | \(O(BdF)\) | 权重 \(O(LdF)\) |
| naive full recompute | 长度随 step 增加 | 生成总 attention 近似 \(O(T^3)\) | 无 cache 但重复算 |
| cached generation | 逐步追加 KV | 生成 attention 约 \(O(T^2)\) | cache 随 \(T\) 线性 |
| speculative verify | \([B,k,d]\) | 一次验证 \(k\) 候选 | draft + target 状态 |

## 9. 分布式与生产 Serving

### 9.1 Replica、TP 与 PP

- **Replica/data parallel serving**：完整模型多副本，便于 request routing，容量按副本复制；
- **Tensor parallel**：每层跨 GPU collective，单 request 可放更大模型，但 TPOT 受通信关键路径；
- **Pipeline parallel**：切 layers，增加 stage latency/bubble，适合模型放不下单节点；
- **Expert parallel**：MoE all-to-all，routing imbalance 影响 tail。

Serving 目标常是最小满足容量的 model parallel degree，再用 replicas 扩吞吐。过高 TP degree
会缩小每 rank GEMM 并增加 collective latency。

### 9.2 Routing 与 autoscaling

Router 需考虑 model/version、prefix locality、KV residency、queue length、SLO 与 tenant。
只按最短 queue 路由可能破坏 prefix-cache 命中；只按 cache affinity 又可能造成热点。
Autoscaling 的冷启动包括加载数十/数百 GB weights、compile/capture 和 warm cache，不能用
CPU web service 的秒级假设。

### 9.3 Reliability 与 cancellation

请求取消应及时释放 KV blocks；worker failure 需清理分布式 request state。Timeout、OOM、
kernel error 和 network partition 不能让其他 ranks 永久 hang。生产系统需：

- request ID / token position 幂等；
- backpressure 与 overload rejection；
- health/readiness（权重加载≠可服务）；
- metrics/logging 不泄露 prompt/response；
- model/tokenizer/template version pinning。

### 9.4 多模型与 adapter serving

LoRA/adapters 可共享 base weights，但 active adapter selection、cache key、batch grouping 与
memory residency 增加调度维度。把不同 adapter requests 混 batch 可能需要 grouped GEMM，
否则频繁切换权重降低吞吐。

## 10. 实现映射

| 概念 | 实现位置/系统 | 验证方法 |
|---|---|---|
| causal attention shape | `assignment1-basics/cs336_basics/model.py` | 比较 full forward 与逐 token logits |
| FlashAttention | `assignment2-systems/cs336_systems/flash_attention.py` | correctness + peak memory + 长序列速度 |
| GQA 配置约束 | A3 `training/model/config.py` | query heads 可被 KV heads 整除 |
| continuous batching | vLLM/SGLang scheduler | 到达时间和长度分布下测 p50/p99 |
| paged attention | vLLM block manager | 碎片率、可服务并发数、prefix sharing |
| quantization | backend-specific kernels | 质量、显存、TTFT、TPOT 同时比较 |

本仓库训练模型没有现成 serving KV-cache API；不要把训练 forward 的 causal mask 误当成缓存实现。正确性测试应将一次性 full-sequence logits 与 token-by-token cached logits 对齐。

## 11. SLO-aware 实验方法

### 11.1 Workload 与 load generator

至少保存 prompt length、requested/actual output length、arrival timestamp、tenant、model 与
sampling config 的联合分布。**Open-loop** 按外生到达过程发请求，能暴露过载排队；
**closed-loop** 等响应后再发，会因系统变慢而自动降载，只适合测单 client 体验。容量实验应
扫 open-loop offered load，直到 goodput 下降。

### 11.2 指标与统计

同时报告 p50/p95/p99 TTFT、TPOT、E2E、request/token throughput、SLO goodput、峰值显存、
功耗与每成功请求成本。重复多个 seeds/时间窗口，给出 bootstrap confidence interval。
失败、timeout、rejection 与 OOM 计入分母，不能从 latency 样本中静默删除。

### 11.3 控制变量与等价性

固定 model revision、tokenizer/chat template、sampling、driver/CUDA/backend、power limit；
区分 cold load、compile/capture warm-up、cold/warm prefix cache。优化前后先验证：

1. greedy token/logit equivalence（容许精度误差）；
2. stochastic decoding 的分布检验，而非同 seed 字符串相同；
3. quantization 的 perplexity、任务质量与长尾/长上下文；
4. request cancellation、OOM recovery 和并发隔离。

### 11.4 可证伪假设

实验前写明预测，例如：“chunked prefill 在 p99 TPOT SLO 下提高 goodput，但增加低负载
TTFT”。然后预先指定 baselines、sweep、stopping rule 与主要指标。这样的设计比跑完再挑
最好看的 tokens/s 更接近可复现 systems research。

## 12. 常见误区

- **KV cache 降低单步 attention 的渐近读取量：** 它避免重算 K/V，但每步仍需读取历史 cache。
- **batch 越大越好：** throughput 上升，TTFT、TPOT 和显存可能恶化。
- **prefill 与 decode 用同一调度策略：** 两者一个偏 compute-bound、一个偏 memory-bound。
- **speculative decoding 是近似采样：** 正确 rejection correction 下是 target 的精确样本。
- **量化位数减半就必然 2×：** 受 kernel、packing、反量化和非量化算子限制。
- **只测平均 prompt：** 长尾长度和突发流量通常决定 p99 与 OOM。
- **只报 tokens/s：** 必须说明 input/output tokens、并发、batch、SLO 和硬件。

## 13. Checklist

- [ ] 分开测 TTFT、TPOT、E2E、throughput、goodput。
- [ ] 记录 prompt/output 长度与请求到达分布。
- [ ] 用公式和 profiler 同时核对权重/KV/workspace 显存。
- [ ] prefill 与 decode 分开 benchmark。
- [ ] batching 实验同时报告 p50/p95/p99。
- [ ] speculative decoding 验证输出分布与 acceptance。
- [ ] quantization 对质量、显存和真实延迟做联合回归。
- [ ] OOM、碎片、prefix sharing 和取消请求进入压力测试。

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| TTFT 高、TPOT 正常 | queue/prefill | 长 prompt、prefill blocking |
| TPOT 随 context 急升 | KV traffic | attention bandwidth |
| throughput 高但 p99 差 | scheduler trace | 大 batch/head-of-line |
| KV OOM 早于公式 | allocator/pages | fragmentation/reservation |
| prefix cache 命中低 | token/template key | tokenizer/version 不一致 |
| speculative 不加速 | acceptance/cost | draft 太慢、batch 已饱和 |
| quantized 更慢 | kernel coverage | dequant/unsupported ops |
| output 与 base 不同 | sampling correction | speculative/residual bug |
| CUDA graph 命中低 | shape buckets | 动态 batch/length |
| worker hang | distributed state | rank failure/collective mismatch |

### 故障排查速查

| 现象 | 优先检查 | 常见根因 |
|---|---|---|
| TTFT 尾部尖刺 | prefill 调度与 chunked prefill | 长 prompt 独占计算、阻塞 decode |
| decode 吞吐低 | batch 组成与 KV 读取带宽 | batch 过小、权重/KV 每步全量读取 |
| KV cache OOM | 容量公式×batch×上下文 | 未分页、未用 GQA/KV 量化 |
| goodput 远低于吞吐 | SLO 违约率与准入 | admission control 缺失、排队策略不当 |
| 量化后质量下降 | scale 校准与 outlier 通道 | 未做 per-channel scale、激活异常值未处理 |
| speculative 加速为负 | 接受率与 draft 成本 | draft 接受率低或 draft 推理本身过慢 |
| 输出退化/重复 | 采样实现审计 | 温度/top-p 边界条件 bug |

## 14. 讨论：效度威胁与结论边界

### Construct validity
- tokens/s 不区分 input/output，也不体现 SLO；
- average latency 隐藏 p99/tail；
- model-only benchmark 排除 queue/sampling/network；
- memory capacity 不等于可稳定服务并发。

### Internal validity
- synthetic fixed lengths、closed-loop clients 会夸大 batching；
- prefix cache warm/cold、compile/capture 状态影响结果；
- quantization 方法若 kernel coverage 不同，不能只按 bits 比；
- speculative sampling 参数/seed 不一致会混入质量差异。

### External validity
- 单一 GPU/traffic trace 不外推多租户生产；
- offline throughput 不外推 bursty arrival goodput；
- 小模型 acceptance/quantization 规律不外推 70B；
- benchmark prompt 可能不代表真实语言/领域/安全负载。

学术报告应公开 arrival/length distribution、SLO、client model、concurrency、batch scheduler、
cache state、sampling config、hardware/software 和质量评测；否则“serves X tok/s”不可比较。

## 面试要点速记

**高频问题与答题要点**

1. **Q：prefill 和 decode 谁是 memory-bound？** 要点：decode——每步读全部
   权重与 KV 却只算一个 token（GEMV、低 arithmetic intensity）；prefill 是
   大 GEMM，通常 compute-bound。两者 SLO 不同（TTFT vs TPOT）。
2. **Q：KV cache 怎么估？** 要点：2·l·h_kv·d_h·seq·B·bytes；70B 级 GQA
   模型 + 128k 上下文可达每请求 GB 量级 → 必须分页/量化/分层。
3. **Q：continuous batching 的收益来源？** 要点：迭代级调度，异质长度请求
   不再互相等整批；GPU 利用率随到达率平滑变化。
4. **Q：speculative decoding 为什么无损？** 要点：对 target 分布做 rejection
   sampling（接受/重采样），输出分布不变；净加速 = 接受率×草稿并行度 − 验证开销。
5. **Q：goodput 与 throughput 的区别？** 要点：吞吐不区分 SLO 违约；
   goodput 只计满足 SLO 的有效吞吐，是容量规划的正确目标。

**必背数字**

- KV cache 公式；TTFT/TPOT/goodput 三指标；chunked prefill 与
  prefill-decode 分离是尾延迟治理的两个主杠杆。

## 15. 本讲小结

推理包含可并行、偏 compute-bound 的 prefill，以及串行、偏 memory-bound 的 decode。KV cache 消除历史 K/V 重算，却把显存容量和带宽推到核心位置；GQA、量化、连续批处理和分页管理都是围绕这一瓶颈展开。Speculative decoding 则利用“并行验证比逐 token 生成高效”的不对称，在保持 target 分布不变的前提下换取速度。

## 参考文献

<a id="ref-1"></a>[1] G.-I. Yu et al. “Orca: A Distributed Serving System for
Transformer-Based Generative Models.” *OSDI*, 2022.

<a id="ref-2"></a>[2] W. Kwon et al. “Efficient Memory Management for Large Language
Model Serving with PagedAttention.” *SOSP*, 2023.
[arXiv](https://arxiv.org/abs/2309.06180)

<a id="ref-3"></a>[3] A. Agrawal et al. “Taming Throughput-Latency Tradeoff in LLM
Inference with Sarathi-Serve.” *OSDI*, 2024.

<a id="ref-4"></a>[4] Y. Zhong et al. “DistServe: Disaggregating Prefill and Decoding
for Goodput-optimized Large Language Model Serving.” *OSDI*, 2024.

<a id="ref-5"></a>[5] Y. Leviathan, M. Kalman, and Y. Matias. “Fast Inference from
Transformers via Speculative Decoding.” *ICML*, 2023.
[arXiv](https://arxiv.org/abs/2211.17192)

<a id="ref-6"></a>[6] C. Chen et al. “Accelerating Large Language Model Decoding with
Speculative Sampling.” arXiv:2302.01318, 2023.

<a id="ref-7"></a>[7] N. Shazeer. “Fast Transformer Decoding: One Write-Head is All
You Need.” arXiv:1911.02150, 2019.

<a id="ref-8"></a>[8] J. Ainslie et al. “GQA: Training Generalized Multi-Query
Transformer Models from Multi-Head Checkpoints.” *EMNLP*, 2023.

<a id="ref-9"></a>[9] T. Dettmers et al. “LLM.int8(): 8-bit Matrix Multiplication for
Transformers at Scale.” *NeurIPS*, 2022.

<a id="ref-10"></a>[10] E. Frantar et al. “GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers.” *ICLR*, 2023.

<a id="ref-11"></a>[11] G. Xiao et al. “SmoothQuant: Accurate and Efficient
Post-Training Quantization for Large Language Models.” *ICML*, 2023.

<a id="ref-12"></a>[12] J. Lin et al. “AWQ: Activation-aware Weight Quantization for
LLM Compression and Acceleration.” *MLSys*, 2024.

<a id="ref-13"></a>[13] G. Xiao et al. “Efficient Streaming Language Models with
Attention Sinks.” *ICLR*, 2024.

<a id="ref-14"></a>[14] Z. Zhang et al. “H2O: Heavy-Hitter Oracle for Efficient
Generative Inference of Large Language Models.” *NeurIPS*, 2023.

## 延伸阅读

- Stanford CS336, [Lecture 10 — Inference](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py).
- JAX Scaling Book, [Inference](https://jax-ml.github.io/scaling-book/inference/).
- Y. Sheng et al. “FlexGen: High-Throughput Generative Inference of Large Language
  Models with a Single GPU.” *ICML*, 2023.
- L. Zheng et al. “SGLang: Efficient Execution of Structured Language Model
  Programs.” arXiv:2312.07104, 2024.
