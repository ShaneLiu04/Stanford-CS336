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
- **PagedAttention 是新的 attention 算法：** 它只是 KV cache 的分页内存
  管理，数学结果与连续 cache 一致；把它与 continuous batching（调度
  策略）混为一谈也是常见错误。
- **MLA/GQA/量化是免费提速：** MLA 训练与 kernel 复杂度高；GQA 质量
  须实测；量化收益受 kernel coverage 限制——压缩比不等于加速比。

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
6. **Q：PagedAttention 到底解决什么问题？** 要点：解决 KV cache 内存
   管理——固定大小页按需分配/释放、消除内外碎片、支持动态扩容与
   prefix 跨请求共享（copy-on-write）；不改变 attention 数学结果。
7. **Q：PD 分离何时值得上？** 要点：prefill compute-bound、decode
   bandwidth-bound，混部互相干扰尾延迟；分离后独立扩缩容、稳定时延；
   代价是 KV 跨实例传输与路由复杂度，低 QPS 小集群不划算。
8. **Q：MLA 与 GQA/MQA 的本质区别？** 要点：MQA/GQA 靠共享 KV head
   硬省；MLA 把 KV 低秩压缩到潜空间（KV cache −93.3%，DeepSeek-V2
   官方口径），推理时上投影恢复表达，容量与质量兼得，代价是 kernel
   与训练复杂度。
9. **Q：投机解码什么情况下反而变慢？** 要点：draft 成本占比高、接受率
   低、大 batch 下 target 本身已高效、树形验证与 KV 管理开销失控；
   必须端到端测 latency，而非只看 acceptance rate。

**必背数字**

- KV cache 公式；TTFT/TPOT/goodput 三指标；chunked prefill 与
  prefill-decode 分离是尾延迟治理的两个主杠杆。

**工业界参照（2024–2026，口径见括注）**

- 推理显存构成：模型权重 50%–70%、KV cache 20%–40%、临时计算张量
  5%–10%（vLLM 优化实践口径）。
- continuous batching 较静态 batching 吞吐约 2.5×；TensorRT-LLM 在
  Hopper 上 FP16 吞吐约达理论峰值 75%（行业口径）。
- MLA：KV cache −93.3%、最大吞吐 +576%（DeepSeek-V2 官方口径）。
- INT4/INT8 权重量化减少 50%–75% 显存；FP8 需 Hopper 及以上。
- 投机解码端侧实践吞吐约 2×；稀疏/检索式 attention 压缩约 70% KV
  而长序列效果不显著受损（实践口径）。
- 前沿模型单用户 TPS 已普遍 >200 tokens/s（2025–2026 行业口径）。

## 行业现状与最新进展（2024–2026）

### 推理显存构成与 serving 栈：vLLM 与 TensorRT-LLM

推理显存构成（vLLM 优化实践口径）：

| 构成 | 占比 | 主要优化手段 |
|---|---:|---|
| 模型权重 | 50%–70% | 权重量化（INT4/INT8 省 50%–75%）、TP 切分 |
| KV cache | 20%–40% | PagedAttention、prefix cache、GQA/MLA、KV 量化 |
| 临时计算张量 | 5%–10% | CUDA Graph 复用、workspace 控制 |

- **vLLM**：PagedAttention 把 KV cache 划分为固定大小「页」，按需
  分配/释放，消除显存碎片、支持动态扩容；continuous batching 以步为
  单位调度请求进出，较静态 batching 吞吐约 2.5×（行业口径）；prefix
  cache 跨请求复用公共前缀 KV；支持 PD 分离与 FlashAttention-3。
  关键参数：gpu_memory_utilization（显存上限）、max_num_batched_tokens、
  max_seq_len、tensor_parallel_size（head 数须能被 TP 卡数整除）、
  swap-space。
- **TensorRT-LLM**：Hopper 深度优化，FP16 吞吐约达理论峰值 75%；
  集成 Medusa 并行解码与前缀缓存。
- **PD 分离**：Prefill（计算密集）与 Decode（访存密集）部署到独立
  实例，避免资源竞争、稳定时延；另有 SplitFuse 类动态切分的中间
  方案（MindIE：splitChunkTokens 建议 512 倍数、cacheBlockSize
  默认 128）。

| 维度 | vLLM | TensorRT-LLM |
|---|---|---|
| 定位 | 开源通用引擎，社区迭代快 | NVIDIA 官方栈，硬件深度协同 |
| KV 管理 | PagedAttention 分页 + prefix cache | 分页/前缀缓存 + Hopper 专属 kernel |
| 批调度 | continuous batching、PD 分离 | in-flight batching |
| 生态 | 多后端、新模型覆盖快 | FP16/FP8 官方路径，峰值性能导向 |

### KV 压缩谱系：MHA → MQA/GQA → MLA，再到稀疏化

| 方案 | 机制 | KV cache 相对 MHA | 代价/收益 |
|---|---|---:|---|
| MHA | 每 query head 独立 KV | 1× | 基线；显存与带宽压力最大 |
| MQA | 全部 query 共享 1 组 KV | 降至 1/N | 牺牲表达，质量下降 |
| GQA | 分组共享 KV | K/N | 平衡；主流训练标配 |
| MLA | KV 低秩压缩到潜空间，推理上投影恢复 | −93.3%（官方口径） | 最大吞吐 +576%；kernel/训练复杂 |

稀疏化实践：Minference 动态稀疏模式（A-shape、垂直划块、分块稀疏）
与检索式 head 压缩，据报道可压缩约 70% KV 而长序列效果不显著受损
（实践口径）；attention 稀疏性随任务变化，须按 workload 实测。

### 投机解码工业方案与量化

- **drafter 三路线**：蒸馏小模型（如 7B 蒸馏）、自起草（Medusa 附加
  头）、检索式；EAGLE-2 进一步用动态草稿树，按验证置信度扩展分支。
- **验证机制**：树形 attention（mask 隔离无效分支）+ 拒绝采样或
  typical acceptance；端侧实践吞吐约 2×——每周期主模型一次前向验证
  多个 token。KV 管理二选一：缓存迁移（额外拷贝）或重计算。
- **量化**：INT4/INT8 权重量化减少 50%–75% 显存；W8A8/W8A16/W4A16
  配合 SmoothQuant 缓解激活离散化；FP8 需 Hopper 及以上。

**对本讲学习者的启示：** 工业界 2024–2026 的主线仍是本讲的两条公式：
KV 容量 \(M_{\text{KV}}=2BLSKHb_e\)（催生分页、GQA/MLA、量化与稀疏化）
和 arithmetic intensity（催生 continuous batching、chunked prefill 与
PD 分离）。面试时从公式推导工程决策——为何分页、为何分离、为何量化
「省显存不一定等比提速」——比罗列系统名更显深度；引用任何行业数字都
应标注硬件/模型/SLO 前提，与第 11 节的实验方法论保持一致。

## 大厂面试真题与答题框架

以下均为高频面试题（公开面经风格），不指向任何特定公司真题。

**题目 1：为什么 prefill 通常 compute-bound，而 decode 通常 bandwidth-bound？**

- 考点：arithmetic intensity / Roofline；prefill 与 decode 的 shape 差异。
- 答题框架：
  1. prefill 一次处理全部 prompt，shape \([B,S,d]\)，大 GEMM，MLP intensity 约 \(BS\)；
  2. decode 每步 shape \([B,1,d]\)，GEMV 为主，却要读全部权重与 KV，intensity 约 \(B\)；
  3. 用 Roofline 判断：prefill 落在屋顶右侧，decode 落在左侧；
  4. 推论：优化方向不同——prefill 抓算力（chunked prefill、PD 分离），decode 抓字节（量化、GQA/MLA、大 batch 摊销权重读取）。
- 加分项：attention intensity prefill 约 \(S/2\)、decode < 1；长上下文后 KV 读取可能取代权重成为 decode 主瓶颈。
- 踩坑：绝对化——极短 prompt 的 prefill 也可能 bandwidth-bound。

**题目 2：PagedAttention 解决什么？不解决什么？**

- 考点：KV cache 内存管理；碎片来源。
- 答题框架：
  1. 传统为每请求预留 max_len 连续显存 → 内/外碎片，容量浪费大；
  2. PagedAttention 把 KV 切成固定大小页，逻辑块经 block table 映射到非连续物理块，按需分配/释放；
  3. 收益：消除碎片、动态扩容、prefix 跨请求共享、分叉生成 copy-on-write；
  4. 边界：只做内存管理，attention 数学不变；block 查找与 kernel 复杂度是代价。
- 加分项：联系 vLLM 参数 gpu_memory_utilization、swap-space；block size 的取舍。
- 踩坑：把 PagedAttention 说成注意力算法创新，或与 continuous batching（调度策略）混为一谈。

**题目 3：PD 分离的收益与代价？**

- 考点：资源画像差异；goodput；分布式 trade-off。
- 答题框架：
  1. prefill compute-bound、decode bandwidth-bound，混部时大 prefill 推高 decode 的 TPOT 尾延迟；
  2. 分离后两池独立扩缩容、按各自瓶颈选卡，稳定时延、提升 SLO goodput；
  3. 代价：KV 跨实例传输（网络带宽/时延）、路由与 placement 复杂、pool 失衡、故障/重试语义；
  4. 结论：高并发、SLO 严苛场景收益大；小规模低 QPS 不值得。
- 加分项：提 SplitFuse 类动态切分作为中间路线（MindIE：splitChunkTokens 建议 512 倍数、cacheBlockSize 默认 128）。
- 踩坑：忽略 KV transfer 成本，把 PD 分离当免费午餐。

**题目 4：投机解码什么时候反而变慢？**

- 考点：speedup model；acceptance 与 draft 成本的权衡。
- 答题框架：
  1. 写成本模型：cost/token ≈ \((k c_d + c_t(k))/\mathbb{E}[\text{accepted}]\)；
  2. 变慢情形：draft 串行成本占比高、接受率低、验证并行效率差；
  3. 大 batch 下 target 已高效，speculation 边际收益下降甚至为负；
  4. 树形验证与 KV 管理（缓存迁移 vs 重计算）的额外开销可能吞掉收益。
- 加分项：无损性来自拒绝采样修正，输出严格服从 target 分布；typical acceptance 放松无损换速度。
- 踩坑：只报 acceptance rate 不报端到端 latency；声称无损却用了 typical acceptance。

**题目 5：continuous batching 为什么比 static batching 快？**

- 考点：调度粒度；head-of-line blocking。
- 答题框架：
  1. static：整批等最慢请求，短请求被拖住、槽位空转、padding 浪费；
  2. continuous：以 iteration 为调度单位，完成即出、到达即入；
  3. 异质长度下 GPU 利用率随负载平滑变化，行业口径吞吐约 2.5×；
  4. 代价：调度器复杂、动态 shape（CUDA Graph 需 bucket 化）。
- 加分项：Orca 首创 iteration-level scheduling；与 chunked prefill 配合防 prefill 阻塞 decode。
- 踩坑：把吞吐提升归因于「批更大」——真实原因是消除空转与等待。

**题目 6：GQA/MQA/MLA 各自如何省 KV cache？代价是什么？**

- 考点：attention 结构；KV 容量公式中的 \(K\)。
- 答题框架：
  1. 公式 \(M_{\text{KV}}=2BLSKHb_e\) 中 KV heads 数 \(K\) 是杠杆；
  2. MQA：\(K=1\)，省最多但表达受损；GQA：分组共享，质量/容量平衡，主流；
  3. MLA：低秩压缩，KV cache −93.3%、最大吞吐 +576%（DeepSeek-V2 官方口径）；
  4. 共同前提：质量须实测；MLA 另有 kernel 与训练复杂度。
- 加分项：稀疏化路线——Minference 动态稀疏与检索式 head 压缩约 70% KV（实践口径）。
- 踩坑：把 MLA 说成「GQA 的极端情况」——低秩压缩与 head 共享机制不同。

**题目 7：量化压缩 4× 为什么没快 4×？**

- 考点：kernel coverage；dequant 开销；activation outlier。
- 答题框架：
  1. 显存/字节减少不等于时间等比减少：非量化算子、dequant、packing 都在关键路径；
  2. weight-only 只省权重读取；W8A8 才可能吃到底层整数矩阵单元；
  3. activation outlier 使 W8A8 困难，需 SmoothQuant 把难度平滑迁移到权重；
  4. 数字锚点：INT4/INT8 权重量化减 50%–75% 显存；FP8 需 Hopper+。
- 加分项：KV 量化单独核算；按 kernel coverage 报告而不是只报 bits。
- 踩坑：不区分 weight/activation/KV 三类张量的精度配置。

## 系统设计题

**设计题 1：为 70B 模型（GQA）设计在线 serving 集群**

- 需求澄清：SLO（TTFT p99、TPOT p99）、目标 QPS/并发、输入输出长度
  分布、卡型与预算、是否多租户。
- 规模估算（锚点数字）：
  - 权重 bf16 约 140 GB → 单卡放不下，TP=4×80GB 起步（head 数须能被 TP 卡数整除）；
  - 显存按权重 50%–70%、KV 20%–40%、临时张量 5%–10% 规划（实践口径）→ 4 卡 320GB 中 KV 预算约 64–128GB；
  - 每请求 KV（80 层、8 KV heads、\(d_h=128\)、8K 上下文、bf16）约 2.5 GiB → 单实例并发约 25–50；
  - 若对标前沿模型单用户 >200 tokens/s（2025–2026 行业口径），TPOT 需 ≤5ms——通常仅单用户/端侧可达，集群 SLO 应按业务实测定，再反推 batch 上限。
- 架构：LB → router（prefix locality + queue length）→ vLLM 实例
  （TP=4；PagedAttention + continuous batching + prefix cache）→
  流量足够大时上 PD 分离（prefill 池 compute 导向、decode 池
  bandwidth 导向，KV transfer 连接）。
- trade-off 表：

| 决策 | 选项 A | 选项 B | 取舍要点 |
|---|---|---|---|
| 并行度 | TP=4 + 多 replica | 更高 TP | TP 大则单请求快但通信占比升；replica 才扩吞吐 |
| prefill 调度 | chunked prefill | 整段 prefill | chunk 保护 TPOT；太小则 TTFT/launch 开销升 |
| 拓扑 | 混部 | PD 分离 | 分离稳定时延，但加 KV 传输与路由复杂度 |

- 评测方案：open-loop 扫 offered load 画 goodput-latency curve；报告
  p50/p95/p99 TTFT/TPOT、SLO goodput、峰值显存；prefix 命中率冷/热分开。
- 追问预案：OOM → admission control + preemption/swap（swap-space）；
  冷启动 → 权重加载 + compile/capture 预热；多租户 → quota 与优先级队列。

**设计题 2：128K 长上下文推理服务的显存与调度设计**

- 需求澄清：128K 是 p99 还是均值；输出长度；并发目标；质量红线
  （retrieval/长文摘要）。
- 规模估算：沿用上题配置（80 层、8 KV heads、\(d_h=128\)、bf16），
  128K 每请求 KV 约 40 GiB；KV 预算 20%–40% 显存（实践口径）→
  单实例并发仅 1–3 → 必须 KV 量化（int8 减半）+ 稀疏/检索压缩
  （约 70% KV，实践口径）+ 多机并行，才能把并发拉回十级。
- 架构：PagedAttention 按需分配；prefix cache 复用系统 prompt/共享
  文档；chunked prefill 限制单 iteration prefill 量（防 128K prompt
  独占计算）；超长序列推动多机并行成为必然（参照 Gemini 1.5 支持
  1M–10M token 序列）。
- trade-off 表：

| 手段 | 收益 | 代价 |
|---|---|---|
| KV int8/int4 量化 | 容量约 ×2 | 长上下文误差累积，须测 retrieval/perplexity |
| 稀疏/检索式压缩 | 约 −70% KV（实践口径） | 语义近似，任务相关 |
| CPU offload/swap | 突发容量 | TPOT 尾部抖动 |
| 多机并行 | 突破单机显存 | 通信与调度复杂度 |

- 评测方案：TPOT 随 context 长度增长曲线；p99 TTFT（chunk 效果）；
  长上下文质量（retrieval、长生成稳定性）；OOM/preemption 率。
- 追问预案：eviction 策略（sliding window / heavy hitter / sink）；
  prefix cache 失效（template/version 变化）；与投机解码组合时长
  context 接受率变化。

**设计题 3：给现有 chat 服务引入投机解码 + 量化的改造方案**

- 需求澄清：目标是 TPOT 还是成本；质量红线（是否必须严格无损）；
  当前 batch 水位；硬件是否 Hopper+（决定 FP8 可用性）。
- 方案选型：drafter 三路线（蒸馏小模型如 7B、自起草 Medusa 附加头、
  检索式）+ EAGLE 类动态草稿树；量化 W8A8 + SmoothQuant 或
  W4A16/INT4（省 50%–75% 显存）。
- 架构：树形 attention 验证（mask 隔离无效分支）+ 拒绝采样或
  typical acceptance；KV 管理选缓存迁移或重计算；量化按 kernel
  coverage 分阶段 rollout。
- trade-off 表：

| 维度 | 严格拒绝采样 | typical acceptance |
|---|---|---|
| 输出分布 | 严格等于 target | 近似 |
| 速度 | 较慢 | 较快（端侧实践约 2×） |
| 大 batch | 收益下降甚至为负 | 同样下降 |

- 评测方案：分布一致性检验（token frequency / sequence likelihood）；
  端到端 TPOT/吞吐（不是 acceptance rate）；量化质量回归
  （perplexity、下游、罕见 token、长上下文）。
- 追问预案：高并发自动降级关闭 speculation；draft 与 target 版本
  同步；量化 + 投机叠加的误差交互。

## 代码实现题

**实现题 1：KV cache 字节数计算器（MHA/GQA/MQA/MLA 对比）**

- 考察点：\(M_{\text{KV}}=2BLSKHb_e\) 的结构；GQA/MQA/MLA 改变的是哪一项。

```python
def kv_cache_bytes(layers, seq, kv_heads, head_dim, batch, bytes_per=2):
    """M_KV = 2 (K and V) * B * L * S * K * H * bytes_per_element"""
    return 2 * batch * layers * seq * kv_heads * head_dim * bytes_per


def compare(layers=80, seq=8192, q_heads=64, kv_heads=8, head_dim=128, batch=1):
    mha = kv_cache_bytes(layers, seq, q_heads, head_dim, batch)
    gqa = kv_cache_bytes(layers, seq, kv_heads, head_dim, batch)
    mqa = kv_cache_bytes(layers, seq, 1, head_dim, batch)
    mla = kv_cache_bytes(layers, seq, 1, 512, batch)  # 低秩潜空间, 例 d_c=512
    for name, v in [("MHA", mha), ("GQA", gqa), ("MQA", mqa), ("MLA", mla)]:
        print(f"{name}: {v / 1024**3:.2f} GiB  ({v / mha:.2%} of MHA)")


compare()
```

- 验收标准：与正文例对齐——\(B=64,L=40,S=4096,K=8,H=128\)、bf16 约
  40 GiB；MLA 相对 MHA 的缩减与 −93.3% 官方口径同量级（取决于
  \(d_c/d_h\) 配置）。

**实现题 2：PagedAttention 块分配模拟器**

- 考察点：block table；按需分配/释放；内部碎片统计。

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free = list(range(num_blocks))        # 空闲物理块池
        self.tables: dict[str, list[int]] = {}     # req -> 物理块列表
        self.lengths: dict[str, int] = {}          # req -> 已缓存 token 数

    def _blocks_needed(self, length: int) -> int:
        return (length + self.block_size - 1) // self.block_size

    def append_tokens(self, req: str, n: int) -> bool:
        new_len = self.lengths.get(req, 0) + n
        blocks = self.tables.setdefault(req, [])
        while len(blocks) < self._blocks_needed(new_len):
            if not self.free:                      # 显存耗尽: 上层做抢占/拒绝
                return False
            blocks.append(self.free.pop())
        self.lengths[req] = new_len
        return True

    def release(self, req: str) -> None:
        self.free.extend(self.tables.pop(req, [])) # 请求结束, 块全部归还
        self.lengths.pop(req, None)

    def internal_fragmentation(self) -> float:
        allocated = sum(
            self._blocks_needed(l) * self.block_size for l in self.lengths.values()
        )
        if not allocated:
            return 0.0
        return 1 - sum(self.lengths.values()) / allocated
```

- 验收标准：随机长度请求下每请求内部碎片 < block_size；分配/释放
  循环后空闲池完整回收；显存耗尽时 append_tokens 返回 False 且状态
  一致；扩展点：prefix 共享 + copy-on-write。

**实现题 3：continuous batching 调度循环骨架**

- 考察点：iteration-level scheduling；步级进出；token 预算
  （max_num_batched_tokens）。

```python
from collections import deque


class Req:
    def __init__(self, rid: str, prompt_len: int, max_new: int):
        self.rid, self.prompt_len, self.max_new = rid, prompt_len, max_new
        self.generated = 0
        self.finished = False


def scheduler_loop(waiting: deque, max_batched_tokens: int, max_batch: int) -> int:
    running: list[Req] = []
    steps = 0
    while waiting or running:
        # 1) 完成的请求步级退出, 槽位与预算立即释放
        for r in running:
            if r.generated >= r.max_new:
                r.finished = True
        running = [r for r in running if not r.finished]
        # 2) decode 预算: 每个活跃请求本步消耗 1 个 token 位
        budget = max_batched_tokens - len(running)
        # 3) admission: 用剩余预算吸收新请求做 prefill (简化为一次灌入)
        while (waiting and len(running) < max_batch
               and waiting[0].prompt_len <= budget):
            r = waiting.popleft()
            budget -= r.prompt_len
            running.append(r)
        # 4) 执行一步: 活跃请求各 decode 1 个 token
        for r in running:
            r.generated += 1
        steps += 1
    return steps
```

- 验收标准：异质长度 workload 下平均在途请求数高于 static batching；
  无请求饿死；budget 恒不被突破；扩展点：chunked prefill（把 prompt
  拆多步灌入）与 preemption。

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
- [vLLM 官方文档（引擎参数与 PD 分离）](https://docs.vllm.ai)（访问日期 2026-10-04）
- [DeepSeek-V2：MLA 低秩 KV 压缩](https://arxiv.org/abs/2405.04434)（访问日期 2026-10-04）
- [Medusa：自起草多头投机解码](https://arxiv.org/abs/2401.10774)（访问日期 2026-10-04）
- [EAGLE：动态草稿树投机解码](https://arxiv.org/abs/2401.15077)（访问日期 2026-10-04）
