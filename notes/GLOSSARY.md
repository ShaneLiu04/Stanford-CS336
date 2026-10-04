# CS336 术语与符号表

## 通用符号
| 符号 | 含义 |
|---|---|
| \(B,T,V\) | batch size、sequence length、vocabulary size |
| \(d,d_{ff},h,d_h\) | model width、FFN width、heads、head dimension |
| \(L,N,D,C\) | layers、parameters、training tokens、training FLOPs |
| \(Q,K,V,O\) | attention query、key、value、output |
| \(S,P\) | attention scores 与 softmax probabilities |
| \(\theta,\pi_\theta\) | 参数与 policy |

## Tokenization / Basics
- **BPE**：反复合并最高频 byte/token pair 的 tokenizer 训练算法。
- **pre-tokenization**：在 BPE 前按 regex/边界拆分，防止跨不合理边界合并。
- **bytes/token**：压缩指标；越高表示相同文本需要更少 tokens。
- **fertility**：每词/每 normalization unit 的平均 token 数；多语言公平性核心指标。
- **nats/byte**：把 token loss 按压缩率归一，跨 tokenizer 可比。
- **RMSNorm**：按 root-mean-square 缩放，不减均值。
- **RoPE**：用位置相关二维旋转把相对位置信息注入 Q/K。
- **SwiGLU**：\(\mathrm{SiLU}(W_1x)\odot W_3x\) 后投影的 gated FFN。
- **GQA / MQA / MLA**：KV 共享/低秩压缩谱系；MLA KV cache −93.3%、吞吐 +576%（DeepSeek-V2 口径）。
- **AdamW**：把 weight decay 与梯度更新解耦的 Adam。
- **MFU**：实际 TFLOPS / 峰值 TFLOPS；第一梯队集群 40–55%。

## Systems
- **arithmetic intensity**：FLOPs / memory bytes；判断 compute-bound 或 memory-bound。
- **roofline**：用峰值算力与带宽上界解释 kernel 性能。
- **warp / block / SM**：GPU 线程执行与调度层级。
- **online softmax**：分块更新 row max 与 normalization denominator。
- **FlashAttention**：减少 HBM 往返、不物化 \(T^2\) attention matrix 的精确算法。
- **collective**：all-reduce、all-gather、reduce-scatter、broadcast 等多进程通信。
- **DDP / TP / PP**：data、tensor、pipeline parallelism。
- **ZeRO / FSDP**：分片 optimizer、gradient、parameter state；Stage-3 显存 ≈ 16Φ/P。
- **FSDP2**：torch 2.4 起 per-parameter sharding + 动态重分片。
- **3D parallel**：机内 TP + 跨机 PP + 外层 DP 的超大模型标准组合。
- **EP**：expert parallelism，MoE 专家分布到多卡（DeepSeek-V3 EP64）。
- **1F1B**：PP 调度；气泡占比 ≈ (p−1)/(m+p−1)。
- **auxiliary-loss-free balancing**：用可学习偏置而非辅助损失做 MoE 负载均衡（DeepSeek-V3）。

## Scaling / Inference / Evaluation
- **IsoFLOP**：固定 \(C\approx6ND\)，扫描 \(N,D\) 找 minimum loss。
- **compute-optimal**：给定 compute 下使 loss 最低的模型/数据分配。
- **Chinchilla-optimal**：D*/N* ≈ 20 tokens/param；Llama-3 走"过训练"路线换推理经济性。
- **test-time compute**：推理时更长思考/搜索/验证换能力（o1/R1 路线）。
- **prefill / decode**：一次处理 prompt 与逐 token 生成阶段。
- **KV cache**：缓存历史 key/value，避免 decode 重算。
- **PagedAttention**：KV cache 分页管理消除显存碎片（vLLM）。
- **PD 分离**：prefill 与 decode 部署到独立实例，稳定时延。
- **continuous batching**：动态合并不同到达/结束时间的 requests。
- **speculative decoding**：draft model 提议、target model 并行验证；树形 attention + typical acceptance。
- **calibration**：预测置信度与真实正确率的一致性。
- **contamination**：训练数据包含 benchmark 内容导致评估虚高。
- **循环评估 / circular eval**：多选题按选项轮换重测，防位置偏差（MMBench）。

## Data
- **WARC / WET**：保留 HTTP/HTML 的 Web archive 与仅文本的转换格式。
- **fastText classifier**：基于 hashed n-gram 的高吞吐线性分类器。
- **Gopher rules**：长度、词长、符号、ellipsis 等可解释质量规则。
- **MinHash**：近似 Jaccard similarity 的集合签名。
- **LSH banding**：将 MinHash 分 band 生成候选对，避免 \(O(n^2)\)。
- **语义去重**：聚类后簇内余弦相似度阈值去重，保留簇中心最近样本。
- **DoReMi**：用小模型 PPL 驱动的 domain 配比优化。
- **去污染 / decontamination**：从训练语料中剔除评测集内容（LLM Decontaminator 类工具）。
- **DSIR**：用 raw/target density ratio 做 data importance resampling。
- **model collapse**：多代替换真实数据的合成数据导致性能退化。

## Alignment
- **SFT**：在 instruction-response pairs 上做 supervised next-token training。
- **reward model / RLHF**：学习偏好 reward，再优化 policy。
- **DPO**：直接优化 chosen/rejected 相对 reference 的 preference margin。
- **RLVR**：用可自动验证 reward（数学答案、代码测试）训练。
- **GRPO**：用同 prompt group 的相对 reward 构造 advantage；省 critic。
- **Dr.GRPO / MaxRL / RFT**：不同 baseline、normalization 与 positive-only objective。
- **DAPO dynamic sampling**：丢弃全同奖励组，保持组内可比性。
- **aha moment**：纯 RL 中自发涌现的自我反思行为（R1-Zero）。
- **staleness**：异步 RL 中 rollout 数据相对当前权重的陈旧度。
- **importance ratio**：\(\rho=\pi_\theta/\pi_{\mathrm{old}}\)。
- **GSPO**：sequence-level importance ratio 与 clipping。
- **alignment tax**：对齐后基础能力或某些 benchmark 的下降。

## 常用工业数字（面试可引用）

- 词表：GPT-2 50,257；GPT-4 级 ~100k；Qwen 15 万+。
- 数据：GPT-2 ~100B tokens（2019，约 $4 万训练）；Qwen 2.2T；FineWeb 15T（96 个 CC 快照）；Llama-3 15T；Common Crawl 2024 已索引约 27 亿网页。
- Chinchilla：70B/1.4T 优于 280B Gopher；D*/N* ≈ 20；Llama-3 8B 过训至 ~1875 tokens/param。
- DeepSeek-V3：671B 总参/37B 激活；256 路由专家+1 共享、top-8；EP64；负载均衡率 99.5%+。
- GPU（SXM dense 口径）：A100 BF16 312 TF/2TB/s；H100 989 TF/3.35TB/s；B200 2.25 PF/8TB/s；NVLink4 900GB/s、NVLink5 1.8TB/s。
- ZeRO：Stage-1 ≈ 12Φ/P+4Φ、Stage-2 ≈ 14Φ/P+2Φ、Stage-3 ≈ 16Φ/P。
- 推理显存：权重 50–70%、KV cache 20–40%；vLLM continuous batching 吞吐 ~2.5×；MLA KV −93.3%。
- 评测：MMLU 15,908 题/57 学科；GPQA 448 题（专家 65%/非专家 34%）；HumanEval+ 测试 ×80。
- R1-Zero：AIME 15.6%→86.7%（纯 RL 规则奖励）；蒸馏 600k 推理+200k 通用样本。
- VSI-Bench：SOTA Gemini-1.5 Pro 空间推理仅 48.8%。
