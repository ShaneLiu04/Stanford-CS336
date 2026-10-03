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
- **RMSNorm**：按 root-mean-square 缩放，不减均值。
- **RoPE**：用位置相关二维旋转把相对位置信息注入 Q/K。
- **SwiGLU**：\(\mathrm{SiLU}(W_1x)\odot W_3x\) 后投影的 gated FFN。
- **AdamW**：把 weight decay 与梯度更新解耦的 Adam。

## Systems
- **arithmetic intensity**：FLOPs / memory bytes；判断 compute-bound 或 memory-bound。
- **roofline**：用峰值算力与带宽上界解释 kernel 性能。
- **warp / block / SM**：GPU 线程执行与调度层级。
- **online softmax**：分块更新 row max 与 normalization denominator。
- **FlashAttention**：减少 HBM 往返、不物化 \(T^2\) attention matrix 的精确算法。
- **collective**：all-reduce、all-gather、reduce-scatter、broadcast 等多进程通信。
- **DDP / TP / PP**：data、tensor、pipeline parallelism。
- **ZeRO / FSDP**：分片 optimizer、gradient、parameter state。

## Scaling / Inference / Evaluation
- **IsoFLOP**：固定 \(C\approx6ND\)，扫描 \(N,D\) 找 minimum loss。
- **compute-optimal**：给定 compute 下使 loss 最低的模型/数据分配。
- **prefill / decode**：一次处理 prompt 与逐 token 生成阶段。
- **KV cache**：缓存历史 key/value，避免 decode 重算。
- **continuous batching**：动态合并不同到达/结束时间的 requests。
- **speculative decoding**：draft model 提议、target model 并行验证。
- **calibration**：预测置信度与真实正确率的一致性。
- **contamination**：训练数据包含 benchmark 内容导致评估虚高。

## Data
- **WARC / WET**：保留 HTTP/HTML 的 Web archive 与仅文本的转换格式。
- **fastText classifier**：基于 hashed n-gram 的高吞吐线性分类器。
- **Gopher rules**：长度、词长、符号、ellipsis 等可解释质量规则。
- **MinHash**：近似 Jaccard similarity 的集合签名。
- **LSH banding**：将 MinHash 分 band 生成候选对，避免 \(O(n^2)\)。
- **DSIR**：用 raw/target density ratio 做 data importance resampling。

## Alignment
- **SFT**：在 instruction-response pairs 上做 supervised next-token training。
- **reward model / RLHF**：学习偏好 reward，再优化 policy。
- **DPO**：直接优化 chosen/rejected 相对 reference 的 preference margin。
- **RLVR**：用可自动验证 reward（数学答案、代码测试）训练。
- **GRPO**：用同 prompt group 的相对 reward 构造 advantage。
- **Dr.GRPO / MaxRL / RFT**：不同 baseline、normalization 与 positive-only objective。
- **importance ratio**：\(\rho=\pi_\theta/\pi_{\mathrm{old}}\)。
- **GSPO**：sequence-level importance ratio 与 clipping。
- **alignment tax**：对齐后基础能力或某些 benchmark 的下降。
