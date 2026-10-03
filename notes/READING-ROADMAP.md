# CS336 阅读与复习路线

## 按时间预算

### 2 小时速览
1. L1：LM 全流程与 tokenization。
2. L3：现代 Transformer block。
3. L6：FlashAttention 为什么减少 IO。
4. L9：\(C\approx6ND\) 与 compute-optimal。
5. L14：过滤+去重流水线。
6. L16：GRPO 与 verifiable reward。

只读每篇的学习目标、核心概念、小结和易错点。

### 1 天工程复习
- 上午：L1–4，手算 Transformer shapes/FLOPs。
- 下午：L5–8，画 GPU memory hierarchy 与 DDP/FSDP 通信图。
- 晚上：L9–12，复现 IsoFLOP 拟合并设计 evaluation sheet。

### 1 周完整路线
| 天 | 内容 | 输出 |
|---|---|---|
| 1 | L1–2 | BPE、tensor shape、FLOPs/bytes worksheet |
| 2 | L3–4 | Transformer/MoE 模块图与超参表 |
| 3 | L5–6 | Triton kernel、online softmax 推导 |
| 4 | L7–8 | parallelism communication accounting |
| 5 | L9–12 | scaling fit、inference/eval checklist |
| 6 | L13–14 | data provenance/filter/dedup pipeline |
| 7 | L15–17 | SFT/DPO/GRPO/multimodal alignment map |

## 按作业
- **A1**：L1–4 → [A1 导读](../experiments/official/a1-basics.md)
- **A2**：L5–8 → [A2 导读](../experiments/official/a2-systems.md)
- **A3**：L9、L11–12 → [A3 导读](../experiments/official/a3-scaling.md)
- **A4**：L12–14 → [A4 导读](../experiments/official/a4-data.md)
- **A5**：L12、L15–17 → [A5 导读](../experiments/official/a5-alignment.md)

## 按目标
- **实现小语言模型**：L1→L2→L3→A1。
- **做 GPU kernel / distributed systems**：L2→L5→L6→L7→L8→A2。
- **做训练预算规划**：L2→L9→L11→L12→A3。
- **做 data engineering**：L13→L14→A4。
- **做 post-training / reasoning**：L12→L15→L16→A5。
- **做 multimodal systems**：L3→L4→L10→L17。

## 每讲复习闭环
1. 不看笔记写出 5 个关键词。
2. 手推至少一个公式，并标注假设。
3. 写出关键 tensor shape 或通信量。
4. 找一个本仓库测试/图验证结论。
5. 给出一个反例或失效边界。
6. 把未解决问题加入下一轮阅读。
