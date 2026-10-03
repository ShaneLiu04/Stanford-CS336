# A2：Fused FlashAttention 与 Systems

- **目的**：量化 IO-aware attention、mixed precision、checkpointing 与分布式语义。
- **Raw**：[`report/results/raw`](../../assignments/spring2026/assignment2-systems/report/results/raw/)
- **报告**：[`writeup.pdf`](../../assignments/spring2026/assignment2-systems/report/writeup.pdf)
- **验证**：CUDA/Triton 6/6；CPU/Gloo DDP/FSDP/sharding 8/8。

## Controlled variables
GPU 固定 RTX 6000D；attention 固定 batch/head dimension/dtype 后扫描 sequence；
Transformer 固定 vocab/batch/context 后扫描 model size。所有计时含 warmup 和 CUDA sync。

## 关键结果
- BF16、sequence 32768、d=64：fused Triton forward+backward 10.06 ms，
  SDPA 88.56 ms；peak memory 48.5 MiB vs 16,444 MiB。
- 旧 PyTorch tiled backward 是 end-to-end 瓶颈；融合 delta、dQ、dK/dV 后
  sequence 4096 加速 1864.7×。
- Medium context 512→2048：latency 11.0×、显存 6.78×，体现 quadratic attention。
- `torch.compile` 对 BF16 small/medium 分别 1.98×/1.72× steady-state speedup。

## 局限
单卡不能产生 NCCL multi-GPU performance evidence；Gloo 多进程只证明数值语义。
