# A3：IsoFLOP 与 Proxy Scaling

- **官方数据**：72 records、9 compute tiers，固定 SHA 见
  [`data/SOURCE.md`](../../assignments/spring2026/assignment3-scaling/data/SOURCE.md)。
- **Proxy**：22 个 RTX 6000D TinyStories runs，4 compute tiers。
- **报告**：[`writeup.pdf`](../../assignments/spring2026/assignment3-scaling/report/writeup.pdf)

## 方法
每个 compute tier 直接选择离散最低 loss，拟合
\(N_\text{opt}=A C^a\)、\(D_\text{opt}=B C^b\)；另做联合 loss law、
bootstrap、residual、held-out 与 extrapolation sensitivity。

## 结果
- 官方：\(a=0.469,b=0.531\)。
- \(10^{23}\) FLOPs：70.05B parameters、237.91B tokens。
- \(10^{24}\) FLOPs：206.12B parameters、808.60B tokens。
- Proxy exponent 为 0.676/0.324，显示 domain、grid boundary 与小样本会改变外推。

## 使用边界
没有 Stanford A3 API key；proxy 不代表 B200/DCLM leaderboard。
