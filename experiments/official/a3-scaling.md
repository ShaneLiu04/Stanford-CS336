# A3 Scaling：用有限试验预算外推最优训练配置

> 中文导读；不替代[官方 handout](../../assignments/spring2026/assignment3-scaling/cs336_assignment3_scaling.pdf)，不披露拟合答案或最优配置。

## 目标

在“最终模型可使用 48 B200-hours、前期探索只有 12 B200-hours”的约束下，选择模型规模、训练 token 数和优化超参数，使最终 validation loss 最低。重点不是训练一个现成模型，而是设计高信息量的实验、拟合 scaling law、量化残差/不确定性并进行大跨度外推。

## Problem 与 deliverable

| Problem | 分值 | 交付物 |
| --- | ---: | --- |
| `chinchilla_isoflops` | 5 | 用 `data/isoflops_curves.json` 重现 IsoFLOP 方法；分别画 `N_opt(C)`、`D_opt(C)` 拟合与外推图，并给出指定 FLOP 预算预测 |
| `scaling_laws` | 50 | 在 API 的 12 B200-hour 硬预算内设计训练矩阵；完整说明拟合、选择与诊断方法；预测 48 B200-hour 最优 architecture/optimizer/training config 和 final loss；向 `/final_submission` 提交 |

Gradescope 交付为 `writeup.pdf` 和 `code.zip`；此外 API 中必须存在最终预测提交。报告要足以复现实验选择、拟合和外推，不能只给最终数字。

## API 接口

所有请求以 `X-API-Key` 携带 8 位 Stanford ID，并需 Stanford 网络/VPN。服务地址在本地题面/README 中为 `http://hyperturing.stanford.edu:8000`。

- `GET /budget`：`used_seconds`、`remaining_seconds`、`total_budget_seconds`；
- `POST /submit`：提交 `architecture_config`、`optimizer_config`、train/val batch、`n_evals`、`total_train_tokens`、`max_runtime_seconds`、`model_seed`；
- `GET /experiments`、`GET /experiment/{id}`：状态与 `val_losses`；
- `POST/GET /final_submission`：最终配置和 `predicted_final_loss`。

关键约束：

- `max_runtime_seconds` 为 1 秒至 12 小时；queued/running 会先按上限占用预算，完成/失败后按规则结算；
- 重复配置返回 409，不重复占预算；超余额返回 400；
- 只有 `completed` run 的最后一个 `status.val_losses` 才是正式 final loss，timeout 的 partial loss 不能混入相同口径；
- `seq_len=512`、validation tokens 固定；`total_train_tokens` 必须可被 `512 * train_batch_size` 整除；
- `hidden_size = num_attention_heads * head_dim`，attention heads 可整除 KV heads；
- 数据顺序固定且无 epoching；`model_seed` 只控制初始化。

## 推荐实验矩阵

不要一次性铺满网格。可采用分阶段设计：

1. **校准阶段：** 极短 run 验证吞吐、可行 batch、参数量估算和 API 字段；严格限制其预算占比。
2. **IsoFLOP 主矩阵：** 选择若干相隔近似对数均匀的 compute levels；每层预算覆盖明显 under-sized、近最优、over-sized 的模型，确保 loss 曲线两侧都有点。
3. **超参数切片：** 在代表性小/中模型上控制变量比较 LR、batch、warmup、weight decay、architecture ratio；避免把架构和 compute 同时改变后无法归因。
4. **重复与边界：** 对拟合最敏感点做多 seed；在预测 ridge 附近补点，不平均浪费到所有格点。
5. **验证阶段：** 留出预算检验外推而不是继续拟合；比较 IsoFLOP、joint loss law 或 robust fit，报告 residual 与 bootstrap interval。

每个 run 保存原始请求 JSON、experiment id、时间戳、状态历史、实际 runtime、完整 val curve、最终 loss、参数量和预算账本。最终模型参数量按题面近似 `12 * n_layers * d_model^2`（非 embedding 参数）统一口径。

## 硬件与预算

- API 中每个训练 job 在单张 B200 上运行。
- scaling-law 探索总预算是 12 B200-hours，服务端硬限制；最终候选按 48 B200-hours 评估。
- 12 小时是 wall-clock 预算，不等同于 `6ND` FLOPs；要从实测 runtime/throughput 建模二者关系。
- 队列、启动、抢占可能让状态 wall time 超过 `max_runtime_seconds`；以 API 的结算字段为准。

## 资源受限替代

- **无 API key：** 完成官方 synthetic IsoFLOP 数据分析；再用本地小模型构造 proxy matrix，练习实验设计、bootstrap 和外推。
- **只有单张非 B200 GPU：** 用 token-aligned runs 和实际 wall time 作为本地预算；结论仅适用于该硬件/模型/数据。
- **极少算力：** 对已有 curves 做 resampling、leave-one-budget-out、拟合模型比较和敏感性分析，这比虚构 API runs 更有价值。
- 不得伪造 experiment id、`/final_submission` 或 B200 leaderboard 结果；本地 proxy loss 与 DCLM/32K/512 官方环境不直接可比。

## 提交检查

- `writeup.pdf`：实验选择理由、拟合公式与方法、数据图、残差/不确定性、48 B200-hour 预测、最终超参数和 predicted loss；
- `code.zip`：查询、缓存、清洗、拟合和绘图代码，不含 API key；
- API：最后一次 `/final_submission` 已成功且可 GET 回读；
- 对失败、timeout、重复请求和预算结算有明确处理；所有图能追溯到 experiment ids。

## 版本风险与本仓库报告

- 本地 handout 版本 `26.0.5`；固定 commit `03e9372992e913061b9e78b5cfcb62ad8a87de35`，见 [`UPSTREAM.md`](../../UPSTREAM.md)。
- API schema、服务可用性、数据与 leaderboard 可能脱离静态仓库变化；课程通知优先。
- 本仓库未使用 Stanford API，仅做官方 synthetic 数据分析和 RTX 6000D proxy；不能视作官方提交。
- 报告：[源码](../../assignments/spring2026/assignment3-scaling/report/main.tex) · [PDF](../../assignments/spring2026/assignment3-scaling/report/writeup.pdf) · [资产说明](../../assignments/spring2026/assignment3-scaling/report/README.md) · [验证记录](../../assignments/spring2026/assignment3-scaling/VERIFICATION.md)。
