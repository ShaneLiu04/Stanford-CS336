# CS336 社区资料索引

本目录只做外部资料的评估与链接，不镜像第三方代码、正文、图片、数据或权重。核查日期：**2026-09-28**。GitHub 的 `main`/`master` 链接会移动；真正复用前应固定 commit，并再次检查许可证。

## 怎么读

- [A1 — Basics](a1.md)
- [A2 — Systems](a2.md)
- [A3 — Scaling](a3.md)
- [A4 — Data](a4.md)
- [A5 — Alignment](a5.md)

每条资料均记录：

- **版本匹配**：与本仓库 Spring 2026 作业是否直接匹配；Spring 2025 资料只作思路参考。
- **覆盖范围**：公开页面实际展示的主题，不由仓库名推断完成度。
- **测试 / 实验完整度**：区分“存在官方测试目录”“作者声称通过”“有可审计结果”“由本仓库复现”。本索引没有复跑第三方训练。
- **License**：未发现仓库级明确许可时一律写 `UNKNOWN`；公开可读不等于允许复制、修改或再发布。
- **Spoiler**：`S0` 仅导航，`S1` 概念讲解，`S2` 实现提示或局部推导，`S3` 完整/近完整实现及结果。先独立完成再看 `S2–S3`。
- **可借鉴点 / 风险**：只提炼方法，不复制无许可证代码或正文。

## 跨作业讲义与导航

### LOGO127 — CS336 2026 notes

- URL：[仓库](https://github.com/LOGO127/cs336.2026)；[作业进度页](https://github.com/LOGO127/cs336.2026/blob/main/assignments/README.md)
- **版本匹配**：明确面向 CS336 2026；适合对照本仓库 Spring 2026。
- **覆盖范围**：按讲次组织 tokenization、架构、GPU / kernels、并行训练、scaling、数据与 alignment；作业区显式区分未完成内容。
- **测试 / 实验完整度**：讲义仓库，不应当作五份作业均已完成的证据；公开树中只有 A1 作业页。
- **License**：代码为 **MIT**（[`LICENSE-CODE`](https://github.com/LOGO127/cs336.2026/blob/main/LICENSE-CODE)）；原创文字和图为 **CC BY-NC-SA 4.0**（[`LICENSE-NOTES`](https://github.com/LOGO127/cs336.2026/blob/main/LICENSE-NOTES)）。
- **Spoiler**：`S1–S2`。
- **可借鉴点**：版本化笔记、作业状态声明、事实与实验结论分离。
- **风险**：非商业、相同方式共享条款不适合直接并入本 MIT 仓库；这里只链接。

### QihongRuan — Spring 2025 illustrated notes

- URL：[仓库](https://github.com/QihongRuan/stanford-cs336-notes)；[站点](https://qihongruan.github.io/cs336/)
- **版本匹配**：明确为 **Spring 2025**；主题大体延续，但不能据此确认 2026 API、题目或评分要求。
- **覆盖范围**：17 讲图文笔记、算法 deep dives，以及五份作业的“guidance, not solutions”学习指南。
- **测试 / 实验完整度**：讲义，无代码测试或课程实验完成声明。
- **License**：**UNKNOWN**（核查时仓库根目录未发现 LICENSE/COPYING）。
- **Spoiler**：讲义 `S1`，作业指南约 `S1–S2`。
- **可借鉴点**：从讲次回链到作业难点，适合查概念和关键词。
- **风险**：包含大量课程幻灯片截图，且仓库许可未知；不要复制正文或图片。

### Aleksandr Timashov — CS336 blog series

- URL：[CS336 分类页](https://timashov.ai/blog/category/cs336/)
- **版本匹配**：文章始于 2025；2026-01 的 FlashAttention 文章也早于 Spring 2026 作业发布，宜视为 **2025 讲次脉络的补充**，不是 2026 作业答案。
- **覆盖范围**：BPE、张量/视图/FLOPs、反向传播、内存基础和 FlashAttention-2 / Triton。
- **测试 / 实验完整度**：博客文章含推导、kernel 讲解和性能讨论，但不是五份作业的测试归档。
- **License**：**UNKNOWN**（页面未见可用于复制正文/代码的明确开放许可）。
- **Spoiler**：通常 `S1–S2`；FlashAttention 文章接近 `S2–S3`。
- **可借鉴点**：把 GPU 心智模型、算术强度和 kernel 设计连起来。
- **风险**：博客代码片段与正文不可因“公开可访问”而默认复用；只链接并自行重写思路。

## 收录边界

本索引偏向能核验版本、目录、许可证或实验声明的资料。结果数字均为**原作者自述**，除非条目明确写“已复现”。`fwukendall` 的五个 2026 fork 经 GitHub compare 核查均为相对官方上游 **ahead 0**；因此只在各页作为版本骨架镜像说明，不当作社区解答。
