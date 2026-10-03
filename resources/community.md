# 社区精选资料

筛选标准：覆盖度、解释质量、可追溯来源、维护状态、复现信息和原创性。星标数只作辅助，不代表正确性。访问日期：2026-09-26。

## 讲义与笔记

### [QihongRuan/stanford-cs336-notes](https://github.com/QihongRuan/stanford-cs336-notes)

- 版本/语言：Spring 2025，英文。
- 覆盖：17 讲、约 200 张关键截图、推导、deep dives 和作业学习指南。
- 推荐理由：结构完整、图文结合、每讲链接回原视频；作业部分以思路和易错点为主而非直接答案。
- 使用建议：课前浏览框架，课后用 deep dives 补足 BPE、FlashAttention、并行、scaling 与 RL 数学。
- 许可提醒：使用前以该仓库当前 LICENSE 为准；本项目仅链接。

### [hqhq1025/ai-course-notes](https://github.com/hqhq1025/ai-course-notes)

- 版本/语言：Spring 2025 与 2026，中文。
- 覆盖：长篇 LaTeX/PDF 中文讲义，保留英文术语，并提供网页阅读。
- 推荐理由：强调教学逻辑、推导、工程解释和质量检查，适合中文精读。
- 注意：内容由字幕、幻灯片及公开材料整理，应回到官方资料核验细节。
- 许可提醒：仓库体量大且包含多门课程；按其 LICENSE/各材料声明使用，本项目仅链接。

### [Niujunbo2002/stanford-cs336-pku](https://github.com/Niujunbo2002/stanford-cs336-pku)

- 版本/语言：Spring 2025，中文大纲与中英字幕资源。
- 推荐理由：提供按周学习规划和硬件需求估算，适合制定节奏。
- 完整度：翻译与课程组织资料为主，不应替代官方 handout。

## 作业实现（卡住后再看）

查看这些仓库前，建议先提交自己的失败尝试、测试和问题分析。引用思路必须写入作业 writeup。

### [zhasion/CS336](https://github.com/zhasion/CS336)

- 覆盖：五个作业的大部分实现，每项附 writeup；作者明确说明部分内容因硬件限制跳过。
- 推荐理由：学习过程和取舍记录较清楚，适合完成后交叉检查。
- 风险：作者声明可能存在不准确之处，不能作为标准答案。

### [Melody-Zhou/stanford-cs336-spring2025-assignments](https://github.com/Melody-Zhou/stanford-cs336-spring2025-assignments)

- 覆盖：Spring 2025 Assignment 1–5，README 对功能点描述较完整。
- 注意：作者注明开发中使用了 ChatGPT；必须通过官方测试与 handout 独立核验。

### [Louisym/Stanford-CS336-spring25](https://github.com/Louisym/Stanford-CS336-spring25)

- 覆盖：Spring 2025 五项作业、测试与说明。
- 推荐理由：目录清晰，明确排除了大型数据和模型文件。
- 注意：课程主页链接和实现细节可能随时间过期。

### [YYZhang2025/Stanford-CS336](https://github.com/YYZhang2025/Stanford-CS336)

- 覆盖：A1、A2、A4、A5 以及 MoE 扩展。
- 不完整项：公开说明中 A3 仍是 placeholder，因此不列为完整答案。

### [Zian-2/cs336_assignments_and_notes](https://github.com/Zian-2/cs336_assignments_and_notes)

- 版本/语言：Spring 2025，中文。
- 覆盖：较详细中文笔记；截至检索时主要完成 A1 和部分 A5。
- 适用：中文概念对照，不适合作为五项作业的完整基线。

## 使用顺序

1. 官方 handout、测试和讲义。
2. 自己的设计、实现、测试与实验。
3. QihongRuan 或中文讲义补概念。
4. 定位到具体失败后，再对照一个社区实现。
5. 关闭参考资料，独立重写并在 writeup 注明获得的启发。
