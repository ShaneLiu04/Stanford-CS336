# 资料条目贡献规范

新增来源时必须：

- 在 `SOURCES.yaml` 添加唯一 `id`、作者、URL、访问日期、版本、topics、type、
  license、mirror、sha256、confidence。
- 优先引用官方/论文原始来源；社区转述不能替代 primary source。
- 摘要必须原创，区分“原文结论”与“本仓库解释”。
- 标记版本匹配度和 spoiler 等级。
- 本地镜像前核对许可证；`UNKNOWN`、`SEE_SOURCE`、`ARXIV` 不得镜像正文。
- 不提交数据集、权重、凭据、付费墙内容或未经许可的课程答案。

运行：

```bash
python experiments/scripts/check_sources.py
```

确保 metadata、重复 URL、本地链接和敏感信息检查通过。
