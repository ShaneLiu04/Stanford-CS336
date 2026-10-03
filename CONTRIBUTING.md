# Contributing

本仓库首先是一份可审计的个人学习记录。欢迎纠错、补充资料和改进实验复现性。

## 提交原则

1. 注明内容对应 Spring 2025 还是 Spring 2026。
2. 新结论附上实验配置、命令、随机种子和结果。
3. 引用外部内容时给出作者、标题、URL、访问日期及许可证（若可得）。
4. 不提交数据集、模型权重、访问令牌、课程私有 API key 或个人信息。
5. 不直接复制第三方作业解答；可在 `resources/` 中增加带评价的链接。

## 建议的提交格式

使用清晰、单一目的的提交，例如：

```text
notes(lecture-03): explain BPE complexity
feat(a1-2025): implement rotary embeddings
exp(a2): record flash-attention benchmark
docs(resources): add scaling-law reading
```

提交前运行 `powershell -File scripts/check-repo.ps1`。
