# Assignment 2 report artifacts

该目录沿用 Assignment 1 的可追溯报告结构：

```text
report/
├── main.tex                  # 中文主体、英文术语
├── writeup.tex               # 稳定 PDF 入口
├── writeup.pdf               # 已编译报告
└── results/
    ├── raw/                  # 不可变实验指标与测试日志
    ├── figures/              # 同名 PNG（README）与 PDF（LaTeX）
    ├── tables/               # 自动生成的 LaTeX tables
    ├── summary.json          # 机器可读关键结论
    ├── experiment_log.csv    # 全部 benchmark 的扁平索引
    └── manifest.sha256       # 所有报告资产的 SHA-256
```

从原始 CSV 重建全部资产：

```bash
uv run python scripts/build_report_assets.py --results report/results
cd report
make
```

当前数据包含 182 条 performance measurements、256 条 attention
correctness measurements、12 条 activation-checkpointing measurements，
以及本地/远端三套测试环境记录。所有图表均由脚本生成，未手工录入数值。

