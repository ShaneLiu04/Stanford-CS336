# Assignment 1 report artifacts

```text
report/
├── main.tex / main.pdf       # 中文 LaTeX 报告
├── problem_walkthrough.tex   # 逐题索引
├── figures/                  # PNG（README）与 PDF（LaTeX）
└── results/
    ├── raw/                  # 每个 run 的 config/environment/metrics/summary
    ├── tables/               # 自动生成的 LaTeX 表格
    ├── summary.json          # 机器可读 headline 结果
    ├── experiment_log.csv    # 全部 run 的扁平索引
    └── manifest.sha256       # 报告资产校验清单
```

重建全部图表、表格和摘要：

```bash
uv run python scripts/build_report_assets.py \
  --raw report/results/raw \
  --figures report/figures \
  --summary report/results/summary.json
cd report && make
```

当前归档包含 75 个训练 runs、27 组 PNG/PDF figures、200-document tokenizer
bootstrap、4-checkpoint generation panel，以及 RTX 4080 SUPER / RTX 6000D 两代
硬件记录。跨硬件绝对吞吐不直接比较；算法结论均来自同机 controlled comparisons。
