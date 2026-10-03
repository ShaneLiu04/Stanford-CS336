# Assignment 5 report artifacts

`results/raw/` 保存 4480 条 RTX 6000D proxy metrics；`results/figures/`
包含 15 组 PDF/PNG；`summary.json`、experiment log 与 manifest 提供机器可读索引。

```bash
python scripts/run_proxy_alignment.py
python scripts/build_report_assets.py
cd report && make
```

这是 **非课程提交**的自学报告，未使用 SUNET_ID/Modal，
不声称官方 OLMo-2/B200 结果。作者：
[ShaneLiu04](https://github.com/ShaneLiu04)。
