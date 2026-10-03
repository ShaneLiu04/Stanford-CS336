# Assignment 3 report artifacts

`results/raw/` 保存官方 IsoFLOP provenance 与 22 个 RTX 6000D proxy runs；
`results/figures/` 同时提供 PDF/PNG；`summary.json`、experiment logs 和
`manifest.sha256` 提供机器可读索引与校验。

```bash
uv run python scripts/analyze_scaling.py
uv run python scripts/analyze_proxy.py
cd report && make
```

该报告是 **非课程提交**的自学记录，未调用 Stanford B200 API。
作者：[ShaneLiu04](https://github.com/ShaneLiu04)。
