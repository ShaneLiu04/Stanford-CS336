# Official upstream snapshots

下列目录是 Stanford CS336 官方作业仓库的无 Git 历史快照。快照通过 GitHub API 按 commit SHA 下载，便于复现且不会形成嵌套仓库。同步日期：2026-09-26。

| 版本 | 作业 | 上游仓库 | 固定 commit |
| --- | --- | --- | --- |
| Spring 2025 | Assignment 1: Basics | https://github.com/stanford-cs336/assignment1-basics | `430e2c844e29f8aad8f9330e8706db9cb508241f` |
| Spring 2025 | Assignment 2: Systems | https://github.com/stanford-cs336/assignment2-systems | `e495ed740080661ab084914674d3e10048778889` |
| Spring 2025 | Assignment 3: Scaling | https://github.com/stanford-cs336/assignment3-scaling | `09d205bde59e5c533368c0209dc86a4d5e4323ea` |
| Spring 2025 | Assignment 4: Data | https://github.com/stanford-cs336/assignment4-data | `5a5f890cd9b72733e6273a596c40b696df3aa9df` |
| Spring 2025 | Assignment 5: Alignment | https://github.com/stanford-cs336/assignment5-alignment | `a88071d1272112306f59257d946e43b56bb31773` |
| Spring 2026 | Assignment 1: Basics | https://github.com/stanford-cs336/assignment1-basics | `a158843b20107949f1a8d7df1b05cd33b9166712` |
| Spring 2026 | Assignment 2: Systems | https://github.com/stanford-cs336/assignment2-systems | `ca8bc81a59b70516f7ebb2da4808daade877c736` |
| Spring 2026 | Assignment 3: Scaling | https://github.com/stanford-cs336/assignment3-scaling | `03e9372992e913061b9e78b5cfcb62ad8a87de35` |
| Spring 2026 | Assignment 4: Data | https://github.com/stanford-cs336/assignment4-data | `0555bea66369872d912652debf10b115ca0688c8` |
| Spring 2026 | Assignment 5: Alignment | https://github.com/stanford-cs336/assignment5-alignment | `c2734a26308710949fe13226960a1e8cece94b7e`（仅索引，未导入） |

## 版本选择说明

官方仓库会将默认分支更新到最新课程。Spring 2025 行选取的是课程结束后、2026 改版前的最终修订 commit；Spring 2026 行固定的是同步当日默认分支 commit。Assignment 5 的 2025 快照来自当时保留的 `master` 分支。

每个快照内的 `LICENSE` 优先于根目录许可证。不要删除原始 README、PDF 或版权信息。Spring 2026 Assignment 5 在检索 commit 中没有许可证，因此源码未导入；无许可证的公开代码不能推定为允许再发布。

## 更新

固定 SHA 默认不会自动漂移。如需有意识地更新快照：

1. 在 GitHub 核对课程版本和 commit。
2. 更新 `scripts/sync-upstream.ps1` 与本文件中的 SHA。
3. 运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync-upstream.ps1 -Force
```

4. 审阅差异，尤其是 handout、测试适配器和许可证。
