from __future__ import annotations

import argparse
import itertools
import json
import traceback
from pathlib import Path

from cs336_basics.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible, failure-isolated experiment sweep")
    parser.add_argument("--base-config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sweep", type=Path, help="JSON mapping config keys to value lists")
    group.add_argument("--manifest", type=Path, help="JSON list of named, explicit config overrides")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    if args.manifest:
        experiments = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
        keys = list(sweep)
        experiments = [
            {"name": f"grid-{index:03d}", "overrides": dict(zip(keys, values))}
            for index, values in enumerate(itertools.product(*(sweep[key] for key in keys)))
        ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "sweep-status.jsonl"
    for run_index, experiment in enumerate(experiments):
        overrides = experiment.get("overrides", {})
        name = experiment.get("name", f"run-{run_index:03d}")
        config = base.copy()
        config.update(overrides)
        config["output_dir"] = str(args.output_root / f"{run_index:03d}-{name}")
        event: dict[str, object] = {"run": run_index, "name": name, "overrides": overrides}
        print(json.dumps(event), flush=True)
        try:
            event["summary"] = train(TrainConfig(**config))
            event["status"] = "completed"
        except Exception as error:
            event["status"] = "failed"
            event["error"] = repr(error)
            (Path(config["output_dir"]) / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        with status_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(event) + "\n")


if __name__ == "__main__":
    main()

