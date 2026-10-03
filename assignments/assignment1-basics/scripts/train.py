from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs336_basics.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CS336 Transformer language model")
    parser.add_argument("--config", required=True, help="Path to a TrainConfig JSON file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config value")
    args = parser.parse_args()

    values = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for override in args.set:
        key, raw_value = override.split("=", 1)
        try:
            values[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            values[key] = raw_value
    summary = train(TrainConfig(**values))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

