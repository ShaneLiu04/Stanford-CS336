"""Single entry point for rebuilding all Assignment 3 report assets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts/analyze_scaling.py")], cwd=root, check=True)
    subprocess.run([sys.executable, str(root / "scripts/analyze_proxy.py")], cwd=root, check=True)
    figures = list((root / "report/results/figures").glob("*.png"))
    if len(figures) < 15:
        raise RuntimeError(f"expected at least 15 report figures, found {len(figures)}")
    print(f"rebuilt {len(figures)} report figures")


if __name__ == "__main__":
    main()
