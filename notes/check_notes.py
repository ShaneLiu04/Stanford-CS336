"""Validate completeness and local links of the 17-lecture note set."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = (
    ("学习目标", "## 目标"),
    ("实现", "代码"),
    ("易错", "反例", "误区"),
    ("练习", "实践", "Checklist"),
    ("小结",),
    ("延伸阅读", "阅读"),
)


def main() -> None:
    notes = sorted(ROOT.glob("lecture-*.md"))
    errors = []
    total_characters = 0
    if len(notes) != 17:
        errors.append(f"expected 17 lecture notes, found {len(notes)}")
    numbers = []
    for path in notes:
        match = re.match(r"lecture-(\d{2})-", path.name)
        if not match:
            errors.append(f"invalid filename: {path.name}")
            continue
        numbers.append(int(match.group(1)))
        text = path.read_text(encoding="utf-8")
        total_characters += len(text)
        missing = [
            "/".join(group)
            for group in REQUIRED
            if not any(term in text for term in group)
        ]
        if missing:
            errors.append(f"{path.name}: missing sections {missing}")
        if len(text) < 3500:
            errors.append(f"{path.name}: too short ({len(text)} characters)")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                errors.append(f"{path.name}: broken link {target}")
    if numbers != list(range(1, 18)):
        errors.append(f"lecture numbers are not 01..17: {numbers}")
    if errors:
        raise SystemExit("\n".join(errors))
    (ROOT / "VALIDATION.md").write_text(
        "\n".join(
            [
                "# Notes validation",
                "",
                f"- Lecture files: {len(notes)}",
                f"- Total characters: {total_characters:,}",
                "- Numbering: 01–17 complete",
                "- Required sections: complete",
                "- Local links: valid",
                "- Sensitive data: checked separately before commit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"validated {len(notes)} lecture notes")


if __name__ == "__main__":
    main()
