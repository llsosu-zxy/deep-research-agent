from __future__ import annotations

import re

from core.models import Source

CITATION_RE = re.compile(r"\[(\d{1,3})\]")


def extract_citations(text: str) -> list[int]:
    return [int(match) for match in CITATION_RE.findall(text)]


def validate_citations(text: str, sources: list[Source]) -> tuple[bool, list[str]]:
    cited = extract_citations(text)
    if not cited:
        return False, ["no citations found"]
    issues: list[str] = []
    for number in cited:
        if number < 1 or number > len(sources):
            issues.append(f"citation [{number}] out of range")
    for idx in range(1, len(sources) + 1):
        if idx not in cited:
            issues.append(f"source [{idx}] never cited")
    return not issues, issues
