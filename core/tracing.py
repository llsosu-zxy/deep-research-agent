from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TraceLogger:
    """Append-only JSONL trace store with budget accounting."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


class BudgetTracker:
    def __init__(self, daily_limit_usd: float = 0.50) -> None:
        self.daily_limit_usd = daily_limit_usd
        self.spent_usd = 0.0

    def add(self, prompt_tokens: int, completion_tokens: int, price_per_1m: float = 0.3) -> None:
        self.spent_usd += (prompt_tokens + completion_tokens) / 1_000_000 * price_per_1m

    def remaining_usd(self) -> float:
        return max(0.0, self.daily_limit_usd - self.spent_usd)

    def over_budget(self) -> bool:
        return self.spent_usd >= self.daily_limit_usd
