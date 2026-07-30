from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepsearch.types import Paper, TopicRecord


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_papers(path: str | Path) -> list[Paper]:
    value = read_json(path)
    rows = value.get("papers", []) if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("Paper input must be a list or an object containing `papers`.")
    return [Paper.from_dict(row) for row in rows]


def load_topic(path: str | Path) -> TopicRecord:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Topic input must be one JSON object.")
    return TopicRecord.from_dict(value)

