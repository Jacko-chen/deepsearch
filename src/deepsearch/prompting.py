from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def prompt_path(name: str) -> Path:
    """Resolve a prompt from the source tree or an installed package."""
    filename = name if name.endswith(".txt") else f"{name}.txt"
    candidates = []
    if configured := os.getenv("DEEPSEARCH_PROMPT_DIR"):
        candidates.append(Path(configured) / filename)
    candidates.extend(
        [
            Path(__file__).resolve().parents[2] / "prompts" / filename,
            Path(sys.prefix) / "share" / "deepsearch" / "prompts" / filename,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Prompt {filename!r} was not found. Searched: {searched}")


def load_prompt(name: str) -> str:
    return prompt_path(name).read_text(encoding="utf-8").strip()


def render_prompt(name: str, **values: Any) -> str:
    text = load_prompt(name)
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text
