from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from typing import Protocol

from deepsearch.types import Action, Paper


class ActionSelector(Protocol):
    def select(
        self,
        *,
        topic: str,
        collection: Sequence[Paper],
        action_history: Sequence[Action],
        used_keywords: Sequence[str],
        step: int,
        max_steps: int,
        unused_seed_ids: Sequence[str],
    ) -> tuple[Action, list[str]]:
        """Choose an action and optional search keywords."""


class HeuristicSelector:
    """Deterministic replacement for the learned selector in the demo."""

    def select(
        self,
        *,
        topic: str,
        collection: Sequence[Paper],
        action_history: Sequence[Action],
        used_keywords: Sequence[str],
        step: int,
        max_steps: int,
        unused_seed_ids: Sequence[str],
    ) -> tuple[Action, list[str]]:
        if step > max_steps:
            return Action.STOP, []
        if not collection:
            return Action.SEARCH, generate_keywords(topic, used_keywords=used_keywords)
        if unused_seed_ids and (not action_history or action_history[-1] != Action.CITATION):
            return Action.CITATION, []
        keywords = generate_keywords(topic, used_keywords=used_keywords)
        if keywords:
            return Action.SEARCH, keywords
        return Action.STOP, []


def generate_keywords(
    topic: str,
    *,
    section_names: Sequence[str] = (),
    used_keywords: Sequence[str] = (),
    limit: int = 10,
) -> list[str]:
    used = {item.casefold() for item in used_keywords}
    phrases = [topic, *section_names]
    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]*", " ".join(phrases))
        if len(token) > 2
    ]
    candidates = [topic, *section_names, *tokens]
    result: list[str] = []
    seen = set()
    for value in candidates:
        key = value.strip().casefold()
        if not key or key in seen or key in used:
            continue
        seen.add(key)
        result.append(value.strip())
        if len(result) >= limit:
            break
    return result


def parse_selector_output(text: str) -> tuple[Action | None, list[str]]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return None, []
    first = lines[0].casefold()
    if first in {"end", "stop"}:
        return Action.STOP, []
    if first == "citation":
        return Action.CITATION, []
    if first != "search":
        return None, []
    if len(lines) < 2:
        return Action.SEARCH, []
    try:
        values = ast.literal_eval("\n".join(lines[1:]))
    except (SyntaxError, ValueError):
        return Action.SEARCH, []
    if not isinstance(values, list):
        return Action.SEARCH, []
    return Action.SEARCH, [str(value).strip() for value in values if str(value).strip()]

