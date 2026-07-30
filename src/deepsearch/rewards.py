from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Sequence

from deepsearch.types import Action


@dataclass(slots=True)
class ParsedAction:
    action: Action | None
    keywords: list[str] = field(default_factory=list)
    format_penalty: float = 0.0
    violations: list[str] = field(default_factory=list)


def parse_action_with_penalty(
    output: str,
    *,
    current_step: int,
    max_steps: int,
    has_papers: bool,
    used_keywords: Sequence[str] = (),
    min_keywords: int = 10,
    max_keywords: int = 15,
) -> ParsedAction:
    text = str(output or "").strip()
    if not text:
        return ParsedAction(None, format_penalty=-1.0, violations=["empty_output"])

    penalty = 0.0
    violations: list[str] = []
    think_pattern = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
    while (match := think_pattern.match(text)) is not None:
        text = text[match.end() :].strip()
        penalty -= 0.10
        violations.append("leading_reasoning_block")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ParsedAction(None, format_penalty=-1.0, violations=violations + ["empty_output"])

    action = _parse_action(lines[0])
    body = "\n".join(lines[1:]).strip()
    if action is None:
        return ParsedAction(None, format_penalty=-1.0, violations=["invalid_first_line"])
    if current_step >= max_steps and action != Action.STOP:
        return ParsedAction(
            None,
            format_penalty=_clip(penalty - 0.5),
            violations=violations + ["step_limit_requires_stop"],
        )
    if not has_papers and action != Action.SEARCH:
        return ParsedAction(
            None,
            format_penalty=_clip(penalty - 0.8),
            violations=violations + ["empty_collection_requires_search"],
        )
    if action in {Action.CITATION, Action.STOP}:
        if body:
            penalty -= 0.45
            violations.append("unexpected_extra_text")
        return ParsedAction(action, format_penalty=_clip(penalty), violations=violations)

    if not body:
        return ParsedAction(
            Action.SEARCH,
            format_penalty=_clip(penalty - 0.7),
            violations=violations + ["missing_keyword_list"],
        )
    try:
        raw_keywords = ast.literal_eval(body)
    except (SyntaxError, ValueError):
        return ParsedAction(
            Action.SEARCH,
            format_penalty=_clip(penalty - 0.7),
            violations=violations + ["unparseable_keyword_list"],
        )
    if not isinstance(raw_keywords, list):
        return ParsedAction(
            Action.SEARCH,
            format_penalty=_clip(penalty - 0.7),
            violations=violations + ["keyword_list_not_a_list"],
        )
    keywords = [str(item).strip() for item in raw_keywords if str(item).strip()]
    if len(keywords) < min_keywords:
        penalty -= min(0.75, 0.20 + 0.05 * (min_keywords - len(keywords)))
        violations.append("insufficient_keywords")
    if len(keywords) > max_keywords:
        penalty -= min(0.75, 0.20 + 0.05 * (len(keywords) - max_keywords))
        violations.append("too_many_keywords")
        keywords = keywords[:max_keywords]
    deduped = _dedupe(keywords)
    if len(deduped) != len(keywords):
        penalty -= 0.15
        violations.append("duplicate_keywords")
        keywords = deduped
    if any(re.match(r"^\s*\(?\d+\)?[.)]\s+", keyword) for keyword in keywords):
        penalty -= 0.20
        violations.append("numbered_keywords")
    if set(keyword.casefold() for keyword in keywords).intersection(
        item.casefold() for item in used_keywords
    ):
        penalty -= 0.20
        violations.append("reused_keywords")
    return ParsedAction(Action.SEARCH, keywords, _clip(penalty), violations)


def action_penalty(history: Sequence[Action], action: Action, *, repeat_limit: int = 4) -> float:
    """Return -0.1 when an action completes four identical retrieval actions."""
    if action not in {Action.SEARCH, Action.CITATION}:
        return 0.0
    recent = [*history, action]
    return -0.10 if len(recent) >= repeat_limit and len(set(recent[-repeat_limit:])) == 1 else 0.0


def trajectory_reward(
    final_collection_reward: float,
    format_penalties: Sequence[float],
    action_penalties: Sequence[float],
    *,
    gamma: float = 0.3,
) -> float:
    return final_collection_reward + gamma * sum(format_penalties) + sum(action_penalties)


def collection_reward(
    section_coverage: float,
    reference_coverage: float,
    new_paper_quality: float,
    *,
    coverage_weight: float = 0.4,
    quality_weight: float = 0.2,
) -> float:
    """Reward a final set using structural, reference, and filter signals."""
    return (
        coverage_weight * section_coverage
        + coverage_weight * reference_coverage
        + quality_weight * new_paper_quality
    )


def _parse_action(value: str) -> Action | None:
    normalized = value.strip().casefold()
    if normalized == "search":
        return Action.SEARCH
    if normalized == "citation":
        return Action.CITATION
    if normalized in {"stop", "end"}:
        return Action.STOP
    return None


def _clip(value: float) -> float:
    return max(-1.0, min(0.0, value))


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
