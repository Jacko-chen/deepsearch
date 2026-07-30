from __future__ import annotations

import json
import random
from collections.abc import Iterable
from typing import Any

from deepsearch.prompting import load_prompt, render_prompt


def build_filter_examples(
    candidates: Iterable[dict[str, Any]],
    *,
    positive_repeat: int = 3,
) -> list[dict[str, Any]]:
    """Apply the asymmetric multi-LLM rule described in the paper."""
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        in_reference = bool(candidate.get("in_reference"))
        raw_votes = candidate.get("judge_votes") or []
        votes = [str(vote).strip().casefold() for vote in raw_votes if str(vote).strip()]
        yes_count = sum(vote in {"yes", "true", "1"} for vote in votes)
        no_count = sum(vote in {"no", "false", "0"} for vote in votes)
        if in_reference:
            label, source = "yes", "reference_positive"
        elif len(votes) >= 3 and yes_count == len(votes):
            label, source = "yes", "unanimous_supplementary_positive"
        elif no_count >= 2:
            label, source = "no", "majority_verified_negative"
        else:
            continue
        paper = dict(candidate.get("paper") or {})
        paper.pop("sim_cal", None)
        paper.pop("similarity_score", None)
        prompt = filter_prompt(str(candidate.get("topic", "")), paper)
        row = {
            "topic_id": candidate.get("topic_id"),
            "paper_id": paper.get("id") or paper.get("_id"),
            "label": label,
            "label_source": source,
            "judge_votes": votes,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": label},
            ],
        }
        rows.extend([row] * (positive_repeat if label == "yes" else 1))
    return rows


def split_by_group(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    validation_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = sorted({str(row.get(group_key, "")) for row in rows})
    random.Random(seed).shuffle(groups)
    validation_count = max(1, round(len(groups) * validation_ratio)) if groups else 0
    validation_groups = set(groups[:validation_count])
    train = [row for row in rows if str(row.get(group_key, "")) not in validation_groups]
    validation = [row for row in rows if str(row.get(group_key, "")) in validation_groups]
    return train, validation


def selector_examples_from_trajectory(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    topic = str(trajectory.get("topic", ""))
    trajectory_id = str(trajectory.get("trajectory_id") or trajectory.get("topic_id") or "")
    rows: list[dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        action = str(step.get("action", "")).casefold()
        if action not in {"search", "citation", "stop", "end"}:
            continue
        state = step.get("state") or {}
        target = "END" if action in {"stop", "end"} else action
        if action == "search":
            target += "\n" + json.dumps(step.get("tool_input") or [], ensure_ascii=False)
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "topic_id": trajectory.get("topic_id"),
                "step": step.get("step"),
                "messages": [
                    {"role": "system", "content": selector_system_prompt()},
                    {"role": "user", "content": selector_state_prompt(topic, state)},
                    {"role": "assistant", "content": target},
                ],
            }
        )
    return rows


def filter_prompt(topic: str, paper: dict[str, Any]) -> str:
    return render_prompt(
        "filter_verification",
        SURVEY_TOPIC=topic,
        TITLE=paper.get("title", ""),
        ABSTRACT=paper.get("abstract", ""),
        AUTHORS=json.dumps(paper.get("authors", []), ensure_ascii=False),
        VENUE=paper.get("venue", ""),
        YEAR=paper.get("year", ""),
        CITATION_COUNT=paper.get("citation_count", paper.get("n_citation", "")),
        KEYWORDS=json.dumps(paper.get("keywords", []), ensure_ascii=False),
    )


def selector_system_prompt() -> str:
    return load_prompt("selector_system")


def selector_state_prompt(topic: str, state: dict[str, Any]) -> str:
    papers = state.get("papers") or state.get("collection_ids", [])
    return render_prompt(
        "selector",
        TOPIC=topic,
        ACTION_HISTORY=json.dumps(state.get("previous_actions", []), ensure_ascii=False),
        USED_KEYWORDS=json.dumps(state.get("used_keywords", []), ensure_ascii=False),
        PAPER_TITLES=json.dumps(papers, ensure_ascii=False),
        CURRENT_STEP=state.get("current_step", ""),
        MAX_STEPS=state.get("max_steps", ""),
    )
