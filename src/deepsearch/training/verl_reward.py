"""VERL-compatible reward entry point.

Point ``custom_reward_function.path`` to this file and set
``custom_reward_function.name=compute_score``.
"""

from __future__ import annotations

from typing import Any

from deepsearch.rewards import action_penalty, parse_action_with_penalty, trajectory_reward
from deepsearch.types import Action


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = extra_info or {}
    step_logs = info.get("step_logs") or []
    if step_logs:
        final_collection_reward = 0.0
        format_penalties: list[float] = []
        action_penalties: list[float] = []
        for step_log in step_logs:
            breakdown = step_log.get("reward_breakdown") or {}
            if "reward_part" in breakdown:
                final_collection_reward = float(breakdown["reward_part"])
            format_penalties.append(
                float(step_log.get("format_penalty", breakdown.get("format_penalty", 0.0)))
            )
            action_penalties.append(float(breakdown.get("action_shaping_reward", 0.0)))
        score = trajectory_reward(
            final_collection_reward,
            format_penalties,
            action_penalties,
            gamma=float(info.get("format_penalty_weight", 0.3)),
        )
        return {
            "score": score,
            "final_collection_reward": final_collection_reward,
            "trajectory_format_penalty": sum(format_penalties),
            "total_action_shaping_reward": sum(action_penalties),
        }

    history = [
        Action(str(value).casefold())
        for value in info.get("action_history", [])
        if str(value).casefold() in {action.value for action in Action}
    ]
    parsed = parse_action_with_penalty(
        solution_str,
        current_step=int(info.get("current_step", 1)),
        max_steps=int(info.get("max_steps", 6)),
        has_papers=bool(info.get("has_papers", False)),
        used_keywords=info.get("used_keywords", []),
    )
    repeated = action_penalty(history, parsed.action) if parsed.action else 0.0
    final_collection_reward = float(info.get("final_collection_reward", 0.0))
    score = trajectory_reward(
        final_collection_reward,
        [parsed.format_penalty],
        [repeated],
        gamma=float(info.get("format_penalty_weight", 0.3)),
    )
    return {
        "score": score,
        "final_collection_reward": final_collection_reward,
        "trajectory_format_penalty": parsed.format_penalty,
        "total_action_shaping_reward": repeated,
    }
