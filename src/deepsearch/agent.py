from __future__ import annotations

from dataclasses import dataclass

from deepsearch.backends.base import RetrievalBackend
from deepsearch.filters import PaperFilter, filter_candidates
from deepsearch.selectors import ActionSelector
from deepsearch.types import Action, Paper


@dataclass(slots=True)
class AgentConfig:
    max_steps: int = 6
    retrieval_limit: int = 50
    max_accepted_per_step: int = 50
    filter_threshold: float = 0.08


class DeepSearchAgent:
    """Inference pipeline that never accesses survey reference targets."""

    def __init__(
        self,
        backend: RetrievalBackend,
        paper_filter: PaperFilter,
        selector: ActionSelector,
        config: AgentConfig | None = None,
    ):
        self.backend = backend
        self.paper_filter = paper_filter
        self.selector = selector
        self.config = config or AgentConfig()

    def retrieve(self, topic: str) -> dict:
        collection: dict[str, Paper] = {}
        action_history: list[Action] = []
        used_keywords: list[str] = []
        expanded_seed_ids: set[str] = set()
        trace: list[dict] = []

        for step in range(1, self.config.max_steps + 1):
            unused_seed_ids = [paper_id for paper_id in collection if paper_id not in expanded_seed_ids]
            action, tool_input = self.selector.select(
                topic=topic,
                collection=list(collection.values()),
                action_history=action_history,
                used_keywords=used_keywords,
                step=step,
                max_steps=self.config.max_steps,
                unused_seed_ids=unused_seed_ids,
            )
            if action == Action.STOP:
                trace.append({"step": step, "action": action.value, "accepted_ids": []})
                break
            if action == Action.SEARCH:
                candidates = self.backend.search(tool_input or [topic], limit=self.config.retrieval_limit)
                used_keywords.extend(tool_input)
            else:
                seeds = tool_input or unused_seed_ids
                candidates = self.backend.expand_citations(seeds, limit=self.config.retrieval_limit)
                expanded_seed_ids.update(seeds)
                tool_input = list(seeds)
            accepted = filter_candidates(
                topic,
                candidates,
                self.paper_filter,
                existing_ids=collection,
                max_papers=self.config.max_accepted_per_step,
                threshold=self.config.filter_threshold,
            )
            for paper in accepted:
                collection[paper.id] = paper
            action_history.append(action)
            trace.append(
                {
                    "step": step,
                    "action": action.value,
                    "tool_input": tool_input,
                    "candidate_ids": [paper.id for paper in candidates],
                    "accepted_ids": [paper.id for paper in accepted],
                    "collection_ids": list(collection),
                }
            )
            if not accepted and step > 1:
                break

        return {
            "topic": topic,
            "papers": [paper.to_dict() for paper in collection.values()],
            "trace": trace,
        }
