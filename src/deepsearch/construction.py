from __future__ import annotations

from dataclasses import dataclass

from deepsearch.backends.base import RetrievalBackend
from deepsearch.filters import PaperFilter, filter_candidates
from deepsearch.metrics import reference_metrics, section_coverage
from deepsearch.selectors import generate_keywords
from deepsearch.types import Action, BranchResult, Paper, TopicRecord, Trajectory, TrajectoryStep


@dataclass(slots=True)
class ConstructionConfig:
    max_steps: int = 5
    retrieval_limit: int = 50
    max_accepted_per_step: int = 50
    filter_threshold: float = 0.08


class TrajectoryConstructor:
    """Section-first compare-then-update trajectory construction."""

    def __init__(
        self,
        backend: RetrievalBackend,
        paper_filter: PaperFilter,
        config: ConstructionConfig | None = None,
    ):
        self.backend = backend
        self.paper_filter = paper_filter
        self.config = config or ConstructionConfig()

    def construct(self, topic: TopicRecord) -> Trajectory:
        collection: dict[str, Paper] = {}
        steps: list[TrajectoryStep] = []
        used_keywords: list[str] = []
        expanded_seed_ids: set[str] = set()

        for step_index in range(1, self.config.max_steps + 1):
            state = {
                "topic": topic.topic,
                "current_step": step_index,
                "max_steps": self.config.max_steps,
                "collection_ids": list(collection),
                "used_keywords": list(used_keywords),
                "previous_actions": [step.action.value for step in steps],
            }
            section_names = self._uncovered_sections(topic, collection)
            keywords = generate_keywords(
                topic.topic,
                section_names=section_names,
                used_keywords=used_keywords,
            )
            search_branch = self._search_branch(topic, collection, keywords)
            branches = [search_branch]

            if collection:
                seed_ids = [paper_id for paper_id in collection if paper_id not in expanded_seed_ids]
                if seed_ids:
                    branches.append(self._citation_branch(topic, collection, seed_ids))

            selected = max(
                branches,
                key=lambda branch: (branch.section_coverage, branch.f1),
            )
            if not selected.accepted:
                break

            for paper in selected.accepted:
                collection[paper.id] = paper
            if selected.action == Action.SEARCH:
                used_keywords.extend(selected.tool_input)
            else:
                expanded_seed_ids.update(selected.tool_input)

            reward = selected.section_coverage + selected.f1
            steps.append(
                TrajectoryStep(
                    step=step_index,
                    state=state,
                    action=selected.action,
                    tool_input=selected.tool_input,
                    candidates=selected.candidates,
                    accepted=selected.accepted,
                    collection=list(collection.values()),
                    reward=reward,
                    branch_results=branches,
                )
            )

        return Trajectory(
            topic_id=topic.topic_id,
            topic=topic.topic,
            steps=steps,
            final_collection=list(collection.values()),
        )

    def _search_branch(
        self,
        topic: TopicRecord,
        collection: dict[str, Paper],
        keywords: list[str],
    ) -> BranchResult:
        candidates = self.backend.search(keywords or [topic.topic], limit=self.config.retrieval_limit)
        return self._branch(topic, collection, Action.SEARCH, keywords or [topic.topic], candidates)

    def _citation_branch(
        self,
        topic: TopicRecord,
        collection: dict[str, Paper],
        seed_ids: list[str],
    ) -> BranchResult:
        candidates = self.backend.expand_citations(seed_ids, limit=self.config.retrieval_limit)
        return self._branch(topic, collection, Action.CITATION, seed_ids, candidates)

    def _branch(
        self,
        topic: TopicRecord,
        collection: dict[str, Paper],
        action: Action,
        tool_input: list[str],
        candidates: list[Paper],
    ) -> BranchResult:
        accepted = filter_candidates(
            topic.topic,
            candidates,
            self.paper_filter,
            existing_ids=collection,
            reference_ids=topic.target_references,
            max_papers=self.config.max_accepted_per_step,
            threshold=self.config.filter_threshold,
        )
        merged_ids = set(collection).union(paper.id for paper in accepted)
        metrics = reference_metrics(merged_ids, topic.target_references)
        return BranchResult(
            action=action,
            tool_input=tool_input,
            candidates=candidates,
            accepted=accepted,
            section_coverage=section_coverage(merged_ids, topic.section_targets),
            f1=metrics["f1"],
        )

    @staticmethod
    def _uncovered_sections(topic: TopicRecord, collection: dict[str, Paper]) -> list[str]:
        current = set(collection)
        return [
            name
            for name, targets in topic.section_targets.items()
            if targets and not current.intersection(targets)
        ]
