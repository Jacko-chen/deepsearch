from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Protocol

from deepsearch.types import Paper


class PaperFilter(Protocol):
    def score(self, topic: str, paper: Paper) -> float:
        """Return a paper-suitability score in [0, 1]."""


class HeuristicPaperFilter:
    """Small deterministic filter used by the offline example.

    The released checkpoints can replace this class without changing the
    retrieval pipeline.
    """

    def __init__(self, *, threshold: float = 0.18):
        self.threshold = threshold

    def score(self, topic: str, paper: Paper) -> float:
        topic_tokens = _tokens(topic)
        paper_tokens = _tokens(" ".join([paper.title, paper.abstract, *paper.keywords]))
        overlap = len(topic_tokens.intersection(paper_tokens)) / max(1, len(topic_tokens))
        influence = min(1.0, math.log1p(max(0, paper.citation_count)) / math.log(1001))
        return min(1.0, 0.85 * overlap + 0.15 * influence)


def filter_candidates(
    topic: str,
    candidates: Sequence[Paper],
    paper_filter: PaperFilter,
    *,
    existing_ids: Iterable[str] = (),
    reference_ids: Iterable[str] = (),
    max_papers: int = 50,
    threshold: float | None = None,
) -> list[Paper]:
    existing = set(existing_ids)
    references = set(reference_ids)
    scored: list[tuple[float, Paper]] = []
    cutoff = threshold if threshold is not None else getattr(paper_filter, "threshold", 0.5)
    for paper in candidates:
        if paper.id in existing:
            continue
        score = 1.0 if paper.id in references else float(paper_filter.score(topic, paper))
        paper.filter_score = score
        if score >= cutoff:
            scored.append((score, paper))
    scored.sort(key=lambda item: (-item[0], -item[1].citation_count, item[1].id))
    return [paper for _, paper in scored[:max_papers]]


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 1
    }

