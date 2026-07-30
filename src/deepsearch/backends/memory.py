from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from deepsearch.backends.base import RetrievalBackend
from deepsearch.types import Paper


class InMemoryBackend(RetrievalBackend):
    """Deterministic backend used for examples and unit tests."""

    def __init__(self, papers: Iterable[Paper]):
        self.papers = {paper.id: paper for paper in papers}

    def search(self, queries: Sequence[str], *, limit: int = 50) -> list[Paper]:
        query_tokens = set()
        for query in queries:
            query_tokens.update(_tokens(query))
        scored: list[tuple[float, Paper]] = []
        for paper in self.papers.values():
            haystack = _tokens(" ".join([paper.title, paper.abstract, *paper.keywords]))
            overlap = len(query_tokens.intersection(haystack))
            if overlap == 0:
                continue
            score = overlap / max(1, len(query_tokens)) + min(paper.citation_count, 1000) / 100_000
            scored.append((score, paper))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [paper for _, paper in scored[:limit]]

    def expand_citations(self, seed_ids: Sequence[str], *, limit: int = 50) -> list[Paper]:
        related: list[str] = []
        for seed_id in seed_ids:
            paper = self.papers.get(seed_id)
            if paper is None:
                continue
            related.extend(paper.references)
            related.extend(paper.cited_by)
        result: list[Paper] = []
        seen = set(seed_ids)
        for paper_id in related:
            if paper_id in seen or paper_id not in self.papers:
                continue
            seen.add(paper_id)
            result.append(self.papers[paper_id])
            if len(result) >= limit:
                break
        return result


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 1
    }

