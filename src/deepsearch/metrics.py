from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from deepsearch.types import Paper, TopicRecord


def reference_metrics(retrieved_ids: Iterable[str], target_ids: Iterable[str]) -> dict[str, float]:
    retrieved = set(retrieved_ids)
    target = set(target_ids)
    hits = len(retrieved.intersection(target))
    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(target) if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def section_coverage(retrieved_ids: Iterable[str], section_targets: Mapping[str, Sequence[str]]) -> float:
    retrieved = set(retrieved_ids)
    valid_sections = [set(ids) for ids in section_targets.values() if ids]
    if not valid_sections:
        return 0.0
    return sum(len(retrieved.intersection(ids)) / len(ids) for ids in valid_sections) / len(valid_sections)


def canonical_influence_coverage(
    retrieved_ids: Iterable[str],
    target_ids: Sequence[str],
    paper_catalog: Mapping[str, Paper],
    *,
    fraction: float = 0.20,
) -> float:
    if not target_ids:
        return 0.0
    ranked = sorted(
        set(target_ids),
        key=lambda paper_id: paper_catalog.get(paper_id, Paper(paper_id, "")).citation_count,
        reverse=True,
    )
    canonical_count = max(1, math.ceil(len(ranked) * fraction))
    canonical = set(ranked[:canonical_count])
    return len(set(retrieved_ids).intersection(canonical)) / len(canonical)


def temporal_alignment(
    papers: Sequence[Paper],
    survey_year: int | None,
    *,
    alpha: float = 0.1,
) -> float:
    if not papers or survey_year is None:
        return 0.0
    scores = []
    for paper in papers:
        if paper.year is None or paper.year > survey_year:
            scores.append(0.0)
        else:
            scores.append(math.exp(-alpha * abs(paper.year - survey_year)))
    return sum(scores) / len(scores)


def evaluate_collection(
    papers: Sequence[Paper],
    topic: TopicRecord,
    *,
    paper_catalog: Mapping[str, Paper] | None = None,
    temporal_alpha: float = 0.1,
) -> dict[str, float]:
    retrieved_ids = [paper.id for paper in papers]
    result = reference_metrics(retrieved_ids, topic.target_references)
    category = section_coverage(retrieved_ids, topic.section_targets)
    catalog = dict(paper_catalog or {})
    catalog.update({paper.id: paper for paper in papers})
    cic = canonical_influence_coverage(retrieved_ids, topic.target_references, catalog)
    tas = temporal_alignment(papers, topic.survey_year, alpha=temporal_alpha)
    result.update(
        {
            "cic": cic,
            "tas": tas,
            "category": category,
            "cscore": (cic + tas + category) / 3,
        }
    )
    return result

