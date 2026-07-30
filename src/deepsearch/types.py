from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Action(str, Enum):
    SEARCH = "search"
    CITATION = "citation"
    STOP = "stop"


@dataclass(slots=True)
class Paper:
    id: str
    title: str
    abstract: str = ""
    year: int | None = None
    citation_count: int = 0
    venue: str = ""
    keywords: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    cited_by: list[str] = field(default_factory=list)
    filter_score: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Paper":
        paper_id = value.get("id") or value.get("_id")
        if not paper_id:
            raise ValueError("A paper record must contain `id` or `_id`.")
        venue = value.get("venue", "")
        if isinstance(venue, dict):
            venue = venue.get("raw", "")
        return cls(
            id=str(paper_id),
            title=str(value.get("title") or value.get("title_zh") or ""),
            abstract=str(value.get("abstract") or value.get("abstract_zh") or ""),
            year=_optional_int(value.get("year")),
            citation_count=_optional_int(value.get("citation_count", value.get("n_citation"))) or 0,
            venue=str(venue or ""),
            keywords=[str(item) for item in value.get("keywords", []) or []],
            authors=[
                str(item.get("name", "")) if isinstance(item, dict) else str(item)
                for item in value.get("authors", []) or []
                if item
            ],
            references=[str(item) for item in value.get("references", []) or []],
            cited_by=[str(item) for item in value.get("cited_by", []) or []],
            filter_score=_optional_float(value.get("filter_score", value.get("paper_score"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TopicRecord:
    topic_id: str
    topic: str
    target_references: list[str] = field(default_factory=list)
    section_targets: dict[str, list[str]] = field(default_factory=dict)
    survey_year: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TopicRecord":
        topic = str(value.get("topic") or value.get("title") or "").strip()
        if not topic:
            raise ValueError("A topic record must contain `topic`.")
        raw_sections = value.get("section_targets") or value.get("sections") or {}
        section_targets: dict[str, list[str]] = {}
        if isinstance(raw_sections, dict):
            for name, ids in raw_sections.items():
                if isinstance(ids, dict):
                    ids = ids.get("ids") or ids.get("references") or []
                section_targets[str(name)] = [str(item) for item in ids or []]
        elif isinstance(raw_sections, list):
            for section in raw_sections:
                if not isinstance(section, dict):
                    continue
                name = section.get("section_title") or section.get("title")
                ids = section.get("section_reference") or section.get("references") or []
                if name:
                    section_targets[str(name)] = [
                        str(item.get("id") if isinstance(item, dict) else item) for item in ids
                    ]
        targets = value.get("target_references") or value.get("ground_truth") or []
        return cls(
            topic_id=str(value.get("topic_id") or value.get("id") or _slug(topic)),
            topic=topic,
            target_references=[
                str(item.get("id") if isinstance(item, dict) else item) for item in targets
            ],
            section_targets=section_targets,
            survey_year=_optional_int(value.get("survey_year", value.get("year"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BranchResult:
    action: Action
    tool_input: list[str]
    candidates: list[Paper]
    accepted: list[Paper]
    section_coverage: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action"] = self.action.value
        return result


@dataclass(slots=True)
class TrajectoryStep:
    step: int
    state: dict[str, Any]
    action: Action
    tool_input: list[str]
    candidates: list[Paper]
    accepted: list[Paper]
    collection: list[Paper]
    reward: float
    branch_results: list[BranchResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "state": self.state,
            "action": self.action.value,
            "tool_input": self.tool_input,
            "candidates": [paper.to_dict() for paper in self.candidates],
            "accepted": [paper.to_dict() for paper in self.accepted],
            "collection": [paper.to_dict() for paper in self.collection],
            "reward": self.reward,
            "branch_results": [branch.to_dict() for branch in self.branch_results],
        }


@dataclass(slots=True)
class Trajectory:
    topic_id: str
    topic: str
    steps: list[TrajectoryStep]
    final_collection: list[Paper]

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic": self.topic,
            "steps": [step.to_dict() for step in self.steps],
            "final_collection": [paper.to_dict() for paper in self.final_collection],
        }


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    return "-".join("".join(char.lower() if char.isalnum() else " " for char in value).split())

