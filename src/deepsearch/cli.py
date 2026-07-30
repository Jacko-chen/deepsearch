from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepsearch.agent import AgentConfig, DeepSearchAgent
from deepsearch.backends.memory import InMemoryBackend
from deepsearch.construction import ConstructionConfig, TrajectoryConstructor
from deepsearch.filters import HeuristicPaperFilter
from deepsearch.io import load_papers, load_topic, read_json, write_json
from deepsearch.metrics import evaluate_collection
from deepsearch.selectors import HeuristicSelector
from deepsearch.types import Paper, TopicRecord


def demo_main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline end-to-end example.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    examples = root / "examples"
    papers = load_papers(examples / "corpus.json")
    topic = load_topic(examples / "topic.json")
    result = run_offline(papers, topic)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "constructed_trajectory.json", result["trajectory"])
    write_json(args.output_dir / "agent_result.json", result["agent"])
    write_json(args.output_dir / "metrics.json", result["metrics"])
    print(json.dumps(result["metrics"], indent=2))
    print(f"Wrote demo outputs to {args.output_dir}")


def construct_main() -> None:
    parser = argparse.ArgumentParser(description="Construct one trajectory with an offline corpus.")
    parser.add_argument("--topic", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--filter-threshold", type=float, default=0.18)
    args = parser.parse_args()
    papers = load_papers(args.corpus)
    topic = load_topic(args.topic)
    constructor = TrajectoryConstructor(
        InMemoryBackend(papers),
        HeuristicPaperFilter(),
        ConstructionConfig(
            max_steps=args.max_steps,
            filter_threshold=args.filter_threshold,
        ),
    )
    write_json(args.output, constructor.construct(topic).to_dict())


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a returned paper collection.")
    parser.add_argument("--topic", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    topic = load_topic(args.topic)
    value = read_json(args.collection)
    rows = value.get("papers", value.get("final_collection", value)) if isinstance(value, dict) else value
    papers = [Paper.from_dict(row) for row in rows]
    metrics = evaluate_collection(papers, topic, paper_catalog={paper.id: paper for paper in papers})
    if args.output:
        write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


def run_offline(papers: list[Paper], topic: TopicRecord) -> dict:
    backend = InMemoryBackend(papers)
    paper_filter = HeuristicPaperFilter()
    trajectory = TrajectoryConstructor(
        backend,
        paper_filter,
        ConstructionConfig(filter_threshold=0.18),
    ).construct(topic)
    agent = DeepSearchAgent(
        backend,
        paper_filter,
        HeuristicSelector(),
        AgentConfig(max_steps=6, filter_threshold=0.18),
    ).retrieve(topic.topic)
    agent_papers = [Paper.from_dict(row) for row in agent["papers"]]
    metrics = evaluate_collection(
        agent_papers,
        topic,
        paper_catalog={paper.id: paper for paper in papers},
    )
    return {"trajectory": trajectory.to_dict(), "agent": agent, "metrics": metrics}
