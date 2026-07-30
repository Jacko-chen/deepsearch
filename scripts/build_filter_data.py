#!/usr/bin/env python3
from __future__ import annotations

import argparse

from deepsearch.data import build_filter_examples, split_by_group
from deepsearch.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--positive-repeat", type=int, default=3)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    args = parser.parse_args()
    examples = build_filter_examples(
        read_jsonl(args.input),
        positive_repeat=args.positive_repeat,
    )
    train, validation = split_by_group(
        examples,
        group_key="topic_id",
        validation_ratio=args.validation_ratio,
    )
    write_jsonl(args.train_output, train)
    write_jsonl(args.validation_output, validation)
    print(f"train={len(train)} validation={len(validation)}")


if __name__ == "__main__":
    main()
