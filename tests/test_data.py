import unittest

from deepsearch.data import (
    build_filter_examples,
    filter_prompt,
    selector_state_prompt,
    selector_system_prompt,
    split_by_group,
)


def candidate(topic_id, paper_id, *, reference=False, votes=None):
    return {
        "topic_id": topic_id,
        "topic": "graph learning",
        "in_reference": reference,
        "judge_votes": votes or [],
        "paper": {"id": paper_id, "title": paper_id, "similarity_score": 0.9},
    }


class DataTest(unittest.TestCase):
    def test_asymmetric_labels(self):
        rows = build_filter_examples(
            [
                candidate("t1", "ref", reference=True),
                candidate("t1", "positive", votes=["yes", "yes", "yes"]),
                candidate("t1", "negative", votes=["no", "no", "yes"]),
                candidate("t1", "uncertain", votes=["yes", "yes", "no"]),
            ],
            positive_repeat=1,
        )
        labels = {row["paper_id"]: row["label"] for row in rows}
        self.assertEqual(labels, {"ref": "yes", "positive": "yes", "negative": "no"})
        self.assertNotIn("similarity_score", rows[0]["messages"][0]["content"])

    def test_split_has_disjoint_topics(self):
        rows = [{"topic_id": f"t{i}", "value": i} for i in range(10)]
        train, validation = split_by_group(rows, group_key="topic_id", validation_ratio=0.2)
        self.assertFalse(
            {row["topic_id"] for row in train}.intersection(
                row["topic_id"] for row in validation
            )
        )

    def test_prompts_are_loaded_and_rendered(self):
        filter_text = filter_prompt("graph learning", {"title": "A paper"})
        selector_text = selector_state_prompt(
            "graph learning",
            {
                "current_step": 1,
                "max_steps": 6,
                "previous_actions": [],
                "used_keywords": [],
                "collection_ids": [],
            },
        )
        self.assertIn("graph learning", filter_text)
        self.assertIn("A paper", filter_text)
        self.assertIn("graph learning", selector_text)
        self.assertNotIn("{TOPIC}", selector_text)
        self.assertIn("expert research assistant", selector_system_prompt())


if __name__ == "__main__":
    unittest.main()
