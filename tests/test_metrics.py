import unittest

from deepsearch.metrics import evaluate_collection, reference_metrics, section_coverage
from deepsearch.types import Paper, TopicRecord


class MetricsTest(unittest.TestCase):
    def test_reference_metrics(self):
        result = reference_metrics(["a", "b"], ["b", "c"])
        self.assertEqual(result, {"precision": 0.5, "recall": 0.5, "f1": 0.5})

    def test_section_coverage(self):
        topic = TopicRecord("t", "topic", section_targets={"A": ["a"], "B": ["b", "c"]})
        self.assertEqual(section_coverage(["a"], topic.section_targets), 0.5)

    def test_evaluate_collection(self):
        topic = TopicRecord("t", "topic", target_references=["a"])
        result = evaluate_collection([Paper("a", "A")], topic, paper_catalog={})
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
