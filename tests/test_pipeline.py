import json
import tempfile
import unittest
from pathlib import Path

from deepsearch.cli import run_offline
from deepsearch.io import load_papers, load_topic


class PipelineTest(unittest.TestCase):
    def test_offline_pipeline(self):
        root = Path(__file__).resolve().parents[1]
        result = run_offline(
            load_papers(root / "examples" / "corpus.json"),
            load_topic(root / "examples" / "topic.json"),
        )
        self.assertTrue(result["trajectory"]["steps"])
        self.assertTrue(result["agent"]["papers"])
        self.assertIn("cscore", result["metrics"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text(json.dumps(result), encoding="utf-8")
            self.assertTrue(output.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
