import unittest

from deepsearch.rewards import (
    action_penalty,
    collection_reward,
    parse_action_with_penalty,
    trajectory_reward,
)
from deepsearch.types import Action


class RewardsTest(unittest.TestCase):
    def test_valid_search(self):
        keywords = [f"keyword {index}" for index in range(10)]
        parsed = parse_action_with_penalty(
            "search\n" + repr(keywords),
            current_step=1,
            max_steps=6,
            has_papers=False,
        )
        self.assertEqual(parsed.action, Action.SEARCH)
        self.assertEqual(parsed.format_penalty, 0.0)

    def test_four_repeated_actions_are_penalized(self):
        history = [Action.SEARCH, Action.SEARCH, Action.SEARCH]
        self.assertEqual(action_penalty(history, Action.SEARCH), -0.1)
        self.assertEqual(action_penalty(history[:2], Action.SEARCH), 0.0)

    def test_reward_weights_format_only(self):
        self.assertAlmostEqual(trajectory_reward(1.0, [-0.5], [-0.1]), 0.75)

    def test_collection_reward(self):
        self.assertAlmostEqual(collection_reward(0.5, 0.75, 1.0), 0.7)


if __name__ == "__main__":
    unittest.main()
