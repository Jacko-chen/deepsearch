"""DeepSearch: survey-oriented literature retrieval."""

from .agent import DeepSearchAgent
from .construction import ConstructionConfig, TrajectoryConstructor
from .filters import HeuristicPaperFilter
from .selectors import HeuristicSelector
from .types import Action, Paper, TopicRecord, Trajectory

__all__ = [
    "Action",
    "ConstructionConfig",
    "DeepSearchAgent",
    "HeuristicPaperFilter",
    "HeuristicSelector",
    "Paper",
    "TopicRecord",
    "Trajectory",
    "TrajectoryConstructor",
]

__version__ = "0.1.0"

