from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from deepsearch.types import Paper


class RetrievalBackend(ABC):
    """Interface shared by offline and live scholarly retrieval backends."""

    @abstractmethod
    def search(self, queries: Sequence[str], *, limit: int = 50) -> list[Paper]:
        """Retrieve papers for keyword queries."""

    @abstractmethod
    def expand_citations(self, seed_ids: Sequence[str], *, limit: int = 50) -> list[Paper]:
        """Retrieve references and citing papers around seed papers."""

