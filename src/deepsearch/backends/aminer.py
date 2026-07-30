from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from deepsearch.backends.base import RetrievalBackend
from deepsearch.types import Paper


class AMinerBackend(RetrievalBackend):
    """AMiner SearchPro, metadata, and citation API adapter.

    Credentials are read from ``AMINER_API_KEY`` and are never stored in
    source files.
    """

    search_url = "https://datacenter.aminer.cn/gateway/api/v3/paper/search/paper/SearchPro"
    detail_url = "https://datacenter.aminer.cn/gateway/api/v3/paper/detail/batch"
    relation_url = "https://datacenter.aminer.cn/gateway/api/v3/paper/pub_relation"

    def __init__(self, api_key: str | None = None, *, timeout: float = 30.0):
        try:
            import requests
        except ImportError as exc:
            raise ImportError("Install the live backend with `pip install -e '.[api]'`.") from exc
        self._requests = requests
        self.api_key = api_key or os.getenv("AMINER_API_KEY", "")
        if not self.api_key:
            raise ValueError("Set AMINER_API_KEY or pass api_key explicitly.")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json;charset=utf-8",
            }
        )

    def search(self, queries: Sequence[str], *, limit: int = 50) -> list[Paper]:
        ids: list[str] = []
        per_query = max(1, limit // max(1, len(queries)))
        for query in queries:
            response = self.session.post(
                self.search_url,
                json={"query": query, "use_topic": True, "size": per_query, "offset": 0},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json().get("data", {}).get("data", [])
            ids.extend(str(item["id"]) for item in payload if item.get("id"))
        return self._details(_dedupe(ids)[:limit])

    def expand_citations(self, seed_ids: Sequence[str], *, limit: int = 50) -> list[Paper]:
        related: list[str] = []
        for seed_id in seed_ids:
            for params, key in (
                ({"cited": seed_id, "offset": 0, "size": limit}, "ref"),
                ({"ref": seed_id, "offset": 0, "size": limit}, "cited"),
            ):
                response = self.session.get(self.relation_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                rows = response.json().get("data", [])
                related.extend(str(row[key]) for row in rows if row.get(key))
        seed_set = set(seed_ids)
        ids = [paper_id for paper_id in _dedupe(related) if paper_id not in seed_set]
        return self._details(ids[:limit])

    def _details(self, paper_ids: Sequence[str]) -> list[Paper]:
        if not paper_ids:
            return []
        response = self.session.post(
            self.detail_url,
            json={"ids": list(paper_ids)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows: list[dict[str, Any]] = response.json().get("data", []) or []
        return [Paper.from_dict(row) for row in rows]


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result

