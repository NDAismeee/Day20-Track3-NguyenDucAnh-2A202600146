"""Search client abstraction for ResearcherAgent."""

import json
from pathlib import Path

import requests

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client skeleton."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        TODO(student): Implement with Tavily, Bing, SerpAPI, internal docs, or a local mock.
        """

        settings = get_settings()
        if settings.tavily_api_key:
            return self._search_tavily(query=query, max_results=max_results, api_key=settings.tavily_api_key)
        return self._search_local_mock(query=query, max_results=max_results)

    def _search_tavily(self, *, query: str, max_results: int, api_key: str) -> list[SourceDocument]:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=20,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise AgentExecutionError(f"Search failed: {exc}") from exc

        out: list[SourceDocument] = []
        for item in payload.get("results", [])[:max_results]:
            title = (item.get("title") or "").strip() or "Untitled"
            url = (item.get("url") or None) if item.get("url") else None
            snippet = (item.get("content") or item.get("snippet") or "").strip()
            if not snippet:
                continue
            out.append(SourceDocument(title=title, url=url, snippet=snippet, metadata={"provider": "tavily"}))
        return out

    def _search_local_mock(self, *, query: str, max_results: int) -> list[SourceDocument]:
        corpus_path = Path("mock_data") / "search_corpus.json"
        if not corpus_path.exists():
            return []
        raw = json.loads(corpus_path.read_text(encoding="utf-8"))
        q = query.lower()

        scored: list[tuple[int, dict]] = []
        for doc in raw:
            text = f"{doc.get('title','')} {doc.get('snippet','')}".lower()
            score = sum(1 for token in q.split() if token and token in text)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)

        out: list[SourceDocument] = []
        for score, doc in scored:
            if len(out) >= max_results:
                break
            snippet = (doc.get("snippet") or "").strip()
            if not snippet:
                continue
            out.append(
                SourceDocument(
                    title=(doc.get("title") or "Untitled").strip(),
                    url=doc.get("url"),
                    snippet=snippet,
                    metadata={"provider": "local_mock", "score": score},
                )
            )
        return out
