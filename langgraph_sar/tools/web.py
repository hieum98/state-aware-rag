"""Optional web-search tool for ``langgraph_sar`` (paper "+Google Search" variant).

Ports ``wemg/langgraph_coe/tools/web.py`` (Serper when keyed, else DuckDuckGo;
optional full-text crawl; per-session URL dedup via a ContextVar). Off by default
for corpus-only parity (``web_search.enabled = false``).
"""

from __future__ import annotations

import asyncio
import os
import re
from contextvars import ContextVar
from typing import Dict, List, Optional

import httpx
from bs4 import BeautifulSoup
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.tools import tool

from ..config import WebSearchConfig
from ..explicit import warn_explicit

_web_search_instance = None
_web_config: Optional[WebSearchConfig] = None


class _WebSession:
    __slots__ = ("visited",)

    def __init__(self) -> None:
        self.visited: set[str] = set()


_cv_web_session: ContextVar[Optional[_WebSession]] = ContextVar(
    "web_research_session", default=None
)


def _get_web_session() -> _WebSession:
    s = _cv_web_session.get(None)
    if s is None:
        s = _WebSession()
        _cv_web_session.set(s)
    return s


def reset_web_research_session() -> None:
    _cv_web_session.set(_WebSession())


def init_web_search(config: WebSearchConfig) -> None:
    global _web_search_instance, _web_config
    _web_config = config
    api_key = config.api_key or os.environ.get("SERPER_API_KEY")

    if api_key:
        _web_search_instance = GoogleSerperAPIWrapper(serper_api_key=api_key)
    else:
        _web_search_instance = DuckDuckGoSearchAPIWrapper(max_results=config.top_k)


async def _crawl_page(url: str, client: httpx.AsyncClient) -> str:
    try:
        response = await client.get(
            url,
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        def _parse(html: str) -> str:
            soup = BeautifulSoup(html, "lxml")
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return re.sub(r"\n{3,}", "\n\n", text)

        return await asyncio.to_thread(_parse, response.text)
    except Exception as exc:
        warn_explicit(
            "Page crawl failed; using empty full_text for this search result.",
            component="web.crawl_page",
            details={"url": url, "error": str(exc)},
        )
        return ""


@tool
async def web_search(query: str) -> List[Dict[str, str]]:
    """Search the web for up-to-date information on a specific query."""
    if not _web_search_instance or not _web_config:
        raise RuntimeError("Web search not initialized. Call init_web_search first.")

    search_results: List[Dict[str, str]] = []

    if isinstance(_web_search_instance, GoogleSerperAPIWrapper):
        res = await asyncio.to_thread(_web_search_instance.results, query)
        organic = res.get("organic", [])[: _web_config.top_k]
        for item in organic:
            search_results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "full_text": "",
                }
            )
    else:
        res = await asyncio.to_thread(_web_search_instance.results, query, _web_config.top_k)
        if isinstance(res, list):
            for item in res:
                search_results.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "url": item.get("link", ""),
                        "full_text": "",
                    }
                )
        else:
            search_results.append(
                {"title": "", "snippet": str(res), "url": "", "full_text": ""}
            )

    if _web_config.crawl_full_text:
        concurrency = max(1, int(_web_config.max_crawl_requests_per_second))
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_crawl(url: str, client: httpx.AsyncClient):
            if not url:
                return ""
            async with semaphore:
                return await _crawl_page(url, client)

        async with httpx.AsyncClient() as client:
            tasks = [bounded_crawl(r["url"], client) for r in search_results]
            full_texts = await asyncio.gather(*tasks)

        for r, text in zip(search_results, full_texts):
            r["full_text"] = text

    session = _get_web_session()
    deduped: List[Dict[str, str]] = []
    for r in search_results:
        url = str(r.get("url", "")).strip()
        if url:
            if url in session.visited:
                continue
            session.visited.add(url)
        deduped.append(
            {
                "title": str(r.get("title", "")),
                "url": url,
                "snippet": str(r.get("snippet", "")),
                "full_text": str(r.get("full_text", "")),
            }
        )

    return deduped
