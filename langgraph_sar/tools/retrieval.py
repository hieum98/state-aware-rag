"""Corpus retrieval tool for ``langgraph_sar``.

Ports the **coe** retrieval pattern (``wemg/langgraph_coe/tools/retrieval.py``):
a local FAISS index queried through the deployed embedding endpoint via
``OpenAIEmbeddings``. Deliberately **NO reranker** and **NO retriever HTTP
server** — the SAR port's only relevance filter is the EXTRACTOR's own
``relevant | not_relevant`` decision (the paper's ``E_consolidate`` design, §4).

``search.top_k`` truncation of the score-ordered hits replaces what a reranker
would otherwise do; that truncation happens in the consuming graph (Phase 1),
so this tool returns the FAISS ``search_k`` candidates in score order.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from ..config import RetrieverConfig


def get_corpus_retriever(retriever_cfg: RetrieverConfig):
    """Build the FAISS retriever, embedding queries via the deployed endpoint."""
    corpus = retriever_cfg.corpus
    emb_cfg = corpus.embedder

    embeddings = OpenAIEmbeddings(
        model=emb_cfg.model_name,
        base_url=emb_cfg.url,
        api_key=emb_cfg.api_key or "EMPTY",
    )

    index_path = os.environ.get("SAR_CORPUS_INDEX_PATH", corpus.index_path)
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path!r}. Set SAR_CORPUS_INDEX_PATH "
            "to the local COE-style corpus index bundle."
        )
    vector_store = FAISS.load_local(
        folder_path=os.path.dirname(index_path),
        embeddings=embeddings,
        index_name=os.path.basename(index_path).replace(".faiss", ""),
        allow_dangerous_deserialization=True,
    )

    return vector_store.as_retriever(search_kwargs={"k": corpus.search_k})


_retriever_instance = None


def init_retrieval_pipeline(retriever_cfg: RetrieverConfig) -> None:
    """One-shot init; build the FAISS retriever before first ``corpus_search`` use."""
    global _retriever_instance
    _retriever_instance = get_corpus_retriever(retriever_cfg)


def set_retriever(retriever) -> None:
    """Inject a retriever instance directly (tests / custom backends)."""
    global _retriever_instance
    _retriever_instance = retriever


def _doc_to_text(doc: Any) -> str:
    """Normalize retriever outputs from LangChain ``Document`` or simple stubs."""
    if isinstance(doc, Document):
        return doc.page_content
    if hasattr(doc, "page_content"):
        return str(doc.page_content)
    if isinstance(doc, dict) and "page_content" in doc:
        return str(doc["page_content"])
    return str(doc)


@tool
async def corpus_search(query: str, top_k: Optional[int] = None) -> List[str]:
    """Retrieve relevant passages from the knowledge corpus for a query.

    Returns the retriever's score-ordered passage contents (``page_content``). The
    tool uses the COE-style local FAISS retriever queried by the deployed Qwen
    embedder, not the legacy retriever HTTP server. ``top_k`` is an optional
    caller-side cap; downstream graphs may still dedup/truncate further.
    """
    if _retriever_instance is None:
        raise RuntimeError(
            "Retriever pipeline not initialized. Call init_retrieval_pipeline first."
        )
    docs = await _retriever_instance.ainvoke(query)
    texts = [_doc_to_text(doc) for doc in docs]
    if top_k is not None:
        texts = texts[:top_k]
    return texts
