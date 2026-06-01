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
    """Build the FAISS retriever, embedding queries via the deployed endpoint.

    Two index layouts are supported (see ``CorpusConfig``):

    * **LangChain bundle** — ``<name>.faiss`` + ``<name>.pkl`` docstore sidecar
      (e.g. the toy index); loaded via :meth:`FAISS.load_local`.
    * **Raw HF-datasets index** — a bare ``<name>.faiss`` written by
      ``state_aware_rag/servers/build_index.py`` (``datasets.save_faiss_index``),
      whose passage text lives in a separate HF dataset (``corpus.corpus_dataset``).
      The docstore is reconstructed from that dataset, row order matching the
      vector order, mirroring the legacy ``servers/retriever.py`` design.
    """
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

    folder = os.path.dirname(index_path) or "."
    index_name = os.path.basename(index_path).replace(".faiss", "")
    pkl_path = os.path.join(folder, index_name + ".pkl")

    if os.path.exists(pkl_path):
        vector_store = FAISS.load_local(
            folder_path=folder,
            embeddings=embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True,
        )
    else:
        # Raw HF-datasets index: no ``.pkl`` sidecar — rebuild the docstore from
        # the companion corpus dataset (legacy build_index.py / retriever.py path).
        vector_store = _load_raw_faiss_corpus(index_path, embeddings, corpus)

    return vector_store.as_retriever(search_kwargs={"k": corpus.search_k})


def _load_raw_faiss_corpus(index_path: str, embeddings: Any, corpus_cfg: Any) -> FAISS:
    """Wrap a bare HF-datasets FAISS index in a LangChain store via a side corpus.

    ``build_index.py`` L2-normalizes passage embeddings and indexes them with
    ``METRIC_INNER_PRODUCT`` (cosine), so we search with ``MAX_INNER_PRODUCT``.
    Query normalization is unnecessary for ranking: every passage is scored
    against the same query vector, so the query norm scales all scores by one
    constant and leaves the top-k order — all this tool consumes — unchanged.
    """
    import faiss  # local import: only needed for the raw-index layout
    import datasets
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores.utils import DistanceStrategy

    dataset_ref = os.environ.get("SAR_CORPUS_DATASET", corpus_cfg.corpus_dataset)
    if not dataset_ref:
        raise FileNotFoundError(
            f"No docstore sidecar {index_name_for(index_path)!r} next to {index_path!r}, "
            "so the index is a raw HF-datasets index. Set retriever.corpus.corpus_dataset "
            "(or env SAR_CORPUS_DATASET) to the corpus that supplies passage text "
            "(e.g. 'Hieuman/wiki23-processed')."
        )

    index = faiss.read_index(index_path)

    if os.path.exists(dataset_ref):
        ds = datasets.load_from_disk(dataset_ref)
        if isinstance(ds, datasets.DatasetDict):
            ds = ds[corpus_cfg.corpus_split]
    else:
        ds = datasets.load_dataset(dataset_ref, split=corpus_cfg.corpus_split)

    if index.ntotal != len(ds):
        raise ValueError(
            f"Index/corpus size mismatch: FAISS index {index_path!r} has "
            f"{index.ntotal} vectors but corpus {dataset_ref!r} "
            f"(split={corpus_cfg.corpus_split!r}) has {len(ds)} rows. The corpus "
            "must be the exact dataset the index was built from, in the same order."
        )

    text_col = corpus_cfg.text_column
    if text_col not in ds.column_names:
        fallback = next(
            (c for c in ("contents", "text", "passage", "content") if c in ds.column_names),
            None,
        )
        if fallback is None:
            raise ValueError(
                f"text_column {text_col!r} not found in corpus {dataset_ref!r}. "
                f"Available columns: {ds.column_names}. Set retriever.corpus.text_column."
            )
        text_col = fallback

    texts = ds[text_col]  # single materialization, preserves row/vector order
    docstore_dict = {
        str(i): Document(page_content="" if t is None else str(t), metadata={"row": i})
        for i, t in enumerate(texts)
    }
    index_to_docstore_id = {i: str(i) for i in range(len(texts))}

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(docstore_dict),
        index_to_docstore_id=index_to_docstore_id,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def index_name_for(index_path: str) -> str:
    """Expected ``.pkl`` sidecar filename for a given ``.faiss`` index path."""
    return os.path.basename(index_path).replace(".faiss", "") + ".pkl"


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
