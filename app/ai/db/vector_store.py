"""Vector store for RAG, backed by ChromaDB.

Namespaces (one Chroma collection each, cosine distance):
    "documents"  chunks of files/text users uploaded       (always owned by a user)
    "database"   rows indexed from PostgreSQL tables         (owned, or shared)

Isolation: every chunk carries an `owner` metadata value: a user id, or
SHARED_OWNER for data every user may see. search() ALWAYS filters on the
caller's user id, so one user can never retrieve another user's chunks.

Embeddings come from the Gemini provider; Chroma never embeds anything itself
(collections are created without an embedding function). The chromadb client is
synchronous, so calls run in a worker thread.

    await add_documents("documents", [VectorItem(content=..., owner=user_id, doc_id=...)])
    hits = await search(["documents", "database"], "refund policy?", user_id=user_id)
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.ai.db.engine import cfg

log = logging.getLogger(__name__)

SHARED_OWNER = "__shared__"
NAMESPACES = ("documents", "database")
_BATCH = 64
_NAMESPACE = re.compile(r"^[a-z0-9_]{1,40}$")
MAX_K = 50

_client: Any = None
_collections: dict[str, Any] = {}
_lock = threading.Lock()


@dataclass
class VectorItem:
    content: str
    owner: str                       # user id, or SHARED_OWNER
    doc_id: str                      # groups the chunks of one document / row
    chunk: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.doc_id}:{self.chunk}"


@dataclass
class SearchHit:
    id: str
    content: str
    score: float  # cosine similarity, higher is better (max 1.0)
    namespace: str
    metadata: dict[str, Any]

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))

    @property
    def source(self) -> str:
        """Human-readable label for citations."""
        m = self.metadata
        if m.get("filename"):
            return str(m["filename"])
        if m.get("table"):
            return f"{m['table']} #{m.get('row_id', '?')}"
        return self.doc_id or self.id


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
def _build_client() -> Any:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    mode = cfg("CHROMA_MODE", "persistent")
    chroma_settings = ChromaSettings(anonymized_telemetry=False)
    if mode == "http":
        token = cfg("CHROMA_AUTH_TOKEN")
        headers = {"Authorization": f"Bearer {token.get_secret_value()}"} if token else None
        return chromadb.HttpClient(
            host=cfg("CHROMA_HOST", "localhost"),
            port=int(cfg("CHROMA_PORT", 8000)),
            ssl=bool(cfg("CHROMA_SSL", False)),
            headers=headers,
            settings=chroma_settings,
        )
    if mode == "memory":  # tests / throwaway
        return chromadb.EphemeralClient(settings=chroma_settings)
    return chromadb.PersistentClient(path=cfg("CHROMA_PATH", "./.chroma"),
                                     settings=chroma_settings)


def _collection(namespace: str) -> Any:
    if not _NAMESPACE.match(namespace or ""):
        raise ValueError("Invalid namespace.")
    global _client
    coll = _collections.get(namespace)
    if coll is not None:
        return coll
    with _lock:
        if _client is None:
            _client = _build_client()
        coll = _collections.get(namespace)
        if coll is None:
            coll = _client.get_or_create_collection(
                name=f"{cfg('CHROMA_COLLECTION_PREFIX', 'vaakdb_')}{namespace}",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
            _collections[namespace] = coll
    return coll


def set_client(client: Any) -> None:
    """Use a specific chromadb client (tests, custom deployments)."""
    global _client
    with _lock:
        _client = client
        _collections.clear()


def close_vector_store() -> None:
    global _client
    with _lock:
        _client = None
        _collections.clear()


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def _dim() -> int:
    return int(cfg("EMBEDDING_DIM", 768))


async def embed(texts: list[str], *, task_type: str | None = None) -> list[list[float]]:
    from app.ai.providers.registry import get_provider  # lazy import

    vectors = await get_provider("gemini").embed(texts, task_type=task_type)
    if vectors and len(vectors[0]) != _dim():
        raise ValueError(
            f"Embedding size {len(vectors[0])} does not match EMBEDDING_DIM={_dim()}."
        )
    return vectors


def _clean_meta(item: VectorItem) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool; reserved keys always win."""
    meta: dict[str, Any] = {}
    for key, value in item.metadata.items():
        if value is None:
            continue
        meta[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    meta.update(owner=item.owner, doc_id=item.doc_id, chunk=item.chunk)
    return meta


def _owner_filter(user_id: str, include_shared: bool) -> dict[str, Any]:
    if include_shared:
        return {"owner": {"$in": [user_id, SHARED_OWNER]}}
    return {"owner": user_id}


def _and(*clauses: dict[str, Any] | None) -> dict[str, Any]:
    parts = [c for c in clauses if c]
    return parts[0] if len(parts) == 1 else {"$and": parts}


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
async def add_documents(namespace: str, items: Sequence[VectorItem]) -> int:
    """Embed and upsert items (same doc_id + chunk replaces the old chunk)."""
    items = [i for i in items if i.content and i.content.strip()]
    if not items:
        return 0
    for i in items:
        if not i.owner:
            raise ValueError("Every vector item needs an owner.")
    coll = _collection(namespace)

    stored = 0
    for start in range(0, len(items), _BATCH):
        batch = items[start : start + _BATCH]
        vectors = await embed([i.content for i in batch], task_type="RETRIEVAL_DOCUMENT")
        await asyncio.to_thread(
            coll.upsert,
            ids=[i.id for i in batch],
            embeddings=vectors,
            documents=[i.content for i in batch],
            metadatas=[_clean_meta(i) for i in batch],
        )
        stored += len(batch)
    return stored


async def search(
    namespaces: str | Iterable[str],
    query: str,
    *,
    user_id: str,
    k: int = 5,
    min_score: float | None = None,
    include_shared: bool = True,
    where: dict[str, Any] | None = None,
) -> list[SearchHit]:
    """The k most similar chunks the user may see, across one or more namespaces."""
    if not user_id:
        raise ValueError("A user is required.")
    if not query or not query.strip():
        return []
    names = [namespaces] if isinstance(namespaces, str) else list(namespaces)
    k = max(1, min(int(k), MAX_K))
    [query_vec] = await embed([query], task_type="RETRIEVAL_QUERY")
    flt = _and(_owner_filter(user_id, include_shared), where)

    hits: list[SearchHit] = []
    for ns in names:
        coll = _collection(ns)
        res = await asyncio.to_thread(
            coll.query,
            query_embeddings=[query_vec],
            n_results=k,
            where=flt,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for hid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - float(dist)
            if min_score is None or score >= min_score:
                hits.append(SearchHit(id=hid, content=doc or "", score=score,
                                      namespace=ns, metadata=dict(meta or {})))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


async def delete_documents(
    namespace: str, *, owner: str, doc_id: str | None = None
) -> int:
    """Delete one document's chunks (or all of an owner's chunks). Returns chunk count."""
    if not owner:
        raise ValueError("An owner is required.")
    coll = _collection(namespace)
    flt = _and({"owner": owner}, {"doc_id": doc_id} if doc_id else None)
    found = await asyncio.to_thread(coll.get, where=flt, include=[])
    ids = found.get("ids") or []
    if ids:
        await asyncio.to_thread(coll.delete, ids=ids)
    return len(ids)


async def list_documents(namespace: str, *, owner: str, limit: int = 1000) -> list[dict[str, Any]]:
    """One entry per doc_id owned by `owner`, with its chunk count and metadata."""
    coll = _collection(namespace)
    found = await asyncio.to_thread(
        coll.get, where={"owner": owner}, include=["metadatas"], limit=max(1, limit) * 50
    )
    docs: dict[str, dict[str, Any]] = {}
    for meta in found.get("metadatas") or []:
        meta = dict(meta or {})
        did = str(meta.get("doc_id", ""))
        entry = docs.setdefault(did, {"doc_id": did, "chunks": 0, **{
            k: meta[k] for k in ("filename", "mime_type", "title", "created_at") if k in meta
        }})
        entry["chunks"] += 1
    return list(docs.values())[:limit]
