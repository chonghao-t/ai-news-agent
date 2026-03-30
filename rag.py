import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level singletons — initialized once, reused across all calls
# ---------------------------------------------------------------------------

CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "articles"

_embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH),
    settings=Settings(anonymized_telemetry=False),
)

_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},  
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_embed_text(article: dict) -> str:
    """
    Produce the text to embed for a given article.
    Prefer full_text, fall back to snippet.
    Truncate to ~1500 words to stay within the model's effective range.
    """
    text = article.get("full_text") or article.get("snippet", "")
    words = text.split()
    if len(words) > 1500:
        words = words[:1500]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert(article: dict, config: dict = None) -> None:
    """
    Embed the article excerpt and store it in ChromaDB.
    """
    text = _get_embed_text(article)
    embedding = _embedder.encode(text).tolist()

    _collection.upsert(
        ids=[article["url"]],
        embeddings=[embedding],
        metadatas=[{
            "headline":     article["headline"],
            "category":     article["category"],
            "source":       article["source"],
            "published_at": article["published_at"],
            "summary":      article.get("why_it_matters", ""),
        }],
        documents=[text],
    )


def query_similar(article: dict, config: dict) -> list[dict]:
    """
    Find the top-5 most similar prior articles to the given article.
    """
    text = _get_embed_text(article)
    embedding = _embedder.encode(text).tolist()

    retention_days = config.get("rag_retention_days", 7)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()

    # Count how many documents exist so we never request more than available
    total = _collection.count()
    if total == 0:
        return []

    n = min(total, 20)  

    try:
        results = _collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["metadatas", "distances", "documents"],
        )
    except Exception:
        return []

    similar = []
    ids       = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for doc_id, distance, meta in zip(ids, distances, metadatas):
        # Exclude self
        if doc_id == article["url"]:
            continue

        # Filter by category in Python
        if meta.get("category") != article["category"]:
            continue

        # Filter by retention window in Python
        if meta.get("published_at", "") < cutoff:
            continue

        similarity = 1.0 - distance
        similar.append({
            "url":          doc_id,
            "headline":     meta.get("headline", ""),
            "category":     meta.get("category", ""),
            "published_at": meta.get("published_at", ""),
            "summary":      meta.get("summary", ""),
            "similarity":   round(similarity, 4),
        })

    similar.sort(key=lambda x: x["similarity"], reverse=True)
    return similar[:5]


def update_summary(url: str, summary: str) -> None:
    """
    After summarization, store the generated summary back into
    the ChromaDB document metadata so future RAG context queries can use it.
    """
    try:
        existing = _collection.get(ids=[url], include=["metadatas", "documents", "embeddings"])
        if not existing["ids"]:
            return

        meta = existing["metadatas"][0]
        meta["summary"] = summary

        _collection.upsert(
            ids=[url],
            embeddings=existing["embeddings"][0],
            metadatas=[meta],
            documents=existing["documents"][0],
        )
    except Exception:
        pass  # non-fatal — summary context is a nice-to-have


def prune(config: dict) -> None:
    """
    Delete all documents older than the configured retention window.
    Call once per pipeline run to keep the collection size bounded.
    """
    retention_days = config.get("rag_retention_days", 7)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()

    try:
        total = _collection.count()
        if total == 0:
            return

        # Fetch all document IDs and their metadata
        existing = _collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(existing["ids"], existing["metadatas"])
            if meta.get("published_at", "") < cutoff
        ]

        if ids_to_delete:
            _collection.delete(ids=ids_to_delete)

    except Exception:
        pass  # non-fatal