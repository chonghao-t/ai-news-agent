import hashlib
import operator
from datetime import datetime, timezone
from typing import Annotated, TypedDict

# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------

STOCKS     = "stocks"
MACRO      = "macro"
AI         = "ai"
CATEGORIES = [STOCKS, MACRO, AI]

# ---------------------------------------------------------------------------
# Article factory
# ---------------------------------------------------------------------------

def make_article(
    url: str,
    headline: str,
    snippet: str,
    source: str,
    tier: int,
    category: str,
    published_at: str,
) -> dict:
    """
    Create a new article dict with all fields initialised to safe defaults.
    Call this in source adapters to guarantee a consistent shape.

    Fields added during processing:
        full_text, snippet_only   — fetch stage in process_category
        content_hash              — embed stage in process_category
        novelty_score,
        final_score, rag_context  — score stage in process_category
        bullets, why_it_matters,
        summary_source            — llm_summarize
    """
    return {
        # Set now
        "url":          url,
        "headline":     headline,
        "snippet":      snippet,
        "source":       source,
        "tier":         tier,
        "category":     category,       # "stocks" | "macro" | "ai"
        "published_at": published_at,   # ISO 8601 datetime string

        # Set during fetch
        "full_text":    None,
        "snippet_only": False,

        # Set during embed
        "content_hash": "",

        # Set during score
        "ticker_tag":    "",            # detected watchlist ticker (STOCKS only)
        "novelty_score": 0.0,
        "final_score":   0.0,
        "rag_context":   [],

        # Set by llm_summarize
        "bullets":        [],
        "market_impact":  "",           # one sentence on market/sector implication
        "investor_angle": "",           # one sentence on what the investor should watch
        "sentiment":      "",           # "bullish" | "bearish" | "mixed"
        "summary_source": "",           # "llm" | "fallback"
    }


def content_hash(article: dict) -> str:
    """
    Return a stable SHA-256 fingerprint of an article's content.
    Falls back to hashing the URL alone if both text fields are empty.
    """
    text = (
        article.get("full_text")
        or article.get("snippet")
        or article.get("url", "")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# LangGraph states
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    """
    Shared state object flowing through every LangGraph node.
    """
    config:     dict
    candidates: list[dict]
    selected:   list[dict]

    # Accumulated across parallel category branches via operator.add.
    # Each process_category branch appends its own ranked articles here.
    ranked:     Annotated[list[dict], operator.add]

    summarized: list[dict]

    # Accumulated across all nodes via operator.add.
    # Each node returns only its NEW errors — LangGraph adds them.
    errors:     Annotated[list[str], operator.add]


class CategoryState(TypedDict):
    """
    State passed to each parallel process_category branch via Send.
    Contains only what that branch needs: config, category name,
    and the selected articles for that category.
    """
    config:   dict
    category: str
    articles: list[dict]


def empty_state() -> dict:
    """Return the initial state dict passed to graph.invoke()."""
    return {
        "config":     {},
        "candidates": [],
        "selected":   [],
        "ranked":     [],
        "summarized": [],
        "errors":     [],
    }


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()