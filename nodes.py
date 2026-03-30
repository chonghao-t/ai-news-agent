import os
import re
import uuid
import json
import yaml
import hashlib
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from langgraph.constants import Send

from schema import (
    PipelineState, CategoryState,
    content_hash, utc_now,
    STOCKS, MACRO, AI,
)
from db  import init_db, is_sent, mark_sent
from rag import upsert, query_similar, prune, update_summary

load_dotenv()

logger = logging.getLogger(__name__)

CONFIG_PATH  = Path("config.yaml")
SOURCES_PATH = Path("sources.yaml")

REQUIRED_KEYS = [
    "tickers", "macro_themes", "ai_themes",
    "article_date_window_days", "max_candidates_per_category",
    "max_triage_stocks", "max_triage_macro", "max_triage_ai",
    "max_items_stocks", "max_items_macro", "max_items_ai",
    'stocks_score_threshold', 'max_items_per_ticker',
    "weight_relevance", "weight_recency", "weight_tier", "weight_novelty",
    "similarity_hard_exclude", "similarity_soft_suppress",
    "openai_model", "openai_max_tokens",
    "llm_temperature_triage", "llm_temperature_summarize",
    "rag_retention_days", "rag_context_min_similarity", "rag_context_max_similarity",
    "summary_max_bullets", "min_articles_to_send",
]

REQUIRED_ENV = ["OPENAI_API_KEY"]


# ===========================================================================
# Node 1: load_config
# ===========================================================================

def load_config(state: PipelineState) -> dict:
    """Load config.yaml, sources.yaml, and .env. Hard-fails if invalid."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH.resolve()}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must be a YAML mapping (key: value pairs)")

    if not SOURCES_PATH.exists():
        raise FileNotFoundError(f"sources.yaml not found at {SOURCES_PATH.resolve()}")

    with open(SOURCES_PATH, "r") as f:
        sources_data = yaml.safe_load(f)

    config["sources"] = sources_data.get("sources", [])

    missing_keys = [k for k in REQUIRED_KEYS if k not in config]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")

    weights = (
        config["weight_relevance"] + config["weight_recency"]
        + config["weight_tier"]    + config["weight_novelty"]
    )
    if not (0.999 <= weights <= 1.001):
        raise ValueError(
            f"Scoring weights must sum to 1.0, got {weights:.4f}. "
            "Check weight_relevance + weight_recency + weight_tier + weight_novelty."
        )

    missing_env = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing_env:
        raise EnvironmentError(
            f"Missing required environment variables: {missing_env}. "
            "Check your .env file."
        )

    config["openai_api_key"]        = os.getenv("OPENAI_API_KEY")
    config["telegram_bot_token"]    = os.getenv("TELEGRAM_BOT_TOKEN", "")
    config["telegram_chat_id"]      = os.getenv("TELEGRAM_CHAT_ID", "")
    config["alpha_vantage_api_key"] = os.getenv("ALPHA_VANTAGE_API_KEY", "")

    if os.getenv("TICKERS"):
        config["tickers"] = [t.strip() for t in os.getenv("TICKERS").split(",") if t.strip()]
    if os.getenv("MACRO_THEMES"):
        config["macro_themes"] = [t.strip() for t in os.getenv("MACRO_THEMES").split(",") if t.strip()]
    if os.getenv("AI_THEMES"):
        config["ai_themes"] = [t.strip() for t in os.getenv("AI_THEMES").split(",") if t.strip()]

    config["run_id"]      = str(uuid.uuid4())
    config["db_conn"]     = init_db()
    config["_started_at"] = utc_now()

    logger.info(f"Config loaded. Run ID: {config['run_id']}")
    logger.info(f"Tickers: {config['tickers']}")
    logger.info(f"Sources enabled: {[s['name'] for s in config['sources'] if s.get('enabled')]}")

    # Returns only new errors (none at this stage).
    return {"config": config, "errors": []}


# ===========================================================================
# Node 2: fetch_candidates
# ===========================================================================

def fetch_candidates(state: PipelineState) -> dict:
    """Run all enabled source adapters, deduplicate, cap per category."""
    from adapters import ADAPTER_REGISTRY

    config  = state["config"]
    conn    = config["db_conn"]
    sources = [s for s in config.get("sources", []) if s.get("enabled")]
    cap     = config.get("max_candidates_per_category", 100)
    new_errors = []

    def run_adapter(source: dict) -> list[dict]:
        name = source["name"]
        fn   = ADAPTER_REGISTRY.get(name)
        if fn is None:
            logger.warning(f"fetch_candidates: no adapter found for '{name}'")
            return []
        try:
            return fn(config)
        except Exception as e:
            msg = f"fetch_candidates: adapter '{name}' failed — {e}"
            logger.warning(msg)
            new_errors.append(msg)
            return []

    async def run_all() -> list[dict]:
        loop    = asyncio.get_event_loop()
        tasks   = [loop.run_in_executor(None, run_adapter, s) for s in sources]
        results = await asyncio.gather(*tasks)
        return [a for batch in results for a in batch]

    try:
        raw = asyncio.run(run_all())
    except RuntimeError:
        raw = []
        for source in sources:
            raw.extend(run_adapter(source))

    logger.info(f"fetch_candidates: {len(raw)} raw articles from {len(sources)} sources")

    # Dedup against sent_items
    not_sent     = []
    skipped_sent = 0
    for article in raw:
        if is_sent(conn, article["url"]):
            skipped_sent += 1
        else:
            not_sent.append(article)

    logger.info(f"fetch_candidates: {skipped_sent} skipped (already sent), {len(not_sent)} remaining")

    # Dedup within batch
    seen_hashes, seen_urls, unique = set(), set(), []
    for article in not_sent:
        url = article["url"]
        h   = hashlib.sha256((article["headline"] + url).encode()).hexdigest()
        if url in seen_urls or h in seen_hashes:
            continue
        seen_urls.add(url)
        seen_hashes.add(h)
        unique.append(article)

    logger.info(f"fetch_candidates: {len(unique)} unique candidates after within-batch dedup")

    # Per-category cap with most recent first
    by_category: dict[str, list[dict]] = {STOCKS: [], MACRO: [], AI: []}
    for article in unique:
        cat = article.get("category")
        if cat in by_category:
            by_category[cat].append(article)

    candidates = []
    for cat, articles in by_category.items():
        articles.sort(key=lambda a: a.get("published_at", ""), reverse=True)
        capped = articles[:cap]
        logger.info(f"fetch_candidates: {cat} → {len(articles)} unique, capped to {len(capped)}")
        candidates.extend(capped)

    logger.info(f"fetch_candidates: {len(candidates)} total candidates entering triage")

    return {"candidates": candidates, "errors": new_errors}


# ===========================================================================
# Node 3: llm_triage
# ===========================================================================

def llm_triage(state: PipelineState) -> dict:
    """
    LLM Gate 1 — score and select candidates by index.
    Also prunes stale ChromaDB documents once per run (runs before fan-out).
    """
    from llm import run_triage

    config     = state["config"]
    candidates = state["candidates"]
    new_errors = []

    # Prune ChromaDB once before the parallel branches start so it runs exactly once per pipeline execution.
    try:
        prune(config)
        logger.debug("llm_triage: ChromaDB pruned")
    except Exception as e:
        msg = f"llm_triage: prune failed — {e}"
        logger.warning(msg)
        new_errors.append(msg)

    if not candidates:
        logger.warning("llm_triage: no candidates to triage")
        return {"selected": [], "errors": new_errors}

    by_cat = {STOCKS: [], MACRO: [], AI: []}
    for a in candidates:
        cat = a.get("category")
        if cat in by_cat:
            by_cat[cat].append(a)

    selections = run_triage(config, candidates)

    selected = []
    for cat in (STOCKS, MACRO, AI):
        for idx in selections.get(cat, []):
            articles = by_cat[cat]
            if 0 <= idx < len(articles):
                article = dict(articles[idx])
                article["_triage_position"] = idx
                selected.append(article)

    logger.info(f"llm_triage: {len(candidates)} candidates → {len(selected)} selected")

    return {"selected": selected, "errors": new_errors}


# ===========================================================================
# Routing function — dispatch_to_categories
# ===========================================================================

def dispatch_to_categories(state: PipelineState) -> list[Send]:
    """
    Routing function called as a conditional edge after llm_triage.
    Splits selected articles by category and dispatches a Send for each,
    causing process_category to run in parallel for stocks, macro, and ai.

    If a category has no selected articles, no Send is dispatched for it
    (the category will appear as empty in the digest).
    """
    config   = state["config"]
    selected = state["selected"]

    by_cat: dict[str, list[dict]] = {STOCKS: [], MACRO: [], AI: []}
    for article in selected:
        cat = article.get("category")
        if cat in by_cat:
            by_cat[cat].append(article)

    sends = []
    for cat in (STOCKS, MACRO, AI):
        articles = by_cat[cat]
        if not articles:
            logger.info(f"dispatch_to_categories: {cat} — no articles, skipping branch")
            continue
        logger.info(f"dispatch_to_categories: {cat} — dispatching {len(articles)} articles")
        sends.append(Send("process_category", {
            "config":   config,
            "category": cat,
            "articles": articles,
        }))

    if not sends:
        # If no categories have articles, dispatch a single empty send so the
        # graph doesn't deadlock waiting for process_category to complete.
        logger.warning("dispatch_to_categories: no articles in any category")
        sends.append(Send("process_category", {
            "config":   config,
            "category": STOCKS,
            "articles": [],
        }))

    return sends


# ===========================================================================
# Node 4: process_category  (runs in parallel: once per category)
# ===========================================================================

def process_category(state: CategoryState) -> dict:
    """
    Parallel node — runs independently for stocks, macro, and ai.

    Executes the full fetch → embed → score pipeline for one category's
    articles. Returns ranked articles and errors for this category only.
    These are accumulated into the main PipelineState via operator.add.
    """
    config   = state["config"]
    category = state["category"]
    articles = state["articles"]
    new_errors = []

    if not articles:
        logger.info(f"process_category [{category}]: no articles")
        return {"ranked": [], "errors": new_errors}

    logger.info(f"process_category [{category}]: starting with {len(articles)} articles")

    # Stage A — fetch full article text
    fetched = _run_fetch(articles, category, new_errors)

    # Stage B — clean, hash, embed, upsert to ChromaDB
    embedded = _run_embed(fetched, config, category, new_errors)

    # Stage C — novelty query, score, rank
    ranked = _run_score(embedded, config, category, new_errors)

    logger.info(f"process_category [{category}]: {len(ranked)} articles ranked for digest")

    return {"ranked": ranked, "errors": new_errors}


# ---------------------------------------------------------------------------
# process_category helpers
# ---------------------------------------------------------------------------

def _run_fetch(articles: list[dict], category: str, errors: list[str]) -> list[dict]:
    """Fetch full article text for a list of articles."""
    import time
    import httpx
    import trafilatura
    from urllib.parse import urlparse

    last_request: dict[str, float] = {}
    MIN_GAP = 2.0

    def fetch_one(article: dict) -> dict:
        article = dict(article)
        url     = article["url"]
        domain  = urlparse(url).netloc

        gap = time.time() - last_request.get(domain, 0)
        if gap < MIN_GAP:
            time.sleep(MIN_GAP - gap)
        last_request[domain] = time.time()

        for attempt in range(2):
            try:
                response = httpx.get(
                    url, timeout=10, follow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; NewsDigestAgent/1.0)"},
                )
                if response.status_code in (401, 403):
                    article["snippet_only"] = True
                    return article
                if response.status_code != 200:
                    if attempt == 0:
                        continue
                    article["snippet_only"] = True
                    return article

                text = trafilatura.extract(
                    response.text,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=False,
                )
                if not text or len(text.strip()) < 200:
                    article["snippet_only"] = True
                    return article

                words = text.split()
                if len(words) > 2500:
                    words = words[:2500]
                article["full_text"]    = " ".join(words)
                article["snippet_only"] = False
                return article

            except Exception as e:
                if attempt == 0:
                    logger.debug(f"_run_fetch [{category}]: {url} attempt 1 failed — {e}")
                    continue
                logger.debug(f"_run_fetch [{category}]: {url} failed after 2 attempts — {e}")
                article["snippet_only"] = True
                return article

        article["snippet_only"] = True
        return article

    fetched       = [fetch_one(a) for a in articles]
    full_count    = sum(1 for a in fetched if a.get("full_text"))
    snippet_count = len(fetched) - full_count
    logger.info(
        f"_run_fetch [{category}]: {full_count} full text, "
        f"{snippet_count} snippet-only"
    )
    return fetched


def _run_embed(
    articles: list[dict],
    config: dict,
    category: str,
    errors: list[str],
) -> list[dict]:
    """Clean text, compute content hash, embed, upsert to ChromaDB."""
    embedded_count = 0
    processed      = []

    for article in articles:
        article = dict(article)

        if article.get("full_text"):
            article["full_text"] = _clean_text(article["full_text"])

        article["content_hash"] = content_hash(article)

        try:
            upsert(article, config)
            embedded_count += 1
        except Exception as e:
            msg = f"_run_embed [{category}]: upsert failed for {article['url']} — {e}"
            logger.warning(msg)
            errors.append(msg)

        processed.append(article)

    logger.info(f"_run_embed [{category}]: {embedded_count}/{len(processed)} embedded")
    return processed


def _detect_ticker(article: dict, tickers: list[str]) -> str:
    """
    Attempt to identify which watchlist ticker an article is primarily about.
 
    Strategy (in priority order):
    1. Parse the source string — Yahoo Finance uses "Yahoo Finance (TICKER)"
    2. Scan the headline for an exact uppercase ticker match
    3. Scan the snippet for an exact uppercase ticker match
    4. Return "" if no match found
    """
    if not tickers:
        return ""
 
    # 1. Source string pattern: "Yahoo Finance (NVDA)"
    source = article.get("source", "")
    import re as _re
    m = _re.search(r"\(([A-Z]{1,5})\)", source)
    if m and m.group(1) in tickers:
        return m.group(1)
 
    # 2 & 3. Scan headline then snippet for ticker as a whole word
    for field in ("headline", "snippet"):
        text = article.get(field, "")
        for ticker in tickers:
            # Match ticker as a standalone word (not inside another word)
            if _re.search(rf"\b{_re.escape(ticker)}\b", text):
                return ticker
 
    return ""


def _run_score(
    articles: list[dict],
    config: dict,
    category: str,
    errors: list[str],
) -> list[dict]:
    """
    Query ChromaDB for novelty, score articles, then select for the digest.
 
    For STOCKS: threshold-based selection with per-ticker deduplication.
      All articles scoring above stocks_score_threshold are kept, then
      capped at max_items_per_ticker per detected ticker to ensure variety.
 
    For MACRO / AI: original top-N behaviour is preserved.
    """
    w_relevance   = config.get("weight_relevance", 0.40)
    w_recency     = config.get("weight_recency",   0.20)
    w_tier        = config.get("weight_tier",      0.15)
    w_novelty     = config.get("weight_novelty",   0.25)
    hard_exclude  = config.get("similarity_hard_exclude",  0.88)
    soft_suppress = config.get("similarity_soft_suppress", 0.75)
    rag_min       = config.get("rag_context_min_similarity", 0.30)
    rag_max       = config.get("rag_context_max_similarity", 0.75)
    tickers       = config.get("tickers", [])

    scored = []
    for article in articles:
        article = dict(article)

        # Tag article with its primary ticker (STOCKS only)
        if category == STOCKS:
            article["ticker_tag"] = _detect_ticker(article, tickers)

        try:
            similar = query_similar(article, config)
        except Exception as e:
            logger.warning(f"_run_score [{category}]: RAG query failed — {e}")
            similar = []

        max_sim = max((r["similarity"] for r in similar), default=0.0)

        if max_sim >= hard_exclude:
            logger.debug(
                f"_run_score [{category}]: excluded "
                f"(similarity={max_sim:.3f}) {article['headline'][:55]}"
            )
            continue

        novelty_score = 1.0 - max_sim
        soft_penalty  = (
            max(0.0, (max_sim - soft_suppress) * 2.0)
            if max_sim >= soft_suppress else 0.0
        )

        article["rag_context"]   = [
            r["summary"] for r in similar
            if rag_min <= r["similarity"] <= rag_max and r.get("summary")
        ]
        article["novelty_score"] = round(novelty_score, 4)

        position = article.get("_triage_position")
        if position is None:
            relevance_proxy = 0.5
        else:
            max_pos = max(
                config.get("max_triage_stocks", 8),
                config.get("max_triage_macro",  6),
                config.get("max_triage_ai",     8),
            ) - 1
            max_pos         = max(max_pos, 1)
            relevance_proxy = round(
                max(0.4, min(1.0, 1.0 - (0.6 * (position / max_pos)))), 4
            )

        recency_score = _recency_score(
            article.get("published_at", ""),
            config.get("article_date_window_days", 7),
        )
        tier_score = {1: 1.0, 2: 0.7, 3: 0.4}.get(article.get("tier", 2), 0.4)

        final_score = round(max(0.0, min(1.0,
            (w_relevance * relevance_proxy)
            + (w_recency  * recency_score)
            + (w_tier     * tier_score)
            + (w_novelty  * novelty_score)
            - soft_penalty
        )), 4)

        article["final_score"]      = final_score
        article["_relevance_proxy"] = relevance_proxy
        article["_recency_score"]   = recency_score
        article["_tier_score"]      = tier_score
        scored.append(article)

    # Sort and apply threshold
    scored.sort(key=lambda a: a["final_score"], reverse=True)

    # STOCKS: threshold + per-ticker cap 
    if category == STOCKS:
        threshold       = config.get("stocks_score_threshold", 0.45)
        max_per_ticker  = config.get("max_items_per_ticker", 2)
 
        above = [a for a in scored if a["final_score"] >= threshold]
 
        # Fallback: if nothing clears threshold, relax novelty weight
        if len(above) < 2 and scored:
            logger.info(
                f"_run_score [{category}]: nothing above stocks threshold "
                f"({threshold}), relaxing novelty weight"
            )
            above = _rescore_without_novelty(scored, config)
 
        # Per-ticker cap — enforce variety across tickers.
        # Articles with no detected ticker share a common "" bucket and are
        # capped at max_per_ticker as well to avoid swamping the digest.
        ticker_counts: dict[str, int] = {}
        ranked = []
        for a in above:
            tag   = a.get("ticker_tag", "")
            count = ticker_counts.get(tag, 0)
            if count < max_per_ticker:
                ranked.append(a)
                ticker_counts[tag] = count + 1
 
        logger.info(
            f"_run_score [{category}]: {len(scored)} scored, "
            f"{len(above)} above threshold ({threshold}), "
            f"{len(ranked)} after per-ticker cap ({max_per_ticker}/ticker)"
        )
 
    # MACRO / AI: original top-N behaviour 
    else:
        max_items_key = f"max_items_{category}"
        max_items     = config.get(max_items_key, 4)
 
        above = [a for a in scored if a["final_score"] >= 0.3]
 
        if len(above) < 2 and scored:
            logger.info(
                f"_run_score [{category}]: fewer than 2 above threshold, "
                f"relaxing novelty weight"
            )
            above = _rescore_without_novelty(scored, config)
 
        ranked = above[:max_items]
        logger.info(
            f"_run_score [{category}]: {len(scored)} scored, "
            f"{len(ranked)} selected for digest"
        )
 
    return ranked


# ---------------------------------------------------------------------------
# Shared private helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _recency_score(published_at: str, window_days: int) -> float:
    if not published_at:
        return 0.4
    try:
        pub = datetime.fromisoformat(published_at)
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.4
    age    = (datetime.now(timezone.utc) - pub).total_seconds()
    window = window_days * 86400
    if age <= 0:
        return 1.0
    if age >= window:
        return 0.4
    return round(max(0.4, min(1.0, 1.0 - (0.6 * (age / window)))), 4)


def _rescore_without_novelty(articles: list[dict], config: dict) -> list[dict]:
    w_r = config.get("weight_relevance", 0.40)
    w_c = config.get("weight_recency",   0.20)
    w_t = config.get("weight_tier",      0.15)
    w_n = config.get("weight_novelty",   0.25)
    scale = (w_r + w_c + w_t + w_n) / (w_r + w_c + w_t)
    rescored = []
    for a in articles:
        a = dict(a)
        a["final_score"] = round(max(0.0, min(1.0,
            (w_r * scale * a.get("_relevance_proxy", 0.5))
            + (w_c * scale * a.get("_recency_score",   0.5))
            + (w_t * scale * a.get("_tier_score",      0.7))
        )), 4)
        rescored.append(a)
    rescored.sort(key=lambda a: a["final_score"], reverse=True)
    return rescored


# ===========================================================================
# Node 5: llm_summarize
# ===========================================================================

def llm_summarize(state: PipelineState) -> dict:
    """
    LLM Gate 2 — summarize each ranked article.
    Receives the combined ranked list from all three parallel branches.
    """
    from llm import run_summarize

    # state["ranked"] is already the merged list from all three branches
    ranked = state["ranked"]
    config = state["config"]
    new_errors = []

    if not ranked:
        logger.warning("llm_summarize: no ranked articles to summarize")
        return {"summarized": [], "errors": new_errors}

    # Sort combined ranked list by category then score for consistent ordering
    ranked_sorted = sorted(
        ranked,
        key=lambda a: (
            [STOCKS, MACRO, AI].index(a.get("category", STOCKS)),
            -a.get("final_score", 0),
        )
    )

    summarized     = []
    llm_count      = 0
    fallback_count = 0

    for article in ranked_sorted:
        article  = dict(article)
        headline = article.get("headline", "")[:60]

        try:
            result = run_summarize(config, article)
            article["bullets"]        = result["bullets"]
            article["market_impact"]  = result["market_impact"]
            article["investor_angle"] = result["investor_angle"]
            article["sentiment"]      = result["sentiment"]
            article["summary_source"] = result["summary_source"]

            if result["summary_source"] == "llm":
                llm_count += 1
                try:
                    summary_text = " ".join(result["bullets"])
                    if result["market_impact"]:
                        summary_text += " " + result["market_impact"]
                    if result["investor_angle"]:
                        summary_text += " " + result["investor_angle"]
                    update_summary(article["url"], summary_text)
                except Exception as e:
                    logger.debug(
                        f"llm_summarize: update_summary failed for {headline} — {e}"
                    )
            else:
                fallback_count += 1
                logger.warning(f"llm_summarize: fallback used for {headline}")

        except Exception as e:
            msg = f"llm_summarize: unexpected error for {headline} — {e}"
            logger.warning(msg)
            new_errors.append(msg)
            article["bullets"]        = [article.get("snippet", "No summary available.")]
            article["market_impact"]  = ""
            article["investor_angle"] = ""
            article["sentiment"]      = "mixed"
            article["summary_source"] = "fallback"
            fallback_count += 1

        summarized.append(article)

    logger.info(
        f"llm_summarize: {llm_count} LLM summaries, "
        f"{fallback_count} fallbacks, {len(summarized)} total"
    )

    return {"summarized": summarized, "errors": new_errors}


# ===========================================================================
# Node 6: format_send
# ===========================================================================

def format_send(state: PipelineState) -> dict:
    """Format digest, deliver via Telegram, record sent articles."""
    from telegram import Bot
    from telegram.constants import ParseMode
    from formatter import build_digest

    summarized  = state["summarized"]
    config      = state["config"]
    # state["errors"] contains the full accumulated list from all previous nodes
    all_errors  = list(state.get("errors", []))
    run_id      = config.get("run_id", "unknown")
    conn        = config["db_conn"]
    include_why = config.get("include_why_it_matters", True)
    new_errors  = []

    if not summarized:
        logger.warning("format_send: no summarized articles — sending empty digest")

    messages = build_digest(summarized, include_why=include_why)
    logger.info(f"format_send: {len(messages)} Telegram messages to send")

    # Backup to disk
    today       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    backup_path = Path("data/digests") / f"{today}.txt"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        backup_path.write_text("\n\n---\n\n".join(messages), encoding="utf-8")
        logger.info(f"format_send: backup written to {backup_path}")
    except Exception as e:
        logger.warning(f"format_send: backup write failed — {e}")

    # Telegram delivery
    bot_token       = config.get("telegram_bot_token", "")
    chat_id         = config.get("telegram_chat_id",   "")
    telegram_status = "skipped"

    if not bot_token or not chat_id:
        logger.warning("format_send: Telegram credentials not set — skipping delivery")
    else:
        async def send_all() -> str:
            bot        = Bot(token=bot_token)
            sent_count = 0
            for i, message in enumerate(messages):
                for attempt in range(3):
                    try:
                        await bot.send_message(
                            chat_id    = chat_id,
                            text       = message,
                            parse_mode = ParseMode.HTML,
                        )
                        sent_count += 1
                        if i < len(messages) - 1:
                            await asyncio.sleep(1)
                        break
                    except Exception as e:
                        if attempt < 2:
                            logger.warning(
                                f"format_send: message {i+1} attempt "
                                f"{attempt+1} failed — {e}"
                            )
                            await asyncio.sleep(2 ** attempt)
                        else:
                            msg = (
                                f"format_send: message {i+1} failed "
                                f"after 3 attempts — {e}"
                            )
                            logger.error(msg)
                            new_errors.append(msg)
            if sent_count == len(messages):
                return "success"
            return "partial" if sent_count > 0 else "failed"

        try:
            telegram_status = asyncio.run(send_all())
        except RuntimeError:
            import nest_asyncio
            nest_asyncio.apply()
            telegram_status = asyncio.get_event_loop().run_until_complete(send_all())

        logger.info(f"format_send: Telegram status — {telegram_status}")

    # Mark sent
    sent_at = datetime.now(timezone.utc).isoformat()
    to_mark = [
        {
            "url":          a["url"],
            "content_hash": a.get("content_hash", ""),
            "category":     a.get("category", ""),
            "sent_at":      sent_at,
        }
        for a in summarized
    ]
    if to_mark:
        try:
            mark_sent(conn, to_mark)
            logger.info(f"format_send: {len(to_mark)} articles marked as sent")
        except Exception as e:
            msg = f"format_send: mark_sent failed — {e}"
            logger.error(msg)
            new_errors.append(msg)

    # Write run record, including all accumulated errors from the whole run
    final_errors = all_errors + new_errors
    status = (
        "success" if telegram_status in ("success", "skipped")
        else "partial" if telegram_status == "partial"
        else "failed"
    )
    try:
        conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, started_at, completed_at, status, items_sent, errors) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                config.get("_started_at", sent_at),
                datetime.now(timezone.utc).isoformat(),
                status,
                len(to_mark),
                json.dumps(final_errors),
            ),
        )
        conn.commit()
        logger.info(
            f"format_send: run record written — "
            f"status={status}, items_sent={len(to_mark)}"
        )
    except Exception as e:
        logger.warning(f"format_send: run record write failed — {e}")

    return {"errors": new_errors}