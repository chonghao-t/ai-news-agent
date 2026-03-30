# Morning News Digest Agent

> A fully automated AI pipeline that delivers a personalised daily financial and technology news digest to Telegram every morning.

## What it does

The agent runs every weekday morning on a cron schedule. Six news sources 
are fetched concurrently, the most relevant articles are selected and 
summarised by an LLM, and then a
structured news briefing is sent directly to Telegram 

### The pipeline

Six news sources are fetched concurrently at startup:

- **Yahoo Finance RSS + Alpha Vantage** — ticker-specific stock news for
  every company in your personal watchlist
- **Google News RSS** — macroeconomic coverage queried per theme
  (e.g. "Fed policy", "inflation", "credit markets")
- **Federal Reserve RSS** — official press releases and policy statements
- **Hugging Face Blog + ArXiv (cs.AI + cs.LG)** — AI research papers and
  industry developments

Raw candidates are deduplicated three ways: against a SQLite database of
previously delivered articles (so the same story never appears twice across
days), by content hash within the current batch, and by a per-category
recency cap.

### LLM triage 

Candidate headlines and snippets are presented to GPT-4o-mini as numbered
lists. The model returns integer indices of the articles worth reading which makes hallucination impossible,
since an out-of-range integer is discarded. Per-category quotas are
enforced set in config regardless of what the model returns.

### Parallel processing

Selected articles are split by category and processed in three simultaneous 
branches using LangGraph's `Send` API. Each branch 
fetches the full article text, strips boilerplate with Trafilatura, embeds 
the content using a local sentence-transformers model (nomic-embed-text-v1), 
and queries ChromaDB for semantic similarity against everything delivered in 
the past 7 days. Articles that are near-duplicates of recent coverage are 
suppressed or excluded entirely.

Each article is then scored on four weighted dimensions:

| Factor | Weight | How it's computed |
|---|---|---|
| Relevance | 40% | LLM triage position (index 0 = highest relevance) |
| Novelty | 25% | 1 − semantic similarity to prior coverage |
| Recency | 20% | Linear decay from publication time to window edge |
| Source quality | 15% | Tier 1 (Reuters, Fed, ArXiv) scores 1.0; lower tiers score less |

For stocks, articles above a configurable score threshold are kept and
capped per ticker to ensure all tickers get fair representation rather than one ticker dominating the section.

### LLM summarisation 

Each selected article is summarised by a category-specific prompt
(financial analyst voice for stocks, macroeconomic analyst for macro, AI
industry analyst for AI). The model produces:

- **Bullets** — 3–4 concise factual sentences grounded in the article text
- **Market impact** — one sentence on the sector or market implication
- **Investor angle** — one sentence on what to watch or act on
- **Sentiment** — the LLM's editorial judgment on whether the news is net 
  positive, net negative, or ambiguous, displayed as 📈 / 📉 / ↔️

Sentiment is determined by the model reading the full article in context. A headline like 
*"Fed holds rates steady"* is bullish or bearish depending on what the 
market expected and what the statement said. The LLM reasons over all of 
that from the article text.

Generated summaries are written back to ChromaDB so tomorrow's run has
today's summaries as prior context.

### Delivery

The digest is formatted as Telegram HTML with dated section headers, numbered
articles, and ticker tags. Three messages are sent, one per section, with
automatic splitting if any section exceeds Telegram's character limit. Each
message is retried up to three times on failure. A plain-text backup is
written to disk regardless of delivery status.

## Architecture

The pipeline is built as a directed graph using **LangGraph**, with a 
linear entry sequence, a parallel fan-out for the three news categories, 
and an automatic fan-in to summarisation.
```
APScheduler (cron)
        │
        ▼
  load_config      ── validates config, opens SQLite, stamps run UUID
        │
        ▼
fetch_candidates   ── 6 adapters concurrently, 3 dedup layers
        │
        ▼
  llm_triage       ── LLM: integer index selection, prune ChromaDB
        │
        ▼ dispatch_to_categories() → list[Send]
        │
  ┌─────┴──────┬──────────────┐
  ▼            ▼              ▼
stocks       macro            ai        ← 3 parallel branches via Send API
fetch+embed  fetch+embed  fetch+embed
detect ticker  RAG query    RAG query
threshold+cap  top-N         top-N
  └─────┬──────┴──────────────┘
        │  operator.add accumulates ranked[] across branches
        ▼
 llm_summarize    ── LLM: bullets, market_impact, investor_angle,
        │             sentiment; writes summaries back to ChromaDB
        ▼
  format_send     ── HTML format, Telegram delivery, mark sent, run record
```

## Tech stack

| Function | Tool |
|---|---|
| Pipeline orchestration | LangGraph |
| LLM prompts / wrappers | LangChain + langchain-openai |
| LLM inference | OpenAI gpt-4o-mini |
| Embeddings | nomic-embed-text-v1 |
| Vector store | ChromaDB |
| Relational persistence | SQLite |
| Article extraction | Trafilatura |
| HTTP client | httpx |
| Scheduling | APScheduler 3.x |
| Delivery | python-telegram-bot |

Sentiment is determined by the model reading the full article in context. A headline like 
*"Fed holds rates steady"* is bullish or bearish depending on what the 
market expected and what the statement said. The LLM reasons over all of 
that from the article text.

Generated summaries are written back to ChromaDB so tomorrow's run has
today's summaries as prior context.

### Delivery

The digest is formatted as Telegram HTML with dated section headers, numbered
articles, and ticker tags. Three messages are sent, one per section, with
automatic splitting if any section exceeds Telegram's character limit. Each
message is retried up to three times on failure. A plain-text backup is
written to disk regardless of delivery status.

## Architecture

The pipeline is built as a directed graph using **LangGraph**, with a 
linear entry sequence, a parallel fan-out for the three news categories, 
and an automatic fan-in to summarisation.
```
APScheduler (cron)
        │
        ▼
  load_config      ── validates config, opens SQLite, stamps run UUID
        │
        ▼
fetch_candidates   ── 6 adapters concurrently, 3 dedup layers
        │
        ▼
  llm_triage       ── LLM: integer index selection, prune ChromaDB
        │
        ▼ dispatch_to_categories() → list[Send]
        │
  ┌─────┴──────┬──────────────┐
  ▼            ▼              ▼
stocks       macro            ai        ← 3 parallel branches via Send API
fetch+embed  fetch+embed  fetch+embed
detect ticker  RAG query    RAG query
threshold+cap  top-N         top-N
  └─────┬──────┴──────────────┘
        │  operator.add accumulates ranked[] across branches
        ▼
 llm_summarize    ── LLM: bullets, market_impact, investor_angle,
        │             sentiment; writes summaries back to ChromaDB
        ▼
  format_send     ── HTML format, Telegram delivery, mark sent, run record
```

## Tech stack

| Function | Tool |
|---|---|
| Pipeline orchestration | LangGraph |
| LLM prompts / wrappers | LangChain + langchain-openai |
| LLM inference | OpenAI gpt-4o-mini |
| Embeddings | nomic-embed-text-v1 |
| Vector store | ChromaDB |
| Relational persistence | SQLite |
| Article extraction | Trafilatura |
| HTTP client | httpx |
| Scheduling | APScheduler 3.x |
| Delivery | python-telegram-bot |
