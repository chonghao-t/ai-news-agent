import json
import logging
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from schema import STOCKS, MACRO, AI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM = """You are a financial and technology news editor.
Your job is to select the most relevant articles for a personal morning digest.
 
The reader's stock watchlist: {tickers}
Macroeconomic themes of interest: {macro_themes}
AI/technology themes of interest: {ai_themes}
 
You will receive three numbered lists of article candidates — one for stocks, \
one for macro, one for AI. Each entry shows: index. [Source] Headline — Snippet.
 
Return a JSON object with exactly three keys: "stocks", "macro", "ai".
Each value is a list of integer indices from the corresponding input list.
 
Selection rules:
- Stocks: prefer direct company events (earnings, guidance, analyst upgrades/\
downgrades, M&A, leadership changes, regulatory actions). Deprioritise general \
market commentary that only mentions a ticker in passing.
- Macro: prefer Fed communications, official data releases (CPI, NFP, GDP, PCE),\
 major central bank decisions, significant policy shifts. Deprioritise opinion \
pieces restating already-covered data.
- AI: prefer major model releases, significant research advances, regulatory \
developments, enterprise deployment news. Deprioritise incremental benchmarks \
and hype pieces without new factual content.
 
Return FEWER indices if fewer articles are genuinely relevant.
Return at most {max_stocks} stocks indices, {max_macro} macro indices, \
{max_ai} ai indices.
Return only the JSON object. No explanation. No markdown. No other text."""
 
TRIAGE_USER = """STOCKS CANDIDATES:
{stocks_list}
 
MACRO CANDIDATES:
{macro_list}
 
AI CANDIDATES:
{ai_list}
 
Return JSON:"""
 
# ---------------------------------------------------------------------------
 
SUMMARIZE_SYSTEM_STOCKS = """You are a concise financial analyst writing a \
personal morning briefing.
 
Summarize the following article for an investor whose watchlist includes: \
{tickers}
 
Return a JSON object with exactly four keys:
- "bullets": a list of {max_bullets} strings, each 1-2 sentences, covering \
the most important facts
- "market_impact": a single sentence on the direct implication for the stock \
or sector (e.g. "Positive read for semiconductor equipment names.")
- "investor_angle": a single sentence on what the investor should watch or \
act on as a result of this development
- "sentiment": exactly one of "bullish", "bearish", or "mixed" — your \
assessment of whether this news is net positive, net negative, or ambiguous \
for the relevant stock or sector
 
Focus on: the key event (earnings/guidance/rating change/corporate action), \
magnitude if stated, and direct implication for the stock or sector.
 
The summary MUST be grounded in the ARTICLE text.
Prior context is background only — do not introduce facts not in the ARTICLE.
Return only the JSON object. No markdown. No other text."""
 
SUMMARIZE_SYSTEM_MACRO = """You are a concise macroeconomic analyst writing a \
personal morning briefing.
 
Summarize the following article for an investor interested in: {macro_themes}
 
Return a JSON object with exactly four keys:
- "bullets": a list of {max_bullets} strings, each 1-2 sentences, covering \
the most important facts
- "market_impact": a single sentence on the market or policy implication \
(e.g. "Likely to sustain elevated rate expectations across the curve.")
- "investor_angle": a single sentence on what the investor should watch or \
position around as a result
- "sentiment": exactly one of "bullish", "bearish", or "mixed" — your \
assessment of whether this development is net positive, net negative, or \
ambiguous for risk assets
 
Focus on: what changed, versus prior expectation, central bank reaction, \
and market implication.
 
The summary MUST be grounded in the ARTICLE text.
Prior context is background only — do not introduce facts not in the ARTICLE.
Return only the JSON object. No markdown. No other text."""
 
SUMMARIZE_SYSTEM_AI = """You are a concise AI industry analyst writing a \
personal morning briefing.
 
Summarize the following article for a reader interested in: {ai_themes}
 
Return a JSON object with exactly four keys:
- "bullets": a list of {max_bullets} strings, each 1-2 sentences, covering \
the most important facts
- "market_impact": a single sentence on the industry or commercial significance \
(e.g. "Strengthens the case for enterprise AI adoption in regulated sectors.")
- "investor_angle": a single sentence on what this means for investors or \
practitioners in the AI space
- "sentiment": exactly one of "bullish", "bearish", or "mixed" — your \
assessment of whether this development is net positive, net negative, or \
ambiguous for the AI sector
 
Focus on: what the development is, who made it, what is technically or \
commercially meaningful, and whether it represents a genuine advance.
 
The summary MUST be grounded in the ARTICLE text.
Prior context is background only — do not introduce facts not in the ARTICLE.
Return only the JSON object. No markdown. No other text."""
 
SUMMARIZE_USER = """ARTICLE:
{article_text}
 
{prior_context_block}"""
 
PRIOR_CONTEXT_BLOCK = """PRIOR CONTEXT (background only — do not summarise \
this; use only for framing):
{prior_summaries}"""

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_llm(config: dict, temperature: float) -> ChatOpenAI:
    """Return a ChatOpenAI instance for the given temperature."""
    return ChatOpenAI(
        model       = config["openai_model"],
        temperature = temperature,
        max_tokens  = config["openai_max_tokens"],
        api_key     = config["openai_api_key"],
        model_kwargs = {"response_format": {"type": "json_object"}},
    )


# ---------------------------------------------------------------------------
# JSON output parser
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    """
    Extract a JSON object from an LLM response string.
    Handles cases where the model wraps output in markdown code fences
    despite being instructed not to.
    """
    # Strip markdown code fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",          "", text)
    text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

def _format_candidate_list(articles: list[dict]) -> str:
    """
    Format a list of article candidates into a numbered string for the prompt.
    """
    if not articles:
        return "(none)"
    lines = []
    for i, a in enumerate(articles):
        snippet = a.get("snippet", "")[:150].replace("\n", " ")
        lines.append(
            f"{i}. [{a['source']}] {a['headline']} — {snippet}"
        )
    return "\n".join(lines)


def run_triage(config: dict, candidates: list[dict]) -> dict:
    """
    Ask the LLM to score and select candidate articles by index.

    Returns a dict:
        {
            "stocks": [2, 5, 8],
            "macro":  [0, 3],
            "ai":     [1, 4, 6],
        }

    Falls back to deterministic top-N selection on any failure.
    """

    # Split candidates by category
    by_cat = {STOCKS: [], MACRO: [], AI: []}
    for a in candidates:
        cat = a.get("category")
        if cat in by_cat:
            by_cat[cat].append(a)

    # Build prompt
    system_prompt = TRIAGE_SYSTEM.format(
        tickers      = ", ".join(config.get("tickers", [])),
        macro_themes = ", ".join(config.get("macro_themes", [])),
        ai_themes    = ", ".join(config.get("ai_themes", [])),
        max_stocks   = config.get("max_triage_stocks", 8),
        max_macro    = config.get("max_triage_macro",  6),
        max_ai       = config.get("max_triage_ai",     8),
    )

    user_prompt = TRIAGE_USER.format(
        stocks_list = _format_candidate_list(by_cat[STOCKS]),
        macro_list  = _format_candidate_list(by_cat[MACRO]),
        ai_list     = _format_candidate_list(by_cat[AI]),
    )

    llm = get_llm(config, temperature=config.get("llm_temperature_triage", 0.1))

    # Try LLM, retry once on failure
    raw_text = None
    for attempt in range(2):
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            raw_text = response.content
            break
        except Exception as e:
            logger.warning(f"run_triage: LLM call failed (attempt {attempt+1}): {e}")

    if raw_text is None:
        logger.warning("run_triage: both attempts failed — using deterministic fallback")
        return _triage_fallback(by_cat, config)

    # Parse output
    try:
        result = _parse_json(raw_text)
    except Exception as e:
        logger.warning(f"run_triage: JSON parse failed ({e}) — using fallback")
        return _triage_fallback(by_cat, config)

    # Validate and enforce quotas
    validated = _validate_triage_output(result, by_cat, config)
    logger.info(
        f"run_triage: selected "
        f"{len(validated[STOCKS])} stocks, "
        f"{len(validated[MACRO])} macro, "
        f"{len(validated[AI])} ai"
    )
    return validated


def _validate_triage_output(
    raw: dict,
    by_cat: dict,
    config: dict,
) -> dict:
    """
    Validate LLM triage output:
    - Each value must be a list of integers
    - Each index must be in range for its category list
    - Enforce per-category hard quotas
    """

    quotas = {
        STOCKS: config.get("max_triage_stocks", 8),
        MACRO:  config.get("max_triage_macro",  6),
        AI:     config.get("max_triage_ai",     8),
    }

    validated = {}
    for cat in (STOCKS, MACRO, AI):
        raw_indices = raw.get(cat, [])

        # Ensure it's a list
        if not isinstance(raw_indices, list):
            raw_indices = []

        # Keep only valid in-range integers
        max_idx = len(by_cat[cat])
        clean = [
            int(i) for i in raw_indices
            if isinstance(i, (int, float)) and 0 <= int(i) < max_idx
        ]

        # Enforce quota
        validated[cat] = clean[:quotas[cat]]

    return validated


def _triage_fallback(by_cat: dict, config: dict) -> dict:
    """
    Deterministic fallback: take top-N per category sorted by
    source tier (ascending = best first) then recency (descending).
    """

    quotas = {
        STOCKS: config.get("max_triage_stocks", 8),
        MACRO:  config.get("max_triage_macro",  6),
        AI:     config.get("max_triage_ai",     8),
    }

    result = {}
    for cat in (STOCKS, MACRO, AI):
        articles = by_cat[cat]
        sorted_articles = sorted(
            enumerate(articles),
            key=lambda x: (x[1].get("tier", 9), x[1].get("published_at", "")),
            reverse=False,  # tier 1 first; within tier, most recent last so we reverse published_at
        )
        # Re-sort: tier ascending, published_at descending
        sorted_articles = sorted(
            enumerate(articles),
            key=lambda x: (x[1].get("tier", 9), x[1].get("published_at", "")),
        )
        # Reverse published_at within same tier: secondary sort descending
        sorted_articles = sorted(
            enumerate(articles),
            key=lambda x: (x[1].get("tier", 9), "~" + x[1].get("published_at", "")),
        )
        indices = [i for i, _ in sorted_articles[:quotas[cat]]]
        result[cat] = indices

    return result


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

# Sentinel values for sentiment fallback per category
_SENTIMENT_FALLBACK = "mixed"
 
VALID_SENTIMENTS = {"bullish", "bearish", "mixed"}

def run_summarize(config: dict, article: dict) -> dict:
    """
    Summarize a single article. Called once per selected article in Node 7.
 
    Returns:
        {
            "bullets":        ["...", "...", "..."],
            "market_impact":  "...",
            "investor_angle": "...",
            "sentiment":      "bullish" | "bearish" | "mixed",
            "summary_source": "llm" | "fallback",
        }
    """

    category = article.get("category", STOCKS)

    # Pick the right system prompt for this category
    system_templates = {
        STOCKS: SUMMARIZE_SYSTEM_STOCKS,
        MACRO:  SUMMARIZE_SYSTEM_MACRO,
        AI:     SUMMARIZE_SYSTEM_AI,
    }
    system_template = system_templates.get(category, SUMMARIZE_SYSTEM_STOCKS)

    system_prompt = system_template.format(
        tickers      = ", ".join(config.get("tickers", [])),
        macro_themes = ", ".join(config.get("macro_themes", [])),
        ai_themes    = ", ".join(config.get("ai_themes", [])),
        max_bullets  = config.get("summary_max_bullets", 4),
    )

    # Build prior context block if RAG context is available
    rag_context = article.get("rag_context", [])
    if rag_context:
        prior_context_block = PRIOR_CONTEXT_BLOCK.format(
            prior_summaries="\n---\n".join(rag_context)
        )
    else:
        prior_context_block = ""

    # Use full_text if available, fall back to snippet
    article_text = article.get("full_text") or article.get("snippet", "")

    user_prompt = SUMMARIZE_USER.format(
        article_text        = article_text[:6000],  # hard cap for context window
        prior_context_block = prior_context_block,
    )

    llm = get_llm(
        config,
        temperature=config.get("llm_temperature_summarize", 0.3)
    )

    # Try LLM, retry once with a repair instruction on parse failure
    for attempt in range(2):
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            parsed = _parse_json(response.content)

            bullets = parsed.get("bullets", [])
            market_impact = parsed.get("market_impact", "")
            investor_angle = parsed.get("investor_angle", "")
            sentiment     = parsed.get("sentiment", _SENTIMENT_FALLBACK).lower().strip()

            if not isinstance(bullets, list) or len(bullets) == 0:
                raise ValueError("bullets missing or empty")

            return {
                "bullets":        [str(b) for b in bullets],
                "market_impact":  str(market_impact),
                "investor_angle": str(investor_angle),
                "sentiment":      sentiment,
                "summary_source": "llm",
            }

        except Exception as e:
            if attempt == 0:
                logger.warning(
                    f"run_summarize: attempt 1 failed ({e}), retrying..."
                )
                # Add repair instruction to user prompt for retry
                user_prompt += (
                    "\n\nIMPORTANT: Return ONLY a valid JSON object with keys "
                    "'bullets' (list of strings), "
                    "'market_impact' (string), "
                    "'investor_angle' (string), and "
                    "'sentiment' (one of: ""bullish, bearish, mixed). Nothing else."
                )
            else:
                logger.warning(
                    f"run_summarize: both attempts failed ({e}), "
                    "using snippet fallback"
                )

    # Fallback — use snippet as sole bullet
    return {
        "bullets":        [article.get("snippet", "No summary available.")],
        "market_impact":  "",
        "investor_angle": "",
        "sentiment":      _SENTIMENT_FALLBACK,
        "summary_source": "fallback",
    }