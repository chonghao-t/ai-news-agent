# formatter.py
import html
from datetime import datetime, timezone
from schema import STOCKS, MACRO, AI

TELEGRAM_MAX_CHARS = 4096

SENTIMENT_EMOJI = {
    "bullish": "📈",
    "bearish": "📉",
    "mixed":   "↔️",
}

def _today_label() -> str:
    """Return today's date in human-readable form, e.g. 'March 29, 2026'."""
    return datetime.now(timezone.utc).strftime("%B %#d, %Y")

def _section_header(category: str) -> str:
    date = _today_label()
    if category == STOCKS:
        return f"📊 <b>Stocks Brief | {date}</b>"
    if category == MACRO:
        return f"🌐 <b>Macroeconomic News | {date}</b>"
    if category == AI:
        return f"🤖 <b>AI Developments | {date}</b>"
    return f"<b>{category.upper()} | {date}</b>"


def esc(text: str) -> str:
    """Escape text for Telegram HTML parse mode."""
    return html.escape(str(text))

def _ticker_prefix(article: dict) -> str:
    """
    Return a formatted ticker prefix for the headline, e.g. '[NVDA] '.
    Uses ticker_tag if present; falls back to scanning the source string.
    """
    tag = article.get("ticker_tag", "").strip()
    if tag:
        return f"[{tag}] "
    # Fallback: extract from source string "Yahoo Finance (NVDA)"
    import re
    m = re.search(r"\(([A-Z]{1,5})\)", article.get("source", ""))
    if m:
        return f"[{m.group(1)}] "
    return ""

def format_article(article: dict, index: int) -> str:
    """
    Format one article block:
 
        N) [TICKER] Headline 📈
        • Bullet one
        • Bullet two
        → Market impact: ...
 
        Investor angle: ...
        Source: URL
    """
    lines = []
 
    # Headline line 
    sentiment     = article.get("sentiment", "mixed").lower()
    emoji         = SENTIMENT_EMOJI.get(sentiment, "↔️")
    ticker_prefix = _ticker_prefix(article)
    headline      = article.get("headline", "No headline")
 
    lines.append(
        f"{index}) {emoji} <b>{esc(ticker_prefix)}{esc(headline)}</b>"
    )
 
    # Bullets 
    bullets = article.get("bullets", [])
    if bullets:
        for bullet in bullets:
            lines.append(f"• {esc(bullet)}")
    else:
        lines.append(f"• {esc(article.get('snippet', 'No summary available.'))}")
 
    # Market impact 
    market_impact = article.get("market_impact", "").strip()
    if market_impact:
        lines.append(f"\n→ <b>Market impact:</b> {esc(market_impact)}")
 
    # Investor angle 
    investor_angle = article.get("investor_angle", "").strip()
    if investor_angle:
        lines.append(f"\n<b>Investor angle:</b> {esc(investor_angle)}")
 
    # Source URL 
    url = article.get("url", "")
    if url:
        lines.append(f"\n<b>Source:</b> {url}")
 
    return "\n".join(lines)


def format_section(category: str, articles: list[dict]) -> str:
    header    = _section_header(category)
    separator = "——————————————"
 
    lines = [header, separator]
 
    if not articles:
        lines.append("No significant items today.")
    else:
        blocks = [
            format_article(a, index=i + 1)
            for i, a in enumerate(articles)
        ]
        lines.append(("\n\n" + separator + "\n\n").join(blocks))
 
    return "\n".join(lines)


def split_message(text: str, max_chars: int = TELEGRAM_MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    parts   = text.split("\n\n")
    chunks  = []
    current = ""

    for part in parts:
        candidate = current + ("\n\n" if current else "") + part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > max_chars:
                chunks.append(part[:max_chars - 3] + "...")
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:max_chars]]


def build_digest(summarized: list[dict], include_why: bool = True) -> list[str]:
    """
    Group summarised articles by category, format each section,
    split any section that exceeds the Telegram limit.
    """
    by_cat = {STOCKS: [], MACRO: [], AI: []}
    for article in summarized:
        cat = article.get("category")
        if cat in by_cat:
            by_cat[cat].append(article)
 
    messages = []
    for cat in (STOCKS, MACRO, AI):
        section_text = format_section(cat, by_cat[cat])
        chunks = split_message(section_text)
        if len(chunks) > 1:
            for i, chunk in enumerate(chunks, 1):
                messages.append(f"{chunk}\n\n(Part {i}/{len(chunks)})")
        else:
            messages.append(chunks[0])
 
    return messages