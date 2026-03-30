import feedparser
import httpx
import logging
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

from schema import make_article, STOCKS, MACRO, AI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_date(value: str | None) -> datetime | None:
    """
    Parse a date string into a timezone-aware datetime.
    Handles ISO 8601 and RFC 2822 (the two formats RSS feeds use).
    Returns None if parsing fails.
    """
    if not value:
        return None
    try:
        # Try ISO 8601 first (e.g. "2024-01-15T10:30:00Z")
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    try:
        # Try RFC 2822 (e.g. "Mon, 15 Jan 2024 10:30:00 +0000")
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    return None


def _is_recent(published_at: datetime | None, days: int) -> bool:
    """Return True if the article was published within the past N days."""
    if published_at is None:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return published_at >= cutoff


def _safe_snippet(text: str | None, max_chars: int = 400) -> str:
    """Clean and truncate a snippet string."""
    if not text:
        return ""
    # Strip basic HTML tags that sometimes appear in RSS descriptions
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())  # normalise whitespace
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Adapter 1 — Yahoo Finance RSS (stocks)
# ---------------------------------------------------------------------------

def fetch_yahoo_finance_rss(config: dict) -> list[dict]:
    """
    Fetch recent news for each ticker in the watchlist from Yahoo Finance RSS.
    One HTTP request per ticker.
    """
    tickers   = config.get("tickers", [])
    days      = config.get("article_date_window_days", 7)
    articles  = []
    seen_urls = set()

    for ticker in tickers:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                article_url = entry.get("link", "")
                if not article_url or article_url in seen_urls:
                    continue

                published_at = _parse_date(entry.get("published"))
                if not _is_recent(published_at, days):
                    continue

                seen_urls.add(article_url)
                articles.append(make_article(
                    url          = article_url,
                    headline     = entry.get("title", "").strip(),
                    snippet      = _safe_snippet(entry.get("summary")),
                    source       = f"Yahoo Finance ({ticker})",
                    tier         = 1,
                    category     = STOCKS,
                    published_at = published_at.isoformat(),
                ))
        except Exception as e:
            logger.warning(f"yahoo_finance_rss [{ticker}]: {e}")

    logger.info(f"yahoo_finance_rss: {len(articles)} candidates")
    return articles


# ---------------------------------------------------------------------------
# Adapter 2 — Alpha Vantage News API (stocks)
# ---------------------------------------------------------------------------

def fetch_alpha_vantage(config: dict) -> list[dict]:
    """
    Fetch news from Alpha Vantage News & Sentiments API for the watchlist.
    """
    api_key = config.get("alpha_vantage_api_key", "")
    if not api_key:
        logger.warning("alpha_vantage: ALPHA_VANTAGE_API_KEY not set — skipping")
        return []
 
    tickers   = config.get("tickers", [])
    days      = config.get("article_date_window_days", 7)
    articles  = []
    seen_urls = set()
 
    for ticker in tickers:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT"
            f"&tickers={ticker}"
            f"&limit=50"
            f"&apikey={api_key}"
        )
 
        try:
            response = httpx.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
 
            # Rate-limit or plan restriction — surface clearly and stop
            if "Information" in data or "Note" in data:
                msg = data.get("Information") or data.get("Note", "")
                logger.warning(f"alpha_vantage [{ticker}]: API message — {msg}")
                break  # no point hammering a rate-limited key further
 
            for item in data.get("feed", []):
                article_url = item.get("url", "")
                if not article_url or article_url in seen_urls:
                    continue
 
                # Alpha Vantage time format: "20240115T103000"
                raw_time = item.get("time_published", "")
                try:
                    published_at = datetime.strptime(raw_time, "%Y%m%dT%H%M%S")
                    published_at = published_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    logger.debug(
                        f"alpha_vantage [{ticker}]: unparseable date '{raw_time}' — skipping"
                    )
                    continue
 
                if not _is_recent(published_at, days):
                    continue
 
                seen_urls.add(article_url)
                articles.append(make_article(
                    url          = article_url,
                    headline     = item.get("title", "").strip(),
                    snippet      = _safe_snippet(item.get("summary")),
                    source       = f"Alpha Vantage ({ticker})",
                    tier         = 1,
                    category     = STOCKS,
                    published_at = published_at.isoformat(),
                ))
 
            logger.debug(f"alpha_vantage [{ticker}]: {len(articles)} cumulative candidates")
 
        except Exception as e:
            logger.warning(f"alpha_vantage [{ticker}]: {e}")
 
    logger.info(f"alpha_vantage: {len(articles)} candidates across {len(tickers)} tickers")
    return articles

# ---------------------------------------------------------------------------
# Adapter 3 — Reuters RSS (macro)
# ---------------------------------------------------------------------------

def fetch_macro_news(config: dict) -> list[dict]:
    """
    Fetch macroeconomic news via Google News RSS.
    """
    days   = config.get("article_date_window_days", 7)
    themes = config.get("macro_themes", [])
 
    articles  = []
    seen_urls = set()
 
    for theme in themes:
        # Google News RSS — searches across all indexed news sources
        encoded_theme = theme.replace(" ", "+")
        url = (
            f"https://news.google.com/rss/search"
            f"?q={encoded_theme}"
            f"&hl=en-US&gl=US&ceid=US:en"
        )
 
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                article_url = entry.get("link", "")
                if not article_url or article_url in seen_urls:
                    continue
 
                published_at = _parse_date(entry.get("published"))
                if not _is_recent(published_at, days):
                    continue
 
                headline = entry.get("title", "").strip()
                snippet  = _safe_snippet(entry.get("summary"))
 
                seen_urls.add(article_url)
                articles.append(make_article(
                    url          = article_url,
                    headline     = headline,
                    snippet      = snippet,
                    source       = entry.get("source", {}).get("title", "Google News")
                                   if hasattr(entry.get("source", ""), "get")
                                   else "Google News",
                    tier         = 1,
                    category     = MACRO,
                    published_at = published_at.isoformat(),
                ))
 
        except Exception as e:
            logger.warning(f"fetch_macro_news [{theme}]: {e}")
 
    logger.info(f"fetch_macro_news: {len(articles)} candidates across {len(themes)} themes")
    return articles


# ---------------------------------------------------------------------------
# Adapter 4 — Federal Reserve RSS (macro)
# ---------------------------------------------------------------------------

def fetch_fed_rss(config: dict) -> list[dict]:
    """
    Fetch press releases from the Federal Reserve.
    All Fed releases are macro-relevant by definition — no keyword filter needed.
    """
    days = config.get("article_date_window_days", 7)
    url  = "https://www.federalreserve.gov/feeds/press_all.xml"
    articles = []

    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            article_url = entry.get("link", "")
            if not article_url:
                continue

            published_at = _parse_date(entry.get("published"))
            if not _is_recent(published_at, days):
                continue

            articles.append(make_article(
                url          = article_url,
                headline     = entry.get("title", "").strip(),
                snippet      = _safe_snippet(entry.get("summary")),
                source       = "Federal Reserve",
                tier         = 1,
                category     = MACRO,
                published_at = published_at.isoformat(),
            ))

    except Exception as e:
        logger.warning(f"fed_rss: {e}")

    logger.info(f"fed_rss: {len(articles)} candidates")
    return articles


# ---------------------------------------------------------------------------
# Adapter 5 — Hugging Face Blog RSS (AI)
# ---------------------------------------------------------------------------

def fetch_huggingface_blog_rss(config: dict) -> list[dict]:
    """
    Fetch posts from the Hugging Face blog.
    Filters by ai_themes keywords.
    """
    days   = config.get("article_date_window_days", 7)
    themes = [t.lower() for t in config.get("ai_themes", [])]
    url    = "https://huggingface.co/blog/feed.xml"
    articles = []

    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            article_url = entry.get("link", "")
            if not article_url:
                continue

            published_at = _parse_date(entry.get("published"))
            if not _is_recent(published_at, days):
                continue

            headline = entry.get("title", "").strip()
            snippet  = _safe_snippet(entry.get("summary"))

            combined = (headline + " " + snippet).lower()
            if themes and not any(theme in combined for theme in themes):
                continue

            articles.append(make_article(
                url          = article_url,
                headline     = headline,
                snippet      = snippet,
                source       = "Hugging Face Blog",
                tier         = 1,
                category     = AI,
                published_at = published_at.isoformat(),
            ))

    except Exception as e:
        logger.warning(f"huggingface_blog_rss: {e}")

    logger.info(f"huggingface_blog_rss: {len(articles)} candidates")
    return articles


# ---------------------------------------------------------------------------
# Adapter 6 — ArXiv RSS (AI)
# ---------------------------------------------------------------------------

def fetch_arxiv_rss(config: dict) -> list[dict]:
    """
    Fetch recent AI/ML papers from ArXiv.
 
    ArXiv RSS (rss.arxiv.org) contains only papers from the current day's
    announcement batch, published once daily around 20:00 ET on weekdays.
    Outside that window the feed is valid XML but contains zero <item>
    elements — not an error, just an empty batch.

    When RSS returns 0 entries, we fall back to the ArXiv Atom API
    (export.arxiv.org/api/query) which returns papers on demand regardless
    of time of day and is not tied to the announcement cycle.

    feedparser.parse(url) sends 'python-feedparser' as its UA which gets
    blocked in some environments. Both fetches use httpx with a browser UA
    and pass raw bytes to feedparser.parse(), which accepts bytes identically
    to a URL.
    """
    days      = config.get("article_date_window_days", 7)
    ai_themes = config.get("ai_themes", [])
    articles  = []
    seen_urls = set()
 
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }
 
    # RSS 
    for category, feed_url in [
        ("cs.AI", "https://rss.arxiv.org/rss/cs.AI"),
        ("cs.LG", "https://rss.arxiv.org/rss/cs.LG"),
    ]:
        try:
            response = httpx.get(
                feed_url, headers=HEADERS, timeout=20, follow_redirects=True
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            logger.debug(
                f"arxiv_rss [{category}]: {len(feed.entries)} entries in RSS batch"
            )
            for entry in feed.entries:
                article_url = entry.get("link", "")
                if not article_url or article_url in seen_urls:
                    continue
                raw_date     = entry.get("published") or entry.get("updated")
                published_at = _parse_date(raw_date)
                if published_at is None:
                    published_at = datetime.now(timezone.utc)
                elif not _is_recent(published_at, days):
                    continue
                headline = entry.get("title", "").replace("\n", " ").strip()
                snippet  = _safe_snippet(entry.get("summary"), max_chars=600)
                seen_urls.add(article_url)
                articles.append(make_article(
                    url          = article_url,
                    headline     = headline,
                    snippet      = snippet,
                    source       = f"ArXiv ({category})",
                    tier         = 1,
                    category     = AI,
                    published_at = published_at.isoformat(),
                ))
        except Exception as e:
            logger.warning(f"arxiv_rss [{category}]: {e}")
 
    # Atom API fallback
    if not articles:
        logger.info(
            "arxiv_rss: RSS batch empty — falling back to Atom API"
        )
        articles = _fetch_arxiv_atom(ai_themes, days, seen_urls, HEADERS)
 
    logger.info(f"arxiv_rss: {len(articles)} candidates total")
    return articles
 
 
def _fetch_arxiv_atom(
    ai_themes: list[str],
    days: int,
    seen_urls: set,
    headers: dict,
) -> list[dict]:
    articles = []
 
    # Use themes as search terms; fall back to a broad category query if empty
    queries = ai_themes if ai_themes else ["machine learning"]
 
    for theme in queries:
        # Quote multi-word phrases for exact matching in the all: field
        phrase = f'"{theme}"' if " " in theme else theme
        encoded = phrase.replace(" ", "+")
        query = f"(cat:cs.AI+OR+cat:cs.LG)+AND+all:{encoded}"
        url = (
            "https://export.arxiv.org/api/query"
            f"?search_query={query}"
            f"&start=0&max_results=15"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
 
        try:
            response = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
 
            logger.debug(
                f"arxiv_atom [{theme}]: {len(feed.entries)} entries returned"
            )
 
            for entry in feed.entries:
                article_url = entry.get("link", "")
                if not article_url or article_url in seen_urls:
                    continue
                raw_date     = entry.get("published") or entry.get("updated")
                published_at = _parse_date(raw_date)
                if published_at is None:
                    published_at = datetime.now(timezone.utc)
                elif not _is_recent(published_at, days):
                    continue
                headline = entry.get("title", "").replace("\n", " ").strip()
                snippet  = _safe_snippet(entry.get("summary"), max_chars=600)
                seen_urls.add(article_url)
                articles.append(make_article(
                    url          = article_url,
                    headline     = headline,
                    snippet      = snippet,
                    source       = "ArXiv",
                    tier         = 1,
                    category     = AI,
                    published_at = published_at.isoformat(),
                ))
 
        except Exception as e:
            logger.warning(f"arxiv_atom [{theme}]: {e}")
 
    logger.info(f"arxiv_atom: {len(articles)} candidates from Atom API fallback")
    return articles

# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------
# Maps source name (from sources.yaml) to its fetch function.

ADAPTER_REGISTRY = {
    "yahoo_finance_rss":    fetch_yahoo_finance_rss,
    "alpha_vantage":        fetch_alpha_vantage,
    "reuters_rss":          fetch_macro_news,
    "fed_rss":              fetch_fed_rss,
    "huggingface_blog_rss": fetch_huggingface_blog_rss,
    "arxiv_rss":            fetch_arxiv_rss,
}