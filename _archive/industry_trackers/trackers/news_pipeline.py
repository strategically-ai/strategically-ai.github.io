"""RSS-based external news ingestion: per-company feeds (Google News + Yahoo Finance) + industry-wide feeds."""

import os
import re
import time
import urllib.parse
from urllib.parse import urlparse

import pandas as pd

from trackers.io import run_timestamp, write_posts_table
from trackers.universe import load_all_segment_universes

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    import yaml
except ImportError:
    yaml = None

# Delays
FEED_FETCH_DELAY = 1.5          # industry-wide RSS feeds
NEWS_COMPANY_FETCH_DELAY = 2.0  # per-company feeds (Google News, Yahoo Finance)

# Per-company RSS URL templates
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
YAHOO_FINANCE_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

# Repo root for config path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_FEEDS_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "news_feeds.yaml")

# M&A keyword pattern
_MA_KEYWORDS = re.compile(
    r"\b(acquisition|acquired|acquires|merger|m&a|mergers?\s+and\s+acquisitions|"
    r"deal\s+to\s+acquire|buyout|takeover|divestiture|spinoff|spin-off|"
    r"definitive\s+agreement|tender\s+offer)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def _google_news_url(company_name: str, ticker: str) -> str:
    """Build Google News RSS search URL for a company."""
    query = urllib.parse.quote(f'"{company_name}" OR {ticker}')
    return GOOGLE_NEWS_RSS.format(query=query)


def _yahoo_finance_url(ticker: str) -> str:
    """Build Yahoo Finance RSS URL for a ticker."""
    return YAHOO_FINANCE_RSS.format(ticker=urllib.parse.quote(ticker))


# ---------------------------------------------------------------------------
# Industry-wide feed config loader
# ---------------------------------------------------------------------------

def _load_feed_urls(industry_id: str) -> list[str]:
    """Load industry-wide RSS feed URLs from config for the industry (or default)."""
    if not yaml:
        return []
    if not os.path.isfile(NEWS_FEEDS_CONFIG_PATH):
        return []
    try:
        with open(NEWS_FEEDS_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
    except Exception:
        return []
    if not data or not isinstance(data, dict):
        return []
    urls = data.get(industry_id) or data.get("default") or []
    return urls if isinstance(urls, list) else []


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_entry_date(entry) -> str:
    """Return publication date as YYYY-MM-DD string if possible."""
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed:
        try:
            return time.strftime("%Y-%m-%d", parsed)
        except Exception:
            pass
    for attr in ("published", "updated", "created"):
        val = getattr(entry, attr, None)
        if val:
            try:
                s = str(val)[:10]
                if len(s) >= 10 and s[4] == "-" and s[7] == "-":
                    return s
            except Exception:
                pass
    return ""


def _source_label(url: str) -> str:
    """Return a short source label (feed domain) for an RSS feed URL."""
    try:
        netloc = urlparse(url).netloc
        return netloc or "rss"
    except Exception:
        return "rss"


def _matches_company(text: str, company_name: str, ticker: str) -> bool:
    """True if company name or ticker (word-boundary) appears in text."""
    if not text:
        return False
    text_lower = text.lower()
    if company_name and company_name.lower() in text_lower:
        return True
    if ticker:
        ticker_esc = re.escape(str(ticker))
        if re.search(r"\b" + ticker_esc + r"\b", text, re.IGNORECASE):
            return True
    return False


def _is_ma_related(text: str) -> bool:
    """True if text mentions M&A or deal activity."""
    return bool(text and _MA_KEYWORDS.search(text))


def _excerpt(summary: str, max_len: int = 500) -> str:
    """Truncate to max_len, strip HTML tags."""
    if not summary:
        return ""
    s = re.sub(r"<[^>]+>", " ", summary)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] + ("..." if len(s) > max_len else "")


def _make_row(
    industry_id: str,
    company_name: str,
    ticker: str,
    segment: str,
    entry,
    source: str,
    source_type: str,
    run_ts: str,
) -> dict:
    """Build a standardised news row dict from a feedparser entry."""
    title = getattr(entry, "title", "") or ""
    link = getattr(entry, "link", "") or ""
    summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
    post_date = _parse_entry_date(entry)
    excerpt = _excerpt(summary)
    text = f"{title} {summary}"
    theme = "ma" if _is_ma_related(text) else "data_strategy"
    return {
        "industry_id": industry_id,
        "company_name": company_name,
        "ticker": ticker,
        "segment": segment,
        "source": source,
        "source_type": source_type,
        "post_date": post_date,
        "post_url": link,
        "title": title,
        "excerpt": excerpt,
        "post_type": "news",
        "theme": theme,
        "signal_strength": "moderate",
        "why_it_matters": "",
        "confidence": 0.5,
        "run_timestamp": run_ts,
    }


# ---------------------------------------------------------------------------
# Per-company feed fetching
# ---------------------------------------------------------------------------

def _fetch_feed_safe(url: str) -> list:
    """Fetch a single RSS feed; return entries or empty list on error."""
    if not feedparser:
        return []
    try:
        feed = feedparser.parse(url)
        return getattr(feed, "entries", []) or []
    except Exception:
        return []


def _fetch_company_feeds(
    industry_id: str,
    company_name: str,
    ticker: str,
    segment: str,
    source_of_truth: str,
    run_ts: str,
) -> list[dict]:
    """
    Fetch Google News RSS + Yahoo Finance RSS for a single company.
    Skips Yahoo Finance for private companies (source_of_truth == 'private_news').
    Returns list of row dicts.
    """
    rows = []
    is_private = str(source_of_truth).strip() == "private_news"

    # Google News RSS — works for both public and private companies
    gn_url = _google_news_url(company_name, ticker)
    gn_entries = _fetch_feed_safe(gn_url)
    time.sleep(NEWS_COMPANY_FETCH_DELAY)
    for entry in gn_entries[:15]:  # cap per company
        rows.append(_make_row(industry_id, company_name, ticker, segment, entry,
                              "news.google.com", "google_news", run_ts))

    # Yahoo Finance RSS — exchange-listed companies only
    if not is_private and ticker:
        yf_url = _yahoo_finance_url(ticker)
        yf_entries = _fetch_feed_safe(yf_url)
        time.sleep(NEWS_COMPANY_FETCH_DELAY)
        for entry in yf_entries[:10]:
            rows.append(_make_row(industry_id, company_name, ticker, segment, entry,
                                  "finance.yahoo.com", "yahoo_finance", run_ts))

    return rows


# ---------------------------------------------------------------------------
# Industry-wide feed fetching (secondary pass)
# ---------------------------------------------------------------------------

def _fetch_industry_feeds(
    industry_id: str,
    universe: pd.DataFrame,
    run_ts: str,
) -> list[dict]:
    """
    Fetch industry-wide RSS feeds; match articles to universe companies by name/ticker.
    This secondary pass catches macro/sector stories not indexed by company-specific feeds.
    """
    urls = _load_feed_urls(industry_id)
    if not urls:
        return []
    rows = []
    for feed_url in urls:
        entries = _fetch_feed_safe(feed_url)
        time.sleep(FEED_FETCH_DELAY)
        source = _source_label(feed_url)
        for entry in entries:
            title = getattr(entry, "title", "") or ""
            link = getattr(entry, "link", "") or ""
            summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
            text = f"{title} {summary} {link}"
            for _, row in universe.iterrows():
                company_name = str(row.get("company_name", ""))
                ticker = str(row.get("ticker", ""))
                segment = str(row.get("segment", "fortune_100"))
                if _matches_company(text, company_name, ticker):
                    rows.append(_make_row(industry_id, company_name, ticker, segment,
                                          entry, source, "rss_industry", run_ts))
    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_industry_news(industry_id: str, universe_version: str = "v1") -> str | None:
    """
    Fetch news for all companies across all three segments (fortune_100, pe_mid, startup).

    Two passes:
    1. Per-company: Google News RSS + Yahoo Finance RSS — gives 10-25 articles per company.
    2. Industry-wide: configured RSS feeds matched by company name/ticker — catches macro stories.

    Deduplicates on (company_name, post_url). Writes combined company_posts table.
    Returns run_timestamp if any rows written, else None.
    """
    if not feedparser:
        print("Warning: feedparser not installed — news pipeline skipped")
        return None

    try:
        universe = load_all_segment_universes(industry_id)
    except Exception as e:
        print(f"Error loading universe for {industry_id}: {e}")
        return None

    run_ts = run_timestamp()
    all_rows: list[dict] = []

    # Pass 1: per-company feeds
    total = len(universe)
    for i, (_, row) in enumerate(universe.iterrows()):
        company_name = str(row.get("company_name", ""))
        ticker = str(row.get("ticker", ""))
        segment = str(row.get("segment", "fortune_100"))
        source_of_truth = str(row.get("source_of_truth", ""))
        print(f"  News [{i+1}/{total}] {company_name} ({segment})...")
        company_rows = _fetch_company_feeds(
            industry_id, company_name, ticker, segment, source_of_truth, run_ts
        )
        all_rows.extend(company_rows)

    # Pass 2: industry-wide feeds
    industry_rows = _fetch_industry_feeds(industry_id, universe, run_ts)
    all_rows.extend(industry_rows)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)

    # Deduplicate on (company_name, post_url) — keep first (company-specific takes priority)
    df = df.drop_duplicates(subset=["company_name", "post_url"], keep="first")
    df["run_timestamp"] = run_ts

    write_posts_table(df, industry_id, run_ts)
    print(f"News pipeline: {len(df)} articles written for {industry_id} "
          f"({df['segment'].value_counts().to_dict()})")
    return run_ts
