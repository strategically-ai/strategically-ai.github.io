"""Optional LLM-generated sector pulse (2-3 sentences) per industry.

Fetches live news headlines via Google News RSS to ground commentary
in real events — not just static boilerplate.

Uses Anthropic Claude API (Haiku) for generation.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import date as _date
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.parse import quote

from .universe import _repo_root
from .io import get_output_dir
from .env_loader import HAIKU_MODEL

# ── Sector → search terms for Google News RSS ──────────────────────
SECTOR_NEWS_QUERIES: dict[str, list[str]] = {
    "Consumer": ["consumer retail earnings", "consumer spending US"],
    "Energy": ["oil prices energy sector", "crude oil OPEC"],
    "Financials": ["bank earnings financials sector", "interest rates banks"],
    "Healthcare": ["healthcare pharma earnings", "drug pricing FDA"],
    "Industrials": ["industrials manufacturing earnings", "defense spending Boeing"],
    "Real Estate": ["commercial real estate REIT", "housing market US"],
    "Technology": ["tech earnings AI semiconductor", "Nvidia Apple Microsoft"],
}


def _fetch_news_headlines(industry: str, max_headlines: int = 8) -> list[str]:
    """Fetch recent news headlines from Google News RSS for the sector.

    Returns a list of headline strings, or empty list on failure.
    """
    queries = SECTOR_NEWS_QUERIES.get(industry, [industry + " sector stock market"])
    headlines: list[str] = []
    for q in queries:
        if len(headlines) >= max_headlines:
            break
        try:
            url = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-US&gl=US&ceid=US:en"
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as resp:
                tree = ET.parse(resp)
            for item in tree.findall(".//item"):
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    headline = unescape(title_el.text).strip()
                    if headline and headline not in headlines:
                        headlines.append(headline)
                        if len(headlines) >= max_headlines:
                            break
        except Exception:
            continue
    return headlines


SECTOR_PULSE_PROMPT = """You are an equity research analyst writing a brief sector note for institutional investors. Today is {date}.

Recent headlines for this sector:
{headlines}

Given the comparables data and these real-world headlines, write exactly 3 concise sentences as a single plain-text paragraph (no numbered lists, no bullet points, no markdown formatting, no bold):

Sentence 1: Where the sector trades today on valuation (EV/EBITDA, P/E) relative to the broader market — reference the actual median multiples from the data.
Sentence 2: One specific observation grounded in BOTH the data AND recent news — name companies/tickers with real numbers, and connect to a headline or event where relevant (e.g. an oil price shock hitting energy margins, a pharma approval driving a standout multiple, tariff risk compressing industrials).
Sentence 3: One forward-looking implication for investors, tying the data to the current news cycle.

IMPORTANT: Output plain text only. No numbered lists, no "1.", no "**bold**", no markdown. Just three sentences in a single paragraph.

Sector: {industry}
{summary}

Sector pulse:"""


def generate_sector_pulse(industry: str, summary: str, anthropic_client) -> Optional[str]:
    """Return 2-3 sentence sector pulse from Claude Haiku, or None if no client or error."""
    if anthropic_client is None:
        return None
    try:
        headlines = _fetch_news_headlines(industry)
        headline_text = "\n".join(f"- {h}" for h in headlines) if headlines else "- No recent headlines available"
        prompt = SECTOR_PULSE_PROMPT.format(
            industry=industry,
            summary=summary,
            date=_date.today().isoformat(),
            headlines=headline_text,
        )
        message = anthropic_client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip() if message.content else None
        return text
    except Exception:
        return None


def write_sector_pulse(industry: str, pulse: str) -> Path:
    out_dir = get_output_dir()
    safe = industry.lower().replace(" ", "_")
    path = out_dir / f"sector_pulse_{safe}.json"
    with open(path, "w") as f:
        json.dump({"industry": industry, "pulse": pulse}, f, indent=2)
    return path
