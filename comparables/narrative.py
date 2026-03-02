"""Optional LLM-generated sector pulse (2-3 sentences) per industry."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .universe import _repo_root
from .io import get_output_dir

SECTOR_PULSE_PROMPT = """You are an equity research analyst writing a brief sector note for institutional investors. Today's date is {date}. Q4 2025 earnings season is wrapping up with S&P 500 blended EPS growth of ~13% and revenue growth of ~9%. A rotation from mega-cap tech into cyclicals (industrials, financials, energy, healthcare) is underway. The S&P 500 forward P/E is 21.6x vs the 5-year average of 20.0x.

Given the comparables data below, write exactly 3 concise sentences:

1. Where the sector trades today on valuation (EV/EBITDA, P/E) relative to the broader market and its own growth profile — reference the actual median multiples.
2. One specific, data-driven observation — name companies and use real numbers from the data (e.g. a wide valuation spread between the richest and cheapest name, a standout grower or margin leader, compressed margins, or unusual leverage).
3. One forward-looking implication for investors evaluating this sector, tying to the current earnings cycle or macro backdrop where relevant.

Be direct. Use tickers and numbers. No filler. Write like a Bloomberg terminal note.

Sector: {industry}
{summary}

Sector pulse:"""


def generate_sector_pulse(industry: str, summary: str, openai_client) -> Optional[str]:
    """Return 2-3 sentence sector pulse from LLM, or None if no client or error."""
    if openai_client is None:
        return None
    try:
        from datetime import date as _date
        prompt = SECTOR_PULSE_PROMPT.format(industry=industry, summary=summary, date=_date.today().isoformat())
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = r.choices[0].message.content.strip() if r.choices else None
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
