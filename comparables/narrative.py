"""Optional LLM-generated sector pulse (2-3 sentences) per industry."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .universe import _repo_root
from .io import get_output_dir

SECTOR_PULSE_PROMPT = """You are an equity research analyst writing a brief sector note for institutional investors. Given the comparables data below, write exactly 3 concise sentences:

1. Where the sector trades today on valuation (EV/EBITDA, P/E) and how that compares to what you would expect given the growth profile.
2. One specific observation grounded in the data — name the company or companies involved (e.g. a wide spread between the richest and cheapest name, a standout grower, compressed margins across the group, or notable leverage).
3. One forward-looking implication for investors or strategists evaluating this sector.

Be direct and specific. Use the actual numbers and tickers. Avoid generic filler like "this indicates" or "notably." Write as if this will appear on a Bloomberg terminal screen.

Sector: {industry}
{summary}

Sector pulse:"""


def generate_sector_pulse(industry: str, summary: str, openai_client) -> Optional[str]:
    """Return 2-3 sentence sector pulse from LLM, or None if no client or error."""
    if openai_client is None:
        return None
    try:
        prompt = SECTOR_PULSE_PROMPT.format(industry=industry, summary=summary)
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
