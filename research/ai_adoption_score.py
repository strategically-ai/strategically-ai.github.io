"""Compute AI Adoption Score from cached SEC 10-K filings.

Composite score based on:
1. Keyword density — AI keyword frequency per 10,000 words
2. Section analysis — which 10-K sections mention AI (risk factors, business, MD&A)
3. LLM passage classification — Claude Haiku classifies AI passages as:
   strategic_investment, operational_deployment, exploratory_mention, risk_disclosure
4. Temporal trend — YoY change in adoption score

The result is a company × filing_date panel of AI adoption intensity.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from research.sec_backfill import CACHE_DIR, load_cached_filing, load_manifest
from research.env_loader import get_anthropic_client, HAIKU_MODEL

# ── AI Keywords (from trackers/config.py, expanded) ───────────────
AI_KEYWORDS = [
    r"\bartificial intelligence\b",
    r"\bAI\b",
    r"\bmachine learning\b",
    r"\bML\b",
    r"\bautomation\b",
    r"\bgenerative\b",
    r"\bLLM\b",
    r"\bcopilot\b",
    r"\bautonomous\b",
    r"\bpredictive\b",
    r"\bdeep learning\b",
    r"\bgen ai\b",
    r"\blarge language model\b",
    r"\bneural network\b",
    r"\bnatural language processing\b",
    r"\bai[ -]powered\b",
    r"\bai[ -]driven\b",
    r"\bai[ -]enabled\b",
    r"\bautomated decision\b",
    r"\bpredictive model\b",
    r"\bpredictive analytics\b",
    r"\bcomputer vision\b",
    r"\brobotic process automation\b",
    r"\bRPA\b",
    r"\bchatbot\b",
    r"\btransformer\b",
    r"\breinforcement learning\b",
    r"\bfoundation model\b",
]

AI_PATTERN = re.compile("|".join(AI_KEYWORDS), re.IGNORECASE)

# ── Section detection heuristics for 10-K ─────────────────────────
SECTION_PATTERNS = {
    "business": re.compile(
        r"(?:item\s*1[.\s]|description\s+of\s+business|business\s+overview)",
        re.IGNORECASE,
    ),
    "risk_factors": re.compile(
        r"(?:item\s*1a[.\s]|risk\s+factors)",
        re.IGNORECASE,
    ),
    "mda": re.compile(
        r"(?:item\s*7[.\s]|management.s\s+discussion|MD&A)",
        re.IGNORECASE,
    ),
    "financial_statements": re.compile(
        r"(?:item\s*8[.\s]|financial\s+statements\s+and\s+supplementary)",
        re.IGNORECASE,
    ),
}

# Section weights — strategic mentions are worth more than risk disclosures
SECTION_WEIGHTS = {
    "business": 1.5,       # Strategic commitment
    "mda": 1.3,            # Operational discussion
    "risk_factors": 0.7,   # May be defensive/cautionary
    "financial_statements": 0.5,
    "unknown": 1.0,
}

# ── LLM Classification ───────────────────────────────────────────

CLASSIFY_PROMPT = """You are a corporate strategy analyst. Classify the following AI-related passage from a company's SEC 10-K filing.

Classify into exactly ONE category:
- strategic_investment: Company is actively investing in AI as a strategic priority (building products, acquiring AI companies, significant R&D)
- operational_deployment: Company is deploying AI in operations (automation, efficiency, internal tools)
- exploratory_mention: Company mentions AI in a general/exploratory way without concrete commitments
- risk_disclosure: Company mentions AI primarily as a risk factor or competitive threat

Also rate the commitment_level from 1-5:
1 = Minimal/passing mention
2 = Acknowledgment without action
3 = Active exploration/early deployment
4 = Significant investment/deployment
5 = Core strategic pillar

Format your response EXACTLY as:
Category: <one from list>
Commitment: <1-5>
Rationale: <one sentence>"""

# Classification weights for score computation
CATEGORY_WEIGHTS = {
    "strategic_investment": 2.0,
    "operational_deployment": 1.5,
    "exploratory_mention": 0.8,
    "risk_disclosure": 0.5,
}


def compute_keyword_density(text: str) -> dict:
    """Compute AI keyword metrics from filing text.

    Returns dict with:
    - keyword_count: total AI keyword matches
    - keyword_density: matches per 10,000 words
    - unique_keywords: number of distinct keyword patterns matched
    """
    word_count = len(text.split())
    if word_count == 0:
        return {"keyword_count": 0, "keyword_density": 0.0, "unique_keywords": 0}

    matches = AI_PATTERN.findall(text)
    unique = set(m.lower() for m in matches)

    return {
        "keyword_count": len(matches),
        "keyword_density": (len(matches) / word_count) * 10000,
        "unique_keywords": len(unique),
    }


def detect_section(text: str, position: int, window: int = 2000) -> str:
    """Detect which 10-K section a text position falls within."""
    # Look backward from position for section headers
    start = max(0, position - window)
    context = text[start:position]

    best_section = "unknown"
    best_pos = -1
    for section, pattern in SECTION_PATTERNS.items():
        match = pattern.search(context)
        if match and match.start() > best_pos:
            best_pos = match.start()
            best_section = section

    return best_section


def extract_ai_passages_with_sections(text: str, max_passages: int = 10) -> list[dict]:
    """Extract AI-keyword passages with section context.

    Returns list of dicts with: passage, section, position, keyword_matches.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    passages = []
    char_pos = 0

    for i, sent in enumerate(sentences):
        if AI_PATTERN.search(sent) and len(passages) < max_passages:
            # Get surrounding context (1 sentence before, 2 after)
            start = max(0, i - 1)
            end = min(len(sentences), i + 3)
            passage = " ".join(sentences[start:end])

            section = detect_section(text, char_pos)
            keyword_matches = AI_PATTERN.findall(sent)

            passages.append({
                "passage": passage[:1500],  # Cap length
                "section": section,
                "position": char_pos,
                "keyword_matches": len(keyword_matches),
            })
        char_pos += len(sent) + 1

    return passages


def classify_passages_llm(passages: list[dict], company_name: str, client=None) -> list[dict]:
    """Use Claude Haiku to classify AI passages. Returns passages with added classification fields."""
    if client is None or not passages:
        # No LLM available — return with defaults
        for p in passages:
            p["category"] = "exploratory_mention"
            p["commitment_level"] = 2
            p["rationale"] = "No LLM classification available"
        return passages

    # Batch passages into a single LLM call for efficiency
    combined_text = f"Company: {company_name}\n\n"
    for i, p in enumerate(passages[:5]):  # Limit to 5 passages per call
        combined_text += f"--- Passage {i+1} (Section: {p['section']}) ---\n{p['passage']}\n\n"

    combined_text += f"\nFor each passage, provide classification. If multiple passages, number them (Passage 1, Passage 2, etc.).\n\n{CLASSIFY_PROMPT}"

    try:
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": combined_text}],
        )
        response = message.content[0].text.strip()

        # Parse response — try to extract per-passage classifications
        # For simplicity, apply the first classification to all passages
        category = "exploratory_mention"
        commitment = 2
        rationale = ""

        for line in response.split("\n"):
            line = line.strip()
            if line.lower().startswith("category:"):
                val = line.split(":", 1)[-1].strip().lower()
                if val in CATEGORY_WEIGHTS:
                    category = val
            elif line.lower().startswith("commitment:"):
                try:
                    commitment = int(line.split(":", 1)[-1].strip())
                    commitment = max(1, min(5, commitment))
                except ValueError:
                    pass
            elif line.lower().startswith("rationale:"):
                rationale = line.split(":", 1)[-1].strip()

        for p in passages:
            p["category"] = category
            p["commitment_level"] = commitment
            p["rationale"] = rationale

    except Exception as e:
        for p in passages:
            p["category"] = "exploratory_mention"
            p["commitment_level"] = 2
            p["rationale"] = f"LLM error: {e}"

    return passages


def compute_adoption_score(
    keyword_metrics: dict,
    passages: list[dict],
) -> dict:
    """Compute composite AI Adoption Score from keyword metrics and classified passages.

    Score components:
    1. keyword_score: normalized keyword density (0-100 scale)
    2. section_score: weighted section presence (strategic sections worth more)
    3. classification_score: LLM classification × commitment level
    4. breadth_score: how many unique keywords appear

    Final score is weighted combination (0-100 scale).
    """
    # 1. Keyword density score (log-scaled, capped at 100)
    density = keyword_metrics["keyword_density"]
    keyword_score = min(100, density * 5)  # 20 mentions per 10k words = 100

    # 2. Section score — weighted by which sections mention AI
    sections_found = set(p["section"] for p in passages)
    section_score = sum(SECTION_WEIGHTS.get(s, 1.0) for s in sections_found) * 20
    section_score = min(100, section_score)

    # 3. Classification score — category weight × commitment level
    if passages:
        avg_category_weight = sum(CATEGORY_WEIGHTS.get(p.get("category", "exploratory_mention"), 1.0) for p in passages) / len(passages)
        avg_commitment = sum(p.get("commitment_level", 2) for p in passages) / len(passages)
        classification_score = min(100, avg_category_weight * avg_commitment * 10)
    else:
        classification_score = 0

    # 4. Breadth score — diversity of AI keywords used
    breadth_score = min(100, keyword_metrics["unique_keywords"] * 10)

    # Composite (weighted)
    composite = (
        keyword_score * 0.30
        + section_score * 0.20
        + classification_score * 0.35
        + breadth_score * 0.15
    )

    return {
        "ai_adoption_score": round(composite, 2),
        "keyword_score": round(keyword_score, 2),
        "section_score": round(section_score, 2),
        "classification_score": round(classification_score, 2),
        "breadth_score": round(breadth_score, 2),
        "keyword_count": keyword_metrics["keyword_count"],
        "keyword_density": round(keyword_metrics["keyword_density"], 4),
        "unique_keywords": keyword_metrics["unique_keywords"],
        "sections_with_ai": list(sections_found) if passages else [],
        "primary_category": passages[0].get("category", "none") if passages else "none",
        "avg_commitment": round(sum(p.get("commitment_level", 0) for p in passages) / max(1, len(passages)), 2),
        "passage_count": len(passages),
    }


def score_all_filings(use_llm: bool = True) -> pd.DataFrame:
    """Score all cached filings and return AI Adoption panel dataset.

    Returns DataFrame with columns: ticker, name, industry, filing_date, fiscal_year,
    ai_adoption_score, keyword_score, section_score, classification_score, breadth_score, ...
    """
    manifest = load_manifest()
    if not manifest:
        print("No cached filings found. Run sec_backfill.py first.")
        return pd.DataFrame()

    client = get_anthropic_client() if use_llm else None
    if use_llm and client is None:
        print("Warning: No Anthropic API key found. Running without LLM classification.")

    results = []
    entries = list(manifest.values())
    total = len(entries)

    for idx, entry in enumerate(entries):
        ticker = entry["ticker"]
        name = entry["name"]
        industry = entry["industry"]
        filing_date = entry["filing_date"]
        cache_file = entry["cache_file"]

        print(f"  [{idx+1}/{total}] Scoring {ticker} ({filing_date})...")

        text = load_cached_filing(cache_file)
        if not text:
            print(f"    Cache file missing: {cache_file}")
            continue

        # 1. Keyword density
        keyword_metrics = compute_keyword_density(text)

        # 2. Extract passages with sections
        passages = extract_ai_passages_with_sections(text)

        # 3. LLM classification (if enabled)
        if client and passages:
            passages = classify_passages_llm(passages, name, client)

        # 4. Compute composite score
        score = compute_adoption_score(keyword_metrics, passages)

        # Derive fiscal year from filing date (10-K filed ~60-90 days after fiscal year end)
        fiscal_year = int(filing_date[:4])
        filing_month = int(filing_date[5:7])
        if filing_month <= 3:  # Filed in Q1 → fiscal year is previous year
            fiscal_year -= 1

        record = {
            "ticker": ticker,
            "name": name,
            "industry": industry,
            "filing_date": filing_date,
            "fiscal_year": fiscal_year,
            **score,
        }
        results.append(record)

    df = pd.DataFrame(results)
    if not df.empty:
        # Add temporal trend (YoY change in adoption score)
        df = df.sort_values(["ticker", "fiscal_year"])
        df["ai_score_yoy_change"] = df.groupby("ticker")["ai_adoption_score"].diff()
        df["ai_score_yoy_pct"] = df.groupby("ticker")["ai_adoption_score"].pct_change()

        print(f"\nScored {len(df)} filings for {df['ticker'].nunique()} companies")
        print(f"Mean AI Adoption Score: {df['ai_adoption_score'].mean():.1f}")
        print(f"Score range: {df['ai_adoption_score'].min():.1f} - {df['ai_adoption_score'].max():.1f}")

    return df


def save_adoption_panel(df: pd.DataFrame) -> Path:
    """Save AI adoption panel dataset to CSV."""
    out_path = Path(__file__).resolve().parents[1] / "data" / "research" / "ai_adoption_panel.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved panel dataset: {out_path} ({len(df)} rows)")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute AI Adoption Scores from cached filings")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM classification (keyword-only scoring)")
    args = parser.parse_args()

    df = score_all_filings(use_llm=not args.no_llm)
    if not df.empty:
        save_adoption_panel(df)
