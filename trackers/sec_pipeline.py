"""SEC 10-Q (and 10-K) pipeline: fetch filings, extract strategy-relevant passages, use LLM to summarise company strategy."""

import re
import time
from datetime import datetime

import pandas as pd
import requests

from trackers.classification import classify_sec_disclosure
from trackers.config import (
    AI_KEYWORDS,
    BASE_OUTPUT_DIR,
    DEFAULT_FILING_TYPE,
    SEC_HEADERS,
    SEC_RATE_LIMIT_SLEEP,
)
from trackers.io import run_date, run_timestamp, write_disclosures_table
from trackers.universe import load_all_segment_universes

AI_PATTERN = re.compile("|".join(AI_KEYWORDS), re.IGNORECASE)


def fetch_submissions(cik: str, headers: dict) -> dict:
    """Fetch SEC submissions JSON for a CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def get_latest_filing(cik: str, headers: dict, form: str = None) -> dict | None:
    """Get latest 10-Q or 10-K metadata for a CIK."""
    form = form or DEFAULT_FILING_TYPE
    data = fetch_submissions(cik, headers)
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return None
    forms = recent.get("form", [])
    acc_nos = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    for i in range(len(forms)):
        if forms[i] == form:
            return {
                "accession_number": acc_nos[i],
                "filing_date": dates[i],
                "primary_document": primary_docs[i],
            }
    return None


def build_filing_url(accession_number: str, primary_document: str, cik: str) -> str:
    """Build SEC EDGAR URL for the primary document."""
    acc_no_dashes = accession_number.replace("-", "")
    cik_num = cik.lstrip("0") or "0"
    return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_no_dashes}/{primary_document}"


def extract_ai_passages(html_text: str, max_total_chars: int = 6000) -> str | None:
    """Extract sentences containing AI keywords with context; return concatenated text or None."""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    ai_passages = []
    total_chars = 0
    for j, sent in enumerate(sentences):
        if AI_PATTERN.search(sent):
            start = max(0, j - 1)
            end = min(len(sentences), j + 3)
            window = " ".join(sentences[start:end])
            if total_chars + len(window) > max_total_chars:
                break
            ai_passages.append(window)
            total_chars += len(window)
    return " [...] ".join(ai_passages) if ai_passages else None


def fetch_and_extract_ai_passages(url: str, headers: dict, max_total_chars: int = 6000) -> tuple[str | None, int]:
    """Fetch filing HTML and return (extracted_ai_text, mention_count)."""
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    passages = extract_ai_passages(r.text, max_total_chars)
    count = len(AI_PATTERN.findall(passages)) if passages else 0
    return passages, count


def run_industry_sec(
    industry_id: str,
    universe_version: str = "v1",
    filing_type: str = None,
    openai_client=None,
) -> str:
    """
    Run SEC pipeline for an industry: load universe, fetch latest 10-Q (or 10-K),
    extract strategy-relevant passages, use LLM to summarise company strategy, write structured table.
    Returns run_timestamp.
    """
    filing_type = filing_type or DEFAULT_FILING_TYPE
    universe = load_all_segment_universes(industry_id)
    run_ts = run_timestamp()

    rows = []
    for idx, row in universe.iterrows():
        # Skip private companies — no public SEC filings
        if str(row.get("source_of_truth", "")).strip() == "private_news":
            continue
        cik_raw = str(row.get("cik", "")).strip()
        if not cik_raw or cik_raw in ("nan", ""):
            continue
        cik = cik_raw.zfill(10)
        company_name = row["company_name"]
        ticker = row["ticker"]
        segment = str(row.get("segment", "fortune_100"))
        try:
            meta = get_latest_filing(cik, SEC_HEADERS, form=filing_type)
            if not meta:
                continue
            url = build_filing_url(meta["accession_number"], meta["primary_document"], cik)
            time.sleep(SEC_RATE_LIMIT_SLEEP)
            passages, mention_count = fetch_and_extract_ai_passages(url, SEC_HEADERS)
            time.sleep(SEC_RATE_LIMIT_SLEEP)
            if not passages:
                continue
            theme = "data_strategy"
            signal_strength = "moderate"
            why_it_matters = ""
            confidence = 0.7
            if openai_client:
                try:
                    result = classify_sec_disclosure(openai_client, passages)
                    theme = result.get("theme", theme)
                    signal_strength = result.get("signal_strength", signal_strength)
                    why_it_matters = result.get("why_it_matters", why_it_matters)
                    confidence = result.get("confidence", confidence)
                except Exception:
                    pass
            rows.append({
                "industry_id": industry_id,
                "company_name": company_name,
                "ticker": ticker,
                "segment": segment,
                "cik": cik,
                "filing_date": meta["filing_date"],
                "filing_url": url,
                "section": filing_type,
                "snippet": passages[:8000],
                "theme": theme,
                "signal_strength": signal_strength,
                "why_it_matters": why_it_matters,
                "confidence": confidence,
                "run_timestamp": run_ts,
            })
        except Exception as e:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        write_disclosures_table(df, industry_id, run_ts)
    return run_ts
