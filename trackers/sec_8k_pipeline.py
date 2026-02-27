"""SEC 8-K pipeline: fetch recent material event filings, extract M&A / capex / leadership signals."""

import re
import time

import pandas as pd
import requests

from trackers.classification import classify_8k_signal
from trackers.config import (
    DEFAULT_8K_MAX_FILINGS,
    SEC_HEADERS,
    SEC_RATE_LIMIT_SLEEP,
)
from trackers.io import get_output_dir, run_timestamp, write_8k_table
from trackers.universe import load_all_segment_universes

# ---------------------------------------------------------------------------
# Signal keyword patterns
# ---------------------------------------------------------------------------

_MA_TERMS = re.compile(
    r"\b(agreement\s+and\s+plan\s+of\s+merger|acquisition|acquires|acquired|"
    r"definitive\s+agreement|tender\s+offer|merger\s+agreement|"
    r"purchase\s+agreement|buyout|takeover|divestiture|spinoff|spin-off|"
    r"strategic\s+alternatives)\b",
    re.IGNORECASE,
)

_CAPEX_TERMS = re.compile(
    r"\b(capital\s+expenditure|capex|facility\s+investment|expansion|"
    r"new\s+distribution\s+center|data\s+center|manufacturing\s+plant|"
    r"warehouse|automation\s+investment|technology\s+investment|"
    r"infrastructure\s+investment|\$[\d,.]+\s*(?:billion|million)\s+investment)\b",
    re.IGNORECASE,
)

_LEADERSHIP_TERMS = re.compile(
    r"\b(chief\s+ai\s+officer|chief\s+data\s+officer|chief\s+digital\s+officer|"
    r"chief\s+technology\s+officer|evp\s+artificial\s+intelligence|"
    r"head\s+of\s+ai|president\s+ai|svp\s+data|chief\s+analytics\s+officer|"
    r"chief\s+information\s+officer)\b",
    re.IGNORECASE,
)

_DOLLAR_AMOUNT = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|trillion|B|M|T)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# EDGAR helpers (reuse pattern from sec_pipeline)
# ---------------------------------------------------------------------------

def _fetch_submissions(cik: str) -> dict:
    """Fetch SEC submissions JSON for a CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get_recent_8k_filings(cik: str, max_filings: int = DEFAULT_8K_MAX_FILINGS) -> list[dict]:
    """Return list of the N most recent 8-K filing metadata dicts for a CIK."""
    data = _fetch_submissions(cik)
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []
    forms = recent.get("form", [])
    acc_nos = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    results = []
    for i in range(len(forms)):
        if forms[i] == "8-K":
            results.append({
                "accession_number": acc_nos[i],
                "filing_date": dates[i],
                "primary_document": primary_docs[i],
            })
            if len(results) >= max_filings:
                break
    return results


def _build_filing_url(accession_number: str, primary_document: str, cik: str) -> str:
    """Build SEC EDGAR URL for the primary 8-K document."""
    acc_no_dashes = accession_number.replace("-", "")
    cik_num = cik.lstrip("0") or "0"
    return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_no_dashes}/{primary_document}"


def _fetch_filing_text(url: str) -> str | None:
    """Fetch filing HTML and return plain text, or None on error."""
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=60)
        r.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:20000]  # cap to avoid huge docs
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def _extract_dollar_amount(text: str) -> str:
    """Extract first dollar amount mentioned in text, or empty string."""
    m = _DOLLAR_AMOUNT.search(text)
    if m:
        return f"${m.group(1)} {m.group(2)}"
    return ""


def extract_8k_signal(text: str) -> dict:
    """
    Detect signal type from 8-K plain text.
    Returns dict: {signal_type, excerpt, is_ma, is_capex, is_leadership_hire, dollar_amount}
    """
    is_ma = bool(_MA_TERMS.search(text))
    is_capex = bool(_CAPEX_TERMS.search(text))
    is_leadership = bool(_LEADERSHIP_TERMS.search(text))
    dollar_amount = _extract_dollar_amount(text)

    # Signal type priority: ma > capex > leadership > earnings > other
    if is_ma:
        signal_type = "ma"
    elif is_capex:
        signal_type = "capex"
    elif is_leadership:
        signal_type = "leadership"
    else:
        signal_type = "other"

    # Extract a short excerpt (first sentence mentioning a signal keyword)
    excerpt = ""
    for pattern in [_MA_TERMS, _CAPEX_TERMS, _LEADERSHIP_TERMS]:
        m = pattern.search(text)
        if m:
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 200)
            excerpt = text[start:end].strip()
            break

    return {
        "signal_type": signal_type,
        "excerpt": excerpt[:500],
        "is_ma": is_ma,
        "is_capex": is_capex,
        "is_leadership_hire": is_leadership,
        "dollar_amount": dollar_amount,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_industry_8k(
    industry_id: str,
    openai_client=None,
) -> str:
    """
    Run 8-K pipeline for all public companies in all segment universes.
    Fetches the most recent 8-K filings per company, extracts M&A/capex/leadership signals.
    Writes sec_8k_signals_{run_ts}.csv.
    Returns run_timestamp.
    """
    universe = load_all_segment_universes(industry_id)
    run_ts = run_timestamp()
    rows = []

    total = len(universe)
    for idx, row in universe.iterrows():
        company_name = str(row.get("company_name", ""))
        ticker = str(row.get("ticker", ""))
        segment = str(row.get("segment", "fortune_100"))
        source_of_truth = str(row.get("source_of_truth", ""))

        # Skip private companies — no SEC filings
        if source_of_truth.strip() == "private_news":
            continue

        cik = str(row.get("cik", "")).strip()
        if not cik or cik in ("nan", ""):
            continue

        cik_padded = cik.zfill(10)

        try:
            filings = get_recent_8k_filings(cik_padded)
            time.sleep(SEC_RATE_LIMIT_SLEEP)
        except Exception:
            continue

        for filing_meta in filings:
            try:
                url = _build_filing_url(
                    filing_meta["accession_number"],
                    filing_meta["primary_document"],
                    cik_padded,
                )
                text = _fetch_filing_text(url)
                time.sleep(SEC_RATE_LIMIT_SLEEP)
                if not text:
                    continue

                signal = extract_8k_signal(text)

                # Skip 8-Ks with no meaningful signal (signal_type == 'other'
                # and no M&A/capex/leadership) unless LLM upgrade requested
                if signal["signal_type"] == "other" and not openai_client:
                    continue

                # Optionally use LLM to get a cleaner signal classification
                if openai_client and signal["signal_type"] in ("other",):
                    try:
                        llm_result = classify_8k_signal(openai_client, text[:4000])
                        signal["signal_type"] = llm_result.get("signal_type", signal["signal_type"])
                    except Exception:
                        pass

                rows.append({
                    "industry_id": industry_id,
                    "company_name": company_name,
                    "ticker": ticker,
                    "segment": segment,
                    "cik": cik_padded,
                    "filing_date": filing_meta.get("filing_date", ""),
                    "filing_url": url,
                    "signal_type": signal["signal_type"],
                    "excerpt": signal["excerpt"],
                    "is_ma": signal["is_ma"],
                    "is_capex": signal["is_capex"],
                    "is_leadership_hire": signal["is_leadership_hire"],
                    "dollar_amount": signal["dollar_amount"],
                    "run_timestamp": run_ts,
                })
            except Exception:
                continue

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "industry_id", "company_name", "ticker", "segment", "cik",
        "filing_date", "filing_url", "signal_type", "excerpt",
        "is_ma", "is_capex", "is_leadership_hire", "dollar_amount", "run_timestamp",
    ])

    if not df.empty:
        write_8k_table(df, industry_id, run_ts)
        ma_count = int(df["is_ma"].sum())
        capex_count = int(df["is_capex"].sum())
        print(f"8-K pipeline: {len(df)} signals for {industry_id} "
              f"(M&A: {ma_count}, capex: {capex_count})")
    else:
        print(f"8-K pipeline: no signals found for {industry_id}")

    return run_ts
