"""Download and cache historical 10-K filings from SEC EDGAR for the comparables universe.

Idempotent: skips already-cached filings. Respects SEC rate limits (10 req/sec).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

# SEC EDGAR settings
SEC_HEADERS = {"User-Agent": "Mfundo Radebe mtr2149@columbia.edu"}
SEC_RATE_LIMIT = 0.12  # ~8 req/sec (conservative under 10/sec limit)

# Paths
_REPO_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = _REPO_ROOT / "data" / "universes" / "comparables_universe.csv"
CACHE_DIR = _REPO_ROOT / "data" / "research" / "filings_cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"


def load_universe() -> pd.DataFrame:
    """Load comparables universe (industry, ticker, name)."""
    return pd.read_csv(UNIVERSE_PATH)


def fetch_cik_mapping() -> dict[str, str]:
    """Fetch SEC ticker → CIK mapping. Returns {TICKER: CIK_PADDED}."""
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    mapping = {}
    for entry in data.values():
        ticker = str(entry["ticker"]).upper().replace("-", ".")
        cik = str(entry["cik_str"]).zfill(10)
        mapping[ticker] = cik
    return mapping


def get_all_10k_filings(cik: str, start_year: int = 2020) -> list[dict]:
    """Get all 10-K filing metadata for a CIK from start_year onwards.

    Returns list of dicts with keys: accession_number, filing_date, primary_document.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(SEC_RATE_LIMIT)
    data = r.json()

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return filings

    forms = recent.get("form", [])
    acc_nos = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    for i in range(len(forms)):
        if forms[i] not in ("10-K", "10-K/A"):
            continue
        filing_date = dates[i]
        if int(filing_date[:4]) < start_year:
            continue
        filings.append({
            "accession_number": acc_nos[i],
            "filing_date": filing_date,
            "primary_document": primary_docs[i],
        })

    return filings


def build_filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Build SEC EDGAR URL for the primary filing document."""
    acc_clean = accession_number.replace("-", "")
    cik_num = cik.lstrip("0") or "0"
    return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_clean}/{primary_document}"


def download_filing_text(url: str) -> str | None:
    """Download filing HTML and strip tags to plain text."""
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=60)
        r.raise_for_status()
        time.sleep(SEC_RATE_LIMIT)
        html = r.text
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return None


def cache_key(ticker: str, filing_date: str) -> str:
    """Generate cache filename for a filing."""
    return f"{ticker}_{filing_date}.txt"


def load_manifest() -> dict:
    """Load the processing manifest (tracks which filings have been cached)."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def backfill_filings(
    industries: list[str] | None = None,
    start_year: int = 2020,
    max_companies: int | None = None,
) -> pd.DataFrame:
    """Download and cache 10-K filings for the universe.

    Args:
        industries: filter to specific industries (None = all)
        start_year: earliest filing year to fetch
        max_companies: limit number of companies (for testing)

    Returns:
        DataFrame with columns: ticker, name, industry, cik, filing_date,
        accession_number, filing_url, cache_file, word_count
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    universe = load_universe()
    if industries:
        ind_lower = [i.lower() for i in industries]
        universe = universe[universe["industry"].str.lower().isin(ind_lower)]
    if max_companies:
        universe = universe.head(max_companies)

    # Get CIK mapping
    print("Fetching SEC CIK mapping...")
    cik_map = fetch_cik_mapping()
    time.sleep(SEC_RATE_LIMIT)

    results = []
    total = len(universe)

    for idx, row in universe.iterrows():
        ticker = str(row["ticker"]).upper()
        name = row["name"]
        industry = row["industry"]

        cik = cik_map.get(ticker)
        if not cik:
            print(f"  [{idx+1}/{total}] {ticker}: no CIK found, skipping")
            continue

        print(f"  [{idx+1}/{total}] {ticker} ({name}): fetching 10-K list...")

        try:
            filings = get_all_10k_filings(cik, start_year=start_year)
        except Exception as e:
            print(f"    Error fetching filings: {e}")
            continue

        if not filings:
            print(f"    No 10-K filings found since {start_year}")
            continue

        print(f"    Found {len(filings)} 10-K filings")

        for filing in filings:
            key = cache_key(ticker, filing["filing_date"])

            # Skip if already cached
            if key in manifest:
                results.append(manifest[key])
                continue

            url = build_filing_url(cik, filing["accession_number"], filing["primary_document"])
            text = download_filing_text(url)
            if not text:
                print(f"    Failed to download {filing['filing_date']}")
                continue

            # Cache the text
            cache_file = CACHE_DIR / key
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(text)

            word_count = len(text.split())
            record = {
                "ticker": ticker,
                "name": name,
                "industry": industry,
                "cik": cik,
                "filing_date": filing["filing_date"],
                "accession_number": filing["accession_number"],
                "filing_url": url,
                "cache_file": key,
                "word_count": word_count,
            }
            manifest[key] = record
            results.append(record)
            print(f"    Cached {filing['filing_date']} ({word_count:,} words)")

    save_manifest(manifest)
    df = pd.DataFrame(results)
    print(f"\nBackfill complete: {len(df)} filings cached for {df['ticker'].nunique() if not df.empty else 0} companies")
    return df


def load_cached_filing(cache_file: str) -> str | None:
    """Load a cached filing text by its cache filename."""
    path = CACHE_DIR / cache_file
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill SEC 10-K filings")
    parser.add_argument("--industry", default=None, help="Filter to specific industry")
    parser.add_argument("--start-year", type=int, default=2020, help="Earliest year")
    parser.add_argument("--max-companies", type=int, default=None, help="Limit companies (for testing)")
    args = parser.parse_args()

    industries = [args.industry] if args.industry else None
    df = backfill_filings(
        industries=industries,
        start_year=args.start_year,
        max_companies=args.max_companies,
    )
    if not df.empty:
        print(f"\nFiling dates range: {df['filing_date'].min()} to {df['filing_date'].max()}")
        print(f"Industries: {df['industry'].value_counts().to_dict()}")
