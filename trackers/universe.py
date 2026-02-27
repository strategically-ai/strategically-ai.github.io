"""Load and enrich industry universes (ticker, CIK, validation)."""

import os
from pathlib import Path

import pandas as pd
import requests

from trackers.config import BASE_OUTPUT_DIR, SEC_HEADERS

REQUIRED_COLUMNS = ["company_name", "ticker", "cik", "sub_industry", "source_of_truth", "segment"]


def load_universe(path: str) -> pd.DataFrame:
    """Load universe CSV; validate required columns (segment inferred from filename if missing)."""
    df = pd.read_csv(path)
    # Infer segment from filename if column is absent
    if "segment" not in df.columns:
        fname = os.path.basename(path)
        if "pe_mid" in fname:
            df["segment"] = "pe_mid"
        elif "startup" in fname:
            df["segment"] = "startup"
        else:
            df["segment"] = "fortune_100"
    hard_required = [c for c in REQUIRED_COLUMNS if c != "segment"]
    missing = [c for c in hard_required if c not in df.columns]
    if missing:
        raise ValueError(f"Universe missing columns: {missing}")
    return df


def fetch_sec_cik_mapping() -> pd.DataFrame:
    """Fetch SEC company_tickers.json and return DataFrame with cik (padded), ticker_sec."""
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame.from_dict(data, orient="index")
    df.columns = ["cik_raw", "ticker_sec", "company_sec"]
    df["ticker_sec"] = df["ticker_sec"].str.upper().str.replace("-", ".", regex=False)
    df["cik"] = df["cik_raw"].apply(lambda x: str(x).zfill(10))
    return df


def enrich_universe_with_cik(universe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Join universe to SEC ticker->CIK mapping. Universe must have 'ticker'.
    Returns (merged_df_with_cik, list_of_unmatched_company_names).
    """
    cik_df = fetch_sec_cik_mapping()
    ticker_col = "ticker" if "ticker" in universe.columns else "ticker_sec"
    universe = universe.copy()
    # Drop existing cik so merge adds a single cik column from SEC
    if "cik" in universe.columns:
        universe = universe.drop(columns=["cik"])
    universe["ticker_upper"] = universe[ticker_col].str.upper().str.replace("-", ".", regex=False)
    merged = universe.merge(
        cik_df[["ticker_sec", "cik", "company_sec"]],
        left_on="ticker_upper",
        right_on="ticker_sec",
        how="left",
    )
    matched = merged.dropna(subset=["cik"]).copy()
    first_col = "company_name" if "company_name" in merged.columns else merged.columns[0]
    unmatched = merged[merged["cik"].isna()][first_col].tolist()
    return matched, list(set(unmatched))


def get_universe_path(industry_id: str, version: str = "v1") -> str:
    """Return path to universe file for an industry (e.g. consumer_universe_v1.csv)."""
    base = os.path.join(BASE_OUTPUT_DIR, industry_id, "universe")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{industry_id}_universe_{version}.csv")


def load_industry_universe(industry_id: str, version: str = "v1") -> pd.DataFrame:
    """Load and enrich universe for an industry; raise if file not found."""
    path = get_universe_path(industry_id, version)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Universe not found: {path}")
    df = load_universe(path)
    if df["cik"].isna().all() or (df["cik"].astype(str).str.strip() == "").all():
        df, _ = enrich_universe_with_cik(df)
    return df


def load_all_segment_universes(industry_id: str) -> pd.DataFrame:
    """
    Load and merge all universe CSVs for an industry across all three segments.
    Naming convention:
      {industry_id}_universe_v1.csv           -> segment inferred as fortune_100
      {industry_id}_universe_pe_mid_v1.csv    -> segment inferred as pe_mid
      {industry_id}_universe_startup_v1.csv   -> segment inferred as startup
    CIK enrichment is run on any rows with missing CIKs.
    Private companies (source_of_truth == 'private_news') keep CIK empty and are
    skipped by SEC pipelines but included for news tracking.
    """
    base = os.path.join(BASE_OUTPUT_DIR, industry_id, "universe")
    os.makedirs(base, exist_ok=True)
    all_dfs = []
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".csv") or fname == ".gitkeep":
            continue
        path = os.path.join(base, fname)
        try:
            df = load_universe(path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: skipping universe file {fname}: {e}")
    if not all_dfs:
        raise FileNotFoundError(f"No universe CSVs found for {industry_id} in {base}")
    combined = pd.concat(all_dfs, ignore_index=True)
    # Enrich CIKs for public companies that are missing them
    public_mask = combined["source_of_truth"] != "private_news"
    needs_enrich = public_mask & (combined["cik"].isna() | (combined["cik"].astype(str).str.strip() == ""))
    if needs_enrich.any():
        try:
            enriched, _ = enrich_universe_with_cik(combined[needs_enrich])
            # Merge enriched CIKs back into combined
            combined = combined.drop(columns=["cik"], errors="ignore")
            combined["ticker_upper"] = combined["ticker"].str.upper().str.replace("-", ".", regex=False)
            cik_df = fetch_sec_cik_mapping()
            combined = combined.merge(
                cik_df[["ticker_sec", "cik"]],
                left_on="ticker_upper",
                right_on="ticker_sec",
                how="left",
            )
            combined = combined.drop(columns=["ticker_upper", "ticker_sec"], errors="ignore")
        except Exception as e:
            print(f"Warning: CIK enrichment failed: {e}")
    return combined
