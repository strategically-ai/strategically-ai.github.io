#!/usr/bin/env python3
"""Run the comparables pipeline: fetch data, compute metrics, validate, write outputs by industry."""
import argparse
import re
import sys

import pandas as pd

from comparables.universe import load_universe, list_industries
from comparables.fetch import fetch_all
from comparables.metrics import build_comparables_table
from comparables.history_forecast import build_revenue_historical_forecast
from comparables.validate import validate_comparables
from comparables.narrative import generate_sector_pulse, write_sector_pulse
from comparables.io import (
    get_output_dir,
    write_industry_csv,
    write_last_updated,
    write_revenue_forecast,
    write_run_log,
    ensure_output_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Industry comparables pipeline (daily, Mon-Fri).")
    parser.add_argument(
        "--industry",
        default="all",
        help="Industry id (e.g. consumer, industrials, financials) or 'all'",
    )
    parser.add_argument("--no-versioned", action="store_true", help="Skip writing YYYY-MM versioned CSVs")
    args = parser.parse_args()

    df = load_universe()
    industries = list_industries(df)
    if args.industry.lower() != "all":
        chosen = [i for i in industries if i.lower() == args.industry.lower()]
        if not chosen:
            print(f"Unknown industry: {args.industry}. Choose from: {industries}", file=sys.stderr)
            sys.exit(1)
        industries = chosen

    ensure_output_dir()
    openai_client = None
    try:
        from openai import OpenAI
        from comparables.env_loader import get_openai_config
        api_key, base_url = get_openai_config()
        if api_key:
            openai_client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception:
        pass
    run_log = {"industries": [], "rows_written": 0, "revenue_rows": 0, "errors": [], "warnings": []}
    for industry in industries:
        subset = df[df["industry"] == industry]
        tickers = subset["ticker"].dropna().astype(str).unique().tolist()
        if not tickers:
            print(f"Skipping {industry}: no tickers")
            continue
        print(f"Fetching {industry} ({len(tickers)} tickers)...")
        raw = fetch_all(tickers)
        names = subset[["ticker", "name"]].drop_duplicates("ticker").rename(columns={"name": "name_universe"})
        raw = raw.merge(names, on="ticker", how="left")
        raw["name"] = raw["name_universe"].fillna(raw["name"])
        raw = raw.drop(columns=["name_universe"], errors="ignore")
        table = build_comparables_table(raw, industry)
        val = validate_comparables(table)
        run_log["industries"].append(industry)
        run_log["rows_written"] += val.get("row_count", 0)
        run_log["errors"].extend([f"{industry}: {e}" for e in val.get("errors", [])])
        run_log["warnings"].extend([f"{industry}: {w}" for w in val.get("warnings", [])])
        if val.get("errors"):
            print(f"  Validation errors: {val['errors']}")
        paths = write_industry_csv(table, industry, versioned=not args.no_versioned)
        print(f"  Wrote {paths[0]}" + (f" + {paths[1]}" if len(paths) > 1 else ""))
        # Optional sector pulse (LLM) — feed full table data for richer commentary
        if openai_client:
            try:
                n = len(table)
                lines = [f"{n} names in {industry}."]
                # Medians
                for col, label in [("ev_to_ebitda", "EV/EBITDA"), ("trailing_pe", "P/E"), ("revenue_growth", "Rev Growth")]:
                    if col in table.columns:
                        m = table[col].median()
                        if pd.notna(m):
                            val = f"{m*100:.1f}%" if "growth" in col or "margin" in col else f"{m:.1f}x"
                            lines.append(f"Median {label}: {val}")
                # Per-name detail: ticker, EV/EBITDA, P/E, margin, growth
                detail_cols = [c for c in ["ticker", "ev_to_ebitda", "trailing_pe", "profit_margin", "operating_margin", "revenue_growth", "fcf_yield", "net_debt_ebitda"] if c in table.columns]
                if detail_cols:
                    lines.append("\nCompany-level data:")
                    for _, row in table[detail_cols].iterrows():
                        parts = []
                        t = row.get("ticker", "?")
                        for c in detail_cols:
                            if c == "ticker":
                                continue
                            v = row.get(c)
                            if pd.notna(v):
                                if "margin" in c or "growth" in c or "yield" in c:
                                    parts.append(f"{c}={v*100:.1f}%")
                                else:
                                    parts.append(f"{c}={v:.1f}x")
                        lines.append(f"  {t}: {', '.join(parts)}")
                summary = "\n".join(lines)
                pulse = generate_sector_pulse(industry, summary, openai_client)
                if pulse:
                    write_sector_pulse(industry, pulse)
                    print(f"  Wrote sector_pulse_{industry.lower().replace(' ', '_')}.json")
            except Exception:
                pass
        print(f"  Revenue historical + forecast for {industry}...")
        rev_df = build_revenue_historical_forecast(tickers, throttle_seconds=0.2)
        if not rev_df.empty:
            rev_df["industry"] = industry
            write_revenue_forecast(rev_df, industry)
            run_log["revenue_rows"] += len(rev_df)
            print(f"  Wrote revenue_forecast_{industry.lower().replace(' ', '_')}.csv")

    write_last_updated()
    write_run_log(run_log)
    # Copy CSVs to docs/comparables so the site can offer download links
    repo_root = get_output_dir().parent.parent
    docs_comparables = repo_root / "docs" / "comparables"
    if docs_comparables.parent.exists():
        docs_comparables.mkdir(parents=True, exist_ok=True)
        import shutil
        out_dir = get_output_dir()
        for f in out_dir.glob("comparables_*.csv"):
            if not re.search(r"_\d{4}-\d{2}\.csv$", f.name):
                shutil.copy2(f, docs_comparables / f.name)
        for f in out_dir.glob("revenue_forecast_*.csv"):
            shutil.copy2(f, docs_comparables / f.name)
        print("Copied CSVs to docs/comparables for download links.")
    print("Done. Last updated and run_log.json written.")


if __name__ == "__main__":
    main()
