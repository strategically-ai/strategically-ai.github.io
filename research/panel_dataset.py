"""Build research panel dataset: merge AI Adoption Scores with market valuation data.

Creates company × fiscal_year panel with:
- AI adoption metrics (from SEC filing NLP)
- Valuation multiples (from yfinance: P/E, EV/EBITDA, EV/Revenue)
- Operating performance (margins, revenue growth)
- Stock returns (forward 6-month, 12-month)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = _REPO_ROOT / "data" / "research" / "ai_adoption_panel.csv"
MERGED_PANEL_PATH = _REPO_ROOT / "data" / "research" / "ai_adoption_valuation_panel.csv"


def load_adoption_panel() -> pd.DataFrame:
    """Load the AI adoption scores panel."""
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"AI adoption panel not found: {PANEL_PATH}. Run ai_adoption_score.py first.")
    return pd.read_csv(PANEL_PATH)


def fetch_annual_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Fetch annual fundamental data from yfinance for all tickers.

    Returns DataFrame with: ticker, fiscal_year, market_cap, ev, revenue,
    ebitda, net_income, operating_margin, profit_margin, revenue_growth, etc.
    """
    records = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers):
        print(f"  [{idx+1}/{total}] Fetching fundamentals for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            # Get annual financials for historical data
            financials = stock.financials
            if financials is None or financials.empty:
                continue

            # Current snapshot
            record = {
                "ticker": ticker,
                "fiscal_year": pd.Timestamp.now().year,
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "return_on_equity": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "free_cash_flow": info.get("freeCashflow"),
                "beta": info.get("beta"),
            }
            records.append(record)

            # Historical annual financials
            for col_date in financials.columns:
                year = col_date.year
                fin_data = financials[col_date]
                rev = fin_data.get("Total Revenue")
                ebitda = fin_data.get("EBITDA")
                net_inc = fin_data.get("Net Income")
                op_inc = fin_data.get("Operating Income")

                hist_record = {
                    "ticker": ticker,
                    "fiscal_year": year,
                    "revenue": rev if pd.notna(rev) else None,
                    "ebitda": ebitda if pd.notna(ebitda) else None,
                    "net_income": net_inc if pd.notna(net_inc) else None,
                    "operating_income": op_inc if pd.notna(op_inc) else None,
                }
                # Compute margins from absolute values
                if rev and pd.notna(rev) and rev != 0:
                    if op_inc and pd.notna(op_inc):
                        hist_record["operating_margin_calc"] = op_inc / rev
                    if net_inc and pd.notna(net_inc):
                        hist_record["profit_margin_calc"] = net_inc / rev
                records.append(hist_record)

        except Exception as e:
            print(f"    Error: {e}")
            continue

    return pd.DataFrame(records)


def fetch_stock_returns(tickers: list[str], start_year: int = 2019) -> pd.DataFrame:
    """Fetch annual stock returns for computing forward returns.

    Returns DataFrame with: ticker, year, annual_return, price_start, price_end.
    """
    records = []
    start_date = f"{start_year}-01-01"

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, auto_adjust=True)
            if hist.empty:
                continue

            # Compute annual returns
            hist.index = pd.to_datetime(hist.index)
            annual = hist["Close"].resample("YE").last()

            for i in range(1, len(annual)):
                prev = annual.iloc[i - 1]
                curr = annual.iloc[i]
                if prev > 0:
                    records.append({
                        "ticker": ticker,
                        "fiscal_year": annual.index[i].year,
                        "annual_return": (curr - prev) / prev,
                        "price_start": prev,
                        "price_end": curr,
                    })

            # Forward 6-month return from most recent data
            if len(hist) > 126:  # ~6 months of trading days
                six_mo_ago = hist["Close"].iloc[-126]
                latest = hist["Close"].iloc[-1]
                if six_mo_ago > 0:
                    records.append({
                        "ticker": ticker,
                        "fiscal_year": hist.index[-1].year,
                        "forward_6m_return": (latest - six_mo_ago) / six_mo_ago,
                    })

        except Exception:
            continue

    return pd.DataFrame(records)


def build_merged_panel() -> pd.DataFrame:
    """Build the full research panel: AI adoption scores + fundamentals + returns.

    Merges on (ticker, fiscal_year).
    """
    # Load AI adoption scores
    adoption = load_adoption_panel()
    tickers = adoption["ticker"].unique().tolist()

    print(f"Building panel for {len(tickers)} companies...")

    # Fetch fundamentals
    print("\nFetching fundamentals from yfinance...")
    fundamentals = fetch_annual_fundamentals(tickers)

    # Fetch returns
    print("\nFetching stock returns...")
    returns = fetch_stock_returns(tickers)

    # Merge: adoption + fundamentals
    panel = adoption.copy()
    if not fundamentals.empty:
        # Aggregate fundamentals by (ticker, fiscal_year) — take first non-null
        fund_agg = fundamentals.groupby(["ticker", "fiscal_year"]).first().reset_index()
        panel = panel.merge(fund_agg, on=["ticker", "fiscal_year"], how="left", suffixes=("", "_fund"))

    # Merge returns
    if not returns.empty:
        ret_agg = returns.groupby(["ticker", "fiscal_year"]).first().reset_index()
        panel = panel.merge(ret_agg, on=["ticker", "fiscal_year"], how="left", suffixes=("", "_ret"))

    # Add forward return (next year's return as target variable)
    panel = panel.sort_values(["ticker", "fiscal_year"])
    panel["forward_annual_return"] = panel.groupby("ticker")["annual_return"].shift(-1)

    # Clean up
    panel = panel.sort_values(["industry", "ticker", "fiscal_year"]).reset_index(drop=True)

    print(f"\nMerged panel: {len(panel)} rows, {len(panel.columns)} columns")
    print(f"Companies: {panel['ticker'].nunique()}")
    print(f"Year range: {panel['fiscal_year'].min()} - {panel['fiscal_year'].max()}")

    return panel


def save_merged_panel(df: pd.DataFrame) -> Path:
    """Save the merged panel dataset."""
    MERGED_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MERGED_PANEL_PATH, index=False)
    print(f"Saved merged panel: {MERGED_PANEL_PATH} ({len(df)} rows)")
    return MERGED_PANEL_PATH


if __name__ == "__main__":
    panel = build_merged_panel()
    if not panel.empty:
        save_merged_panel(panel)
        print(f"\nColumns: {list(panel.columns)}")
