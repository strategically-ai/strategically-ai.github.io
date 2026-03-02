"""Fetch market and financial data for comparables. Primary: yfinance. Stubs: SEC, external API."""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


def fetch_yfinance(tickers: list[str], throttle_seconds: float = 0.2) -> pd.DataFrame:
    """Fetch quote and key stats from yfinance. Returns one row per ticker with standard columns."""
    if yf is None:
        raise ImportError("yfinance is required. pip install yfinance")
    rows = []
    for i, ticker in enumerate(tickers):
        if throttle_seconds and i > 0:
            time.sleep(throttle_seconds)
        try:
            t = yf.Ticker(ticker)
            info = t.info
            fcf = info.get("freeCashflow")
            mcap = info.get("marketCap")
            ev = info.get("enterpriseValue")
            ebitda = info.get("ebitda")
            total_debt = info.get("totalDebt") or 0
            total_cash = info.get("totalCash") or 0
            net_debt = total_debt - total_cash if (total_debt is not None and total_cash is not None) else None
            rows.append({
                "ticker": ticker,
                "name": info.get("shortName") or info.get("longName") or ticker,
                "market_cap": mcap,
                "enterprise_value": ev,
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "revenue": info.get("totalRevenue"),
                "ebitda": ebitda,
                "net_income": info.get("netIncomeToCommon"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "free_cash_flow": fcf,
                "total_debt": total_debt,
                "total_cash": total_cash,
                "net_debt": net_debt,
                "fcf_yield": (fcf / mcap) if (fcf and mcap and mcap > 0) else None,
                "net_debt_ebitda": (net_debt / ebitda) if (net_debt is not None and ebitda and ebitda > 0) else None,
            })
        except Exception as e:
            rows.append({
                "ticker": ticker,
                "name": ticker,
                "market_cap": None,
                "enterprise_value": None,
                "trailing_pe": None,
                "forward_pe": None,
                "ev_to_revenue": None,
                "ev_to_ebitda": None,
                "revenue": None,
                "ebitda": None,
                "net_income": None,
                "profit_margin": None,
                "operating_margin": None,
                "revenue_growth": None,
                "earnings_growth": None,
                "free_cash_flow": None,
                "fcf_yield": None,
                "net_debt_ebitda": None,
                "_error": str(e),
            })
    return pd.DataFrame(rows)


def fetch_sec_stub(tickers: list[str]) -> pd.DataFrame:
    """Stub: SEC/EDGAR enrichment. Returns empty DataFrame with expected columns; implement later."""
    return pd.DataFrame()


def fetch_external_api_stub(tickers: list[str]) -> pd.DataFrame:
    """Stub: external API (e.g. FMP, Alpha Vantage). Returns empty DataFrame; implement when key available."""
    return pd.DataFrame()


def fetch_all(
    tickers: list[str],
    use_yfinance: bool = True,
    use_sec: bool = False,
    use_external_api: bool = False,
) -> pd.DataFrame:
    """Fetch from enabled sources. Today only yfinance is implemented; SEC and API are stubs."""
    out = None
    if use_yfinance and yf is not None:
        out = fetch_yfinance(tickers)
    if use_sec:
        sec = fetch_sec_stub(tickers)
        if out is not None and not sec.empty:
            out = out.merge(sec, on="ticker", how="left")
        elif out is None:
            out = sec
    if use_external_api:
        ext = fetch_external_api_stub(tickers)
        if out is not None and not ext.empty:
            out = out.merge(ext, on="ticker", how="left")
        elif out is None:
            out = ext
    if out is None:
        raise RuntimeError("No data source enabled; install yfinance or implement SEC/API.")
    return out
