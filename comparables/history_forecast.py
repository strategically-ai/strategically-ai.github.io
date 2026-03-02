"""
Historical revenue from yfinance and a simple growth-rate forecast.
No ML: we use trailing growth to project forward (level 1 data science).
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

REVENUE_ROW_NAMES = ("Total Revenue", "Operating Revenue")
HISTORICAL_QUARTERS = 8
FORECAST_QUARTERS = 4


def _get_quarterly_revenue_series(ticker: str) -> pd.Series:
    """Return a Series of (period_end -> revenue) for the last HISTORICAL_QUARTERS quarters."""
    if yf is None:
        raise ImportError("yfinance is required")
    t = yf.Ticker(ticker)
    q = t.quarterly_income_stmt
    if q is None or q.empty:
        return pd.Series(dtype=float)
    for row_name in REVENUE_ROW_NAMES:
        if row_name in q.index:
            s = q.loc[row_name].dropna()
            s = s[s > 0]
            s = s.sort_index(ascending=False).head(HISTORICAL_QUARTERS)
            return s
    return pd.Series(dtype=float)


def fetch_historical_revenue(
    tickers: list[str],
    throttle_seconds: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch quarterly revenue for each ticker. Returns long-format DataFrame:
    ticker, period_end, revenue, type='historical'.
    """
    rows = []
    for i, ticker in enumerate(tickers):
        if throttle_seconds and i > 0:
            time.sleep(throttle_seconds)
        try:
            s = _get_quarterly_revenue_series(ticker)
            for period_end, rev in s.items():
                rows.append({
                    "ticker": ticker,
                    "period_end": pd.Timestamp(period_end),
                    "revenue": float(rev),
                    "type": "historical",
                })
        except Exception:
            pass
    if not rows:
        return pd.DataFrame(columns=["ticker", "period_end", "revenue", "type"])
    df = pd.DataFrame(rows)
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.normalize()
    return df


def _simple_growth_forecast(
    series: pd.Series,
    n_quarters: int = FORECAST_QUARTERS,
    bear_mult: float = 0.5,
    bull_mult: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    """
    Project next n_quarters using trailing growth. Returns (base_series, bear_series, bull_series, q_growth).
    Bear/bull use growth scaled by bear_mult and bull_mult (e.g. 0.5x and 1.5x).
    """
    empty = pd.Series(dtype=float)
    if series is None or len(series) < 2:
        return empty, empty, empty, 0.0
    s = series.sort_index(ascending=True).dropna()
    if len(s) < 2:
        return empty, empty, empty, 0.0
    first_val = s.iloc[0]
    last_val = s.iloc[-1]
    if first_val <= 0:
        return empty, empty, empty, 0.0
    n_periods = len(s) - 1
    q_growth = (last_val / first_val) ** (1 / n_periods) - 1.0
    last_ts = s.index[-1]
    out_base, out_bear, out_bull = {}, {}, {}
    for i in range(1, n_quarters + 1):
        next_ts = last_ts + pd.DateOffset(months=3 * i)
        out_base[next_ts] = last_val * ((1 + q_growth) ** i)
        out_bear[next_ts] = last_val * ((1 + q_growth * bear_mult) ** i)
        out_bull[next_ts] = last_val * ((1 + q_growth * bull_mult) ** i)
    return pd.Series(out_base), pd.Series(out_bear), pd.Series(out_bull), q_growth


def forecast_revenue(historical: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker, project FORECAST_QUARTERS quarters: base, bear (0.5x growth), bull (1.5x growth).
    Appends rows with type in ('forecast', 'forecast_bear', 'forecast_bull').
    """
    if historical.empty or "ticker" not in historical.columns:
        return historical.copy()
    forecast_rows = []
    for ticker, g in historical.groupby("ticker"):
        g = g.sort_values("period_end")
        rev_series = g.set_index("period_end")["revenue"]
        base, bear, bull, _ = _simple_growth_forecast(rev_series, n_quarters=FORECAST_QUARTERS)
        for period_end, rev in base.items():
            forecast_rows.append({
                "ticker": ticker,
                "period_end": pd.Timestamp(period_end),
                "revenue": float(rev),
                "type": "forecast",
            })
        for period_end, rev in bear.items():
            forecast_rows.append({
                "ticker": ticker,
                "period_end": pd.Timestamp(period_end),
                "revenue": float(rev),
                "type": "forecast_bear",
            })
        for period_end, rev in bull.items():
            forecast_rows.append({
                "ticker": ticker,
                "period_end": pd.Timestamp(period_end),
                "revenue": float(rev),
                "type": "forecast_bull",
            })
    if not forecast_rows:
        return historical.copy()
    combined = pd.concat([
        historical,
        pd.DataFrame(forecast_rows),
    ], ignore_index=True)
    return combined


def build_revenue_historical_forecast(
    tickers: list[str],
    throttle_seconds: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch historical quarterly revenue and append simple growth-based forecasts.
    Returns long-format: ticker, period_end, revenue, type (historical | forecast).
    """
    hist = fetch_historical_revenue(tickers, throttle_seconds=throttle_seconds)
    if hist.empty:
        return hist
    return forecast_revenue(hist)
