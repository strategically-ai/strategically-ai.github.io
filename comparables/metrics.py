"""Build comparables table: multiples and operational metrics for display."""
from __future__ import annotations

import pandas as pd


DISPLAY_COLUMNS = [
    "ticker",
    "name",
    "market_cap",
    "ev_to_ebitda",
    "trailing_pe",
    "forward_pe",
    "ev_to_revenue",
    "fcf_yield",
    "net_debt_ebitda",
    "profit_margin",
    "operating_margin",
    "revenue_growth",
]
OUTLIER_STD_THRESHOLD = 3.0


def _fmt_num(x: float | None, decimals: int = 2) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return f"{float(x):,.{decimals}f}"
    except (TypeError, ValueError):
        return None


def _fmt_pct(x: float | None) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return None


def build_display_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Select and format columns for the Market Comps page. Keeps numeric for CSV; display formatting in Quarto."""
    cols = [c for c in DISPLAY_COLUMNS if c in raw.columns]
    out = raw[cols].copy()
    # Optional: format for readability in CSV (e.g. millions for market_cap)
    if "market_cap" in out.columns:
        out["market_cap_m"] = out["market_cap"].apply(lambda x: round(x / 1e6, 1) if pd.notna(x) and x else None)
    return out


def add_outlier_flags(table: pd.DataFrame, metric: str = "trailing_pe") -> pd.DataFrame:
    """Add outlier_flag column: True if metric > sector median + OUTLIER_STD_THRESHOLD * std (for P/E-style metrics)."""
    if metric not in table.columns or table[metric].dropna().empty:
        return table
    col = pd.to_numeric(table[metric], errors="coerce")
    med = col.median()
    std = col.std()
    if pd.isna(std) or std == 0:
        table["outlier_flag"] = False
        return table
    # High P/E = potential outlier; flag if above median + 3*std
    table["outlier_flag"] = col > (med + OUTLIER_STD_THRESHOLD * std)
    return table


def build_comparables_table(raw: pd.DataFrame, industry: str) -> pd.DataFrame:
    """Build final comparables table for one industry: display columns + industry + outlier flags."""
    table = build_display_table(raw)
    table.insert(0, "industry", industry)
    table = add_outlier_flags(table, "trailing_pe")
    return table
