"""Data validation before writing outputs: schema, null rates, range guards."""
from __future__ import annotations

import pandas as pd
from typing import Any

REQUIRED_COMPS_COLUMNS = {"industry", "ticker", "name"}
NULL_RATE_MAX = 0.95  # fail if >95% of a column is null
PE_RANGE = (0, 500)  # trailing_pe sanity
EV_EBITDA_RANGE = (0, 100)
FCF_YIELD_RANGE = (-0.5, 0.5)


def validate_comparables(df: pd.DataFrame) -> dict[str, Any]:
    """Run checks on comparables table. Returns log dict with errors, warnings, row_count."""
    log = {"errors": [], "warnings": [], "row_count": len(df), "passed": True}
    if df.empty:
        log["errors"].append("Comparables table is empty")
        log["passed"] = False
        return log
    missing = REQUIRED_COMPS_COLUMNS - set(df.columns)
    if missing:
        log["errors"].append(f"Missing required columns: {missing}")
        log["passed"] = False
    for col in df.columns:
        if col.startswith("_") or col == "industry":
            continue
        null_rate = df[col].isna().mean()
        if null_rate > NULL_RATE_MAX:
            log["warnings"].append(f"High null rate in {col}: {null_rate:.1%}")
    if "trailing_pe" in df.columns:
        pe = pd.to_numeric(df["trailing_pe"], errors="coerce")
        pe = pe[pe.notna() & (pe > 0)]
        if not pe.empty and (pe.min() < PE_RANGE[0] or pe.max() > PE_RANGE[1]):
            log["warnings"].append(f"trailing_pe outside typical range {PE_RANGE}: min={pe.min():.1f}, max={pe.max():.1f}")
    if "ev_to_ebitda" in df.columns:
        ev = pd.to_numeric(df["ev_to_ebitda"], errors="coerce")
        ev = ev[ev.notna() & (ev > 0)]
        if not ev.empty and ev.max() > EV_EBITDA_RANGE[1]:
            log["warnings"].append(f"ev_to_ebitda max {ev.max():.1f} above {EV_EBITDA_RANGE[1]}")
    return log
