"""Write comparables outputs: CSV per industry (versioned + latest), last_updated.txt, run_log.json."""
from pathlib import Path
from datetime import datetime
import json

import pandas as pd

from .universe import _repo_root


def get_output_dir() -> Path:
    return _repo_root() / "outputs" / "comparables"


def ensure_output_dir() -> Path:
    d = get_output_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _month_suffix() -> str:
    return datetime.utcnow().strftime("%Y-%m")


def write_industry_csv(df: pd.DataFrame, industry: str, versioned: bool = True) -> list[Path]:
    """Write comparables_<industry>.csv (latest) and optionally comparables_<industry>_YYYY-MM.csv. Returns paths."""
    out_dir = ensure_output_dir()
    safe = industry.lower().replace(" ", "_")
    paths = []
    latest = out_dir / f"comparables_{safe}.csv"
    df.to_csv(latest, index=False)
    paths.append(latest)
    if versioned:
        versioned_path = out_dir / f"comparables_{safe}_{_month_suffix()}.csv"
        df.to_csv(versioned_path, index=False)
        paths.append(versioned_path)
    return paths


def write_last_updated() -> Path:
    """Write last_updated.txt with current date. Returns path."""
    out_dir = ensure_output_dir()
    path = out_dir / "last_updated.txt"
    path.write_text(datetime.utcnow().strftime("%Y-%m-%d"))
    return path


def write_revenue_forecast(df: pd.DataFrame, industry: str) -> Path:
    """Write revenue_forecast_<industry>.csv (long: ticker, period_end, revenue, type). Returns path."""
    out_dir = ensure_output_dir()
    safe = industry.lower().replace(" ", "_")
    path = out_dir / f"revenue_forecast_{safe}.csv"
    df = df.copy()
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False)
    return path


def write_run_log(log: dict) -> Path:
    """Write run_log.json with timestamp, industries, row counts, errors. Returns path."""
    out_dir = ensure_output_dir()
    path = out_dir / "run_log.json"
    log["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    return path
