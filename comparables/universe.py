"""Load comparables universe (industry, ticker, name)."""
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_universe(path: str | Path | None = None) -> pd.DataFrame:
    """Load universe CSV. Default: data/universes/comparables_universe.csv."""
    if path is None:
        path = _repo_root() / "data" / "universes" / "comparables_universe.csv"
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Universe not found: {path}")
    df = pd.read_csv(path)
    for col in ("industry", "ticker", "name"):
        if col not in df.columns:
            raise ValueError(f"Universe must have column: {col}")
    return df


def list_industries(df: pd.DataFrame | None = None) -> list[str]:
    """Return sorted list of industry IDs (e.g. Consumer, Industrials, Financials)."""
    if df is None:
        df = load_universe()
    return sorted(df["industry"].dropna().unique().tolist())
