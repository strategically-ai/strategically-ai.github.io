"""I/O helpers for reading/writing tracker outputs."""

import os
from datetime import datetime

import pandas as pd

from trackers.config import BASE_OUTPUT_DIR


def get_output_dir(industry_id: str, subdir: str) -> str:
    """Return the path to an industry subdir (universe, sec_10q, posts, weekly_summaries)."""
    path = os.path.join(BASE_OUTPUT_DIR, industry_id, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def write_disclosures_table(df: pd.DataFrame, industry_id: str, run_ts: str) -> str:
    """Write sec_ai_disclosures to CSV; return path."""
    out_dir = get_output_dir(industry_id, "sec_10q")
    fname = f"sec_ai_disclosures_{run_ts}.csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False)
    return path


def write_posts_table(df: pd.DataFrame, industry_id: str, run_ts: str) -> str:
    """Write company_posts (e.g. RSS news) to CSV; return path."""
    out_dir = get_output_dir(industry_id, "posts")
    fname = f"company_posts_{run_ts}.csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False)
    return path


def list_latest_posts(industry_id: str, limit: int = 5) -> list[dict]:
    """List latest company_posts files for an industry (path, run_ts) for weekly synthesis."""
    out_dir = get_output_dir(industry_id, "posts")
    results = []
    for f in os.listdir(out_dir):
        if f.startswith("company_posts_") and f.endswith(".csv"):
            ts = f.replace("company_posts_", "").replace(".csv", "")
            results.append({"run_ts": ts, "path": os.path.join(out_dir, f)})
    results.sort(key=lambda x: x["run_ts"], reverse=True)
    return results[:limit]


def write_weekly_summary(content: str, industry_id: str, run_date: str) -> str:
    """Write weekly summary markdown; return path."""
    out_dir = get_output_dir(industry_id, "weekly_summaries")
    fname = f"industry_weekly_summary_{run_date}.md"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        f.write(content)
    return path


def run_timestamp() -> str:
    """Current run timestamp for filenames and table column."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def run_date() -> str:
    """Current date YYYY-MM-DD for weekly summary filenames."""
    return datetime.utcnow().strftime("%Y-%m-%d")


def list_latest_summaries(industry_id: str) -> list[dict]:
    """List weekly summary files for an industry (path, date) for the refreshes page."""
    out_dir = get_output_dir(industry_id, "weekly_summaries")
    results = []
    for f in os.listdir(out_dir):
        if f.startswith("industry_weekly_summary_") and f.endswith(".md"):
            date_part = f.replace("industry_weekly_summary_", "").replace(".md", "")
            results.append({"date": date_part, "path": f})
    results.sort(key=lambda x: x["date"], reverse=True)
    return results[:10]


def get_latest_sec_path(industry_id: str) -> str | None:
    """Return path to the latest sec_ai_disclosures CSV, or None."""
    sec_dir = get_output_dir(industry_id, "sec_10q")
    latest_ts = None
    latest_path = None
    for f in os.listdir(sec_dir):
        if f.startswith("sec_ai_disclosures_") and f.endswith(".csv"):
            ts = f.replace("sec_ai_disclosures_", "").replace(".csv", "")
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
                latest_path = os.path.join(sec_dir, f)
    return latest_path


def get_latest_run_metadata(industry_id: str) -> dict | None:
    """Return latest run timestamp and paths if any sec_10q or weekly_summaries exist."""
    sec_dir = os.path.join(BASE_OUTPUT_DIR, industry_id, "sec_10q")
    summaries = list_latest_summaries(industry_id)
    latest_sec = None
    if os.path.isdir(sec_dir):
        for f in os.listdir(sec_dir):
            if f.startswith("sec_ai_disclosures_") and f.endswith(".csv"):
                ts = f.replace("sec_ai_disclosures_", "").replace(".csv", "")
                if latest_sec is None or ts > latest_sec:
                    latest_sec = ts
    return {
        "latest_sec_run": latest_sec,
        "latest_summaries": summaries,
    }


def list_industries_with_summaries() -> list[str]:
    """List industry_ids that have at least one weekly summary file (for nav)."""
    if not os.path.isdir(BASE_OUTPUT_DIR):
        return []
    industries = []
    for name in sorted(os.listdir(BASE_OUTPUT_DIR)):
        sub = os.path.join(BASE_OUTPUT_DIR, name)
        if not os.path.isdir(sub):
            continue
        summary_dir = os.path.join(sub, "weekly_summaries")
        if not os.path.isdir(summary_dir):
            continue
        for f in os.listdir(summary_dir):
            if f.startswith("industry_weekly_summary_") and f.endswith(".md"):
                industries.append(name)
                break
    return industries


def write_8k_table(df: pd.DataFrame, industry_id: str, run_ts: str) -> str:
    """Write sec_8k_signals to CSV; return path."""
    out_dir = get_output_dir(industry_id, "sec_8k")
    fname = f"sec_8k_signals_{run_ts}.csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False)
    return path


def get_latest_8k_path(industry_id: str) -> str | None:
    """Return path to the latest sec_8k_signals CSV, or None."""
    sec_dir = get_output_dir(industry_id, "sec_8k")
    latest_ts = None
    latest_path = None
    for f in os.listdir(sec_dir):
        if f.startswith("sec_8k_signals_") and f.endswith(".csv"):
            ts = f.replace("sec_8k_signals_", "").replace(".csv", "")
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
                latest_path = os.path.join(sec_dir, f)
    return latest_path


def list_8k_signals(industry_id: str, limit: int = 10) -> list[dict]:
    """Load latest 8-K CSV and return top signals as list of dicts (M&A first, then capex)."""
    path = get_latest_8k_path(industry_id)
    if not path or not os.path.isfile(path):
        return []
    try:
        df = pd.read_csv(path)
        if df.empty:
            return []
        type_order = {"ma": 0, "capex": 1, "leadership": 2, "earnings": 3, "other": 4}
        df["_sort"] = df["signal_type"].map(type_order).fillna(4)
        df = df.sort_values("_sort").drop(columns=["_sort"])
        return df.head(limit).to_dict(orient="records")
    except Exception:
        return []


def get_segment_summary(industry_id: str) -> dict:
    """Return {segment: unique_company_count} from the latest company_posts CSV."""
    posts = list_latest_posts(industry_id, limit=1)
    if not posts or not os.path.isfile(posts[0]["path"]):
        return {}
    try:
        df = pd.read_csv(posts[0]["path"])
        if "segment" not in df.columns:
            return {}
        return df.groupby("segment")["company_name"].nunique().to_dict()
    except Exception:
        return {}


def get_news_feed_summary(industry_id: str) -> dict | None:
    """Return { configured_count, sources_in_run } from config and latest posts CSV, or None."""
    try:
        import yaml
    except ImportError:
        return None
    repo_root = os.path.dirname(os.path.dirname(BASE_OUTPUT_DIR))
    config_path = os.path.join(repo_root, "config", "news_feeds.yaml")
    configured = 0
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            urls = data.get(industry_id) or data.get("default") or []
            configured = len(urls) if isinstance(urls, list) else 0
        except Exception:
            pass
    posts = list_latest_posts(industry_id, limit=1)
    sources_in_run = []
    if posts and os.path.isfile(posts[0]["path"]):
        try:
            df = pd.read_csv(posts[0]["path"])
            sources_in_run = sorted(df["source"].dropna().unique().tolist()) if "source" in df.columns else []
        except Exception:
            pass
    return {"configured_count": configured, "sources_in_run": sources_in_run}
