#!/usr/bin/env python3
"""
Run the industry AI tracker for a given industry.
Usage:
  python run_industry_tracker.py consumer
  python run_industry_tracker.py consumer --pipelines sec
  python run_industry_tracker.py consumer --pipelines sec,8k,news,weekly
"""

import argparse
import os
import sys

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trackers.sec_pipeline import run_industry_sec
from trackers.sec_8k_pipeline import run_industry_8k
from trackers.weekly_synthesis import generate_weekly_summary
from trackers.news_pipeline import run_industry_news


def main():
    parser = argparse.ArgumentParser(description="Run industry AI tracker")
    parser.add_argument("industry_id", choices=["consumer", "industrials", "financials"], help="Industry to run")
    parser.add_argument(
        "--pipelines",
        default="sec,8k,news,weekly",
        help="Comma-separated pipelines: sec, 8k, news, weekly (default: sec,8k,news,weekly)",
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM classification (use defaults)")
    args = parser.parse_args()

    pipelines = [p.strip().lower() for p in args.pipelines.split(",")]
    industry_id = args.industry_id
    openai_client = None
    needs_llm = not args.no_llm and any(p in pipelines for p in ("sec", "8k", "weekly"))
    if needs_llm:
        try:
            from openai import OpenAI
            from trackers.env_loader import get_openai_config
            api_key, base_url = get_openai_config()
            if not api_key:
                print("Warning: No OpenAI key (set OPENAI_API_KEY or add api_keys.py) - running without LLM")
            else:
                openai_client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            print("Warning: OpenAI client not available:", e, "- running without LLM")

    run_ts = None

    if "sec" in pipelines:
        print(f"Running SEC 10-Q pipeline for {industry_id}...")
        run_ts = run_industry_sec(industry_id, openai_client=openai_client)
        print(f"SEC pipeline done. Run timestamp: {run_ts}")

    if "8k" in pipelines:
        print(f"Running SEC 8-K pipeline for {industry_id}...")
        k8_ts = run_industry_8k(industry_id, openai_client=openai_client)
        print(f"8-K pipeline done. Run timestamp: {k8_ts}")

    if "news" in pipelines:
        print(f"Running news pipeline for {industry_id} (per-company RSS + industry feeds)...")
        news_ts = run_industry_news(industry_id)
        if news_ts:
            print(f"News pipeline done. Run timestamp: {news_ts}")
        else:
            print("News pipeline produced no results.")

    if "weekly" in pipelines:
        print(f"Generating weekly summary for {industry_id}...")
        path = generate_weekly_summary(industry_id, run_ts=run_ts, openai_client=openai_client)
        print(f"Weekly summary written: {path}")


if __name__ == "__main__":
    main()
