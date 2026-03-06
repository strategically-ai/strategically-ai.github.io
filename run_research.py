#!/usr/bin/env python3
"""Run the research pipeline: backfill SEC filings, compute AI Adoption Scores, build panel dataset.

Usage:
    # Full pipeline (all industries, from 2020)
    python run_research.py

    # Single industry, test mode
    python run_research.py --industry technology --max-companies 5

    # Backfill only (no scoring)
    python run_research.py --backfill-only

    # Score only (assumes filings already cached)
    python run_research.py --score-only

    # Skip LLM (keyword-only scoring, no API cost)
    python run_research.py --no-llm
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from research.sec_backfill import backfill_filings
from research.ai_adoption_score import score_all_filings, save_adoption_panel
from research.panel_dataset import build_merged_panel, save_merged_panel

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs" / "research"


def write_run_log(log: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / "run_log.json"
    log["timestamp"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Run log: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Strategist research pipeline")
    parser.add_argument("--industry", default=None, help="Filter to specific industry (e.g. technology)")
    parser.add_argument("--start-year", type=int, default=2020, help="Earliest filing year (default: 2020)")
    parser.add_argument("--max-companies", type=int, default=None, help="Limit companies (for testing)")
    parser.add_argument("--backfill-only", action="store_true", help="Only download filings, skip scoring")
    parser.add_argument("--score-only", action="store_true", help="Only score (assumes filings cached)")
    parser.add_argument("--panel-only", action="store_true", help="Only build merged panel (assumes scores exist)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM classification (keyword-only)")
    args = parser.parse_args()

    run_log = {"errors": [], "stages_completed": []}
    industries = [args.industry] if args.industry else None

    # Stage 1: Backfill SEC filings
    if not args.score_only and not args.panel_only:
        print("=" * 60)
        print("STAGE 1: Backfilling SEC 10-K filings from EDGAR")
        print("=" * 60)
        try:
            filings_df = backfill_filings(
                industries=industries,
                start_year=args.start_year,
                max_companies=args.max_companies,
            )
            run_log["filings_cached"] = len(filings_df)
            run_log["companies_with_filings"] = int(filings_df["ticker"].nunique()) if not filings_df.empty else 0
            run_log["stages_completed"].append("backfill")
        except Exception as e:
            print(f"Error in backfill: {e}")
            run_log["errors"].append(f"backfill: {e}")

    if args.backfill_only:
        write_run_log(run_log)
        return

    # Stage 2: Score all filings
    if not args.panel_only:
        print("\n" + "=" * 60)
        print("STAGE 2: Computing AI Adoption Scores")
        print("=" * 60)
        try:
            scores_df = score_all_filings(use_llm=not args.no_llm)
            if not scores_df.empty:
                save_adoption_panel(scores_df)
                run_log["filings_scored"] = len(scores_df)
                run_log["mean_ai_score"] = round(scores_df["ai_adoption_score"].mean(), 2)
                run_log["stages_completed"].append("scoring")
            else:
                print("No scores computed.")
        except Exception as e:
            print(f"Error in scoring: {e}")
            run_log["errors"].append(f"scoring: {e}")

    if args.score_only:
        write_run_log(run_log)
        return

    # Stage 3: Build merged panel
    print("\n" + "=" * 60)
    print("STAGE 3: Building merged panel dataset (AI scores + valuations)")
    print("=" * 60)
    try:
        panel = build_merged_panel()
        if not panel.empty:
            save_merged_panel(panel)
            run_log["panel_rows"] = len(panel)
            run_log["panel_columns"] = len(panel.columns)
            run_log["stages_completed"].append("panel")
    except Exception as e:
        print(f"Error building panel: {e}")
        run_log["errors"].append(f"panel: {e}")

    write_run_log(run_log)
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    if run_log["errors"]:
        print(f"Errors: {run_log['errors']}")
    print(f"Stages completed: {run_log['stages_completed']}")


if __name__ == "__main__":
    main()
