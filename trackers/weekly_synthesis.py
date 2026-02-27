"""Generate weekly summary: 10-Qs as strategy blueprint, 8-K signals as material events, news as current context; LLM synthesizes all three into a segment-aware narrative."""

import os

import pandas as pd

from datetime import datetime

from trackers.io import get_latest_sec_path, get_latest_8k_path, list_latest_posts, run_date, write_weekly_summary


def _format_week_display(ymd: str) -> str:
    """Format YYYY-MM-DD as '25 February 2026'."""
    if not ymd or len(ymd) < 10:
        return ymd
    try:
        d = datetime.strptime(ymd[:10], "%Y-%m-%d")
        return d.strftime("%d %B %Y")
    except Exception:
        return ymd


SYNTHESIS_PROMPT = """You are a senior analyst writing for The AI Strategist newsletter. Your readers are PE investors, growth equity analysts, strategic operators, and M&A advisors. They want crisp, evidence-backed analysis — not consulting speak.

**Voice:** Confident, precise, intellectually direct. No MBA jargon. Replace vague verbs with operational ones: automate, consolidate, replatform, reduce cycle time, standardize, underwrite. Use: mechanism, constraint, tradeoff, wedge, pricing power, unit economics, throughput, roll-up, value creation.

**Avoid:** "aligns with", "reflects the importance of", "signifies", "underscores", "amidst a competitive landscape", "pivotal", "ongoing importance", "enhanced operational execution", "reflective of", "in an effort to", "leveraging synergies". No cheerleading. No breathless futurism. No generic summaries of what companies said.

**Inputs you have:**
1. **Strategy blueprint (from 10-Qs)** — Organized by segment: Fortune 100, PE mid-size ($500M–$10B), Startups (VC-backed). Use as stable context for interpreting what moves mean, not as "news."
2. **8-K material events** — Highest-signal input. Filed within 4 business days of the event. Every M&A item must be called out explicitly with acquirer thesis.
3. **News this week** — Headlines organized by segment. Cite sources naturally using [link](url) format.

**Structural requirement:** The summary must explicitly compare at least two of the three segments (Fortune 100 vs PE mid-size, or PE mid-size vs Startups, etc.). The reader wants competitive dynamics across the value chain, not a single-tier review.

**Narrative moves to use:** (1) Contrast: two tiers reacting differently to the same pressure. (2) Mechanism-first: what changes operationally (cost, throughput, service level) and what that changes (margin, pricing, retention). (3) Signal vs noise: is this structural or a one-quarter anomaly? (4) M&A logic: state the acquirer's strategic thesis and what it implies for competitors.

Output format: No title. No "Industry" line at the top.

Start with a lede (no heading): 2–4 sentences on what this week reveals about AI and automation adoption dynamics across the industry's tiers. State one throughline. No company-by-company listing.

Then use ONLY these section headings:

## The week in signals
2–4 themes drawn from 8-K events, news, and filing signals combined. At least one theme must address contrast between segments. Each theme: one short paragraph, one concrete example, one implication for margin/working capital/competitive position. Use [link](url) where available.

## M&A and deal activity
ONLY include if 8-K filings or news contain M&A signals. If none, omit this section entirely. For each deal or signal: acquirer, target (or nature of deal), price if disclosed, strategic rationale in one sentence, implication for competitors.

## Company moves
3–6 bullets maximum. Each bullet: company name + segment tag [F100], [PE], or [Startup] + action + mechanism + implication. Include [source](url). Choose moves with the most signal, not the most recent. Do not list every company.

## Implications for investors and operators
One paragraph. Address: what PE funds should track in the PE mid-size segment; what makes startups acquirable or threatening to incumbents; where Fortune 100 capex signals structural commitment (not just a press release).

## Watchlist
5–7 questions framed as investment or strategic theses to track. Specific, answerable, forward-looking. Tag each with [F100], [PE], or [Startup].

Do not add sections beyond these five. Do not manufacture quotes. If inputs are thin for a segment, note it briefly rather than padding with generalities."""


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _build_segment_block(df: pd.DataFrame, max_rows: int = 8) -> str:
    """Format a segment's SEC rows as a concise bullet list."""
    lines = []
    for _, row in df.head(max_rows).iterrows():
        company = str(row.get("company_name", ""))
        theme = str(row.get("theme", ""))
        why = str(row.get("why_it_matters", ""))[:200]
        url = str(row.get("filing_url", ""))
        line = f"- {company}: {theme}. {why}"
        if url:
            line += f" [10-Q]({url})"
        lines.append(line)
    return "\n".join(lines) if lines else "(No data.)"


def _build_blueprint_context(sec_df: pd.DataFrame) -> str:
    """Build three-section SEC blueprint context (Fortune 100 / PE Mid / Startups)."""
    if sec_df is None or sec_df.empty:
        return "(No 10-Q blueprint loaded.)"
    segment_col = "segment" if "segment" in sec_df.columns else None
    if not segment_col:
        return _build_segment_block(sec_df)

    sections = []
    for seg, label in [("fortune_100", "Fortune 100"), ("pe_mid", "PE Mid-Size ($500M–$10B)"), ("startup", "Startups (VC-backed)")]:
        sub = sec_df[sec_df[segment_col] == seg]
        if not sub.empty:
            sections.append(f"**{label}:**\n{_build_segment_block(sub)}")
    return "\n\n".join(sections) if sections else _build_segment_block(sec_df)


def _build_8k_context(k8_df: pd.DataFrame) -> str:
    """Format 8-K material events context block: M&A and capex first."""
    if k8_df is None or k8_df.empty:
        return ""
    lines = []
    # M&A first
    ma = k8_df[k8_df["is_ma"] == True]
    for _, row in ma.head(6).iterrows():
        company = str(row.get("company_name", ""))
        seg = str(row.get("segment", ""))
        excerpt = str(row.get("excerpt", ""))[:300]
        url = str(row.get("filing_url", ""))
        dollar = str(row.get("dollar_amount", ""))
        dollar_str = f" ({dollar})" if dollar and dollar != "nan" else ""
        tag = f"[{seg.upper()}]" if seg and seg != "nan" else ""
        line = f"- [M&A] {company} {tag}{dollar_str}: {excerpt}"
        if url:
            line += f" [8-K]({url})"
        lines.append(line)
    # Capex
    capex = k8_df[(k8_df["is_ma"] != True) & (k8_df["is_capex"] == True)]
    for _, row in capex.head(4).iterrows():
        company = str(row.get("company_name", ""))
        seg = str(row.get("segment", ""))
        excerpt = str(row.get("excerpt", ""))[:200]
        url = str(row.get("filing_url", ""))
        dollar = str(row.get("dollar_amount", ""))
        dollar_str = f" ({dollar})" if dollar and dollar != "nan" else ""
        tag = f"[{seg.upper()}]" if seg and seg != "nan" else ""
        line = f"- [Capex] {company} {tag}{dollar_str}: {excerpt}"
        if url:
            line += f" [8-K]({url})"
        lines.append(line)
    # Leadership hires
    leadership = k8_df[
        (k8_df["is_ma"] != True) & (k8_df["is_capex"] != True) & (k8_df["is_leadership_hire"] == True)
    ]
    for _, row in leadership.head(3).iterrows():
        company = str(row.get("company_name", ""))
        seg = str(row.get("segment", ""))
        excerpt = str(row.get("excerpt", ""))[:200]
        url = str(row.get("filing_url", ""))
        tag = f"[{seg.upper()}]" if seg and seg != "nan" else ""
        line = f"- [Leadership] {company} {tag}: {excerpt}"
        if url:
            line += f" [8-K]({url})"
        lines.append(line)
    return "\n".join(lines) if lines else ""


def _build_news_context(posts_df: pd.DataFrame) -> tuple[str, dict]:
    """
    Build news context block: M&A items first, then grouped by segment.
    Returns (context_str, source_counts).
    """
    if posts_df is None or posts_df.empty:
        return "(No news items this week.)", {}

    lines = []
    source_counts: dict[str, int] = {}

    def _format_entry(row) -> str:
        title = str(row.get("title", "")) or "(No title)"
        url = str(row.get("post_url", ""))
        company = str(row.get("company_name", ""))
        source = str(row.get("source", "rss"))
        seg = str(row.get("segment", ""))
        theme = str(row.get("theme", ""))
        source_counts[source] = source_counts.get(source, 0) + 1
        ma_tag = " [M&A]" if theme == "ma" else ""
        seg_tag = f" [{seg.upper()}]" if seg and seg != "nan" else ""
        if url and url != "nan":
            return f"- [{title}]({url}) — {company}{seg_tag} ({source}){ma_tag}"
        return f"- {title} — {company}{seg_tag} ({source}){ma_tag}"

    has_segment = "segment" in posts_df.columns

    # M&A items first
    if has_segment:
        ma_items = posts_df[posts_df["theme"] == "ma"]
        if not ma_items.empty:
            lines.append("**M&A signals:**")
            for _, row in ma_items.head(8).iterrows():
                lines.append(_format_entry(row))
            lines.append("")

    # Segment-grouped news
    if has_segment:
        for seg, label in [("fortune_100", "Fortune 100"), ("pe_mid", "PE Mid-Size"), ("startup", "Startups")]:
            sub = posts_df[(posts_df["segment"] == seg) & (posts_df["theme"] != "ma")]
            if not sub.empty:
                lines.append(f"**{label} news:**")
                for _, row in sub.head(15).iterrows():
                    lines.append(_format_entry(row))
                lines.append("")
    else:
        for _, row in posts_df.head(30).iterrows():
            lines.append(_format_entry(row))

    context = "\n".join(lines).strip() if lines else "(No news items this week.)"
    return context, source_counts


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

def _pretty_source_name(src: str) -> str:
    """Map feed hostnames to readable names for prompt hints."""
    s = src or ""
    if "retaildive" in s:
        return "Retail Dive"
    if "bloomberg" in s:
        return "Bloomberg"
    if "dowjones" in s or "mw_" in s:
        return "MarketWatch"
    if "reuters" in s:
        return "Reuters"
    if "google" in s:
        return "Google News"
    if "yahoo" in s:
        return "Yahoo Finance"
    if "supplychaindive" in s:
        return "Supply Chain Dive"
    if "manufacturingdive" in s:
        return "Manufacturing Dive"
    if "bankingdive" in s:
        return "Banking Dive"
    if "paymentsdive" in s:
        return "Payments Dive"
    return s.split(".")[0].replace("www", "").strip(".") or "news source"


def _synthesize_with_llm(
    client,
    industry_id: str,
    run_d: str,
    blueprint: str,
    news: str,
    k8_context: str = "",
    source_counts: dict | None = None,
) -> str | None:
    """Call OpenAI gpt-4o to generate the weekly narrative from blueprint + 8-K events + news."""
    try:
        k8_section = ""
        if k8_context:
            k8_section = "\n\n---\n\n**8-K material events (M&A, capex, leadership):**\n\n" + k8_context

        sources_hint = ""
        if source_counts and len(source_counts) >= 2:
            names = [_pretty_source_name(src) for src in source_counts]
            sources_hint = (
                "\n\nYou must cite at least one item from each of these sources in your summary: "
                + " and ".join(names[:4])
                + ". Do not default to a single outlet."
            )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Industry: {industry_id}. Week of: {run_d}.\n\n"
                        "---\n\n**Strategy blueprint (from latest 10-Qs, by segment):**\n\n"
                        + blueprint
                        + k8_section
                        + "\n\n---\n\n**News this week (by segment):**\n\n"
                        + news
                        + sources_hint
                        + "\n\n---\n\n"
                        + SYNTHESIS_PROMPT
                    ),
                }
            ],
            max_tokens=3000,
        )
        content = response.choices[0].message.content
        return (content or "").strip() or None
    except Exception as e:
        print(f"Warning: LLM synthesis failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_latest_sec(industry_id: str) -> pd.DataFrame | None:
    """Load latest sec_ai_disclosures CSV."""
    path = get_latest_sec_path(industry_id)
    if not path or not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_latest_posts(industry_id: str) -> pd.DataFrame | None:
    """Load latest company_posts CSV."""
    latest = list_latest_posts(industry_id, limit=1)
    if not latest:
        return None
    path = latest[0]["path"]
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_latest_8k(industry_id: str) -> pd.DataFrame | None:
    """Load latest sec_8k_signals CSV."""
    path = get_latest_8k_path(industry_id)
    if not path or not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fallback body
# ---------------------------------------------------------------------------

def _fallback_body(blueprint: str, news: str, k8_context: str = "") -> str:
    """When LLM is not available or fails: list raw inputs."""
    k8_section = (
        "\n\n---\n\n**8-K material events:**\n\n" + k8_context
        if k8_context else ""
    )
    return (
        "## What changed this week\n\n"
        "Synthesis is not available for this run. Run the weekly pipeline with LLM enabled to generate narrative sections.\n\n"
        "## Top company moves\n\n(Synthesis required.)\n\n"
        "## Themes & evidence\n\n(Synthesis required.)\n\n"
        "## Implications\n\n(Synthesis required.)\n\n"
        "## Forward watchlist\n\n(Synthesis required.)\n\n"
        "---\n\n"
        "**Strategy blueprint (10-Qs):**\n\n"
        + blueprint
        + k8_section
        + "\n\n---\n\n**News this week:**\n\n"
        + news
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_weekly_summary(
    industry_id: str,
    run_ts: str = None,
    sec_rows: list = None,
    posts_rows: list = None,
    k8_rows: list = None,
    openai_client=None,
) -> str:
    """
    Produce industry_weekly_summary_<date>.md.

    - 10-Qs are the point-in-time strategy blueprint (context by segment).
    - 8-K filings are material events (M&A, capex, leadership) — highest-signal input.
    - News/posts are what is happening this week.
    - LLM (gpt-4o) synthesizes all three into a segment-aware narrative.
    Returns path to written file.
    """
    run_d = run_date()

    sec_df = _load_latest_sec(industry_id) if sec_rows is None else pd.DataFrame(sec_rows)
    posts_df = _load_latest_posts(industry_id) if posts_rows is None else (
        posts_rows if isinstance(posts_rows, pd.DataFrame) else pd.DataFrame(posts_rows)
    )
    k8_df = _load_latest_8k(industry_id) if k8_rows is None else (
        k8_rows if isinstance(k8_rows, pd.DataFrame) else pd.DataFrame(k8_rows)
    )

    blueprint = _build_blueprint_context(sec_df) if sec_df is not None and len(sec_df) > 0 else "(No 10-Q blueprint loaded.)"
    k8_context = _build_8k_context(k8_df) if k8_df is not None and len(k8_df) > 0 else ""
    if posts_df is not None and len(posts_df) > 0:
        news, source_counts = _build_news_context(posts_df)
    else:
        news = "(No news items this week.)"
        source_counts = {}

    body: str
    if openai_client:
        synthesized = _synthesize_with_llm(
            openai_client, industry_id, run_d, blueprint, news,
            k8_context=k8_context, source_counts=source_counts,
        )
        if synthesized:
            body = synthesized
            # Strip "## Lede" heading if model still outputs it
            if body.strip().startswith("## Lede"):
                body = body.replace("## Lede", "", 1).strip().lstrip("\n")
        else:
            body = _fallback_body(blueprint, news, k8_context)
    else:
        body = _fallback_body(blueprint, news, k8_context)

    week_label = _format_week_display(run_d)
    header = (
        f"# Weekly Summary — {industry_id.capitalize()}\n\n"
        f"**Week of:** {week_label}\n\n"
        "Strategy context is drawn from the latest 10-Q filings; material events from 8-K filings; "
        "current developments from configured news sources. "
        "The narrative below is synthesised from all three.\n\n---\n\n"
    )
    content = header + body
    path = write_weekly_summary(content, industry_id, run_d)
    return path
