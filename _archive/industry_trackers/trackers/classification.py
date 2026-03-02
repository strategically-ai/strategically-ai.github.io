"""LLM-based company strategy summarisation from SEC filings and company posts."""

from trackers.config import SEC_THEMES, SIGNAL_STRENGTHS

SEC_STRATEGY_PROMPT = """You are a senior equity analyst. Your task is to summarise company strategy from SEC filing excerpts (10-Q or 10-K).

The following text is extracted from a company's regulatory filing. Summarise the strategy-relevant content (business model, priorities, risks, investments, competitive position).

Tasks:
1. Assign the PRIMARY THEME from this list (exactly one): product_features, internal_efficiency, manufacturing_automation, supply_chain_optimization, pricing_forecasting, customer_service, cybersecurity, talent_hiring, partnerships_platforms, regulatory_legal_ip, capex_infrastructure, data_strategy.
2. Assign SIGNAL STRENGTH: weak, moderate, or strong (how material this is for understanding the company's strategy).
3. In one sentence, explain why it matters (why_it_matters).
4. Give confidence 0.0-1.0 (how confident you are in the theme and strength).

Format your response EXACTLY as:
Theme: <one from list>
Signal strength: <weak|moderate|strong>
Why it matters: <one sentence>
Confidence: <number between 0 and 1>
"""


def classify_sec_disclosure(client, text_slice: str, max_chars: int = 5000) -> dict:
    """
    Use LLM to summarise company strategy from SEC filing snippet. Returns dict with theme, signal_strength, why_it_matters, confidence.
    """
    text = (text_slice or "")[:max_chars]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": SEC_STRATEGY_PROMPT + "\n\n---\n\n" + text}
            ],
            max_tokens=200,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {
            "theme": "data_strategy",
            "signal_strength": "moderate",
            "why_it_matters": str(e),
            "confidence": 0.0,
        }

    theme = "data_strategy"
    signal_strength = "moderate"
    why_it_matters = ""
    confidence = 0.5

    for line in content.split("\n"):
        line = line.strip()
        if line.lower().startswith("theme:"):
            val = line.split(":", 1)[-1].strip().lower()
            if val in SEC_THEMES:
                theme = val
        elif "signal strength" in line.lower() or "signal_strength" in line.lower():
            val = line.split(":", 1)[-1].strip().lower()
            if val in SIGNAL_STRENGTHS:
                signal_strength = val
        elif line.lower().startswith("why it matters:"):
            why_it_matters = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("confidence:"):
            try:
                confidence = float(line.split(":", 1)[-1].strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

    return {
        "theme": theme,
        "signal_strength": signal_strength,
        "why_it_matters": why_it_matters,
        "confidence": confidence,
    }


_8K_SIGNAL_TYPES = ["ma", "capex", "leadership", "earnings", "other"]

_8K_CLASSIFICATION_PROMPT = """You are a senior M&A analyst reviewing an SEC 8-K filing excerpt.

Classify the primary signal type of this 8-K filing. Choose EXACTLY ONE from:
- ma: mergers, acquisitions, divestitures, definitive agreements, tender offers
- capex: major capital expenditure, facility investment, new plant/warehouse/data center
- leadership: senior leadership hire (C-suite, SVP) — especially AI/data/digital roles
- earnings: results of operations, financial performance announcement
- other: none of the above

Format your response EXACTLY as:
Signal type: <one from list>
Rationale: <one sentence>
"""


def classify_8k_signal(client, text: str, max_chars: int = 4000) -> dict:
    """
    Use LLM to classify an 8-K filing excerpt into a signal type.
    Returns dict with signal_type and rationale.
    Falls back to 'other' on error.
    """
    text = (text or "")[:max_chars]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": _8K_CLASSIFICATION_PROMPT + "\n\n---\n\n" + text}
            ],
            max_tokens=100,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"signal_type": "other", "rationale": str(e)}

    signal_type = "other"
    rationale = ""
    for line in content.split("\n"):
        line = line.strip()
        if line.lower().startswith("signal type:"):
            val = line.split(":", 1)[-1].strip().lower()
            if val in _8K_SIGNAL_TYPES:
                signal_type = val
        elif line.lower().startswith("rationale:"):
            rationale = line.split(":", 1)[-1].strip()

    return {"signal_type": signal_type, "rationale": rationale}
