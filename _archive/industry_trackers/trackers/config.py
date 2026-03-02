"""Constants and configuration for industry trackers."""

import os

# SEC EDGAR
SEC_HEADERS = {
    "User-Agent": "Mfundo Radebe mtr2149@columbia.edu",
}

# Filing type for the tracker (10-Q for quarterly; 10-K also supported)
DEFAULT_FILING_TYPE = "10-Q"
SEC_RATE_LIMIT_SLEEP = 0.15

# AI query terms for extraction (regex patterns)
AI_KEYWORDS = [
    r"\bartificial intelligence\b",
    r"\bAI\b",
    r"\bmachine learning\b",
    r"\bML\b",
    r"\bautomation\b",
    r"\bgenerative\b",
    r"\bLLM\b",
    r"\bcopilot\b",
    r"\bautonomous\b",
    r"\bpredictive\b",
    r"\bdeep learning\b",
    r"\bgen ai\b",
    r"\blarge language model\b",
    r"\bneural network\b",
    r"\bnatural language processing\b",
    r"\bai[ -]powered\b",
    r"\bai[ -]driven\b",
    r"\bai[ -]enabled\b",
    r"\bautomated decision\b",
    r"\bpredictive model\b",
    r"\bpredictive analytics\b",
]

# Output base (relative to repo root)
BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "industry_trackers",
)

# Strategy theme taxonomy (for LLM summarisation of company strategy)
SEC_THEMES = [
    "product_features",
    "internal_efficiency",
    "manufacturing_automation",
    "supply_chain_optimization",
    "pricing_forecasting",
    "customer_service",
    "cybersecurity",
    "talent_hiring",
    "partnerships_platforms",
    "regulatory_legal_ip",
    "capex_infrastructure",
    "data_strategy",
]
SIGNAL_STRENGTHS = ["weak", "moderate", "strong"]

# Revenue segment taxonomy
SEGMENTS = ["fortune_100", "pe_mid", "startup"]

# Delay for per-company RSS fetches (more conservative than industry-wide feeds)
NEWS_COMPANY_FETCH_DELAY = 2.0

# Number of recent 8-K filings to fetch per company
DEFAULT_8K_MAX_FILINGS = 5

# Themes that indicate capex / infrastructure investment
CAPEX_THEMES = ["capex_infrastructure", "supply_chain_optimization", "manufacturing_automation"]
