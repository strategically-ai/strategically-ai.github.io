"""Load API keys from api_keys.py (if present) or environment."""

import os


def get_openai_config() -> tuple[str, str]:
    """
    Return (api_key, base_url) for OpenAI client.
    Precedence: api_keys.py (openai_api_key, openai_base_url) then env OPENAI_API_KEY, OPENAI_BASE_URL.
    """
    api_key = ""
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    try:
        import api_keys
        api_key = getattr(api_keys, "openai_api_key", "") or api_key
        base_url = getattr(api_keys, "openai_base_url", base_url) or base_url
    except ImportError:
        pass
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    return api_key, base_url
