"""Load OpenAI config from repo root api_keys.py or environment."""
import os
from pathlib import Path

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def get_openai_config() -> tuple[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    try:
        import sys
        root = str(_repo_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        import api_keys
        api_key = getattr(api_keys, "openai_api_key", "") or api_key
        base_url = getattr(api_keys, "openai_base_url", base_url) or base_url
    except ImportError:
        pass
    return api_key, base_url
