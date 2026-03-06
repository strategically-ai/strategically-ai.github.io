"""Load Anthropic API config from environment or local api_keys.py."""
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_anthropic_config() -> str:
    """Return Anthropic API key from env or api_keys.py."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    try:
        import sys
        root = str(_repo_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        import api_keys
        api_key = getattr(api_keys, "anthropic_api_key", "") or api_key
    except ImportError:
        pass
    return api_key


def get_anthropic_client():
    """Return an Anthropic client, or None if no API key."""
    api_key = get_anthropic_config()
    if not api_key:
        return None
    from anthropic import Anthropic
    return Anthropic(api_key=api_key)


# Model constants
HAIKU_MODEL = "claude-haiku-4-5-20251001"
