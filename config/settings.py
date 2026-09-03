"""Cortex configuration loading from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root is two levels up from this file.
ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
CACHE_DIR = ROOT_DIR / "data_cache"
STORE_DIR = ROOT_DIR / "store"

for _d in (REPORTS_DIR, CACHE_DIR, STORE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Load .env from the project root. Does nothing if the file is absent,
# so Cortex can run in "degraded" mode without credentials.
load_dotenv(ROOT_DIR / ".env")


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


class Settings:
    def __init__(self) -> None:
        # --- Alpaca ---
        self.alpaca_api_key = _get("ALPACA_API_KEY")
        self.alpaca_api_secret = _get("ALPACA_API_SECRET")
        self.alpaca_paper = _get("ALPACA_PAPER", "true").lower() == "true"

        # --- LLM ---
        self.llm_api_key = _get("CORTEX_LLM_API_KEY")
        self.llm_base_url = _get("CORTEX_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.llm_model = _get("CORTEX_LLM_MODEL", "gpt-4o-mini")
        self.llm_temperature = float(_get("CORTEX_LLM_TEMPERATURE", "0.3"))

        # --- Risk ---
        self.risk_per_trade = float(_get("CORTEX_RISK_PER_TRADE", "0.01"))
        self.max_position_pct = float(_get("CORTEX_MAX_POSITION_PCT", "0.20"))

        # --- News ---
        self.news_max_items = int(_get("CORTEX_NEWS_MAX_ITEMS", "25"))

        # --- TradingView ---
        # Optional toggle; on by default. Public scanner needs no key.
        self.tv_enabled = _get("CORTEX_TV_ENABLED", "true").lower() == "true"
        self.tv_timeout = int(_get("CORTEX_TV_TIMEOUT", "25"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def alpaca_enabled(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_api_secret)


settings = Settings()
