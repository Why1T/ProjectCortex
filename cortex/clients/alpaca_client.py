"""Alpaca paper-trading client: historical data, account, and positions.

Read-only by design. Cortex uses this account as ground truth for capital,
open positions and historical PnL so it can learn from your wins and losses.
This client NEVER submits orders.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

# Alpaca REST map between our friendly interval names and API endpoints.
BAR_SPECS: dict[str, dict[str, Any]] = {
    "15m": {"path": "15Min", "multiplier": 1, "timeframe": "15Min"},
    "1h": {"path": "1Hour", "multiplier": 1, "timeframe": "1Hour"},
    "4h": {"path": "4Hour", "multiplier": 1, "timeframe": "4Hour"},
    "1d": {"path": "1D", "multiplier": 1, "timeframe": "1Day"},
    "1w": {"path": "1W", "multiplier": 1, "timeframe": "1Week"},
    "1mo": {"path": "1M", "multiplier": 1, "timeframe": "1Month"},
}
SUPPORTED_INTERVALS = list(BAR_SPECS.keys())


class AlpacaClient:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.rest_url = PAPER_URL if paper else "https://api.alpaca.markets"
        self._h = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        self._cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    def _get(self, base: str, path: str, params: dict | None = None) -> Any:
        url = f"{base}{path}"
        resp = requests.get(url, headers=self._h, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            time.sleep(1)
            return self._get(base, path, params)
        log.warning("Alpaca HTTP %s for %s: %s", resp.status_code, path, resp.text[:200])
        return None

    # ---------- account / positions ----------
    def get_account(self) -> dict:
        return self._get(self.rest_url, "/v2/account") or {}

    def get_positions(self) -> list[dict]:
        return self._get(self.rest_url, "/v2/positions") or []

    def get_open_orders(self) -> list[dict]:
        return self._get(self.rest_url, "/v2/orders") or []

    def get_fill_activities(self, days: int = 30) -> list[dict]:
        """Recent fill activities (completed buy/sell transactions)."""
        params = {"activity_types": "FILL", "days": days}
        return self._get(self.rest_url, "/v2/account/activities", params) or []

    # ---------- market data ----------
    def get_bars(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str,
        limit: int = 5000,
        adjustment: str = "all",
    ) -> pd.DataFrame:
        """Return OHLCV bars as a DataFrame indexed by 'timestamp'."""
        if interval not in BAR_SPECS:
            raise ValueError(f"Unsupported interval {interval}. Use {SUPPORTED_INTERVALS}")
        cache_key = (symbol, interval, f"{start}_{end}_{limit}")
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        timeframe = BAR_SPECS[interval]["timeframe"]
        payload = self._get(
            DATA_URL,
            f"/v2/stocks/{symbol}/bars",
            {
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "limit": limit,
                "adjustment": adjustment,
                "feed": "iex",
            },
        )
        df = self._bars_to_df(payload)
        self._cache[cache_key] = df
        return df.copy()

    def get_symbols(self) -> list[str]:
        """Tradeable {asset asset as a list of symbols (active, global/US)."""
        data = self._get(DATA_URL, "/v2/stocks", {"status": "active", "limit": 1000})
        if not data:
            return []
        return [s["symbol"] for s in data.get("stocks", [])]

    @staticmethod
    def _bars_to_df(payload: dict | None) -> pd.DataFrame:
        rows = []
        for bar in (payload or {}).get("bars", []) or []:
            rows.append(
                {
                    "timestamp": pd.Timestamp(bar["t"]),
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar["v"]),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
        return df