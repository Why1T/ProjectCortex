"""Market data facade: pulls OHLCV across all six timeframe intervals.

Sources (in priority order):
  1. Alpaca IEX (free, The Investor Exchange) when credentials exist.
  2. Stooq free data otherwise.

All six intervals for our swing-trading stack:
   15m, 1h, 4h, 1d, 1w, 1mo

Returns normalized pandas DataFrames indexed by 'timestamp' with columns
open/high/low/close/volume.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from config.settings import settings
from cortex.clients.alpaca_client import AlpacaClient, SUPPORTED_INTERVALS
from cortex.clients.free_data import FreeDataClient

INTERVALS: list[str] = ["15m", "1h", "4h", "1d", "1w", "1mo"]

log = logging.getLogger(__name__)

_LOOKBACK = {
    "15m": timedelta(days=90),
    "1h": timedelta(days=180),
    "4h": timedelta(days=365),
    "1d": timedelta(days=400),
    "1w": timedelta(days=1095),
    "1mo": timedelta(days=3650),
}


class MarketData:
    def __init__(self) -> None:
        self.primary: AlpacaClient | None = None
        self.free = FreeDataClient()
        self.source = "none"
        if settings.alpaca_enabled:
            self.primary = AlpacaClient(
                settings.alpaca_api_key,
                settings.alpaca_api_secret,
                paper=settings.alpaca_paper,
            )
            self.source = "alpaca"
        else:
            self.source = "stooq"

    def get_bars(
        self,
        symbol: str,
        interval: str,
        lookback: timedelta | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}. Use {SUPPORTED_INTERVALS}")
        lookback = lookback or _LOOKBACK.get(interval, _LOOKBACK["1d"])
        end = end or datetime.utcnow()
        start = end - lookback
        start_s, end_s = start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Prefer Alpaca, fall back to free source on any failure.
        if self.primary is not None:
            try:
                df = self.primary.get_bars(symbol, interval, start_s, end_s)
                if not df.empty:
                    return df
            except Exception as exc:
                log.debug("Alpaca data failed for %s@%s: %s", symbol, interval, exc)

        df = self.free.get_bars(symbol, interval, start_s, end_s)
        return df

    def get_latest_price(self, symbol: str) -> float | None:
        """Get the latest current-ish price from live or free sources."""
        if self.primary is not None:
            try:
                bars = self.primary.get_bars(
                    symbol,
                    "15m",
                    start=(datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=200,
                )
                if not bars.empty:
                    return float(bars["close"].iloc[-1])
            except Exception as exc:
                log.debug("Alpaca live quote failed for %s: %s", symbol, exc)
        return self.free.get_latest_quote(symbol)

    def all_intervals(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Fetch all six intervals for a symbol, clipping to the same window."""
        end = datetime.utcnow()
        return {
            iv: self.get_bars(symbol, iv, end=end)
            for iv in INTERVALS
        }

    def cached(self, symbol: str) -> dict[str, pd.DataFrame]:
        return self.all_intervals(symbol)