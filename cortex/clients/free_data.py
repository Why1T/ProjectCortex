"""Free historical OHLCV data provider (Yahoo Finance chart API).

Used when Alpaca credentials are unavailable so Cortex can still study,
backtest and advise. Also serves as a cross-check on Alpaca data.

Note: Yahoo has no native 4h interval, so 4h is derived by resampling the
1h bars. 15m/1h are intraday; 1d/1w/1mo are standard.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency fallback
    yf = None

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Yahoo interval per our friendly names, plus a range hint.
_INTERVAL_MAP = {
    "15m": {"interval": "15m", "range": "1mo"},
    "1h": {"interval": "60m", "range": "730d"},
    "4h": {"interval": "60m", "range": "730d"},  # resampled below
    "1d": {"interval": "1d", "range": "5y"},
    "1w": {"interval": "1wk", "range": "max"},
    "1mo": {"interval": "1mo", "range": "max"},
}


class FreeDataClient:
    def get_latest_quote(self, symbol: str) -> float | None:
        """Return a current-ish quote using yfinance when available."""
        if yf is not None:
            try:
                ticker = yf.Ticker(symbol)
                fast = getattr(ticker, "fast_info", None)
                if fast is not None:
                    price = getattr(fast, "last_price", None)
                    if price is not None:
                        return float(price)
                hist = ticker.history(period="5d", interval="1m", auto_adjust=True)
                if hist.empty:
                    return None
                last = hist["Close"].dropna()
                if last.empty:
                    return None
                return float(last.iloc[-1])
            except Exception as exc:
                log.debug("yfinance latest quote failed for %s: %s", symbol, exc)

        # Fallback to Yahoo chart API if yfinance isn’t available or fails.
        try:
            resp = requests.get(
                CHART_URL.format(symbol=symbol),
                params={"interval": "1m", "range": "5d"},
                headers=HEADERS,
                timeout=25,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            result = (data.get("chart") or {}).get("result") or []
            if not result:
                return None
            r = result[0]
            ts = r.get("timestamp") or []
            q = r.get("indicators", {}).get("quote", [{}])[0]
            closes = q.get("close") or []
            for idx in range(len(ts) - 1, -1, -1):
                price = closes[idx]
                if price is not None:
                    return float(price)
        except Exception as exc:
            log.debug("Yahoo quote fallback failed for %s: %s", symbol, exc)
        return None

    def get_bars(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        spec = _INTERVAL_MAP.get(interval)
        if spec is None:
            log.warning("Unsupported free interval %s", interval)
            return pd.DataFrame()
        df = self._fetch(symbol, spec["interval"], spec["range"])
        if df.empty:
            return df
        if interval == "4h":
            df = _resample(df, "4h")
        start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        return df

    def _fetch(self, symbol: str, interval: str, range_: str) -> pd.DataFrame:
        try:
            resp = requests.get(
                CHART_URL.format(symbol=symbol),
                params={"interval": interval, "range": range_},
                headers=HEADERS,
                timeout=25,
            )
            if resp.status_code != 200:
                log.debug("Yahoo HTTP %s for %s/%s", resp.status_code, symbol, interval)
                return pd.DataFrame()
            data = resp.json()
            result = (data.get("chart") or {}).get("result") or []
            if not result:
                return pd.DataFrame()
            r = result[0]
            ts = r.get("timestamp") or []
            q = r.get("indicators", {}).get("quote", [{}])[0]
            rows = []
            for i, t in enumerate(ts):
                o = (q.get("open") or [None])[i]
                h = (q.get("high") or [None])[i]
                low = (q.get("low") or [None])[i]
                c = (q.get("close") or [None])[i]
                v = (q.get("volume") or [None])[i]
                if c is None:
                    continue
                rows.append(
                    {
                        "timestamp": pd.Timestamp(t, unit="s", tz="UTC"),
                        "open": float(o or c),
                        "high": float(h or c),
                        "low": float(low or c),
                        "close": float(c),
                        "volume": float(v or 0),
                    }
                )
            df = pd.DataFrame(rows)
            if df.empty:
                return df
            return df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
        except Exception as exc:
            log.debug("Yahoo fetch failed %s/%s: %s", symbol, interval, exc)
            return pd.DataFrame()


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample intraday bars to a higher timeframe (e.g. 1h -> 4h)."""
    agg = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna(subset=["close"])
    return agg[agg["close"] > 0]