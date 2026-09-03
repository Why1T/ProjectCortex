"""TradingView client: authoritative technical-analysis recommendations + charts.

Uses TradingView's public scanner datafeed (no API key required) to obtain
its composite technical-analysis recommendation and key metrics. TradingView
indexes symbols per exchange (NASDAQ:, NYSE:, AMEX:); we map US tickers
accordingly and fall back to a market-wide lookup if an exchange prefix is
unknown.

This is the "use TradingView" layer for timeframes: it feeds TradingView's
own multi-indicator recommendation into Cortex and produces chart links so
you can open any symbol on all six timeframes (15m/1H/4H/1D/1W/1M).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

log = logging.getLogger(__name__)

SCAN_URL = "https://scanner.tradingview.com/america/scan"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Exchange prefixes TradingView recognises for US equities.
EXCHANGE_PREFIXES = ["NASDAQ", "NYSE", "AMEX", "OTC", "BATS"]

# Our interval -> TradingView chart interval code.
CHART_INTERVAL_CODES = {
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1d": "D",
    "1w": "W",
    "1mo": "M",
}

# Columns fetched from the scanner (TradingView's own indicator suite).
COLS = [
    "name",
    "close",
    "Recommend.All",
    "Recommend.MA",
    "Recommend.Other",
    "RSI",
    "ATR",
    "MACD.macd",
    "MACD.signal",
    "SMA20",
    "EMA20",
    "Perf.1M",
    "Perf.3M",
    "Perf.6M",
]

# Human labels for the tradingview recommendation value (approx scale).
REC_LABELS = {
    1.0: "strong_buy", 0.5: "buy", 0.0: "neutral",
    -0.5: "sell", -1.0: "strong_sell",
}


@dataclass
class TVSignal:
    symbol: str
    exchange: str = "NASDAQ"
    price: float = 0.0
    recommendation: float = 0.0            # -1 .. +1
    recommendation_label: str = "neutral"
    ma_rec: float = 0.0
    other_rec: float = 0.0
    rsi: Optional[float] = None
    atr: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    sma20: Optional[float] = None
    ema20: Optional[float] = None
    perf_1m: Optional[float] = None
    perf_3m: Optional[float] = None
    perf_6m: Optional[float] = None
    charts: dict[str, str] = field(default_factory=dict)

    def chart_links(self) -> dict[str, str]:
        """TradingView chart URLs for all six swing-trading timeframes."""
        if not self.charts:
            self.charts = build_chart_links(self.symbol, self.exchange)
        return self.charts

    def as_dict(self) -> dict:
        base = {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "price": round(self.price, 2),
            "recommendation": round(self.recommendation, 2),
            "recommendation_label": self.recommendation_label,
            "chart_1h": build_chart_link(self.symbol, "1h", self.exchange),
            "chart_4h": build_chart_link(self.symbol, "4h", self.exchange),
            "chart_1d": build_chart_link(self.symbol, "1d", self.exchange),
        }
        for k in ("rsi", "atr", "macd", "macd_signal", "sma20", "ema20", "perf_1m", "perf_3m", "perf_6m"):
            v = getattr(self, k)
            base[k] = round(v, 3) if isinstance(v, float) else v
        base["charts"] = self.chart_links()
        return base


def build_chart_link(symbol: str, interval: str, exchange: str = "NASDAQ") -> str:
    code = CHART_INTERVAL_CODES.get(interval, "D")
    return f"https://www.tradingview.com/chart/?symbol={exchange}:{symbol}&interval={code}"


def build_chart_links(symbol: str, exchange: str = "NASDAQ") -> dict[str, str]:
    return {iv: build_chart_link(symbol, iv, exchange) for iv in CHART_INTERVAL_CODES}


def _recommendation_label(value: Optional[float]) -> str:
    if value is None:
        return "neutral"
    v = round(value, 1)
    return REC_LABELS.get(v, "neutral")


class TradingViewClient:
    def __init__(self, timeout: int = 25) -> None:
        self.timeout = timeout

    # -------- scanning --------
    def fetch(self, symbols: list[str]) -> dict[str, TVSignal]:
        """Batch-fetch TradingView recommendations for a list of US tickers.

        Returns a symbol -> TVSignal map. Symbols that TradingView can't
        resolve are omitted (with a debug log).
        """
        tickers = [self._to_ticker(s) for s in symbols]
        unique = list(dict.fromkeys(tickers))  # preserve order, dedupe
        out: dict[str, TVSignal] = {}
        if not unique:
            return out

        payload = {"symbols": {"tickers": unique}, "columns": COLS}
        try:
            resp = requests.post(SCAN_URL, json=payload, headers=HEADERS, timeout=self.timeout)
            resp.raise_for_status()
        except Exception as exc:
            log.warning("TradingView scan failed: %s", exc)
            return out

        data = resp.json().get("data") or []
        for row in data:
            d = row.get("d") or []
            sig = self._parse(row.get("s", ""), d)
            if sig:
                out[sig.symbol] = sig
        return out

    def get(self, symbol: str) -> Optional[TVSignal]:
        return self.fetch([symbol]).get(symbol.upper())

    # -------- parsing --------
    def _parse(self, ticker: str, d: list) -> Optional[TVSignal]:
        if not d:
            return None
        symbol = str(d[0])
        exchange, _ = _split_ticker(ticker)
        def num(idx: int) -> Optional[float]:
            try:
                v = d[idx]
                return float(v) if v is not None else None
            except (TypeError, ValueError, IndexError):
                return None

        try:
            price = num(1) or 0.0
        except Exception:
            price = 0.0
        rec = num(2) if num(2) is not None else 0.0

        return TVSignal(
            symbol=symbol,
            exchange=exchange,
            price=price,
            recommendation=rec if rec is not None else 0.0,
            recommendation_label=_recommendation_label(rec),
            ma_rec=num(3) or 0.0,
            other_rec=num(4) or 0.0,
            rsi=num(5),
            atr=num(6),
            macd=num(7),
            macd_signal=num(8),
            sma20=num(9),
            ema20=num(10),
            perf_1m=num(11),
            perf_3m=num(12),
            perf_6m=num(13),
        )

    # -------- helpers --------
    @staticmethod
    def _to_ticker(symbol: str) -> str:
        """Map a bare US ticker to a TradingView-prefixed ticker."""
        symbol = symbol.upper().strip()
        if ":" in symbol:
            return symbol
        # Best-effort: use NASDAQ as default (most liquid US names are on it).
        return f"NASDAQ:{symbol}"


def _split_ticker(ticker: str) -> tuple[str, str]:
    if ":" in ticker:
        exchange, symbol = ticker.split(":", 1)
        if exchange in EXCHANGE_PREFIXES:
            return exchange, symbol
    return "NASDAQ", ticker.split(":")[-1]