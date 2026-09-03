"""Market scanner: screens a universe of liquid names for swing candidates.

Cortex scans each symbol across all six timeframes, runs the Swing4H
strategy, and collects signals that clear the score threshold. It then
sorts by score and (optionally) cross-references holdings so it knows
what you are already in.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from config.settings import settings
from cortex.data.market_data import MarketData
from cortex.strategies.swing_4h import Swing4HStrategy, TradeSignal

log = logging.getLogger(__name__)


class Scanner:
    def __init__(
        self,
        data: MarketData,
        strategy: Swing4HStrategy,
        symbols: list[str] | None = None,
        workers: int = 6,
        tv_rec_lookup: "Callable[[str], float | None]" | None = None,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.symbols = symbols or []
        self.workers = workers
        self.tv_rec_lookup = tv_rec_lookup

    def _scan_one(self, symbol: str) -> tuple[str, TradeSignal]:
        try:
            intervals = self.data.cached(symbol)
            tv_rec = self.tv_rec_lookup(symbol) if self.tv_rec_lookup else None
            sig = self.strategy.analyze(symbol, intervals, tv_rec=tv_rec)
            return symbol, sig
        except Exception as exc:
            log.debug("scan %s failed: %s", symbol, exc)
            return symbol, TradeSignal(symbol, "no-trade", 0.0, None, None, 0.0, ["scan error"])

    def scan(self) -> list[TradeSignal]:
        """Return signals sorted by score, filtered to actionable ones."""
        signals: list[TradeSignal] = []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = [ex.submit(self._scan_one, s) for s in self.symbols]
            for fut in as_completed(futures):
                _, sig = fut.result()
                if sig.direction == "long" and sig.score >= self.strategy.params.min_score:
                    signals.append(sig)
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals


def default_universe() -> list[str]:
    """A broad liquid US universe; swap in Alpaca's symbol list when available."""
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "AMD",
        "NFLX", "JPM", "V", "MA", "UNH", "XOM", "JNJ", "LLY", "PG", "HD", "COST",
        "CRM", "ADBE", "ORCL", "KO", "PEP", "WMT", "BAC", "DIS", "MCD", "ABBV",
        "CSCO", "ACN", "TMO", "INTC", "QCOM", "CAT", "IBM", "GE", "T", "PFE",
        "MRK", "CVX", "UPS", "MMM", "SBUX", "BA", "GM", "F", "NKE", "VZ",
        "MU", "PLTR", "SOFI", "COIN", "SHOP", "PANW", "SNOW", "DDOG", "CRWD",
        "MSTR", "RIVN", "LCID", "SQ", "PYPL", "UBER", "ABNB", "NET", "HOOD", "MDB",
    ]