"""Backtest engine: replays historical bars through the Swing4H strategy.

Indicators are computed ONCE per timeframe (precomputed frames) and the
signal at each 4h bar is derived by looking at the indicator values only
up to that bar. This keeps the backtest tractable over 216 optimizer combos.

Cortex NEVER places real trades — this is research-only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from cortex.indicators.engine import compute_all
from cortex.strategies.swing_4h import (
    StrategyParams,
    score_entry,
    higher_tf_score,
    _ema_last,
    SCORE_1H_CONFIRM,
    SCORE_15M_CONFIRM,
)
from cortex.backtest.metrics import compute_metrics, TradeStats

log = logging.getLogger(__name__)


@dataclass
class SimTrade:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    bars_held: int = 0

    @property
    def r_multiple(self) -> float:
        if self.exit_price is None or self.direction == "no-trade":
            return 0.0
        risk = abs(self.entry_price - self.stop_loss)
        if risk == 0:
            return 0.0
        if self.direction == "long":
            return (self.exit_price - self.entry_price) / risk
        return (self.entry_price - self.exit_price) / risk


class Backtester:
    def __init__(
        self,
        params: StrategyParams | None = None,
        max_bars_in_trade: int = 100,
        risk_per_trade: float = 0.01,
    ) -> None:
        self.params = params or StrategyParams()
        self.max_bars = max_bars_in_trade
        self.risk = risk_per_trade

    def run(self, symbol: str, intervals: dict[str, pd.DataFrame]) -> tuple[TradeStats, list[SimTrade]]:
        sec_4h = intervals.get("4h")
        if sec_4h is None or len(sec_4h) < 70:
            return TradeStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None), []

        # Precompute enriched frames once per timeframe.
        pre: dict[str, pd.DataFrame] = {}
        for iv, df in intervals.items():
            if df is None or df.empty:
                continue
            if iv == "4h":
                pre["4h"] = compute_all(df.copy())
            else:
                pre[iv] = df.copy()

        trades: list[SimTrade] = []
        in_trade: SimTrade | None = None
        t4 = pre["4h"]  # enriched 4h with indicator columns

        # Pre-index everything by timestamp for fast lookup.
        idx = t4.index
        n = len(t4)

        for i in range(70, n):
            cur_time = idx[i]
            cur = t4.iloc[i]
            hi, lo, close, t = float(cur["high"]), float(cur["low"]), float(cur["close"]), float(cur["open"])

            # -------- manage open trade --------
            if in_trade is not None:
                if in_trade.direction == "long":
                    exit_price = None
                    if lo <= in_trade.stop_loss:
                        exit_price, reason = in_trade.stop_loss, "stop"
                    elif hi >= in_trade.take_profit:
                        exit_price, reason = in_trade.take_profit, "tp"
                    else:
                        in_trade.bars_held += 1
                        if in_trade.bars_held >= self.max_bars:
                            exit_price, reason = close, "time_exit"
                    if exit_price is not None:
                        in_trade.exit_price, in_trade.exit_reason, in_trade.exit_time = exit_price, reason, cur_time
                        trades.append(in_trade)
                        in_trade = None
                continue

            # -------- scan for entry (4h is decision bar) --------
            sig = self._signal_at(t4, pre, cur_time)
            if sig is not None:
                in_trade = SimTrade(
                    symbol=symbol,
                    direction=sig["direction"],
                    entry_price=sig["entry"],
                    stop_loss=sig["stop"],
                    take_profit=sig["tp"],
                    entry_time=cur_time,
                )

        if in_trade is not None and in_trade.exit_price is None:
            in_trade.exit_price = float(t4.iloc[-1]["close"])
            in_trade.exit_time = idx[-1]
            in_trade.exit_reason = "end_of_data"
            trades.append(in_trade)

        outcomes = [(t, t.r_multiple) for t in trades]
        stats = compute_metrics(outcomes)
        return stats, trades

    def _signal_at(self, t4: pd.DataFrame, pre: dict, cur_time: pd.Timestamp) -> dict | None:
        """Derive a long signal at cur_time from precomputed indicator frames.

        Uses the SAME score_entry() as the live advisor so backtests reflect
        what you would actually be advised. Only the 4h bar itself is used
        for the entry price (backtest uses completed-bar close).
        """
        p = self.params
        last = t4.loc[cur_time]
        price = float(last["close"])
        atr14 = float(last["atr_14"])

        # Higher-TF trend context (1w/1d/4h) from bars up to cur_time only.
        hi_tf_score = higher_tf_score(
            intervals=None, precomputed=pre, end=cur_time
        )

        # Lower-TF confirmation.
        lower_confirm = 0.0
        for tf, w in (("1h", SCORE_1H_CONFIRM), ("15m", SCORE_15M_CONFIRM)):
            frame = pre.get(tf)
            if frame is None or frame.empty:
                continue
            sub = frame.loc[:cur_time]
            if len(sub) < 20:
                continue
            # Sub-frames for 1h/15m are raw (not enriched) -> compute EMA on the fly.
            if float(sub["close"].iloc[-1]) > _ema_last(sub["close"], 21):
                lower_confirm += w

        dec = score_entry(
            price=price,
            atr_value=atr14,
            rsi=float(last["rsi_14"]),
            macd_hist=float(last["macd_hist"]),
            ema9=float(last["ema_9"]),
            ema21=float(last["ema_21"]),
            ema50=float(last["ema_50"]),
            hi_tf_score=hi_tf_score,
            lower_confirm=lower_confirm,
            params=p,
        )
        if dec.direction == "long" and dec.stop and dec.target:
            return {"direction": "long", "entry": price, "stop": dec.stop, "tp": dec.target}
        return None