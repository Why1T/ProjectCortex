"""Swing trading strategy engine.

Cortex's core strategy encodes classic swing-trading methodology:

  * Direction comes from the higher timeframes (weekly -> daily -> 4h).
  * The 4-hour chart is the PRIMARY decision/entry timeframe.
  * Lower timeframes (1h, 15m) fine-tune entry timing.
  * Entries are pullbacks-in-trend (preferred) or confirmed breakouts.
  * Risk is defined by ATR; stops and targets are structural.

Both the live analyzer and the backtester call `score_entry()` so that
what you see in an advisory is exactly what gets validated historically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from cortex.indicators.engine import atr, ema, compute_all


@dataclass
class StrategyParams:
    pullback_rsi_max: float = 55.0
    overbought_rsi_min: float = 72.0
    atr_stop_mult: float = 2.0
    atr_tp_mult: float = 4.0
    min_avg_dollar_vol: float = 5_000_000
    min_score: float = 60.0
    require_daily_trend: bool = True


@dataclass
class EntryDecision:
    direction: str  # 'long' | 'no-trade'
    score: float
    stop: Optional[float]
    target: Optional[float]
    matched_style: str = ""  # 'pullback' | 'breakout' | ''


# Higher-timeframe alignment weights (max achievable = 20 + 15 + 12 = 47).
TF_WEIGHTS = {"1w": 20.0, "1d": 15.0, "4h": 12.0}

# Pullback / momentum contribution when matched.
SCORE_PULLBACK = 30.0
SCORE_MILD_PULLBACK = 15.0
SCORE_BREAKOUT = 20.0
SCORE_MACD = 10.0
SCORE_1H_CONFIRM = 12.0
SCORE_15M_CONFIRM = 6.0
SCORE_TV_CONFLUENCE = 14.0  # TradingView recommendation agrees with an uptrend.


def higher_tf_score(
    intervals: dict[str, pd.DataFrame],
    precomputed: dict[str, pd.DataFrame] | None = None,
    end: pd.Timestamp | None = None,
) -> float:
    """Weighted multi-timeframe trend score (positive=uptrend aligned).

    Intervals may be raw OHLCV frames (indicators computed here) or already
    enriched via `precomputed`. `end` limits each frame to bars up to that
    timestamp (used by the backtester to avoid lookahead).
    """
    total = 0.0
    for tf, w in TF_WEIGHTS.items():
        frame = (precomputed or {}).get(tf) if precomputed else intervals.get(tf)
        if frame is None:
            frame = intervals.get(tf)
        if frame is None or frame.empty:
            continue
        if end is not None:
            sub = frame.loc[:end]
        else:
            sub = frame
        if len(sub) < 30:
            continue
        sign = _trend_sign(sub)
        total += w * sign
    return total


def _trend_sign(frame: pd.DataFrame) -> float:
    """+1 uptrend, -1 downtrend, 0 range, based on last closed bar EMA stack."""
    if len(frame) < 30:
        return 0.0
    last = frame.iloc[-1]
    c = float(last["close"])
    e21 = float(last["ema_21"]) if "ema_21" in frame else _ema_last(frame["close"], 21)
    e50 = float(last["ema_50"]) if "ema_50" in frame else _ema_last(frame["close"], 50)
    if c > e21 > e50:
        return 1.0
    if c < e21 < e50:
        return -1.0
    return 0.0


def _ema_last(series: pd.Series, period: int) -> float:
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def score_entry(
    *,
    price: float,
    atr_value: float,
    rsi: float,
    macd_hist: float,
    ema9: float,
    ema21: float,
    ema50: float,
    hi_tf_score: float,
    lower_confirm: float,
    params: StrategyParams,
    tv_rec: float | None = None,
) -> EntryDecision:
    """Shared entry decision used by live analyzer AND backtester.

    `tv_rec` is TradingView's composite recommendation (-1 .. +1). If it
    agrees with a long (>= Buy, i.e. > 0.25) we add confluence points;
    if it disagrees (<= Sell) the idea is penalised. This lets TradingView's
    own multi-timeframe analysis corroborate (or veto) the 4h signal without
    being the primary driver.
    """
    if atr_value <= 0 or price <= 0:
        return EntryDecision("no-trade", 0.0, None, None)

    score = hi_tf_score

    # 4h trend anchor: price must stay above the EMA50 (uptrend intact).
    if price <= ema50:
        return EntryDecision("no-trade", max(0.0, hi_tf_score - 20), None, None)

    long = False
    style = ""

    # Pullback into trend (preferred swing entry): RSI cooled, price pulled
    # back toward/below EMA21 but the EMA50 trend is intact.
    if rsi <= params.pullback_rsi_max and price <= ema9:
        score += SCORE_PULLBACK
        long = True
        style = "pullback"
    # Early/mild pullback: price drifting toward EMA21 from above.
    elif price < ema21 and rsi <= params.pullback_rsi_max + 10:
        score += SCORE_MILD_PULLBACK
        long = True
        style = "pullback"
    # Confirmed breakout with momentum.
    elif price > ema21 and rsi < params.overbought_rsi_min and macd_hist > 0:
        score += SCORE_BREAKOUT
        long = True
        style = "breakout"

    if macd_hist > 0:
        score += SCORE_MACD

    # TradingView confluence: corroborate (or veto) the long thesis.
    if tv_rec is not None:
        if tv_rec >= 0.25:          # TradingView says Buy / Strong Buy
            score += SCORE_TV_CONFLUENCE
        elif tv_rec <= -0.25:       # TradingView says Sell / Strong Sell
            score -= SCORE_TV_CONFLUENCE

    # Lower-TF confirmation adds precision points.
    score += lower_confirm

    if not long or score < params.min_score:
        return EntryDecision("no-trade", max(0.0, score), None, None)

    return EntryDecision(
        direction="long",
        score=round(score, 1),
        stop=round(price - params.atr_stop_mult * atr_value, 2),
        target=round(price + params.atr_tp_mult * atr_value, 2),
        matched_style=style,
    )


@dataclass
class TradeSignal:
    symbol: str
    direction: str  # 'long' | 'short' | 'no-trade'
    entry: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    score: float
    reasons: list[str] = field(default_factory=list)
    trend: str = ""
    confidence: str = "low"
    timeframe_notes: dict = field(default_factory=dict)
    atr: float = 0.0


class Swing4HStrategy:
    """The 4-hour-first swing strategy (live/analysis path)."""

    def __init__(self, params: StrategyParams | None = None) -> None:
        self.params = params or StrategyParams()

    def analyze(self, symbol: str, intervals: dict[str, pd.DataFrame], tv_rec: float | None = None) -> TradeSignal:
        sec_4h = intervals.get("4h")
        if sec_4h is None or len(sec_4h) < 70:
            return TradeSignal(symbol, "no-trade", 0.0, None, None, 0.0, ["insufficient 4h data"], "unknown")

        sec_4h = compute_all(sec_4h.copy())
        last = sec_4h.iloc[-1]
        price = float(last["close"])
        atr14 = float(last["atr_14"])
        if not atr14 or atr14 <= 0:
            return TradeSignal(symbol, "no-trade", price, None, None, 0.0, ["no ATR"], "unknown")

        trend_4h = "uptrend" if _trend_sign(sec_4h) > 0 else ("downtrend" if _trend_sign(sec_4h) < 0 else "range")
        hi_tf_score = higher_tf_score(intervals)

        # Lower-TF confirmation.
        lower_confirm = 0.0
        tf_notes = {}
        for tf, w in (("1h", SCORE_1H_CONFIRM), ("15m", SCORE_15M_CONFIRM)):
            d = intervals.get(tf)
            if d is None or d.empty or len(d) < 20:
                continue
            c = compute_all(d.copy())
            last_l = c.iloc[-1]
            tf_notes[tf] = {
                "rsi": round(float(last_l["rsi_14"]), 1),
                "above_ema21": bool(last_l["close"] > last_l["ema_21"]),
                "macd_hist_pos": bool(last_l["macd_hist"] > 0),
            }
            if tf_notes[tf]["above_ema21"]:
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
            params=self.params,
            tv_rec=tv_rec,
        )

        reasons = []
        if dec.direction == "long":
            reasons.append(f"4h {dec.matched_style} entry in uptrend (score {dec.score})")
            if dec.matched_style == "pullback":
                reasons.append(f"price pulled back to/below EMA21, RSI cooled to {float(last['rsi_14']):.1f}, EMA50 trend intact")
            else:
                reasons.append(f"momentum breakout, RSI {float(last['rsi_14']):.1f}, MACD hist +")
        else:
            reasons.append(f"no-trade: score {dec.score:.1f} below {self.params.min_score}")

        if tv_rec is not None:
            tv_lab = "buy" if tv_rec >= 0.25 else ("sell" if tv_rec <= -0.25 else "neutral")
            reasons.append(f"TradingView composite: {tv_lab} ({tv_rec:+.2f})")

        trend_lab = f"{'weekly/daily+4h up' if hi_tf_score > 0 else 'weekly/daily+4h neutral/down'} / 4h:{trend_4h}"
        confidence = "high" if dec.score >= 80 else ("medium" if dec.score >= 70 else "low")

        return TradeSignal(
            symbol=symbol,
            direction=dec.direction,
            entry=round(price, 2),
            stop_loss=dec.stop,
            take_profit=dec.target,
            score=dec.score,
            reasons=reasons,
            trend=trend_lab,
            confidence=confidence,
            timeframe_notes=tf_notes,
            atr=round(atr14, 4),
        )

    @classmethod
    def default_params(cls) -> StrategyParams:
        return StrategyParams()