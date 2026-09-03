"""End-to-end smoke tests using synthetic OHLCV data (no network required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cortex.strategies.swing_4h import Swing4HStrategy, StrategyParams, score_entry, SCORE_TV_CONFLUENCE, TradeSignal
from cortex.backtest.engine import Backtester
from cortex.backtest.optimizer import Optimizer
from cortex.backtest.metrics import compute_metrics
from cortex.journal.risk import RiskManager
from cortex.journal.journal import TradeJournal, JournalEntry
from cortex.indicators.engine import compute_all, atr, rsi
from cortex.clients.tradingview_client import TradingViewClient, build_chart_links, TVSignal
from cortex.analysis.advisor import Advisor, Candidate
from cortex.strategies.swing_4h import score_entry, SCORE_TV_CONFLUENCE

def make_bars(n: int, start_price: float = 100.0, seed: int = 1, drift: float = 0.005, pullback_period: int = 40) -> pd.DataFrame:
    """Deterministic strong uptrend with shallow pullbacks.

    Structure: price advances ~drift%/bar. Every `pullback_period` bars a
    shallow 5-bar pullback (~-1.0%) occurs, giving the pullback-in-trend
    strategy clean repeatable long entries while keeping 4h EMA50 support
    well below price.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="4h")
    price = start_price
    closes = []
    for i in range(n):
        phase = i % pullback_period
        if phase >= pullback_period - 3:
            # Meaningful pullback: -1.5% per bar for 3 bars (~-4.4% total).
            d = -0.015 + rng.normal(0, 0.002)
        else:
            d = drift + rng.normal(0, 0.002)
        price *= (1 + d)
        closes.append(price)
    closes = np.array(closes)
    opens = np.r_[start_price, closes[:-1]]
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.004, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.004, n))
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol},
        index=idx,
    )


def make_intervals(n4h: int = 500) -> dict[str, pd.DataFrame]:
    return {
        "15m": make_bars(n4h * 16, drift=0.005),
        "1h": make_bars(n4h * 4, drift=0.005),
        "4h": make_bars(n4h, drift=0.005, seed=7, pullback_period=50),
        "1d": make_bars(n4h // 6 + 1, drift=0.03, seed=3, pullback_period=20),
        "1w": make_bars(n4h // 30 + 1, drift=0.06, seed=4, pullback_period=10),
        "1mo": make_bars(n4h // 130 + 1, drift=0.10, seed=5, pullback_period=8),
    }


def test_indicator_engine_values():
    df = make_bars(300)
    out = compute_all(df)
    assert not out["ema_21"].isna().iloc[-1]
    assert 0 <= out["rsi_14"].iloc[-1] <= 100
    assert out["atr_14"].iloc[-1] > 0
    assert not np.isnan(out["macd_hist"].iloc[-1])


def test_strategy_runs_on_synthetic():
    intervals = make_intervals()
    strategy = Swing4HStrategy()
    sig = strategy.analyze("TEST", intervals)
    assert sig.symbol == "TEST"
    assert sig.direction in ("long", "short", "no-trade")
    assert sig.entry > 0
    # If it's a trade, the bracket must be on the correct side of entry.
    if sig.direction == "long":
        assert sig.stop_loss < sig.entry < sig.take_profit
        assert sig.atr > 0
    elif sig.direction == "short":
        assert sig.take_profit < sig.entry < sig.stop_loss
        assert sig.atr > 0


def test_backtest_runs_and_returns_stats():
    intervals = make_intervals()
    bt = Backtester()
    stats, trades = bt.run("TEST", intervals)
    assert stats.trades >= 0
    for t in trades:
        assert t.exit_price is not None
        assert t.entry_time <= t.exit_time


def test_optimizer_finds_something():
    intervals = make_intervals()
    opt = Optimizer()
    best = opt.optimise("TEST", intervals, max_rounds=1)
    assert best is not None
    assert opt.best_score > -9999


def test_risk_manager_plan():
    rm = RiskManager(risk_per_trade=0.01, max_position_pct=0.20)
    plan = rm.plan_position("AAPL", "long", equity=100_000, entry=150.0, stop=145.0, target=170.0)
    assert plan.qty >= 1
    assert plan.risk_amount == pytest.approx(1000.0)
    assert plan.rr_ratio == pytest.approx((170 - 150) / (150 - 145))
    assert plan.entry < plan.target
    assert plan.stop < plan.entry


def test_journal_loss_lessons(tmp_path):
    j = TradeJournal(db_path=tmp_path / "t.db")
    for i, r in enumerate(["chase", "chase", "chase", "unknown", "unknown"]):
        j.log_trade(
            JournalEntry(
                symbol="AA", entry_price=10, exit_price=9, qty=100, direction="long",
                entry_time="t", exit_time="t", exit_reason="stop", pnl=-100,
                pnl_pct=-10, r_multiple=-1.0, loss_reason=r, loss_detail=""
            )
        )
    lessons = j.lessons()
    assert any("chasing" in l.lower() or "pullback" in l.lower() for l in lessons)
    counts = j.loss_reason_counts()
    assert counts.get("chase", 0) == 3


def test_metrics_healthy_flag():
    trades = [(1, 1.5), (1, 2.0), (1, 1.2), (0, -0.8), (0, -1.0)]
    stats = compute_metrics(trades)
    assert stats.healthy

def test_metrics_negative_flag():
    trades = [(0, -1.0), (0, -1.2), (0, -0.9), (1, 0.3), (0, -0.7)]
    stats = compute_metrics(trades)
    assert not stats.healthy


def test_journal_sync_activities(tmp_path):
    j = TradeJournal(db_path=tmp_path / "sync.db")
    activities = [
        {"id": "1", "activity_type": "FILL", "symbol": "AAPL", "side": "buy",
         "qty": "10", "price": "100", "transaction_time": "2026-01-01T10:00:00Z"},
        {"id": "2", "activity_type": "FILL", "symbol": "AAPL", "side": "sell",
         "qty": "10", "price": "95", "transaction_time": "2026-01-05T10:00:00Z",
         "net_amount": "-50"},
    ]
    n = j.sync_activities(activities)
    assert n == 1
    trades = j.all_trades()
    assert trades[0]["symbol"] == "AAPL"
    assert trades[0]["pnl"] == -50.0
    assert j.losses()  # recognised as a loss
    # Idempotent: syncing again adds nothing.
    assert j.sync_activities(activities) == 0


# ---------------- TradingView integration ----------------

def test_build_chart_links_all_timeframes():
    links = build_chart_links("AAPL", "NASDAQ")
    expected = ["15m", "1h", "4h", "1d", "1w", "1mo"]
    assert sorted(links) == sorted(expected)
    assert links["4h"] == "https://www.tradingview.com/chart/?symbol=NASDAQ:AAPL&interval=240"
    assert links["1d"] == "https://www.tradingview.com/chart/?symbol=NASDAQ:AAPL&interval=D"
    assert links["15m"] == "https://www.tradingview.com/chart/?symbol=NASDAQ:AAPL&interval=15"


def test_tv_parse_fake_row():
    tv = TradingViewClient()
    # d = [name, close, Recommend.All, MA, Other, RSI, ATR]
    row = ["AAPL", 200.0, 0.5, 0.5, 0.5, 60.0, 2.5]
    sig = tv._parse("NASDAQ:AAPL", row)
    assert sig is not None
    assert sig.symbol == "AAPL"
    assert sig.recommendation == 0.5
    assert sig.recommendation_label == "buy"
    assert sig.rsi == 60.0
    assert sig.atr == 2.5
    d = sig.as_dict()
    assert d["chart_4h"] == "https://www.tradingview.com/chart/?symbol=NASDAQ:AAPL&interval=240"


def test_tv_parse_none_metrics_guarded():
    tv = TradingViewClient()
    sig = tv._parse("NYSE:MSFT", ["MSFT", 300.0, None, 0.0, None, None, None])
    assert sig is not None
    assert sig.recommendation == 0.0
    assert sig.recommendation_label == "neutral"
    assert sig.rsi is None


def test_score_entry_tv_confluence():
    params = StrategyParams()
    kwargs = dict(
        price=105.0, atr_value=2.0, rsi=50.0, macd_hist=1.0,
        ema9=100.0, ema21=101.0, ema50=95.0, hi_tf_score=5.0,
        lower_confirm=0.0, params=params,
    )
    base = score_entry(**kwargs, tv_rec=0.0)
    agree = score_entry(**kwargs, tv_rec=0.8)
    disagree = score_entry(**kwargs, tv_rec=-0.8)
    assert agree.score == base.score + SCORE_TV_CONFLUENCE
    assert disagree.score == base.score - SCORE_TV_CONFLUENCE


def test_advisor_rule_reason_populates_fields():
    adv = Advisor()
    sig_t = TradeSignal(
        symbol="AAPL", direction="long", entry=100.0, stop_loss=95.0, take_profit=110.0,
        score=80.0, reasons=["4h pullback into EMA21", "higher-timeframe uptrend"],
        trend="uptrend", confidence="high",
    )
    c = Candidate(signal=sig_t, position=None, tv=TVSignal("AAPL", recommendation=0.6, recommendation_label="buy"))
    adv._apply_rule_reason(c)
    assert c.deep_reason
    assert c.bull_case
    assert isinstance(c.risks, list) and c.risks
    assert c.verdict == "advise"
    assert "$95.0" in c.risks[0]  # stop referenced in the risk wording