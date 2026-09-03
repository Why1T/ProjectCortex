"""Advisor: orchestrates everything into actionable guidance for you.

Cortex synthesises:
  * account equity + current paper positions (Alpaca),
  * a multi-timeframe scan of the universe,
  * ATR-based risk management and position sizing,
  * backtest health checks with automatic self-optimisation,
  * loss lessons from the trade journal,
  * a market news brief + social (X) sentiment,
  * an LLM deep-reasoning pass that sets out WHY and points to sources.

It NEVER submits orders; it only produces recommendations/guidance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config.settings import settings
from cortex.analysis.news import Headline, NewsAggregator
from cortex.analysis.scanner import Scanner, default_universe
from cortex.backtest.optimizer import Optimizer
from cortex.backtest.engine import Backtester
from cortex.backtest.metrics import TradeStats
from cortex.clients.alpaca_client import AlpacaClient
from cortex.clients.llm_client import LLMClient
from cortex.clients.tradingview_client import TradingViewClient, TVSignal, build_chart_links
from cortex.data.market_data import MarketData
from cortex.journal.journal import TradeJournal
from cortex.journal.risk import PositionPlan, RiskManager
from cortex.strategies.swing_4h import StrategyParams, Swing4HStrategy, TradeSignal

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    signal: TradeSignal
    position: Optional[PositionPlan]
    deep_reason: str = ""
    bull_case: str = ""
    risks: list[str] = field(default_factory=list)
    verdict: str = ""
    sources: list[str] = field(default_factory=list)
    news: list[dict] = field(default_factory=list)
    tv: Optional[TVSignal] = None
    already_holding: bool = False


@dataclass
class AdvisoryReport:
    generated_at: str
    equity: float
    buying_power: float
    positions: list[dict]
    candidates: list[Candidate]
    watchlist_week: list[str]
    watchlist_month: list[str]
    backtest: dict
    strategy_health: str
    lessons: list[str]
    news_brief: dict
    events_brief: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class DefaultStats:
    """An account summary used when Alpaca credentials are unavailable."""


class Advisor:
    def __init__(
        self,
        data: MarketData | None = None,
        strategy: Swing4HStrategy | None = None,
        journal: TradeJournal | None = None,
        news: NewsAggregator | None = None,
        scanner: Scanner | None = None,
    ) -> None:
        self.data = data or MarketData()
        self.strategy = strategy or Swing4HStrategy()
        self.journal = journal or TradeJournal()
        self.llm = LLMClient(
            settings.llm_api_key, settings.llm_base_url, settings.llm_model, settings.llm_temperature
        )
        self.news = news or NewsAggregator(llm=self.llm, max_items=settings.news_max_items)
        self.scanner = scanner
        self.risk = RiskManager(settings.risk_per_trade, settings.max_position_pct)
        self.tv = TradingViewClient(timeout=settings.tv_timeout) if settings.tv_enabled else None
        self._tv_signals: dict[str, TVSignal] = {}
        self.account = self._describe_account()

    # ---------------- account ----------------
    def _describe_account(self) -> dict:
        if self.data.primary is None:
            return {"equity": 100_000.0, "buying_power": 100_000.0, "positions": [], "source": "default"}
        try:
            acct = self.data.primary.get_account()
            if not acct or "equity" not in acct:
                return {
                    "equity": 100_000.0,
                    "buying_power": 100_000.0,
                    "positions": [],
                    "source": "alpaca-error",
                    "error": "Alpaca account could not be read. Check your API keys and paper/live setting.",
                }
        except Exception as exc:
            log.warning("Alpaca account lookup failed: %s", exc)
            return {
                "equity": 100_000.0,
                "buying_power": 100_000.0,
                "positions": [],
                "source": "alpaca-error",
                "error": "Alpaca account could not be read. Check your API keys and paper/live setting.",
            }
        equity = float(acct.get("equity", 100_000.0))
        bp = float(acct.get("buying_power", equity))
        positions = []
        for p in self.data.primary.get_positions():
            positions.append(
                {
                    "symbol": p.get("symbol"),
                    "qty": float(p.get("qty", 0)),
                    "avg_entry": float(p.get("avg_entry_price", 0)),
                    "unrealized_pl": float(p.get("unrealized_pl", 0)),
                    "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
                }
            )
        return {"equity": equity, "buying_power": bp, "positions": positions, "source": "alpaca"}

    # ---------------- scan + size ----------------
    def _scan(self, symbols: list[str] | None = None) -> list[TradeSignal]:
        universe = symbols or default_universe()
        # Fetch TradingView composite recommendations once for the universe.
        self._tv_signals = self.tv.fetch(universe) if self.tv else {}

        def _tv_lookup(sym: str) -> float | None:
            tv = self._tv_signals.get(sym.upper())
            return tv.recommendation if tv else None

        if self.scanner is None or not hasattr(self.scanner, "tv_rec_lookup"):
            self.scanner = Scanner(self.data, self.strategy, symbols=universe, tv_rec_lookup=_tv_lookup)
        signals = self.scanner.scan()
        for sig in signals:
            live_price = self.data.get_latest_price(sig.symbol)
            if live_price and live_price > 0 and sig.entry and sig.entry > 0 and sig.direction in {"long", "short"}:
                delta = live_price - sig.entry
                sig.entry = round(live_price, 2)
                if sig.stop_loss is not None:
                    sig.stop_loss = round(sig.stop_loss + delta, 2)
                if sig.take_profit is not None:
                    sig.take_profit = round(sig.take_profit + delta, 2)
        return signals

    def _size(self, sig: TradeSignal) -> Optional[PositionPlan]:
        if sig.direction != "long" or sig.entry <= 0 or not sig.stop_loss or not sig.take_profit:
            return None
        try:
            return self.risk.plan_position(
                sig.symbol, sig.direction, self.account["equity"], sig.entry, sig.stop_loss, sig.take_profit
            )
        except ValueError:
            return None

    # ---------------- backtest + self-optimise ----------------
    def _backtest(self, symbols: list[str], param_source: Optional[StrategyParams] = None) -> dict:
        params = param_source or self.strategy.params
        bt = Backtester(params=params)
        agg: dict = {"total_trades": 0, "profitable_symbols": 0, "symbols": {}}
        worst: TradeStats | None = None
        for sym in symbols:
            try:
                intervals = self.data.cached(sym)
            except Exception:
                continue
            stats, trades = bt.run(sym, intervals)
            agg["total_trades"] += stats.trades
            agg["symbols"][sym] = {"stats": stats.as_dict(), "n": stats.trades}
            if stats.trades > 0:
                if stats.total_return > 0:
                    agg["profitable_symbols"] += 1
                if worst is None or stats.total_return < worst.total_return:
                    worst = stats
        # Aggregate expectancy + PF across the whole basket.
        agg["expectancy_r"] = round(
            sum(v["stats"]["expectancy_r"] * v["stats"]["trades"] for v in agg["symbols"].values())
            / max(1, agg["total_trades"]), 3
        )
        agg["profit_factor"] = _weighted_pf(agg["symbols"])
        agg["profitable_symbols"] = agg["profitable_symbols"]
        agg["total_symbols"] = len(agg["symbols"])
        agg["worst_symbol"] = worst.as_dict() if worst else None
        return agg

    def _ensure_healthy(self, symbols: list[str]) -> dict:
        """Backtest current params; if unhealthy, self-optimise and report."""
        before = self._backtest(symbols)
        healthy = before["expectancy_r"] > 0 and before["profit_factor"] > 1.1
        if not healthy and before["total_trades"] >= 10:
            log.info("Strategy unhealthy -> running self-optimisation.")
            opt = Optimizer(starting_params=self.strategy.params)
            sample_sym = next((s for s in symbols if before["symbols"].get(s, {}).get("n", 0) >= 5), symbols[0] if symbols else "AAPL")
            try:
                intervals = self.data.cached(sample_sym)
                new_params = opt.optimise(sample_sym, intervals)
                self.strategy.params = new_params
                self.journal.save_strategy_state("strategy_params", _params_to_dict(new_params))
            except Exception as exc:
                log.warning("Optimisation failed: %s", exc)
            after = self._backtest(symbols, param_source=self.strategy.params)
            return {"before": before, "after": after, "optimised": True}
        return {"before": before, "optimised": False}

    # ---------------- build report ----------------
    def get_daily_look(self, symbols: list[str] | None = None, deep_analysis: bool = True) -> AdvisoryReport:
        symbols = symbols or default_universe()
        self._sync_paper_trades()
        candidates_raw = self._scan(symbols)
        # Don't re-recommend names we're already holding (unless higher confidence).
        held = {p["symbol"] for p in self.account["positions"]}

        candidates: list[Candidate] = []
        for sig in candidates_raw[:8]:
            if sig.symbol in held and sig.score < 90:
                continue
            plan = self._size(sig)
            tv_sig = self._tv_signals.get(sig.symbol.upper())
            cand = Candidate(
                signal=sig,
                position=plan,
                tv=tv_sig,
                already_holding=sig.symbol in held,
            )
            # News tie-in per symbol.
            cand.news = [n.as_dict() for n in self.news.enrich(self._symbol_news(sig.symbol))[:3]]
            self._apply_rule_reason(cand)
            candidates.append(cand)

        if deep_analysis:
            self._deep_reason(candidates)
            self._attach_sources(candidates)

        backtest = self._ensure_healthy(symbols)

        watch_week = [c.signal.symbol for c in candidates[:6] if c.signal.confidence != "low"]
        watch_month = [c.signal.symbol for c in candidates if c.signal.confidence in ("medium", "high")]

        lessons = self.journal.lessons()
        news_brief = self.news.get_news_brief()
        events_brief = self.news.market_events_brief() if deep_analysis else {"headlines": news_brief["headlines"], "summary": ""}

        health = (
            "HEALTHY" if backtest["before"]["expectancy_r"] > 0 and backtest["before"]["profit_factor"] > 1.1
            else "WEAK - self-optimisation ran" if backtest.get("optimised") else "UNKNOWN"
        )

        notes = []
        if self.data.primary is None:
            notes.append("No Alpaca credentials: using default $100k equity. Add .env to use your paper account.")
        elif self.account.get("source") == "alpaca-error":
            notes.append(self.account.get("error", "Alpaca account unavailable; using default $100k equity."))

        return AdvisoryReport(
            generated_at=pd.Timestamp.utcnow().isoformat(),
            equity=self.account["equity"],
            buying_power=self.account["buying_power"],
            positions=self.account["positions"],
            candidates=[{
                "signal": c.signal,
                "position": c.position,
                "deep_reason": c.deep_reason,
                "bull_case": c.bull_case,
                "risks": c.risks,
                "verdict": c.verdict,
                "sources": c.sources,
                "news": c.news,
                "tv": c.tv.as_dict() if c.tv else None,
                "already_holding": c.already_holding,
            } for c in candidates],
            watchlist_week=watch_week,
            watchlist_month=watch_month,
            backtest={"before": backtest["before"], "after": backtest.get("after"), "optimised": backtest.get("optimised", False)},
            strategy_health=health,
            lessons=lessons,
            news_brief=news_brief,
            events_brief=events_brief,
            notes=notes,
        )

    # ---------------- helpers ----------------
    def _sync_paper_trades(self) -> None:
        """Auto-ingest closed paper trades into the journal so Cortex learns."""
        if self.data.primary is None:
            return
        try:
            activities = self.data.primary.get_fill_activities(days=30)
            if activities:
                self.journal.sync_activities(activities)
        except Exception as exc:
            log.warning("Paper-trade sync skipped: %s", exc)

    def _symbol_news(self, symbol: str) -> list[dict]:
        """Fetch a small set of current headlines mentioning the symbol."""
        brief = []
        for it in self.news.fetch_market_news():
            if symbol.lower() in it.get("title", "").lower():
                brief.append({"title": it["title"], "link": it.get("link", ""), "source": it.get("source", "")})
        return brief

    def _deep_reason(self, candidates: list[Candidate]) -> None:
        if not self.llm.enabled:
            for c in candidates:
                self._apply_rule_reason(c)
            return
        for c in candidates:
            s = c.signal
            tv = c.tv
            tv_txt = "unavailable"
            if tv is not None:
                tv_txt = (
                    f"{tv.recommendation_label} ({tv.recommendation:+.2f}), "
                    f"RSI {tv.rsi:.0f}, ATR ${tv.atr:.2f}" if tv.rsi and tv.atr else
                    f"{tv.recommendation_label} ({tv.recommendation:+.2f})"
                )
            charts = build_chart_links(s.symbol, c.tv.exchange if c.tv else "NASDAQ") if c.tv else build_chart_links(s.symbol)

            news_bits = []
            for item in c.news[:3]:
                if hasattr(item, "title") and hasattr(item, "source"):
                    news_bits.append(f"{item.title} ({item.source})")
                elif isinstance(item, dict):
                    news_bits.append(f"{item.get('title', 'News')} ({item.get('source', 'unknown')})")
            news_txt = "; ".join(news_bits) or "none"

            tf_notes = ", ".join(
                f"{tf}: RSI{n.get('rsi')}{',>EMA21' if n.get('above_ema21') else ',<EMA21'}"
                for tf, n in s.timeframe_notes.items()
            )
            prompt = (
                f"Analyse {s.symbol} as a LONG swing trade on the 4h chart for a retail trader.\n\n"
                f"Signal: score {s.score}, confidence {s.confidence}, trend {s.trend}.\n"
                f"Entry ${s.entry}, Stop loss ${s.stop_loss}, Take profit ${s.take_profit}, ATR {s.atr}.\n"
                f"4h pullback/breakout logic: {' '.join(s.reasons)}\n"
                f"Timeframe notes: {tf_notes or 'n/a'}\n\n"
                f"TradingView composite: {tv_txt}\n"
                f"Recent headlines for {s.symbol}: {news_txt}\n"
                "Charts (all timeframes) are available at these TradingView links for reference:\n"
                f"  4H: {charts['4h']}\n  1D: {charts['1d']}\n\n"
                "Produce a structured recommendation. Return JSON with exactly these keys:\n"
                "1) 'bull_case' (string): WHY this is a good buy right now — cite the technical "
                "setup, any news/catalyst, and the TradingView view. Be concrete, 3-4 sentences.\n"
                "2) 'risks' (list of strings): the specific risks that could invalidate the trade "
                "(e.g. earnings, macro news, technical invalidation below stop, overextension). 2-4 items.\n"
                "3) 'reason' (string): a 1-2 sentence bottom-line summary.\n"
                "4) 'verdict' (string): one of 'advise' | 'hold' | 'wait' — your final call for entering today.\n"
                "Be disciplined and honest; a conservative verdict is a good verdict."
            )
            try:
                j = self.llm.ask_json(prompt)
                c.bull_case = j.get("bull_case", "")
                risk_val = j.get("risks", [])
                c.risks = [r for r in risk_val if isinstance(r, str)] if isinstance(risk_val, list) else []
                c.deep_reason = j.get("reason", "")
                c.verdict = j.get("verdict", "")
                c.sources = sorted(set(c.sources) | {charts["4h"], charts["1d"]})
            except Exception as exc:
                log.warning("LLM deep reason failed for %s: %s", s.symbol, exc)
                self._apply_rule_reason(c)

    def _apply_rule_reason(self, c: Candidate) -> None:
        """Rules-based fallback that still explains WHY and the risk honestly."""
        s = c.signal
        c.deep_reason = self._rule_reason(s)
        if c.tv is not None and c.tv.recommendation >= 0.25:
            c.bull_case = (
                f"{s.symbol} shows a {s.trend} and a {'pullback' if 'pullback' in ' '.join(s.reasons) else 'breakout'} "
                f"setup on the 4h chart (score {s.score}). TradingView's composite is 'buy' ({c.tv.recommendation:+.2f}), "
                f"so the trend, 4h entry and TradingView alignment all agree. Entry ~${s.entry}."
            )
        else:
            c.bull_case = (
                f"{s.symbol} is in a {s.trend} with a 4h {'pullback' if 'pullback' in ' '.join(s.reasons) else 'breakout'} "
                f"signal (score {s.score}). Momentum and higher-timeframe trend support a long swing toward ${s.take_profit}."
            )
        c.risks = [
            f"Technical invalidation if price closes below stop ${s.stop_loss} (below the swing/ATR stop).",
            "Earnings/news gap risk — always check the earnings date and scheduled macro events before entry.",
            f"Overextension risk: current RSI on 4h indicates the move could be late; a pullback may improve entry.",
        ]
        c.verdict = "advise" if s.confidence in ("high", "medium") else "wait"

    @staticmethod
    def _rule_reason(sig: TradeSignal) -> str:
        return f"Signal on {sig.symbol}: {sig.trend}; score {sig.score}. Reasoning: " + " ".join(sig.reasons)

    def _attach_sources(self, candidates: list[Candidate]) -> None:
        """Point to concrete source material for each candidate."""
        for c in candidates:
            srcs = set()
            for n in c.news:
                link = getattr(n, "link", None)
                if isinstance(n, dict):
                    link = n.get("link")
                if link:
                    srcs.add(link)
            # TradingView charts across the swing timeframes.
            exchange = c.tv.exchange if c.tv else "NASDAQ"
            charts = build_chart_links(c.signal.symbol, exchange)
            srcs.update([charts["4h"], charts["1d"], charts["1h"]])
            # The market data / indicator evidence as a usable reference.
            srcs.add(f"4h chart indicators: RSI/MACD/EMA/ATR on {c.signal.symbol}")
            c.sources = sorted(srcs)


def _weighted_pf(symbols: dict) -> float:
    gross_w = sum(v["stats"]["avg_win_r"] * v["stats"]["wins"] for v in symbols.values() if v["stats"]["wins"])
    gross_l = sum(v["stats"]["avg_loss_r"] * v["stats"]["losses"] for v in symbols.values() if v["stats"]["losses"])
    if gross_l == 0:
        return gross_w / max(1, len(symbols) or 1)
    return gross_w / gross_l


def _params_to_dict(p: StrategyParams) -> dict:
    return {
        "pullback_rsi_max": p.pullback_rsi_max,
        "overbought_rsi_min": p.overbought_rsi_min,
        "atr_stop_mult": p.atr_stop_mult,
        "atr_tp_mult": p.atr_tp_mult,
        "min_avg_dollar_vol": p.min_avg_dollar_vol,
        "min_score": p.min_score,
        "require_daily_trend": p.require_daily_trend,
    }