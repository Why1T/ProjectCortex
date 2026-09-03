"""Single-symbol and market-news commands."""
from __future__ import annotations

from config.settings import settings
from cortex.analysis.advisor import Candidate
from cortex.analysis.news import NewsAggregator
from cortex.clients.llm_client import LLMClient
from cortex.clients.tradingview_client import build_chart_links
from cortex.cli.commands.common import make_advisor


def run_symbol(args) -> int:
    advisor = make_advisor()
    symbol = args.symbol.upper()
    intervals = advisor.data.all_intervals(symbol)
    if intervals.get("4h") is None or intervals["4h"].empty:
        print(f"No data available for {symbol}.")
        return 1
    tv_signal = advisor.tv.get(symbol) if advisor.tv else None
    tv_recommendation = tv_signal.recommendation if tv_signal else None
    signal = advisor.strategy.analyze(symbol, intervals, tv_rec=tv_recommendation)
    candidate = Candidate(
        signal=signal,
        position=None,
        tv=tv_signal,
        news=[headline.as_dict() for headline in advisor.news.enrich(advisor._symbol_news(symbol))[:3]],
    )
    advisor._apply_rule_reason(candidate)
    if advisor.llm.enabled:
        advisor._deep_reason([candidate])

    print(f"\n=== {symbol} — {signal.direction.upper()}  (Cortex verdict: {candidate.verdict}) ===")
    print(f"Trend: {signal.trend}")
    print(f"Score: {signal.score} ({signal.confidence})")
    print(f"Entry: ${signal.entry}  Stop: ${signal.stop_loss}  Target: ${signal.take_profit}  ATR: {signal.atr}")
    if tv_signal is not None:
        print(f"TradingView composite: {tv_signal.recommendation_label} ({tv_signal.recommendation:+.2f}) | RSI {tv_signal.rsi:.0f} | ATR ${tv_signal.atr:.2f}")
        charts = tv_signal.chart_links()
        print(f"TradingView charts: 4H {charts['4h']} | 1D {charts['1d']} | 1W {charts['1w']}")
    else:
        charts = build_chart_links(symbol)
        print(f"TradingView charts: 4H {charts['4h']} | 1D {charts['1d']}")
    print("\nWHY IT'S A GOOD BUY:")
    print(f"  {candidate.bull_case}")
    print("\nRISKS:")
    for risk in candidate.risks:
        print(f"  - {risk}")
    print("\nReasons (signal):")
    for reason in signal.reasons:
        print(f"  - {reason}")
    print("Timeframe detail:")
    for timeframe, note in signal.timeframe_notes.items():
        print(f"  {timeframe}: {note}")
    if candidate.news:
        print("\nRecent headlines:")
        for headline in candidate.news:
            source = headline.get("source", "?")
            title = headline.get("title", "")
            link = headline.get("link", "")
            print(f"  - [{source}] {title} {link}")
    if signal.direction == "long":
        plan = advisor.risk.plan_position(signal.symbol, "long", advisor.account["equity"], signal.entry, signal.stop_loss, signal.take_profit)
        print(f"\nSizing (equity ${plan.equity:,.2f}): {plan.qty} shares = ${plan.dollar_amount:,.2f} | "
              f"risk ${plan.risk_amount:,.2f} ({plan.risk_pct}%) | R:R {plan.rr_ratio}")
    return 0


def run_news(args) -> int:
    llm = LLMClient(settings.llm_api_key, settings.llm_base_url, settings.llm_model, settings.llm_temperature)
    news = NewsAggregator(llm=llm, max_items=settings.news_max_items)
    brief = news.get_news_brief()
    print("\n=== Market News Brief ===")
    for headline in brief["headlines"][:args.limit]:
        print(f"[{headline['source']:12s}] ({headline['sentiment']:7s}) {headline['title']}")
        if headline["link"]:
            print(f"            {headline['link']}")
    social = brief["social"]
    print(f"\nSocial (X): {social['count']} posts, {social['label']} - {social['summary']}")
    return 0
