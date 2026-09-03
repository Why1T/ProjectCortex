"""Watchlist and backtest commands."""
from __future__ import annotations

from cortex.analysis.scanner import default_universe
from cortex.cli.commands.common import make_advisor
from cortex.reports.generator import generate_backtest_report, generate_watchlist_report


def run_watchlist(args) -> int:
    advisor = make_advisor()
    report = advisor.get_daily_look(deep_analysis=False)
    path = generate_watchlist_report(report)
    print(f"Watchlist written to: {path}")
    print(f"Week: {', '.join(report.watchlist_week) or '—'}")
    print(f"Month: {', '.join(report.watchlist_month) or '—'}")
    return 0


def run_backtest(args) -> int:
    advisor = make_advisor()
    symbols = args.symbols or default_universe()
    result = advisor._ensure_healthy(symbols)
    path = generate_backtest_report(result)
    before = result["before"]
    print(f"Backtest report written to: {path}")
    print(f"Trades: {before['total_trades']} | Expectancy: {before['expectancy_r']} R | PF: {before['profit_factor']:.2f} | "
          f"Optimised: {result.get('optimised', False)}")
    print("Top performers:")
    for symbol, info in sorted(before.get("symbols", {}).items(), key=lambda item: -item[1]["stats"]["expectancy_r"])[:5]:
        stats = info["stats"]
        print(f"  {symbol}: {stats['trades']} trades, {stats['expectancy_r']} R, PF {stats['profit_factor']:.2f}")
    return 0
