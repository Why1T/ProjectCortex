"""Argument parsing and application entry point for Cortex."""
from __future__ import annotations

import argparse
import logging

from cortex.cli.commands.daily import run_daily
from cortex.cli.commands.journal import run_journal
from cortex.cli.commands.market import run_news, run_symbol
from cortex.cli.commands.reports import run_backtest, run_watchlist
from cortex.cli.commands.status import run_status

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
for noisy_logger in ("urllib3", "requests", "asyncio"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortex",
        description="Cortex — swing trading research & advisory bot (never trades)",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    daily = subcommands.add_parser("daily", help="Generate the full daily advisory report")
    daily.add_argument("--no-deep", action="store_true", help="Skip LLM deep-reasoning pass")
    daily.set_defaults(func=run_daily)

    symbol = subcommands.add_parser("symbol", help="Analyse a single symbol across all timeframes")
    symbol.add_argument("symbol")
    symbol.set_defaults(func=run_symbol)

    watchlist = subcommands.add_parser("watchlist", help="Quick watchlist report")
    watchlist.set_defaults(func=run_watchlist)

    backtest = subcommands.add_parser("backtest", help="Run backtest + self-optimisation")
    backtest.add_argument("--symbols", nargs="+", help="Restrict backtest to these symbols")
    backtest.set_defaults(func=run_backtest)

    news = subcommands.add_parser("news", help="Market news brief + social sentiment")
    news.add_argument("--limit", type=int, default=8)
    news.set_defaults(func=run_news)

    journal = subcommands.add_parser("journal", help="Trade journal (show / log / sync from Alpaca paper)")
    journal.add_argument("--show", action="store_true")
    journal.add_argument("--sync", action="store_true", help="Pull closed trades from Alpaca paper account")
    journal.add_argument("--sync-days", type=int, default=30)
    journal.add_argument("--add", action="store_true")
    journal.add_argument("--symbol")
    journal.add_argument("--entry", type=float)
    journal.add_argument("--exit", type=float)
    journal.add_argument("--qty", type=float)
    journal.add_argument("--direction", default="long")
    journal.add_argument("--entry-time")
    journal.add_argument("--exit-time")
    journal.add_argument("--exit-reason", default="manual")
    journal.add_argument("--r", type=float, default=0.0)
    journal.add_argument("--loss-reason", default="unknown")
    journal.add_argument("--loss-detail", default="")
    journal.add_argument("--limit", type=int, default=25)
    journal.set_defaults(func=run_journal)

    status = subcommands.add_parser("status", help="Show account + configuration status")
    status.set_defaults(func=run_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        logging.getLogger(__name__).exception("Cortex crashed")
        print(f"\nError: {exc}")
        return 1
