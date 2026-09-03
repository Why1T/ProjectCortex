"""Account and configuration status command."""
from __future__ import annotations

from config.settings import settings
from cortex.cli.commands.common import make_advisor


def run_status(args) -> int:
    advisor = make_advisor()
    account = advisor.account
    print(f"\nData source: {advisor.data.source}")
    print(f"Equity: ${account['equity']:,.2f}")
    print(f"Buying power: ${account['buying_power']:,.2f}")
    print(f"Open positions: {len(account['positions'])}")
    for position in account["positions"]:
        print(f"  {position['symbol']:6s} qty={position['qty']:.0f} @ ${position['avg_entry']:.2f} "
              f"uP/L=${position['unrealized_pl']:.2f} ({position['unrealized_plpc']*100:.2f}%)")
    print(f"\nLLM analysis: {'enabled' if settings.llm_enabled else 'disabled (rules-based only)'}")
    return 0
