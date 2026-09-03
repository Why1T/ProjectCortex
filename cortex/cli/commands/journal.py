"""Trade journal command."""
from __future__ import annotations

from cortex.analysis.advisor import Advisor
from cortex.journal.journal import JournalEntry, LOSS_CATEGORIES, TradeJournal


def run_journal(args) -> int:
    journal = TradeJournal()
    if args.sync:
        advisor = Advisor(journal=journal)
        if advisor.data.primary is None:
            print("Alpaca not configured — nothing to sync. Log trades manually with --add.")
            return 0
        activities = advisor.data.primary.get_fill_activities(days=args.sync_days)
        count = journal.sync_activities(activities)
        print(f"Synced {count} closed trade(s) from your Alpaca paper account.")
        return 0
    if args.show:
        trades = journal.all_trades()
        if not trades:
            print("No trades logged.")
        else:
            print(f"\nLogged trades ({len(trades)}):")
            for trade in trades[:args.limit]:
                print(f"  #{trade['id']} {trade['symbol']} {trade['direction']} pnl=${trade['pnl']:.2f} "
                      f"exit={trade['exit_reason']} r={trade['r_multiple']:.2f} reason={trade['loss_reason']}")
            print("\nLoss reasons:")
            print(journal.aggregate_loss_reasons())
            print("\nLessons:")
            for lesson in journal.lessons():
                print(f"  - {lesson}")
        return 0
    if args.add:
        try:
            direction_multiplier = 1 if args.direction == "long" else -1
            entry = JournalEntry(
                symbol=args.symbol.upper(), entry_price=args.entry, exit_price=args.exit, qty=args.qty,
                direction=args.direction, entry_time=args.entry_time or "now", exit_time=args.exit_time or "now",
                exit_reason=args.exit_reason,
                pnl=(args.exit - args.entry) * args.qty * direction_multiplier,
                pnl_pct=(args.exit / args.entry - 1) * direction_multiplier,
                r_multiple=args.r, loss_reason=args.loss_reason or "unknown", loss_detail=args.loss_detail or "",
            )
            trade_id = journal.log_trade(entry)
            print(f"Logged trade #{trade_id} for {entry.symbol}.")
            if entry.pnl < 0:
                print(f"  Loss category notes:\n  {LOSS_CATEGORIES.get(entry.loss_reason, '')}")
        except Exception as exc:
            print(f"Error logging trade: {exc}")
            return 1
        return 0
    print("Use --show or --add sub-options. See --help.")
    return 1
