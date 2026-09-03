"""Daily advisory command."""
from __future__ import annotations

from cortex.cli.commands.common import make_advisor
from cortex.reports.generator import generate_daily_report


def run_daily(args) -> int:
    advisor = make_advisor()
    print("Scanning universe and running analysis (this may take a moment)...")
    report = advisor.get_daily_look(deep_analysis=not args.no_deep)
    path = generate_daily_report(report)
    print(f"\nDaily advisory written to: {path}\n")
    print(f"Strategy status: {report.strategy_health}")
    print(f"Equity: ${report.equity:,.2f} | Watchlist (week): {', '.join(report.watchlist_week) or '—'}")
    for candidate in report.candidates:
        signal = candidate["signal"]
        position = candidate["position"]
        size = f" {position.qty} sh" if position else ""
        verdict = candidate.get("verdict") or "n/a"
        tv = candidate.get("tv")
        tv_text = f"tv={tv.get('recommendation_label')}" if isinstance(tv, dict) and tv else "tv=n/a"
        print(
            f"  {signal.symbol:6s} {signal.direction:5s} score={signal.score:5.1f} conf={signal.confidence:6s} verdict={verdict:6s} {tv_text:12s}"
            f"entry=${signal.entry:.2f} stop=${signal.stop_loss} tp=${signal.take_profit}{size}"
        )
        print(f"    Why: {candidate.get('bull_case') or candidate.get('deep_reason') or ' '.join(signal.reasons)}")
        for risk in (candidate.get("risks") or [])[:3]:
            print(f"    Risk: {risk}")
        if candidate.get("deep_reason") and candidate.get("deep_reason") != candidate.get("bull_case"):
            print(f"    Bottom line: {candidate['deep_reason']}")
    return 0
