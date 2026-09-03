"""Backtest reporting metrics and R-multiple-audit helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


class TradeStats:
    def __init__(
        self,
        trades: int,
        wins: int,
        losses: int,
        win_rate: float,
        profit_factor: float,
        expectancy_r: float,
        avg_win: float,
        avg_loss: float,
        max_drawdown: float,
        total_return: float,
        sharpe: float | None,
    ) -> None:
        self.trades = trades
        self.wins = wins
        self.losses = losses
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.expectancy_r = expectancy_r
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.max_drawdown = max_drawdown
        self.total_return = total_return
        self.sharpe = sharpe

    @property
    def healthy(self) -> bool:
        """Cortex considers a strategy healthy when it's net positive after costs."""
        return self.total_return > 0 and self.profit_factor > 1.0

    def as_dict(self) -> dict:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 3),
            "profit_factor": round(self.profit_factor, 3),
            "expectancy_r": round(self.expectancy_r, 3),
            "avg_win_r": round(self.avg_win, 3),
            "avg_loss_r": round(self.avg_loss, 3),
            "max_drawdown_pct": round(self.max_drawdown, 3),
            "total_return_pct": round(self.total_return, 3),
            "sharpe": round(self.sharpe, 3) if self.sharpe is not None else None,
        }


def compute_metrics(trades: list[tuple[float, float]]) -> TradeStats:
    """Compute metrics from a list of R-multiple outcomes (win, R), plus realised trades."""
    if not trades:
        return TradeStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)
    wins = [r for _, r in trades if r > 0]
    losses = [abs(r) for _, r in trades if r <= 0]
    n = len(trades)
    nw, nl = len(wins), len(losses)
    win_rate = nw / n
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit / max(1e-6, len(wins)))
    avg_win = gross_profit / nw if nw else 0.0
    avg_loss = gross_loss / nl if nl else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) if n > 1 else 0.0

    # Equity curve from R outcomes for drawdown + sharpe; notional risk per trade = 1R.
    equity = np.cumsum([r for _, r in trades])
    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max
    max_dd = float(dd.min() / (running_max.max() if running_max.max() > 0 else 1.0))

    rets = np.array([r for _, r in trades])
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(252)) if rets.std() > 0 else None
    total_return = float((sum(wins) - sum(losses)) / n * 100.0)

    return TradeStats(
        trades=n,
        wins=nw,
        losses=nl,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy_r=expectancy,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown=max_dd,
        total_return=total_return,
        sharpe=sharpe,
    )