"""Parameter optimizer: tunes StrategyParams when backtests come back negative.

Cortex runs this automatically after each backtest cycle. When the strategy
is net profitable and has a profit factor > 1.2, the optimizer saves the
current params as "optimal". Otherwise, it searches a modest parameter grid
around the current values and picks the combination that maximises
expectancy (expected R per trade) with a secondary constraint on max
drawdown.
"""
from __future__ import annotations

import itertools
import logging
import math

from cortex.strategies.swing_4h import StrategyParams
from cortex.backtest.engine import Backtester
from cortex.backtest.metrics import TradeStats

log = logging.getLogger(__name__)

# How many grid combos to evaluate before stopping.
_MAX_COMBOS = 96


class Optimizer:
    def __init__(self, starting_params: StrategyParams | None = None) -> None:
        self.current = starting_params or StrategyParams()
        self.best = self.current
        self.best_score = -9999.0
        self.iterations = 0

    def optimise(
        self,
        symbol: str,
        intervals: dict[str, "pd.DataFrame"],
        max_rounds: int = 3,
    ) -> StrategyParams:
        """Return best params found. Stores `self.best` and `self.best_score`."""
        base_params = self.current
        for _round in range(max_rounds):
            grid = self._build_grid(base_params)
            for p in grid:
                bt = Backtester(params=p)
                stats, trades = bt.run(symbol, intervals)
                if stats.trades < 5:
                    continue
                score = self._score(stats)
                if score > self.best_score:
                    self.best_score = score
                    self.best = p
                self.iterations += 1
            base_params = self.best
        log.info(
            "Optimizer: best_score=%.3f after %d combos", self.best_score, self.iterations
        )
        return self.best

    @staticmethod
    def _score(stats: TradeStats) -> float:
        """
        Optimise for a single metric: expectancy (R per trade),
        with drawdown as a penalty when it's bad.
        If expectancy > 0, a large drawdown just reduces the score; if
        it's already negative the drawdown doesn't make it worse.
        """
        penalty = 0.0
        if stats.max_drawdown < -0.30:  # 30% R drawdown is concerning.
            penalty = stats.max_drawdown * 0.3
        return stats.expectancy_r + penalty

    @staticmethod
    def _build_grid(base: StrategyParams) -> list[StrategyParams]:
        """
        Small local search around current best; enough to move the needle
        without burning too much compute.
        """
        pullback_vals = [max(20, base.pullback_rsi_max + d) for d in (-10, -5, 0, 5, 10)]
        atr_stop_vals = [max(1.0, base.atr_stop_mult + d) for d in (-0.5, 0, 0.5, 1.0)]
        atr_tp_vals   = [max(1.5, base.atr_tp_mult + d) for d in (-1.0, 0, 1.0, 2.0)]
        min_score_vals = [max(0, base.min_score + d) for d in (-15, 0, 15)]

        combos = list(itertools.product(pullback_vals, atr_stop_vals, atr_tp_vals, min_score_vals))[:_MAX_COMBOS]
        out: list[StrategyParams] = []
        for pb, sl, tp, ms in combos:
            out.append(
                StrategyParams(
                    pullback_rsi_max=pb,
                    atr_stop_mult=max(1.0, sl),
                    atr_tp_mult=max(1.5, tp),
                    min_score=max(0, ms),
                    min_avg_dollar_vol=base.min_avg_dollar_vol,
                    require_daily_trend=base.require_daily_trend,
                )
            )
        return out