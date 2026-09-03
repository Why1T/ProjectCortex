"""Risk management: position sizing, take-profit and stop-loss derivation.

Cortex uses fixed-fractional risk sizing: you define a % of equity you're
willing to lose per trade (default 1%). The number of shares is derived
from that risk and the ATR-based stop distance, giving consistent
R-multiple management across every idea we present.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionPlan:
    symbol: str
    direction: str
    equity: float
    entry: float
    stop: float
    target: float
    risk_per_share: float
    qty: int
    dollar_amount: float
    risk_amount: float
    reward_amount: float
    risk_pct: float
    rr_ratio: float
    cap_exceeded: bool = False


class RiskManager:
    def __init__(self, risk_per_trade: float = 0.01, max_position_pct: float = 0.20) -> None:
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct

    @staticmethod
    def atr_stop_distance(atr_value: float, mult: float = 2.0) -> float:
        """Structural stop distance from ATR (recommended over fixed pips)."""
        return atr_value * mult

    def plan_position(
        self,
        symbol: str,
        direction: str,
        equity: float,
        entry: float,
        stop: float,
        target: float,
    ) -> PositionPlan:
        if direction == "long":
            risk_per_share = entry - stop
            reward_per_share = target - entry
        else:
            risk_per_share = stop - entry
            reward_per_share = entry - target

        if risk_per_share <= 0 or reward_per_share <= 0:
            raise ValueError("stop/target must bracket entry on correct side")

        risk_amount = equity * self.risk_per_trade
        qty = max(1, int(risk_amount / risk_per_share))
        dollar_amount = qty * entry

        cap = equity * self.max_position_pct
        cap_exceeded = dollar_amount > cap
        if cap_exceeded:
            qty = max(1, int(cap / entry))
            dollar_amount = qty * entry

        return PositionPlan(
            symbol=symbol,
            direction=direction,
            equity=equity,
            entry=round(entry, 2),
            stop=round(stop, 2),
            target=round(target, 2),
            risk_per_share=round(risk_per_share, 2),
            qty=qty,
            dollar_amount=round(dollar_amount, 2),
            risk_amount=round(risk_amount, 2),
            reward_amount=round(qty * reward_per_share, 2),
            risk_pct=round(self.risk_per_trade * 100, 2),
            rr_ratio=round(reward_per_share / risk_per_share, 2),
            cap_exceeded=cap_exceeded,
        )