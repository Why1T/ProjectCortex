"""Trade journal with loss-learning.

Cortex logs every completed trade (paper or advised), classifies the reason
for each loss, and maintains adaptive rules so it can avoid the same mistake.
Persistence is a lightweight SQLite database under the project's store/ dir.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import STORE_DIR

DB_PATH = STORE_DIR / "cortex.db"

LOSS_CATEGORIES = {
    "chase": "Bought after an extension instead of waiting for a pullback.",
    "no_stop": "Held without a stop and took a larger loss than planned.",
    "wrong_trend": "Traded against the higher-timeframe direction.",
    "news_risk": "Entered right into a scheduled news/earnings event.",
    "overtrade": "Too many simultaneous or high-frequency positions.",
    "premature": "Stopped out by volatility immediately after entry (too tight).",
    "market_regime": "Lost during a broad market downtrend / regime change.",
    "unknown": "No clear mechanical cause identified.",
}


@dataclass
class JournalEntry:
    symbol: str
    entry_price: float
    exit_price: float
    qty: float
    direction: str
    entry_time: str
    exit_time: str
    exit_reason: str  # 'tp' | 'stop' | 'manual' | 'time'
    pnl: float
    pnl_pct: float
    r_multiple: float
    loss_reason: str = "unknown"
    loss_detail: str = ""
    notes: str = ""


class TradeJournal:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, entry_price REAL, exit_price REAL, qty REAL,
                direction TEXT, entry_time TEXT, exit_time TEXT,
                exit_reason TEXT, pnl REAL, pnl_pct REAL, r_multiple REAL,
                loss_reason TEXT, loss_detail TEXT, notes TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_state (
                key TEXT PRIMARY KEY, value TEXT
            )
            """
        )
        self._conn.commit()

    # ---------- writes ----------
    def log_trade(self, entry: JournalEntry) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO trades (symbol, entry_price, exit_price, qty, direction,
                entry_time, exit_time, exit_reason, pnl, pnl_pct, r_multiple,
                loss_reason, loss_detail, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                entry.symbol, entry.entry_price, entry.exit_price, entry.qty,
                entry.direction, entry.entry_time, entry.exit_time, entry.exit_reason,
                entry.pnl, entry.pnl_pct, entry.r_multiple,
                entry.loss_reason, entry.loss_detail, entry.notes,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def update_loss_reason(self, trade_id: int, reason: str, detail: str = "") -> None:
        self._conn.execute(
            "UPDATE trades SET loss_reason=?, loss_detail=? WHERE id=?",
            (reason, detail, trade_id),
        )
        self._conn.commit()

    # ---------- reads ----------
    def all_trades(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM trades ORDER BY exit_time").fetchall()
        return [dict(r) for r in rows]

    def closed_trades(self) -> list[dict]:
        return self.all_trades()

    def losses(self) -> list[dict]:
        return [t for t in self.all_trades() if t["pnl"] < 0]

    def wins(self) -> list[dict]:
        return [t for t in self.all_trades() if t["pnl"] >= 0]

    def loss_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.losses():
            r = t["loss_reason"] or "unknown"
            counts[r] = counts.get(r, 0) + 1
        return counts

    def aggregate_loss_reasons(self) -> str:
        """Human-readable summary of the most common loss causes."""
        counts = self.loss_reason_counts()
        if not counts:
            return "No losses logged yet."
        total = sum(counts.values())
        lines = [f"Total closed losses: {total}"]
        for reason in sorted(counts, key=lambda k: -counts[k]):
            pct = counts[reason] / total * 100
            lines.append(f"  - {reason} ({counts[reason]}, {pct:.0f}%): {LOSS_CATEGORIES.get(reason, '')}")
        return "\n".join(lines)

    # ---------- auto-sync from Alpaca paper account ----------
    def sync_activities(self, activities: list[dict]) -> int:
        """Turn Alpaca FILL activities into closed-trade journal entries.

        Groups fills per symbol and logs a round-trip using `net_amount`
        (realised P/L) when a sell fully closes a side, so your paper
        account's wins and losses feed Cortex's learning automatically.
        Returns the number of new entries created.
        """
        seen = set(self.load_strategy_state("synced_activity_ids", []))
        fills = [
            a for a in activities
            if a.get("activity_type", "").upper() == "FILL" and a.get("id")
        ]
        fills.sort(key=lambda a: a.get("transaction_time", ""))
        open_pos: dict[str, dict] = {}
        new_entries = 0
        for a in fills:
            aid = a.get("id", "")
            if aid in seen:
                continue
            sym = a.get("symbol", "")
            side = (a.get("side") or "").lower()
            qty = float(a.get("qty") or 0)
            price = float(a.get("price") or 0)
            ts = a.get("transaction_time", "")
            if side == "buy":
                open_pos[sym] = {"price": price, "qty": qty, "ts": ts}
            elif side == "sell" and sym in open_pos:
                entry = open_pos.pop(sym)
                realized = float(a.get("net_amount") or 0)
                pnl_pct = (price / entry["price"] - 1) * 100 if entry["price"] else 0.0
                risk = abs(entry["price"] - price) * qty
                r_mult = round(realized / risk, 2) if risk else 0.0
                self.log_trade(
                    JournalEntry(
                        symbol=sym,
                        entry_price=entry["price"],
                        exit_price=price,
                        qty=qty,
                        direction="long",
                        entry_time=entry["ts"],
                        exit_time=ts,
                        exit_reason="manual",
                        pnl=realized,
                        pnl_pct=pnl_pct,
                        r_multiple=r_mult,
                    )
                )
                new_entries += 1
            if aid:
                seen.add(aid)
        self.save_strategy_state("synced_activity_ids", sorted(seen))
        return new_entries

    def lessons(self) -> list[str]:
        """Adaptive rules derived from historical loss patterns."""
        lessons: list[str] = []
        counts = self.loss_reason_counts()
        total = len(self.losses())
        if total == 0:
            return lessons
        if counts.get("chase", 0) / max(1, total) >= 0.3:
            lessons.append("Avoid chasing: require a 4h pullback (RSI<=threshold) or a confirmed breakout; never buy extension.")
        if counts.get("no_stop", 0) / max(1, total) >= 0.25:
            lessons.append("Never enter without a stop; always size so the stop = 1R from entry.")
        if counts.get("wrong_trend", 0) / max(1, total) >= 0.2:
            lessons.append("Only take longs aligned with 1w/1d uptrend; respect higher-timeframe direction.")
        if counts.get("news_risk", 0) / max(1, total) >= 0.15:
            lessons.append("Halt entries within 2 days of scheduled earnings/news.")
        if counts.get("premature", 0) / max(1, total) >= 0.25:
            lessons.append("Widen stops: use ATR-based stops instead of tight price stops.")
            lessons.append("Consider raising the ATR stop multiplier to avoid premature exits.")
        if counts.get("market_regime", 0) / max(1, total) >= 0.25:
            lessons.append("In a broad market downturn, reduce position count and size; prefer cash/short bias.")
        if counts.get("overtrade", 0) / max(1, total) >= 0.2:
            lessons.append("Cap the number of simultaneous open positions to control overtrading.")
        return lessons

    # ---------- strategy state persistence ----------
    def save_strategy_state(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO strategy_state (key, value) VALUES (?,?)",
            (key, json.dumps(value)),
        )
        self._conn.commit()

    def load_strategy_state(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value FROM strategy_state WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def close(self) -> None:
        self._conn.close()