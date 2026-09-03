# Cortex — Swing Trading Research & Advisory Bot

Cortex is an Alpaca-backed swing trading **research and advisory** bot. It studies the
market on multiple timeframes, backtests its own strategy, tracks your paper account,
maintains a trade journal from which it learns, and guides YOU — it **never places a
trade**. All trading decisions are yours.

---

## What Cortex does

| Capability | Details |
|---|---|
| **Multi-timeframe study** | Pulls OHLCV on **15m, 1h, 4h, 1d, 1w, 1mo** (Alpaca IEX / The Investor Exchange). |
| **4h-first swing logic** | The **4-hour chart is the primary entry timeframe**. Higher timeframes set direction; 1h/15m fine-tune entries. |
| **Trade guidance** | Scans a liquid universe, scores candidates, and reports **entry, stop-loss, take-profit, suggested size, reasoning, and sources**. |
| **Fixed risk** | Position sizing caps risk at 1% of equity per trade (configurable). Stops are ATR-based, targets are R-multiple based. |
| **Backtesting** | Replays historical bars through the same `score_entry()` used live — what you're advised is what gets validated. |
| **Self-optimisation** | If backtests come back negative, Cortex tunes `StrategyParams` (RSI thresholds, ATR stop/target multipliers, score floor) to find positive-expectancy settings, then persists them. |
| **Loss learning** | Every closed trade in your paper account can be logged with a loss reason. Cortex tracks loss *causes* and converts recurring patterns into adaptive "lessons". |
| **News & social (X)** | Fetches no-key news (BBC, MarketWatch, Reuters) with deterministic sentiment — LLM-upgraded when configured. Social posts are pluggable. |
| **Reports** | Generates `reports/cortex_daily_YYYY-MM-DD.md`, watchlist and backtest reports. |
| **LLM deep reasoning** | When an API key is configured, an LLM produces concrete per-trade reasoning with sources and a verdict. |

---

## Quick start

```bash
# 1. Install
pip3 install -e .            # or: pip3 install -r requirements.txt

# 2. Configure
cp .env.example .env         # then edit .env with your Alpaca paper keys + optional LLM key

# 3. Generate your daily advisory (full deep analysis)
cortex daily

# 4. Quick single-symbol deep dive
cortex symbol NVDA

# 5. Weekly/monthly watchlists
cortex watchlist

# 6. Backtest + auto self-optimisation
cortex backtest

# 7. News brief + social sentiment
cortex news

# 8. Trade journal (log a closed trade, then view lessons)
cortex journal --show
cortex journal --add --symbol AAPL --entry 150 --exit 158 \
    --qty 100 --exit-reason tp --r 2.0 --loss-reason unknown
```

After editing `pyproject.toml`, rerun `pip install -e .` once so the `cortex` command is installed.

### Streamlit dashboard

On Streamlit Cloud, set the app's main file to `streamlit_app.py`. Add these values under
**Settings → Secrets** using TOML syntax:

```toml
ALPACA_API_KEY = "your_paper_key"
ALPACA_API_SECRET = "your_paper_secret"
ALPACA_PAPER = "true"
```

The dashboard reads these secrets for the account equity, buying power, and paper positions.
The **Research** tab includes an interactive TradingView chart; it is a public chart widget
and does not display your TradingView account or private broker information.

The **LLM hub** tab provides a conversation with the configured model. It remembers the
conversation during the current dashboard session and includes the latest Cortex trade ideas
as context. It uses the same `CORTEX_LLM_API_KEY`, base URL, model, and temperature settings
as the deep-reasoning analysis.

---

## Configuration (`config/settings.py` ← `.env`)

| Variable | Default | Purpose |
|---|---|---|
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | — | Alpaca **paper** trading credentials. |
| `ALPACA_PAPER` | `true` | Force paper environment (do not disable). |
| `CORTEX_LLM_API_KEY` | — | OpenAI-compatible key for deep analysis. |
| `CORTEX_LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint. |
| `CORTEX_LLM_MODEL` | `gpt-4o-mini` | Chat model. |
| `CORTEX_LLM_TEMPERATURE` | `0.3` | Lower = more disciplined output. |
| `CORTEX_RISK_PER_TRADE` | `0.01` | Max % of equity risked per trade. |
| `CORTEX_MAX_POSITION_PCT` | `0.20` | Max % of equity per position. |
| `CORTEX_NEWS_MAX_ITEMS` | `25` | Max news items collected. |

If Alpaca credentials are absent, Cortex falls back to **free data** (Stooq) so it still
backtests, studies, and advises. If no LLM key exists, it uses deterministic rules-based
reasoning and sentiment.

---

## How the strategy works (swing-trading method)

1. **Higher-timeframe bias (1w, 1d, 4h):** each timeframe contributes to a weighted trend
   score (20 / 15 / 12). The market is only tradable long when it agrees.
2. **4h decision:** the primary entry bar. Price must be above the 4h **EMA50** (trend
   intact).
3. **Entry styles** on the 4h chart:
   - **Pullback (preferred):** RSI cooled ≤ 55 while price pulled back to/below the 4h EMA21.
   - **Mild pullback / early entry:** price drifting under EMA21, RSI ≤ 65.
   - **Breakout:** price above EMA21, RSI < 72, MACD histogram positive.
4. **Entry precision:** 1h/15m confirmation (price above their EMA21) scores points.
5. **Risk:** stop = `entry − ATR×atr_stop_mult` (default 2×), target = `entry + ATR×atr_tp_mult`
   (default 4×), giving ~2:1 R:R on entry. Position size derived from fixed-fractional risk.

All of the above parameters live in `StrategyParams` and are tuned by the optimizer.

---

## Project layout

```
cortex/
├── main.py                 # CLI entrypoint
├── clients/
│   ├── alpaca_client.py    # Alpaca REST (paper + market data), read-only
│   ├── free_data.py        # Stooq fallback data
│   └── llm_client.py       # OpenAI-compatible deep analysis + fallback sentiment
├── data/
│   └── market_data.py      # 6-interval data facade (Alpaca → Stooq)
├── indicators/
│   └── engine.py           # SMA/EMA/RSI/MACD/ATR/Bollinger/volume vectorised
├── strategies/
│   └── swing_4h.py         # 4h-first swing strategy + shared score_entry()
├── backtest/
│   ├── engine.py           # replay engine (same score_entry as live)
│   ├── metrics.py          # win rate, expectancy, PF, drawdown, Sharpe
│   └── optimizer.py        # self-optimisation on negative results
├── journal/
│   ├── journal.py          # trade DB + loss-reason learning + lessons
│   └── risk.py             # position sizing, TP/SL derivation
├── analysis/
│   ├── scanner.py          # universe scanner
│   ├── news.py             # news + social (X) aggregator + sentiment
│   └── advisor.py          # orchestrates account, scan, backtest, lessons, news, LLM
└── reports/
    └── generator.py         # markdown reports → reports/
```

---

## Self-improvement loop

```
 Study 6 timeframes
    │
    ▼
 Backtest current StrategyParams over history
    │
    ├── healthy (expectancy > 0, PF > 1.1)  →  keep params, persist in journal
    │
    └── unhealthy  →  Optimizer searches grid, picks best expectancy/R
                     with drawdown penalty, saves new params
    │
    ▼
 Paper account / journal losses  →  loss-reason tallies  →  lessons
                           (e.g. "avoid chasing", "widen stops", "respect news")
```

Journal lessons are surfaced in every daily report so you can see what Cortex has learned.

---

## Important caveats

- **Cortex does not place orders.** It provides analysis, guidance, watchlists, and reports.
- Historical backtests are simulated on completed bars and do **not** include real spread,
  slippage, or commissions. Treat results as directional, not gospel.
- Alpaca news content requires a paid data subscription; Cortex uses free RSS by default
  and only attempts Alpaca news when a key is configured.
- This is not financial advice. Always do your own research and risk management.