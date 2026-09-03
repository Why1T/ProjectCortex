"""Local Streamlit dashboard for Cortex."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
import os

import streamlit as st
import streamlit.components.v1 as components


def load_cloud_secrets() -> None:
    """Expose Streamlit secrets through the same environment settings as local runs."""
    aliases = {
        "ALPACA_API_KEY": ("ALPACA_API_KEY", "APCA_API_KEY_ID"),
        "ALPACA_API_SECRET": ("ALPACA_API_SECRET", "APCA_API_SECRET_KEY"),
        "ALPACA_PAPER": ("ALPACA_PAPER",),
        "CORTEX_LLM_API_KEY": ("CORTEX_LLM_API_KEY",),
        "CORTEX_LLM_BASE_URL": ("CORTEX_LLM_BASE_URL",),
        "CORTEX_LLM_MODEL": ("CORTEX_LLM_MODEL",),
        "CORTEX_LLM_TEMPERATURE": ("CORTEX_LLM_TEMPERATURE",),
    }
    for setting_name, secret_names in aliases.items():
        if os.environ.get(setting_name):
            continue
        for secret_name in secret_names:
            try:
                value = st.secrets.get(secret_name)
            except Exception:
                value = None
            if value:
                os.environ[setting_name] = str(value)
                break


load_cloud_secrets()

from cortex.analysis.advisor import Advisor
from cortex.analysis.scanner import default_universe


st.set_page_config(
    page_title="Cortex | Market Desk",
    page_icon="◒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink:#17211d; --muted:#66736d; --paper:#f4f6f1; --line:#dce3dc; --lime:#d8f36b; --coral:#ff8067; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background:#1d2924; border-right:0; }
    [data-testid="stSidebar"] * { color:#f4f6f1 !important; }
    h1,h2,h3,h4,p,span,div { font-family:'Space Grotesk', sans-serif; }
    code, .mono { font-family:'DM Mono', monospace; }
    h1 { font-size:2.7rem !important; letter-spacing:0 !important; line-height:1.05 !important; }
    h2 { margin-top:1.2rem !important; }
    .eyebrow { color:#617067; font-family:'DM Mono',monospace; font-size:.75rem; letter-spacing:.08em; text-transform:uppercase; }
    .hero { padding:1.4rem 0 1rem; border-bottom:1px solid var(--line); margin-bottom:1.25rem; }
    .hero p { color:var(--muted); font-size:1.05rem; margin:0; }
    .signal-card { background:#fff; border:1px solid var(--line); border-left:5px solid var(--lime); padding:1.15rem 1.25rem; margin:.65rem 0; box-shadow:0 5px 18px rgba(23,33,29,.04); }
    .signal-card.wait { border-left-color:var(--coral); }
    .signal-title { font-size:1.3rem; font-weight:700; }
    .tag { display:inline-block; background:#eaf1e6; border-radius:999px; padding:.2rem .55rem; margin-left:.4rem; font-family:'DM Mono',monospace; font-size:.7rem; text-transform:uppercase; }
    .why { color:#405048; line-height:1.5; margin:.65rem 0; }
    .risk { color:#8b4c3d; font-size:.88rem; margin:.2rem 0; }
    .stMetric { background:#fff; border:1px solid var(--line); padding:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_advisor() -> Advisor:
    return Advisor()


@st.cache_data(ttl=300, show_spinner=False)
def load_report(symbols: tuple[str, ...], deep_analysis: bool):
    return get_advisor().get_daily_look(symbols=list(symbols), deep_analysis=deep_analysis)


def money(value: float | None) -> str:
    return f"${value:,.2f}" if value is not None else "—"


def render_tradingview(symbol: str, interval: str) -> None:
        """Render TradingView's public interactive chart widget."""
        components.html(
                f"""
                <div class="tradingview-widget-container" style="height:610px;width:100%">
                    <div id="tradingview_{symbol.lower()}" style="height:100%;width:100%"></div>
                    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                    <script type="text/javascript">
                        new TradingView.widget({{
                            "autosize": true,
                            "symbol": "NASDAQ:{symbol}",
                            "interval": "{interval}",
                            "timezone": "exchange",
                            "theme": "light",
                            "style": "1",
                            "locale": "en",
                            "toolbar_bg": "#f4f6f1",
                            "enable_publishing": false,
                            "hide_top_toolbar": false,
                            "hide_legend": false,
                            "save_image": false,
                            "container_id": "tradingview_{symbol.lower()}"
                        }});
                    </script>
                </div>
                """,
                height=620,
        )


def render_candidate(candidate: dict) -> None:
    signal = candidate["signal"]
    verdict = candidate.get("verdict") or "wait"
    card_class = "signal-card" if verdict == "advise" else "signal-card wait"
    position = candidate.get("position")
    sizing = f"{position.qty:g} shares / {money(position.dollar_amount)}" if position else "No position plan"
    st.markdown(
        f"""
        <div class="{card_class}">
          <div class="signal-title">{signal.symbol} <span class="tag">{signal.direction}</span><span class="tag">{verdict}</span></div>
          <div class="mono">Entry {money(signal.entry)} &nbsp; Stop {money(signal.stop_loss)} &nbsp; Target {money(signal.take_profit)} &nbsp; Score {signal.score:.1f}</div>
          <div class="why"><strong>Why Cortex likes it:</strong> {candidate.get('bull_case') or candidate.get('deep_reason') or ' '.join(signal.reasons)}</div>
          <div><strong>Plan:</strong> {sizing}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    risks = candidate.get("risks") or []
    if risks:
        with st.expander(f"Risks and reasoning for {signal.symbol}"):
            for risk in risks:
                st.markdown(f"<div class='risk'>• {risk}</div>", unsafe_allow_html=True)
            if candidate.get("deep_reason"):
                st.write(candidate["deep_reason"])
            if candidate.get("sources"):
                st.caption("Sources: " + " · ".join(candidate["sources"]))


def main() -> None:
    st.sidebar.markdown("## ◒ CORTEX")
    st.sidebar.caption("Personal market desk")
    symbols_text = st.sidebar.text_input("Universe", ", ".join(default_universe()))
    symbols = tuple(symbol.strip().upper() for symbol in symbols_text.split(",") if symbol.strip())
    deep_analysis = st.sidebar.toggle("LLM deep reasoning", value=True)
    if st.sidebar.button("Refresh analysis", type="primary", use_container_width=True):
        load_report.clear()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption("Research and advisory only. Cortex never places orders.")

    st.markdown('<div class="hero"><div class="eyebrow">Personal market desk · ' + datetime.now().strftime("%d %b %Y") + '</div><h1>Cortex overview</h1><p>Signals, risk, and reasoning in one quiet workspace.</p></div>', unsafe_allow_html=True)
    with st.spinner("Scanning markets and assembling the desk..."):
        report = load_report(symbols, deep_analysis)

    candidates = report.candidates
    advised = sum(1 for candidate in candidates if candidate.get("verdict") == "advise")
    data_source = get_advisor().data.source
    st.caption(f"Last analysis: {report.generated_at} · Data: {data_source}")
    if report.notes:
        for note in report.notes:
            st.error(note)
    metrics = st.columns(4)
    metrics[0].metric("Account equity", money(report.equity))
    metrics[1].metric("Buying power", money(report.buying_power))
    metrics[2].metric("Trade ideas", len(candidates))
    metrics[3].metric("Advise", advised)

    overview, ideas, positions, research = st.tabs(["Overview", "Trade ideas", "Positions", "Research"])
    with overview:
        left, right = st.columns([1.35, 1])
        with left:
            st.subheader("What stands out")
            if candidates:
                for candidate in candidates[:5]:
                    render_candidate(candidate)
            else:
                st.info("No candidates cleared the current strategy threshold.")
        with right:
            st.subheader("Watchlists")
            st.write("**This week**")
            st.write(" · ".join(report.watchlist_week) or "Nothing yet")
            st.write("**This month**")
            st.write(" · ".join(report.watchlist_month) or "Nothing yet")
            st.subheader("Strategy health")
            st.metric("Status", report.strategy_health)
            st.caption(f"Expectancy: {report.backtest.get('before', {}).get('expectancy_r', 0)} R · Profit factor: {report.backtest.get('before', {}).get('profit_factor', 0):.2f}")
            for note in report.notes:
                st.warning(note)
    with ideas:
        st.subheader("All current trade ideas")
        for candidate in candidates:
            render_candidate(candidate)
    with positions:
        st.subheader("Paper account")
        if report.positions:
            st.dataframe(report.positions, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions reported.")
    with research:
        st.subheader("Live TradingView chart")
        chart_symbol = st.selectbox("Chart symbol", options=list(symbols) or ["AAPL"])
        chart_interval = st.selectbox("Chart timeframe", options=["15", "60", "240", "D", "W", "M"], index=2)
        render_tradingview(chart_symbol, chart_interval)
        st.subheader("Recent market events")
        events = report.events_brief or {}
        if events.get("summary"):
            st.write(events["summary"])
        for headline in (events.get("headlines") or [])[:8]:
            st.markdown(f"**{headline.get('title')}**  \\  :gray[{headline.get('source')} · {headline.get('sentiment')}]" )
        st.subheader("Backtest")
        backtest = report.backtest.get("before", {})
        st.write(f"{backtest.get('total_trades', 0)} trades across {backtest.get('total_symbols', 0)} symbols")
        st.json({key: backtest.get(key) for key in ("expectancy_r", "profit_factor", "profitable_symbols")})


def launch() -> int:
    """Start this dashboard through Streamlit's local web server."""
    environment = os.environ.copy()
    environment["CORTEX_STREAMLIT_CHILD"] = "1"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__)), "--server.headless=true"],
        env=environment,
    )


if __name__ == "__main__" and not os.environ.get("CORTEX_STREAMLIT_CHILD"):
    main()
