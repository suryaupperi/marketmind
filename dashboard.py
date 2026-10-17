"""Interactive RSI/Bollinger swing-trading dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marketmind.backtest import run_swing_backtest, swing_positions
from marketmind.data import load_prices
from marketmind.features import build_features
from marketmind.metrics import calculate_metrics
from marketmind.strategies import generate_signals
from marketmind.tickers import MAX_TICKERS, parse_tickers

STRATEGIES = ["RSI", "Bollinger Bands"]
PERIODS = {
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
    "10 years": "10y",
    "Maximum available": "max",
}
DEFAULT_TICKERS = "NVDA, AAPL, MSFT, SPY"

st.set_page_config(
    page_title="MarketMind | Strategy Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --mm-bg:#0a1020; --mm-panel:#111a2d; --mm-line:#26344c; --mm-muted:#91a2bd; --mm-text:#eef4ff; --mm-blue:#69a9ff; --mm-green:#54d6a0; --mm-red:#ff7188; }
      .stApp { background: radial-gradient(ellipse at 10% 0%, #142542 0%, #0a1020 46%); color:var(--mm-text); }
      html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
      [data-testid="stSidebar"] { background:#0c1425; border-right:1px solid #26344c; }
      [data-testid="stMetric"] { background:linear-gradient(145deg,#141f34,#10192b); border:1px solid #26344c; border-radius:15px; padding:16px 18px; }
      [data-testid="stMetricLabel"] { color:#91a2bd; font-size:0.78rem; }
      [data-testid="stMetricValue"] { color:#eef4ff; font-weight:800; }
      .mm-eyebrow { color:#69a9ff; letter-spacing:.16em; font-size:.72rem; font-weight:800; text-transform:uppercase; }
      .mm-title { font-size:2.25rem; font-weight:800; letter-spacing:-.055em; line-height:1.1; margin:.25rem 0 .6rem; color:#f4f7ff; }
      .mm-subtitle { color:#a5b4cc; font-size:.98rem; max-width:800px; }
      .mm-pill { display:inline-block; padding:5px 10px; margin:3px 5px 3px 0; border:1px solid #2a4567; border-radius:99px; color:#b9d8ff; background:#132641; font-size:.75rem; }
      div[data-testid="stDataFrame"] { border:1px solid #26344c; border-radius:12px; overflow:hidden; }
      .stTabs [data-baseweb="tab-list"] { gap:8px; }
      .stTabs [data-baseweb="tab"] { background:#111a2d; border:1px solid #26344c; border-radius:10px 10px 0 0; padding:9px 16px; }
      .stTabs [aria-selected="true"] { background:#193153 !important; }
      hr { border-color:#26344c; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner="Downloading adjusted market prices…")
def load_dashboard_prices(ticker: str, period: str) -> pd.DataFrame:
    """Cache downloaded prices by ticker and period."""
    return load_prices(ticker=ticker, period=period)


def add_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI and Bollinger indicators for chart display."""
    df = prices.copy()
    close = df["Close"]

    change = close.diff()
    gain = change.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-change.clip(upper=0)).rolling(14, min_periods=14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    df["RSI"] = (100 - 100 / (1 + relative_strength)).where(
        loss.ne(0), 100.0
    )

    df["BB_Middle"] = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    df["BB_Upper"] = df["BB_Middle"] + 2 * std
    df["BB_Lower"] = df["BB_Middle"] - 2 * std
    return df


def completed_trade_returns(daily: pd.DataFrame) -> list[float]:
    """Compound returns for completed contiguous long positions."""
    if daily.empty:
        return []

    positions = daily["signal"].fillna(0).astype(int).to_numpy()
    returns = daily["strategy_return"].fillna(0).astype(float).to_numpy()
    completed: list[float] = []
    i = 0

    while i < len(positions):
        if positions[i] != 1:
            i += 1
            continue

        start = i
        while i + 1 < len(positions) and positions[i + 1] == 1:
            i += 1

        end = i
        if end + 1 < len(positions) and positions[end + 1] == 0:
            completed.append(
                float(np.prod(1 + returns[start : end + 1]) - 1)
            )
        i += 1

    return completed


def analyze_ticker(
    ticker: str,
    period: str,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Run both strategies on a chronological holdout for one ticker."""
    prices = load_dashboard_prices(ticker, period)

    if len(prices) < 100:
        raise ValueError(
            f"Only {len(prices)} price rows were returned. "
            "Choose a longer history."
        )

    features = build_features(prices)
    if len(features) < 80:
        raise ValueError("Not enough usable history after calculating features.")

    split_at = max(1, int(len(features) * 0.60))
    test_dates = features.index[split_at:]

    if len(test_dates) < 10:
        raise ValueError("Not enough out-of-sample sessions for this ticker.")

    test_start = test_dates[0]
    cost_per_side = cost_bps / 10_000
    price_frame = add_indicators(prices)
    summaries: list[dict[str, object]] = []
    daily_by_strategy: dict[str, pd.DataFrame] = {}

    for strategy in STRATEGIES:
        signals = generate_signals(
            prices,
            strategy,
            start_date=test_start,
        )
        signals = signals.where(
            signals.index.isin(test_dates),
            0,
        ).astype(int)

        strategy_returns, _ = run_swing_backtest(
            prices,
            signals,
            cost_per_side=cost_per_side,
            liquidate_at_end=True,
        )
        positions = swing_positions(signals, liquidate_at_end=True)
        open_to_open = prices["Open"].shift(-1) / prices["Open"] - 1
        signal_dates = pd.Series(
            prices.index,
            index=prices.index,
        ).shift(1)

        daily = pd.DataFrame({
            "date": prices.index,
            "ticker": ticker,
            "strategy": strategy,
            "signal_date": signal_dates.to_numpy(),
            "signal": positions.to_numpy(),
            "strategy_return": strategy_returns.to_numpy(),
            "buy_hold_return": open_to_open.to_numpy(),
        })

        daily = daily[
            daily["signal_date"].isin(test_dates)
        ].copy().sort_values("date")
        daily_for_trades = daily.copy()
        valid = daily[daily["strategy_return"].notna()].copy()

        if valid.empty:
            raise ValueError("No usable out-of-sample returns for this ticker.")

        # Apply benchmark entry and liquidation costs.
        valid.loc[valid.index[0], "buy_hold_return"] -= cost_per_side
        valid.loc[valid.index[-1], "buy_hold_return"] -= cost_per_side

        strategy_metrics = calculate_metrics(valid["strategy_return"])
        benchmark_metrics = calculate_metrics(valid["buy_hold_return"])
        trades = completed_trade_returns(daily_for_trades)
        previous = valid["signal"].shift(1, fill_value=0)
        trade_count = int(
            ((valid["signal"] == 1) & (previous == 0)).sum()
        )

        summaries.append({
            "ticker": ticker,
            "strategy": strategy,
            "period": period,
            "test_start": str(valid["date"].min().date()),
            "test_end": str(valid["date"].max().date()),
            "oos_days": len(valid),
            "active_days": int((valid["signal"] == 1).sum()),
            "trade_count": trade_count,
            "completed_trade_win_rate": (
                float(np.mean(np.asarray(trades) > 0))
                if trades else np.nan
            ),
            "average_trade_return": (
                float(np.mean(trades)) if trades else np.nan
            ),
            **{
                f"strategy_{key}": value
                for key, value in strategy_metrics.items()
            },
            "buy_hold_return": benchmark_metrics["total_return"],
            "buy_hold_sharpe": benchmark_metrics["sharpe"],
        })
        daily_by_strategy[strategy] = valid

    return pd.DataFrame(summaries), daily_by_strategy, price_frame


def pct(value: object, decimals: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{decimals}%}"


def number(value: object, decimals: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{decimals}f}"


def make_price_chart(
    df: pd.DataFrame,
    strategy: str,
    daily: pd.DataFrame,
    chart_period: str,
) -> go.Figure:
    period_rows = {
        "3M": 63,
        "6M": 126,
        "1Y": 252,
        "2Y": 504,
        "5Y": 1260,
        "10Y": 2520,
        "Max": len(df),
    }
    view = df.tail(min(period_rows[chart_period], len(df))).copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.72, 0.28],
        subplot_titles=(
            "Adjusted close & trade signals",
            "RSI · 14 sessions",
        ),
    )

    if strategy == "Bollinger Bands":
        fig.add_trace(
            go.Scatter(
                x=view.index,
                y=view["BB_Upper"],
                name="Upper band",
                line=dict(color="#6889b8", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=view.index,
                y=view["BB_Lower"],
                name="Lower band",
                line=dict(color="#6889b8", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(105,169,255,0.09)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=view.index,
                y=view["BB_Middle"],
                name="20-day mean",
                line=dict(color="#f4c76b", width=1.4),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["Close"],
            name="Adjusted close",
            line=dict(color="#69a9ff", width=2.2),
            hovertemplate="Close: $%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if not daily.empty:
        markers = daily.sort_values("date").copy()
        markers["previous_signal"] = markers["signal"].shift(1).fillna(0)
        entries = markers[
            (markers["signal"] == 1)
            & (markers["previous_signal"] == 0)
        ]
        exits = markers[
            (markers["signal"] == 0)
            & (markers["previous_signal"] == 1)
        ]

        open_by_date = {
            pd.Timestamp(index).date(): float(value)
            for index, value in view["Open"].items()
        }

        if not entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries["date"],
                    y=entries["date"].map(
                        lambda d: open_by_date.get(
                            pd.Timestamp(d).date(), np.nan
                        )
                    ),
                    mode="markers",
                    name="Long entry",
                    marker=dict(
                        symbol="triangle-up",
                        size=11,
                        color="#54d6a0",
                    ),
                ),
                row=1,
                col=1,
            )

        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits["date"],
                    y=exits["date"].map(
                        lambda d: open_by_date.get(
                            pd.Timestamp(d).date(), np.nan
                        )
                    ),
                    mode="markers",
                    name="Exit to cash",
                    marker=dict(
                        symbol="triangle-down",
                        size=10,
                        color="#ff7188",
                    ),
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["RSI"],
            name="RSI",
            line=dict(color="#c4a1ff", width=1.8),
        ),
        row=2,
        col=1,
    )

    for level, color, label in [
        (30, "#54d6a0", "Oversold · 30"),
        (50, "#91a2bd", "Midline · 50"),
        (70, "#ff7188", "Overbought · 70"),
    ]:
        fig.add_hline(
            y=level,
            line_dash="dot",
            line_color=color,
            opacity=0.65,
            annotation_text=label,
            annotation_position="bottom right",
            row=2,
            col=1,
        )

    fig.update_yaxes(
        title_text="Price",
        row=1,
        col=1,
        gridcolor="rgba(145,162,189,.12)",
        tickprefix="$",
        tickformat=".0f",
    )
    fig.update_yaxes(
        title_text="RSI",
        row=2,
        col=1,
        range=[0, 100],
        gridcolor="rgba(145,162,189,.12)",
    )
    fig.update_xaxes(
        gridcolor="rgba(145,162,189,.10)",
        rangeslider_visible=False,
    )
    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin=dict(l=12, r=18, t=58, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Manrope, sans-serif",
            color="#dce7f8",
            size=11,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
    )
    return fig


def make_equity_chart(daily: pd.DataFrame) -> go.Figure:
    frame = daily.sort_values("date").copy()
    frame["strategy_equity"] = (
        1 + frame["strategy_return"].fillna(0)
    ).cumprod()
    frame["buy_hold_equity"] = (
        1 + frame["buy_hold_return"].fillna(0)
    ).cumprod()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["strategy_equity"],
            name="Strategy",
            line=dict(color="#69a9ff", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["buy_hold_equity"],
            name="Buy & hold",
            line=dict(color="#54d6a0", width=1.8),
        )
    )
    fig.add_hline(
        y=1,
        line_color="#91a2bd",
        line_dash="dot",
        opacity=0.5,
    )
    fig.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Manrope, sans-serif",
            color="#dce7f8",
            size=11,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
        yaxis=dict(
            title="Growth of $1",
            tickprefix="$",
            gridcolor="rgba(145,162,189,.12)",
        ),
        xaxis=dict(gridcolor="rgba(145,162,189,.10)"),
    )
    return fig


def make_drawdown_chart(daily: pd.DataFrame) -> go.Figure:
    frame = daily.sort_values("date").copy()
    equity = (1 + frame["strategy_return"].fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1

    fig = go.Figure(
        go.Scatter(
            x=frame["date"],
            y=drawdown,
            name="Drawdown",
            line=dict(color="#ff7188", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(255,113,136,.14)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=10, r=10, t=18, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Manrope, sans-serif",
            color="#dce7f8",
            size=11,
        ),
        yaxis=dict(
            title="Drawdown",
            tickformat=".0%",
            gridcolor="rgba(145,162,189,.12)",
        ),
        xaxis=dict(gridcolor="rgba(145,162,189,.10)"),
    )
    return fig


def display_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "strategy",
        "strategy_total_return",
        "strategy_annualized_return",
        "strategy_sharpe",
        "strategy_max_drawdown",
        "completed_trade_win_rate",
        "trade_count",
        "buy_hold_return",
    ]
    result = frame[columns].copy()
    result = result.rename(columns={
        "ticker": "Ticker",
        "strategy": "Strategy",
        "strategy_total_return": "Total return",
        "strategy_annualized_return": "Annualized return",
        "strategy_sharpe": "Sharpe",
        "strategy_max_drawdown": "Max drawdown",
        "completed_trade_win_rate": "Trade win rate",
        "trade_count": "Trades",
        "buy_hold_return": "Buy & hold return",
    })

    for column in [
        "Total return",
        "Annualized return",
        "Max drawdown",
        "Trade win rate",
        "Buy & hold return",
    ]:
        result[column] = result[column].map(pct)

    result["Sharpe"] = result["Sharpe"].map(number)
    result["Trades"] = result["Trades"].map(
        lambda value: "—" if pd.isna(value) else str(int(value))
    )
    return result


st.markdown(
    '<div class="mm-eyebrow">MarketMind · Quantitative research</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="mm-title">Strategy Lab'
    '<span style="color:#69a9ff">.</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="mm-subtitle">Test RSI and Bollinger Bands on '
    'user-selected tickers, with matched swing-trading benchmarks '
    'and out-of-sample results.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<span class='mm-pill'>User-selected tickers</span>"
    "<span class='mm-pill'>Out-of-sample evaluation</span>"
    "<span class='mm-pill'>Configurable costs</span>"
    "<span class='mm-pill'>Swing positions</span>",
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.markdown("### Analysis controls")
    ticker_input = st.text_input(
        "Ticker symbols (comma-separated)",
        value=DEFAULT_TICKERS,
        help=(
            f"Examples: TSLA, AMD, GOOGL, BRK-B, SPY. "
            f"Up to {MAX_TICKERS} symbols per run."
        ),
    )
    period_label = st.selectbox(
        "Historical price data",
        list(PERIODS.keys()),
        index=2,
    )
    cost_bps = st.number_input(
        "Transaction cost per side (basis points)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
        help="5 bps = 0.05% per entry or exit.",
    )
    run_clicked = st.button(
        "Run analysis",
        type="primary",
        use_container_width=True,
    )
    st.markdown("---")
    st.markdown("**Strategy rules**")
    st.markdown(
        "**RSI:** enter below 30; exit above 50.\n\n"
        "**Bollinger Bands:** enter when close crosses back above "
        "the lower band; exit at the 20-session mean."
    )
    st.caption(
        "Signals use the close and execute at the next open. "
        "Swing positions hold across sessions. Historical results "
        "are estimates, not guarantees."
    )

if "marketmind_request" not in st.session_state:
    st.session_state.marketmind_request = {
        "tickers": ["NVDA", "AAPL", "MSFT", "SPY"],
        "period": "5y",
        "cost_bps": 5.0,
    }

if run_clicked:
    try:
        st.session_state.marketmind_request = {
            "tickers": parse_tickers(ticker_input),
            "period": PERIODS[period_label],
            "cost_bps": float(cost_bps),
        }
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

request = st.session_state.marketmind_request
if not request["tickers"]:
    st.info("Enter ticker symbols in the sidebar and click Run analysis.")
    st.stop()

all_summaries: list[pd.DataFrame] = []
daily_by_ticker: dict[str, dict[str, pd.DataFrame]] = {}
prices_by_ticker: dict[str, pd.DataFrame] = {}
failures: dict[str, str] = {}

with st.spinner("Running out-of-sample swing backtests for selected tickers…"):
    for symbol in request["tickers"]:
        try:
            ticker_summary, ticker_daily, price_frame = analyze_ticker(
                symbol,
                request["period"],
                request["cost_bps"],
            )
            all_summaries.append(ticker_summary)
            daily_by_ticker[symbol] = ticker_daily
            prices_by_ticker[symbol] = price_frame
        except Exception as exc:
            failures[symbol] = str(exc)

for symbol, message in failures.items():
    st.warning(f"{symbol}: {message}")

if not all_summaries:
    st.error(
        "No tickers could be analyzed. Check the symbols and try a longer history."
    )
    st.stop()

summary = pd.concat(all_summaries, ignore_index=True)
valid_tickers = list(daily_by_ticker)

selected_ticker = st.selectbox(
    "Ticker to inspect",
    valid_tickers,
    index=0,
)
selected_strategy = st.radio(
    "Strategy",
    STRATEGIES,
    horizontal=True,
)
chart_period = st.select_slider(
    "Price chart range",
    options=["3M", "6M", "1Y", "2Y", "5Y", "10Y", "Max"],
    value="1Y",
)

selected_summary = summary[
    (summary["ticker"] == selected_ticker)
    & (summary["strategy"] == selected_strategy)
].iloc[0]
selected_daily = daily_by_ticker[selected_ticker][selected_strategy].copy()
selected_prices = prices_by_ticker[selected_ticker]

st.markdown(f"### {selected_ticker} · {selected_strategy}")

metric_cols = st.columns(5)
metric_values = [
    ("Out-of-sample return", pct(selected_summary["strategy_total_return"])),
    ("Annualized return", pct(selected_summary["strategy_annualized_return"])),
    ("Sharpe ratio", number(selected_summary["strategy_sharpe"])),
    ("Maximum drawdown", pct(selected_summary["strategy_max_drawdown"])),
    ("Completed-trade win rate", pct(selected_summary["completed_trade_win_rate"])),
]

for col, (label, value) in zip(metric_cols, metric_values):
    col.metric(label, value)

meta = st.columns(4)
meta[0].caption(f"**Test start:** {selected_summary['test_start']}")
meta[1].caption(f"**Test end:** {selected_summary['test_end']}")
meta[2].caption(f"**Test sessions:** {int(selected_summary['oos_days'])}")
meta[3].caption(f"**Entries:** {int(selected_summary['trade_count'])}")

st.markdown("### Price action & indicators")
st.plotly_chart(
    make_price_chart(
        selected_prices,
        selected_strategy,
        selected_daily,
        chart_period,
    ),
    use_container_width=True,
)
st.caption(
    f"Historical data period: {request['period']} · "
    f"Transaction cost: {request['cost_bps']:g} bps per side · "
    "Metrics use the final 40% of usable feature dates as a chronological holdout."
)

st.markdown("### Equity curve & drawdown")
left, right = st.columns([1.6, 1])

with left:
    st.plotly_chart(
        make_equity_chart(selected_daily),
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        make_drawdown_chart(selected_daily),
        use_container_width=True,
    )

st.markdown("### Strategy comparison")
st.dataframe(
    display_summary(summary[summary["ticker"] == selected_ticker]),
    hide_index=True,
    use_container_width=True,
)

st.markdown("### All selected tickers")
st.dataframe(
    display_summary(summary),
    hide_index=True,
    use_container_width=True,
)

with st.expander("Methodology & limitations"):
    st.markdown(
        """
        - **Ticker symbols:** Enter up to 10 Yahoo Finance symbols separated
          by commas. Examples include `TSLA`, `AMD`, `GOOGL`, `BRK-B`,
          and `SPY`. Availability depends on the data provider.
        - **Out-of-sample split:** The first 60% of usable feature dates is
          excluded from performance scoring; the final 40% is evaluated
          chronologically.
        - **Execution:** A signal generated after close is applied at the next
          session's open. Returns are measured open-to-open.
        - **Costs:** The selected cost is charged per entry and exit.
          Buy-and-hold uses matching open-to-open intervals and assumed
          entry and liquidation costs.
        - **Risk:** Sharpe is annualized using 252 sessions. Drawdown is
          computed from compounded strategy returns. Completed-trade win
          rate excludes trades still open at the end of the test window.
        - **Limitations:** Taxes, market impact, borrow costs, and actual
          broker fills are not modeled. Historical results do not guarantee
          future performance.
        """
    )

st.markdown("---")
st.caption("MarketMind · Quantitative research dashboard · Educational use only")