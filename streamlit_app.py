"""SentiFolio - systematic multi-asset funds with news-sentiment analytics.

The deployed app only reads precomputed artifacts from results/ (written by
scripts/run_part_b.py) - it never scores sentiment or recomputes a backtest
at runtime, so the sentiment-scoring library stays out of this file.

Run locally:   streamlit run streamlit_app.py
Deploy:        push this folder to a public GitHub repo, then connect it on
               share.streamlit.io with entrypoint streamlit_app.py.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

st.set_page_config(page_title="SentiFolio", page_icon="📊", layout="wide")

# Presentation only - no data or metric logic below this point depends on it.
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }

    .senti-header {
        padding: 1.15rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #0F2942 0%, #1B4F72 100%);
        margin-bottom: 1.3rem;
    }
    .senti-header h1 {
        margin: 0; color: #FFFFFF; font-size: 2.0rem;
    }
    .senti-header p {
        margin: 0.3rem 0 0 0; color: #D7E3EC; font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background-color: #F4F6F8;
        border: 1px solid rgba(15, 41, 66, 0.12);
        border-radius: 10px;
        padding: 0.9rem 1.1rem 0.7rem 1.1rem;
        box-shadow: 0 1px 2px rgba(15, 41, 66, 0.06);
    }
    div[data-testid="stMetricLabel"] p {
        font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;
        font-size: 0.72rem; opacity: 0.7;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700; color: #0F2942;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(15, 41, 66, 0.12);
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 1.6rem; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }

    h3 { margin-top: 1.4rem; }

    /* KPI rows: every st.columns(4) metric row on this page follows the same
       return / volatility / Sharpe / drawdown order, so a fixed left-accent
       colour per column position reads as reward (blue), risk-adjusted
       reward (green), and downside risk (red) without touching any values. */
    div[data-testid="stHorizontalBlock"] div[data-testid="stMetric"] {
        border-left: 4px solid #1B4F72;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-of-type(3)
        div[data-testid="stMetric"] { border-left-color: #1E8449; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-of-type(4)
        div[data-testid="stMetric"] { border-left-color: #C0392B; }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(15, 41, 66, 0.10);
        border-radius: 10px;
        overflow: hidden;
    }

    hr { border-top: 1px solid rgba(15, 41, 66, 0.12); margin: 1.6rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=86_400, show_spinner="Loading fund results...")
def _load_results():
    fund_returns = pd.read_csv(RESULTS / "data" / "fund_returns.csv", parse_dates=["date"])
    fund_weights = pd.read_csv(RESULTS / "data" / "fund_weights.csv", parse_dates=["date"])
    sector_sentiment = pd.read_csv(RESULTS / "data" / "sector_sentiment_index.csv", parse_dates=["date"])
    performance = pd.read_csv(RESULTS / "tables" / "performance_metrics.csv")
    return fund_returns, fund_weights, sector_sentiment, performance


try:
    fund_returns, fund_weights, sector_sentiment, performance = _load_results()
except FileNotFoundError:
    st.error("No precomputed results found. Run `python scripts/run_part_b.py` first "
             "to populate results/data and results/tables.")
    st.stop()

ALL_FUNDS = performance["fund"].tolist()


def growth_of_dollar(fund_name: str) -> pd.Series:
    r = fund_returns.loc[fund_returns["fund"] == fund_name].sort_values("date").set_index("date")["ret"]
    return (1 + r).cumprod()


def max_drawdown_series(growth: pd.Series) -> pd.Series:
    return growth / growth.cummax() - 1


def current_holdings(fund_name: str, top_n: int = 10) -> pd.DataFrame:
    w = fund_weights.loc[fund_weights["fund"] == fund_name]
    last_date = w["date"].max()
    latest = (w.loc[w["date"] == last_date, ["ticker", "weight"]]
                .sort_values("weight", ascending=False))
    latest = latest.loc[latest["weight"] > 1e-4].head(top_n)
    latest["weight"] = (latest["weight"] * 100).round(2)
    return latest.rename(columns={"weight": "weight (%)"}), last_date


st.markdown(
    '<div class="senti-header"><h1>SentiFolio</h1>'
    "<p>Systematically managed equity, crypto, and combined funds, "
    "with a news-sentiment index across equity sectors.</p></div>",
    unsafe_allow_html=True,
)

with st.expander("Important disclosures"):
    st.markdown(
        "- Backtested performance is not actual performance - no money was invested and no trades were placed.\n"
        "- Past performance, backtested or actual, does not guarantee future results.\n"
        "- Transaction costs, taxes, and management fees are excluded from every figure shown here.\n"
        "- Crypto funds involve materially higher volatility and drawdown risk than equity funds - "
        "see each fund's fact sheet.\n"
        "- Sentiment is based on news headlines only, scored automatically - it is a noisy proxy for "
        "market mood, not a verified signal."
    )

tab_funds, tab_sentiment, tab_allocate = st.tabs(["Funds", "Sentiment", "Build a portfolio"])

with tab_funds:
    st.subheader("📊 Compare funds")
    perf_display = performance.copy()
    perf_display["annualised_return"] = (perf_display["annualised_return"] * 100).round(2)
    perf_display["annualised_volatility"] = (perf_display["annualised_volatility"] * 100).round(2)
    perf_display["max_drawdown"] = (perf_display["max_drawdown"] * 100).round(2)
    perf_display["sharpe_ratio"] = perf_display["sharpe_ratio"].round(3)
    perf_display["avg_turnover"] = (perf_display["avg_turnover"] * 100).round(1)
    perf_display["latest_max_weight"] = (perf_display["latest_max_weight"] * 100).round(1)
    perf_display["effective_n_holdings"] = perf_display["effective_n_holdings"].round(1)
    perf_display = perf_display.drop(columns=["herfindahl_index"])
    perf_display = perf_display.rename(columns={
        "annualised_return": "Ann. return (%)", "annualised_volatility": "Ann. vol (%)",
        "sharpe_ratio": "Sharpe", "max_drawdown": "Max drawdown (%)", "fund": "Fund",
        "periods_per_year": "Periods/yr", "avg_turnover": "Avg turnover (%)",
        "latest_max_weight": "Largest holding (%)", "effective_n_holdings": "Effective N holdings",
    })
    # Colour is read-only sugar on top of the same numbers in the table: higher
    # Sharpe shades greener, a deeper (more negative) drawdown shades redder.
    # No value is changed, only how the existing figure is highlighted.
    styled_perf = (
        perf_display.style
        .background_gradient(cmap="Greens", subset=["Sharpe"])
        .background_gradient(cmap="Reds_r", subset=["Max drawdown (%)"])
        .format({"Ann. return (%)": "{:.2f}", "Ann. vol (%)": "{:.2f}", "Sharpe": "{:.3f}",
                  "Max drawdown (%)": "{:.2f}", "Avg turnover (%)": "{:.1f}",
                  "Largest holding (%)": "{:.1f}", "Effective N holdings": "{:.1f}"})
    )
    st.dataframe(styled_perf, width="stretch", hide_index=True)
    st.caption("Avg turnover: mean one-way turnover per rebalance. Effective N holdings: 1/Herfindahl "
               "index, computed from the most recent rebalance - lower means more concentrated. "
               "Sharpe shading: greener = higher. Max drawdown shading: redder = deeper.")

    st.divider()
    st.subheader("🎯 Fund fact sheet")
    fund_choice = st.selectbox("Choose a fund", ALL_FUNDS, index=ALL_FUNDS.index("Combined Max-Sharpe")
                                if "Combined Max-Sharpe" in ALL_FUNDS else 0)

    metrics_row = performance.loc[performance["fund"] == fund_choice].iloc[0]
    growth = growth_of_dollar(fund_choice)
    drawdown = max_drawdown_series(growth)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualised return", f"{metrics_row['annualised_return']*100:.2f}%")
    c2.metric("Annualised volatility", f"{metrics_row['annualised_volatility']*100:.2f}%")
    c3.metric("Sharpe ratio", f"{metrics_row['sharpe_ratio']:.2f}")
    c4.metric("Max drawdown", f"{metrics_row['max_drawdown']*100:.2f}%")

    asset_class = fund_choice.split(" ")[0]
    first_live = growth.index.min().date()
    st.caption(
        f"**Asset class:** {asset_class}  |  **First live date:** {first_live}  |  "
        f"**Estimation window:** 252 trading days, rolled forward  |  **Rebalance:** monthly  |  "
        f"**Current concentration:** largest holding {metrics_row['latest_max_weight']*100:.1f}%, "
        f"effective {metrics_row['effective_n_holdings']:.1f} equal-sized holdings"
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.caption(f"Growth of $1 - {fund_choice} (out-of-sample)")
        st.line_chart(growth)
    with col_right:
        st.caption(f"Drawdown - {fund_choice}")
        st.line_chart(drawdown)

    holdings, as_of = current_holdings(fund_choice)
    st.caption(f"Current holdings (target weights from the most recent rebalance, {as_of.date()})")
    st.dataframe(holdings, width="stretch", hide_index=True)

with tab_sentiment:
    st.subheader("🗞️ Sector news-sentiment index")
    st.caption("Equal-weight finVADER sentiment across each sector's tickers, standardised against "
               "each sector's own history (z-score) so 0 = normal, positive = relatively greedy, "
               "negative = relatively fearful. 21-trading-day rolling mean. Lagged by at least one "
               "trading day before any fund uses it.")
    sectors = sorted(sector_sentiment["sector"].unique())
    chosen_sectors = st.multiselect("Sectors", sectors, default=sectors[:4])
    if chosen_sectors:
        wide = (sector_sentiment[sector_sentiment["sector"].isin(chosen_sectors)]
                .pivot(index="date", columns="sector", values="sentiment_index_z")
                .rolling(21, min_periods=5).mean())
        st.line_chart(wide)
    else:
        st.info("Pick at least one sector.")

with tab_allocate:
    st.subheader("🧮 Build your own allocation")
    st.caption("Set a weight per fund (must sum to 100%) and see the blended portfolio's out-of-sample performance.")
    default_pct = round(100 / len(ALL_FUNDS), 1)
    alloc = {}
    cols = st.columns(3)
    for i, fund in enumerate(ALL_FUNDS):
        with cols[i % 3]:
            alloc[fund] = st.number_input(fund, min_value=0.0, max_value=100.0,
                                           value=0.0 if i else 100.0, step=5.0, key=f"alloc_{fund}")

    total_pct = sum(alloc.values())
    st.write(f"Total allocated: {total_pct:.1f}%")
    if abs(total_pct - 100.0) > 0.01:
        st.warning("Allocation must sum to 100% before the blended fact sheet updates.")
    else:
        weights_vec = pd.Series({f: w for f, w in alloc.items() if w > 0}) / 100.0
        # Restrict to dates where every selected fund is live (crypto funds start
        # 2020-10-01 on a 365-day calendar, equity/combined funds start 2021-01-04
        # on a 252-day calendar) - zero-filling a fund's pre-launch dates instead
        # would silently drag down its annualised return and volatility.
        wide_returns = (fund_returns[fund_returns["fund"].isin(weights_vec.index)]
                         .pivot(index="date", columns="fund", values="ret")
                         .dropna())
        blended_ret = wide_returns[weights_vec.index] @ weights_vec
        blended_growth = (1 + blended_ret).cumprod()
        # The blend's dates are the INTERSECTION of every selected fund's live
        # dates (see dropna() above). Equity/Combined funds only have data on
        # equity trading days (~252/year), so including even one of them caps
        # the blend at the 252-day calendar regardless of what else is picked.
        # Only a blend of crypto-only funds actually runs on crypto's 365-day
        # calendar - annualising a mixed or equity-containing blend with 365
        # would overstate it, and always using 252 (the previous behaviour)
        # understated a pure-crypto blend.
        all_crypto = all(f.startswith("Crypto") for f in weights_vec.index)
        periods_per_year = 365 if all_crypto else 252
        ann_return = blended_ret.mean() * periods_per_year
        ann_vol = blended_ret.std() * (periods_per_year ** 0.5)
        sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")
        max_dd = (blended_growth / blended_growth.cummax() - 1).min()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annualised return", f"{ann_return*100:.2f}%")
        c2.metric("Annualised volatility", f"{ann_vol*100:.2f}%")
        c3.metric("Sharpe ratio", f"{sharpe:.2f}")
        c4.metric("Max drawdown", f"{max_dd*100:.2f}%")
        st.caption("Growth of $1 - your blended allocation (out-of-sample)")
        st.line_chart(blended_growth)
