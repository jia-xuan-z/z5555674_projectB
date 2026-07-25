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

st.set_page_config(page_title="SentiFolio", layout="wide")


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


st.title("SentiFolio")
st.caption("Systematically managed equity, crypto, and combined funds, with a news-sentiment index across equity sectors.")

tab_funds, tab_sentiment, tab_allocate = st.tabs(["Funds", "Sentiment", "Build a portfolio"])

with tab_funds:
    st.subheader("Compare funds")
    perf_display = performance.copy()
    perf_display["annualised_return"] = (perf_display["annualised_return"] * 100).round(2)
    perf_display["annualised_volatility"] = (perf_display["annualised_volatility"] * 100).round(2)
    perf_display["max_drawdown"] = (perf_display["max_drawdown"] * 100).round(2)
    perf_display["sharpe_ratio"] = perf_display["sharpe_ratio"].round(3)
    perf_display = perf_display.rename(columns={
        "annualised_return": "Ann. return (%)", "annualised_volatility": "Ann. vol (%)",
        "sharpe_ratio": "Sharpe", "max_drawdown": "Max drawdown (%)", "fund": "Fund",
        "periods_per_year": "Periods/yr",
    })
    st.dataframe(perf_display, width="stretch", hide_index=True)

    st.subheader("Fund fact sheet")
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
    st.subheader("Sector news-sentiment index")
    st.caption("Equal-weight VADER compound sentiment across each sector's tickers, "
               "21-trading-day rolling mean. Lagged by at least one trading day before any fund uses it.")
    sectors = sorted(sector_sentiment["sector"].unique())
    chosen_sectors = st.multiselect("Sectors", sectors, default=sectors[:4])
    if chosen_sectors:
        wide = (sector_sentiment[sector_sentiment["sector"].isin(chosen_sectors)]
                .pivot(index="date", columns="sector", values="sentiment_index")
                .rolling(21, min_periods=5).mean())
        st.line_chart(wide)
    else:
        st.info("Pick at least one sector.")

with tab_allocate:
    st.subheader("Build your own allocation")
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
        ann_return = blended_ret.mean() * 252
        ann_vol = blended_ret.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")
        max_dd = (blended_growth / blended_growth.cummax() - 1).min()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annualised return", f"{ann_return*100:.2f}%")
        c2.metric("Annualised volatility", f"{ann_vol*100:.2f}%")
        c3.metric("Sharpe ratio", f"{sharpe:.2f}")
        c4.metric("Max drawdown", f"{max_dd*100:.2f}%")
        st.caption("Growth of $1 - your blended allocation (out-of-sample)")
        st.line_chart(blended_growth)
