"""Reproduce Part B results. Run from the project root:

    python scripts/run_part_b.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import etl, features, portfolios, sentiment, fusion  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
(RESULTS / "data").mkdir(parents=True, exist_ok=True)
(RESULTS / "tables").mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)

WINDOW = 252
UNIVERSES = ("Equity", "Crypto", "Combined")
UNIVERSE_PERIODS = {"Equity": 252, "Crypto": 365, "Combined": 252}
METHOD_LABELS = {"min_variance": "Min-Variance", "max_sharpe": "Max-Sharpe", "risk_parity": "Risk-Parity"}
FUSION_BASE_FUND = "Equity Max-Sharpe"
TILT_STRENGTH = 0.5


def build_universes():
    eq, eq_report = etl.load_clean_equities()
    cr, cr_report = etl.load_clean_crypto()
    news, news_report = etl.load_clean_news()

    eq_ret = features.daily_returns(eq)
    cr_ret = features.daily_returns(cr)
    combined = features.combined_returns_panel(eq_ret, cr_ret)[["ticker", "date", "ret"]]

    trading_calendar = pd.DatetimeIndex(sorted(eq["date"].unique()))
    headline_panel = features.assemble_headline_panel(news, trading_calendar)

    return {
        "Equity": eq_ret, "Crypto": cr_ret, "Combined": combined,
        "headline_panel": headline_panel, "eq_report": eq_report,
        "cr_report": cr_report, "news_report": news_report,
    }


def build_funds(universe_returns: dict) -> dict:
    """Run all (universe, method) backtests. Returns {fund_name: backtest_result}."""
    funds = {}
    for universe in UNIVERSES:
        for method in portfolios.METHODS:
            name = f"{universe} {METHOD_LABELS[method]}"
            print(f"  backtesting {name}...")
            funds[name] = portfolios.oos_backtest(universe_returns[universe], method=method, window=WINDOW)
    return funds


def build_sentiment(headline_panel: pd.DataFrame):
    scores = sentiment.score_headlines(headline_panel)
    sector_index = sentiment.sector_sentiment_index(scores)
    return scores, sector_index


def build_fusion(funds: dict, scores: pd.DataFrame, equity_wide: pd.DataFrame):
    base = funds[FUSION_BASE_FUND]
    tilted_weights = fusion.apply_sentiment(base["weights"], scores, tilt_strength=TILT_STRENGTH)
    fused_returns = fusion.apply_weights_to_returns(tilted_weights, equity_wide)
    fused_growth = (1 + fused_returns).cumprod()
    fund_name = f"{FUSION_BASE_FUND} + Sentiment Tilt"
    return fund_name, {
        "method": "sentiment_tilt", "window": WINDOW, "first_live_date": base["first_live_date"],
        "daily_returns": fused_returns, "weights": tilted_weights, "growth": fused_growth,
    }


def save_required_csvs(funds: dict):
    ret_rows = []
    for name, res in funds.items():
        ret_rows.append(pd.DataFrame({"date": res["daily_returns"].index, "fund": name,
                                       "ret": res["daily_returns"].values}))
    pd.concat(ret_rows, ignore_index=True).to_csv(RESULTS / "data" / "fund_returns.csv", index=False)

    w_rows = []
    for name, res in funds.items():
        w = res["weights"].reset_index().rename(columns={"index": "date"})
        w_long = w.melt(id_vars="date", var_name="ticker", value_name="weight")
        w_long["fund"] = name
        w_rows.append(w_long[["date", "fund", "ticker", "weight"]])
    pd.concat(w_rows, ignore_index=True).to_csv(RESULTS / "data" / "fund_weights.csv", index=False)


def save_performance_table(funds: dict) -> pd.DataFrame:
    rows = []
    for name, res in funds.items():
        universe = name.split(" ")[0] if name.split(" ")[0] in UNIVERSE_PERIODS else "Equity"
        ppy = UNIVERSE_PERIODS.get(universe, 252)
        m = portfolios.performance_metrics(res["daily_returns"], periods_per_year=ppy)
        rows.append({"fund": name, "periods_per_year": ppy, **m})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "tables" / "performance_metrics.csv", index=False)
    return table


def growth_figure(funds: dict):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, res in funds.items():
        if name.endswith("Sentiment Tilt"):
            continue
        ax.plot(res["growth"].index, res["growth"].values, label=name, linewidth=1.2)
    ax.set_title("Growth of $1 - out-of-sample, by fund and method (2021-2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "growth_of_dollar.png", dpi=150)
    plt.close(fig)


def drawdown_figure(funds: dict, fund_name: str = "Combined Max-Sharpe"):
    res = funds[fund_name]
    growth = res["growth"]
    running_max = growth.cummax()
    drawdown = growth / running_max - 1
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.6)
    ax.set_title(f"Drawdown - {fund_name} fund (out-of-sample, 2021-2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown from running peak")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "drawdown.png", dpi=150)
    plt.close(fig)


def weights_over_time_figure(funds: dict, fund_name: str = "Equity Min-Variance", top_n: int = 8):
    w = funds[fund_name]["weights"]
    avg_w = w.mean().sort_values(ascending=False)
    top_tickers = avg_w.head(top_n).index
    other = w.drop(columns=top_tickers).sum(axis=1)
    plot_df = w[top_tickers].copy()
    plot_df["Other"] = other

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(plot_df.index, plot_df.T.values, labels=plot_df.columns)
    ax.set_title(f"Portfolio weights over time - {fund_name} fund (top {top_n} holdings)")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "weights_over_time.png", dpi=150)
    plt.close(fig)


def sharpe_barplot_figure(perf_table: pd.DataFrame):
    plot_df = perf_table[~perf_table["fund"].str.endswith("Sentiment Tilt")].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"Equity": "#4C72B0", "Crypto": "#DD8452", "Combined": "#55A868"}
    bar_colors = [colors[f.split(" ")[0]] for f in plot_df["fund"]]
    ax.bar(plot_df["fund"], plot_df["sharpe_ratio"], color=bar_colors)
    ax.set_title("Annualised Sharpe ratio by fund and method (out-of-sample, rf=0)")
    ax.set_ylabel("Sharpe ratio")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "sharpe_barplot.png", dpi=150)
    plt.close(fig)


def sentiment_index_figure(sector_index: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for sector, g in sector_index.groupby("sector"):
        g = g.sort_values("date")
        smoothed = g["sentiment_index"].rolling(21, min_periods=5).mean()
        ax.plot(g["date"], smoothed, label=sector, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title("Sector news-sentiment index (21-trading-day rolling mean VADER compound, 2020-2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment index")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "sentiment_index.png", dpi=150)
    plt.close(fig)


def fusion_comparison(funds: dict, fused_name: str, perf_table: pd.DataFrame):
    base = perf_table.loc[perf_table["fund"] == FUSION_BASE_FUND].iloc[0]
    fused = perf_table.loc[perf_table["fund"] == fused_name].iloc[0]
    comparison = pd.DataFrame([base, fused]).set_index("fund")
    comparison.to_csv(RESULTS / "tables" / "fusion_comparison.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(funds[FUSION_BASE_FUND]["growth"].index, funds[FUSION_BASE_FUND]["growth"].values,
            label=FUSION_BASE_FUND, linewidth=1.3)
    ax.plot(funds[fused_name]["growth"].index, funds[fused_name]["growth"].values,
            label=fused_name, linewidth=1.3, linestyle="--")
    ax.set_title(f"Sentiment fusion: {FUSION_BASE_FUND} before vs. after (out-of-sample)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "fusion_comparison.png", dpi=150)
    plt.close(fig)
    return comparison


def main():
    print("loading and cleaning data...")
    data = build_universes()

    print("building funds (9 base backtests: 3 universes x 3 methods)...")
    funds = build_funds(data)

    print("scoring sentiment...")
    scores, sector_index = build_sentiment(data["headline_panel"])
    sector_index.to_csv(RESULTS / "data" / "sector_sentiment_index.csv", index=False)

    print("building sentiment fusion...")
    equity_wide = portfolios.long_to_wide(data["Equity"])
    fused_name, fused_result = build_fusion(funds, scores, equity_wide)
    funds[fused_name] = fused_result

    print("saving required CSVs...")
    save_required_csvs(funds)
    perf_table = save_performance_table(funds)
    print(perf_table.round(4).to_string(index=False))

    print("building figures...")
    growth_figure(funds)
    drawdown_figure(funds, fund_name="Combined Max-Sharpe")
    weights_over_time_figure(funds, fund_name="Equity Min-Variance")
    sharpe_barplot_figure(perf_table)
    sentiment_index_figure(sector_index)
    fusion_comparison(funds, fused_name, perf_table)

    print("\ndone - results/ populated.")


if __name__ == "__main__":
    main()
