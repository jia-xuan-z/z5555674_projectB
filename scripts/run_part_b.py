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
    # A min-variance fund over 50 names diversifies hard: the top-8 tickers by
    # average weight are typically a small slice of the book, so stacking them
    # under an "Other" band showing everyone else buries them under one huge,
    # undifferentiated block. Dropping the stack and plotting each top holding
    # as its own line shows what's actually going on (initial concentration
    # that fades as the fund diversifies) instead of hiding it under "Other".
    surface, ink_primary, ink_secondary, ink_muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    gridline, baseline = "#e1e0d9", "#c3c2b7"
    categorical = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                   "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

    w = funds[fund_name]["weights"]
    avg_w = w.mean().sort_values(ascending=False)
    top_tickers = avg_w.head(top_n).index
    other_share = w.drop(columns=top_tickers).sum(axis=1).mean()

    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.patch.set_facecolor(surface)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.09, right=0.98)

    for color, ticker in zip(categorical, top_tickers):
        ax.plot(w.index, w[ticker].values * 100, color=color, linewidth=2, label=ticker)

    ax.set_facecolor(surface)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(baseline)
    ax.grid(axis="y", color=gridline, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=ink_muted, labelsize=8)
    ax.set_xlabel("Rebalance date", color=ink_secondary, fontsize=9)
    ax.set_ylabel("Weight (%)", color=ink_secondary, fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=ink_secondary, ncol=2)

    fig.text(0.09, 0.96, f"Portfolio weights over time - {fund_name} fund", color=ink_primary,
              fontsize=13, ha="left", va="top")
    fig.text(0.09, 0.915, f"Top {top_n} holdings by average weight (the remaining "
                          f"{w.shape[1] - top_n} names average {other_share*100:.0f}% combined)",
              color=ink_secondary, fontsize=9, ha="left", va="top")

    fig.savefig(RESULTS / "figures" / "weights_over_time.png", dpi=150, facecolor=surface)
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
    # 10 sectors as overlapping lines is a rainbow spaghetti chart - no single
    # sector is traceable and the legend does more work than the plot. Small
    # multiples (one thin panel per sector, sharing one y-scale) trade a
    # single glance for the ability to actually read any one sector, and a
    # shared scale keeps them comparable at a glance.
    surface, ink_primary, ink_secondary, ink_muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    gridline, baseline, blue = "#e1e0d9", "#c3c2b7", "#2a78d6"

    sectors = sorted(sector_index["sector"].unique())
    y_min = sector_index["sentiment_index"].rolling(21, min_periods=5).mean().min()
    y_max = sector_index["sentiment_index"].rolling(21, min_periods=5).mean().max()
    pad = (y_max - y_min) * 0.1

    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(surface)
    fig.subplots_adjust(top=0.85, bottom=0.11, left=0.06, right=0.98, hspace=0.45, wspace=0.12)

    for ax, sector in zip(axes.flat, sectors):
        g = sector_index.loc[sector_index["sector"] == sector].sort_values("date")
        smoothed = g["sentiment_index"].rolling(21, min_periods=5).mean()
        ax.set_facecolor(surface)
        ax.axhline(0, color=baseline, linewidth=0.8, zorder=1)
        ax.plot(g["date"], smoothed, color=blue, linewidth=1.3, zorder=2)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(sector, fontsize=10, color=ink_primary, loc="left", pad=4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(baseline)
        ax.grid(axis="y", color=gridline, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink_muted, labelsize=7)
        ax.tick_params(axis="x", rotation=45)

    fig.text(0.06, 0.965, "Sector news-sentiment index", color=ink_primary, fontsize=13, ha="left", va="top")
    fig.text(0.06, 0.925, "21-trading-day rolling mean VADER compound score, one panel per sector, "
                          "same scale (2020-2023)", color=ink_secondary, fontsize=9, ha="left", va="top")

    fig.savefig(RESULTS / "figures" / "sentiment_index.png", dpi=150, facecolor=surface)
    plt.close(fig)


SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def fusion_comparison(funds: dict, fused_name: str, perf_table: pd.DataFrame):
    base = perf_table.loc[perf_table["fund"] == FUSION_BASE_FUND].iloc[0]
    fused = perf_table.loc[perf_table["fund"] == fused_name].iloc[0]
    comparison = pd.DataFrame([base, fused]).set_index("fund")
    comparison.to_csv(RESULTS / "tables" / "fusion_comparison.csv")

    base_growth = funds[FUSION_BASE_FUND]["growth"]
    fused_growth = funds[fused_name]["growth"]
    spread = (fused_growth - base_growth) * 100  # cents per $1, easier to read than a 4th decimal

    # The base and fused growth paths are within a few cents of each other
    # almost every day - that IS the finding (the tilt is a small nudge, not
    # a redesign of the fund), and no amount of line styling makes two nearly
    # identical series look visually distinct without misrepresenting them.
    # So: drop the overlapping-lines chart entirely. The spread (fused - base)
    # is the only view that actually carries information, so it becomes the
    # single hero chart, with the headline metrics as text stat-callouts above
    # it rather than fought for on the chart itself.
    fig = plt.figure(figsize=(9, 5.6))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3.2], hspace=0.05,
                           top=0.80, bottom=0.11, left=0.1, right=0.97)
    ax_stats = fig.add_subplot(gs[0])
    ax_spread = fig.add_subplot(gs[1])

    fig.text(0.1, 0.96, "Sentiment fusion: does tilting toward positive sentiment change the fund?",
              color=INK_PRIMARY, fontsize=12.5, ha="left", va="top")
    fig.text(0.1, 0.915, f"{FUSION_BASE_FUND} vs. {fused_name} - out-of-sample daily spread, in cents per $1 invested",
              color=INK_MUTED, fontsize=9, ha="left", va="top")

    ax_stats.axis("off")
    stat_specs = [
        ("Ann. return", base["annualised_return"], fused["annualised_return"], "pp", 100),
        ("Ann. volatility", base["annualised_volatility"], fused["annualised_volatility"], "pp", 100),
        ("Sharpe ratio", base["sharpe_ratio"], fused["sharpe_ratio"], "", 1),
        ("Max drawdown", base["max_drawdown"], fused["max_drawdown"], "pp", 100),
    ]
    for i, (label, b, f, unit, scale) in enumerate(stat_specs):
        x = 0.02 + i * 0.25
        delta = (f - b) * scale
        delta_color = BLUE if delta >= 0 else RED
        fmt = "{:+.2f}" if unit == "" else "{:+.2f}" + unit
        ax_stats.text(x, 0.85, label, transform=ax_stats.transAxes, fontsize=9,
                       color=INK_MUTED, ha="left", va="top")
        base_str = f"{b:.2f}" if unit == "" else f"{b*scale:.1f}{unit}"
        fused_str = f"{f:.2f}" if unit == "" else f"{f*scale:.1f}{unit}"
        ax_stats.text(x, 0.48, f"{base_str} → {fused_str}", transform=ax_stats.transAxes,
                       fontsize=13, color=INK_PRIMARY, ha="left", va="top")
        ax_stats.text(x, 0.05, fmt.format(delta), transform=ax_stats.transAxes,
                       fontsize=9.5, color=delta_color, ha="left", va="top", weight="bold")

    ax_spread.fill_between(spread.index, spread.values, 0, where=(spread.values >= 0),
                            color=BLUE, alpha=0.6, linewidth=0, zorder=2)
    ax_spread.fill_between(spread.index, spread.values, 0, where=(spread.values < 0),
                            color=RED, alpha=0.6, linewidth=0, zorder=2)
    ax_spread.axhline(0, color=BASELINE, linewidth=1, zorder=3)
    _style_axis(ax_spread)
    ax_spread.set_ylabel("Tilt effect (cents per $1)", color=INK_SECONDARY, fontsize=9)
    ax_spread.set_xlabel("Date", color=INK_SECONDARY, fontsize=9)
    ax_spread.margins(x=0.01)
    end_spread = spread.iloc[-1]
    ax_spread.text(0.995, 0.05, f"{'+' if end_spread >= 0 else ''}{end_spread:.1f}¢ by {spread.index[-1].date()}",
                    transform=ax_spread.transAxes, ha="right", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax_spread.text(0.005, 0.95, "sentiment tilt ahead", transform=ax_spread.transAxes,
                    ha="left", va="top", fontsize=8, color=BLUE)
    ax_spread.text(0.005, 0.05, "sentiment tilt behind", transform=ax_spread.transAxes,
                    ha="left", va="bottom", fontsize=8, color=RED)

    fig.savefig(RESULTS / "figures" / "fusion_comparison.png", dpi=150, facecolor=SURFACE)
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
