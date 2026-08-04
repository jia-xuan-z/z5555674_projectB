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
    """Score headlines with finVADER (VADER + finance lexicons) - see
    src/sentiment.py for why plain VADER under-covers finance vocabulary.
    Also scores a plain-VADER pass for a documented before/after comparison
    (innovation evidence), saved to results/tables/vader_vs_finvader.csv.
    """
    scores = sentiment.score_headlines(headline_panel, model="finvader")
    scores_vader = sentiment.score_headlines(headline_panel, model="vader")
    comparison = pd.DataFrame([
        {"model": "VADER", "mean_sentiment": scores_vader["sentiment"].mean(),
         "pct_exact_zero": (scores_vader["sentiment"] == 0).mean()},
        {"model": "finVADER", "mean_sentiment": scores["sentiment"].mean(),
         "pct_exact_zero": (scores["sentiment"] == 0).mean()},
    ])
    comparison.to_csv(RESULTS / "tables" / "vader_vs_finvader.csv", index=False)
    print(comparison.to_string(index=False))

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


TILT_STRENGTH_GRID = (0.25, 0.5, 1.0)


def fusion_robustness_table(scores: pd.DataFrame, equity_wide: pd.DataFrame, base_weights: pd.DataFrame):
    """Sensitivity of the fusion tilt to tilt_strength, at three PRE-SET
    values (0.25, 0.5, 1.0) chosen before looking at any result. This is
    reported as a robustness check across all three, not used to pick a
    "best" tilt_strength - selecting whichever value performs best on this
    same out-of-sample period and then reporting only that one would be
    data snooping (using the test period to both choose and evaluate a
    parameter). TILT_STRENGTH (0.5) stays the one used everywhere else in
    the pipeline; this table only exists to show whether the qualitative
    conclusion (roughly neutral effect) holds across a range of strengths.
    """
    base_metrics = portfolios.performance_metrics(
        fusion.apply_weights_to_returns(base_weights, equity_wide), periods_per_year=252)
    rows = [{"tilt_strength": 0.0, "annualised_return": base_metrics["annualised_return"],
             "annualised_volatility": base_metrics["annualised_volatility"],
             "sharpe_ratio": base_metrics["sharpe_ratio"], "max_drawdown": base_metrics["max_drawdown"],
             "turnover": portfolios.average_turnover(base_weights)}]
    for strength in TILT_STRENGTH_GRID:
        tilted = fusion.apply_sentiment(base_weights, scores, tilt_strength=strength)
        tilted_returns = fusion.apply_weights_to_returns(tilted, equity_wide)
        m = portfolios.performance_metrics(tilted_returns, periods_per_year=252)
        rows.append({"tilt_strength": strength, "annualised_return": m["annualised_return"],
                      "annualised_volatility": m["annualised_volatility"], "sharpe_ratio": m["sharpe_ratio"],
                      "max_drawdown": m["max_drawdown"], "turnover": portfolios.average_turnover(tilted)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "tables" / "fusion_robustness.csv", index=False)
    print(table.round(4).to_string(index=False))
    return table


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
    # Concentration/turnover diagnostics added after a review flagged Combined
    # Max-Sharpe's ~64% weight in its top 2 holdings (GE + NVDA) - a known
    # failure mode of naive mean-variance optimisation (max-Sharpe is highly
    # sensitive to noisy historical mean-return estimates and long-only
    # constraints push that sensitivity into corner solutions), not a bug.
    # Reporting these numbers rather than acting on them (e.g. capping any
    # single weight): a cap changes every fund's actual weights and results,
    # which is a bigger, riskier change this late - the diagnostics let the
    # report state the risk honestly and turn it into a concrete
    # recommendation (Section 6) instead.
    rows = []
    for name, res in funds.items():
        universe = name.split(" ")[0] if name.split(" ")[0] in UNIVERSE_PERIODS else "Equity"
        ppy = UNIVERSE_PERIODS.get(universe, 252)
        m = portfolios.performance_metrics(res["daily_returns"], periods_per_year=ppy)
        c = portfolios.concentration_metrics(res["weights"])
        t = portfolios.average_turnover(res["weights"])
        rows.append({"fund": name, "periods_per_year": ppy, **m, "avg_turnover": t, **c})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "tables" / "performance_metrics.csv", index=False)
    return table


def growth_figure(funds: dict):
    # All 9 base funds on one shared linear axis crowds the plot, and
    # crypto's much larger growth (up to ~10x) compresses equity/combined
    # (which stay under ~2x) into an unreadable band near the bottom - the
    # same "one dominant series drowns the rest" problem as the earlier
    # weights-over-time and sentiment-index redesigns, just in growth-of-$1
    # form. Splitting into one panel per universe, each with its OWN y-scale,
    # fixes both: no more 9-way legend collision, and equity/combined get
    # to use their own full vertical range instead of crypto's.
    surface, ink_primary, ink_secondary, ink_muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    gridline, baseline = "#e1e0d9", "#c3c2b7"
    method_colors = {"Min-Variance": "#2a78d6", "Max-Sharpe": "#eb6834", "Risk-Parity": "#1baf7a"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    fig.patch.set_facecolor(surface)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.06, right=0.98, wspace=0.22)

    for ax, universe in zip(axes, UNIVERSES):
        ax.set_facecolor(surface)
        for label, color in method_colors.items():
            res = funds[f"{universe} {label}"]
            ax.plot(res["growth"].index, res["growth"].values, color=color, linewidth=1.4, label=label, zorder=2)
        ax.set_title(universe, fontsize=11.5, color=ink_primary, loc="left", pad=6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(baseline)
        ax.grid(axis="y", color=gridline, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink_muted, labelsize=8)
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_ylabel("Growth of $1", color=ink_secondary, fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.99),
               frameon=False, fontsize=9.5, labelcolor=ink_secondary, ncol=3)

    fig.text(0.06, 0.94, "Growth of $1, out-of-sample, by universe and method (2020-2023)",
              color=ink_primary, fontsize=13, ha="left", va="top")
    fig.text(0.06, 0.885, "Each panel has its own vertical scale - Crypto's growth is 3-5x "
                          "larger than Equity/Combined, so a shared axis would flatten the latter two",
              color=ink_secondary, fontsize=9, ha="left", va="top")

    fig.savefig(RESULTS / "figures" / "growth_of_dollar.png", dpi=150, facecolor=surface)
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


def weights_over_time_figure(funds: dict, universe: str = "Equity", n_tickers: int = 6):
    # The brief asks for weights over time ACROSS METHODS for one fund/
    # universe - not across holdings within one method (an earlier version
    # of this figure compared the 8 largest holdings under a single method,
    # which does not answer that). Comparing methods only makes sense
    # ticker by ticker (one method's weight in isolation says nothing about
    # another method), so this is small multiples of TICKERS, each panel
    # overlaying all three methods' weight for that one name - the panels
    # are chosen as the tickers where the three methods disagree most (by
    # max-minus-min average weight), since a ticker all three methods treat
    # the same way has nothing to show.
    surface, ink_primary, ink_secondary, ink_muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    gridline, baseline = "#e1e0d9", "#c3c2b7"
    method_colors = {"Min-Variance": "#2a78d6", "Max-Sharpe": "#eb6834", "Risk-Parity": "#1baf7a"}

    method_weights = {label: funds[f"{universe} {label}"]["weights"] * 100 for label in METHOD_LABELS.values()}
    avg_by_method = pd.DataFrame({label: w.mean() for label, w in method_weights.items()})
    spread = avg_by_method.max(axis=1) - avg_by_method.min(axis=1)
    tickers = spread.sort_values(ascending=False).head(n_tickers).index.tolist()
    y_max = max(w[tickers].max().max() for w in method_weights.values())

    ncols = 3
    nrows = -(-n_tickers // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), sharex=True, sharey=True)
    fig.patch.set_facecolor(surface)
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.06, right=0.98, hspace=0.5, wspace=0.12)

    for ax, ticker in zip(np.atleast_1d(axes).flat, tickers):
        ax.set_facecolor(surface)
        for label, w in method_weights.items():
            ax.plot(w.index, w[ticker].values, color=method_colors[label], linewidth=1.5, label=label, zorder=2)
        ax.set_ylim(-y_max * 0.05, y_max * 1.08)
        ax.set_title(ticker, fontsize=10.5, color=ink_primary, loc="left", pad=4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(baseline)
        ax.grid(axis="y", color=gridline, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink_muted, labelsize=7)
        ax.tick_params(axis="x", rotation=45)

    axes2d = np.atleast_2d(axes)
    for row in range(axes2d.shape[0]):
        axes2d[row, 0].set_ylabel("Weight (%)", color=ink_secondary, fontsize=8.5)
    handles, labels = axes2d.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.99),
               frameon=False, fontsize=9, labelcolor=ink_secondary, ncol=3)

    fig.text(0.06, 0.965, f"Portfolio weights over time, across methods - {universe} funds",
              color=ink_primary, fontsize=13, ha="left", va="top")
    fig.text(0.06, 0.91, "The tickers where Min-Variance, Max-Sharpe, and Risk-Parity disagree "
                         "most about how much weight to hold, one panel per ticker, same scale",
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


SECTOR_DISPLAY_NAMES = {
    "Comm": "Communication Services", "RealEstate": "Real Estate", "Tech": "Information Technology",
}
# GICS order, with Consumer left as one merged label - the course's "Consumer"
# sector mixes Discretionary names (DIS, NKE, SBUX) and Staples names (WMT, KO),
# so relabelling it as either official GICS sector would misstate the data; the
# report text notes the merge instead of the chart pretending it's pure GICS.
SECTOR_ORDER = ["Comm", "Consumer", "Energy", "Financials", "Healthcare", "Industrials",
                "Tech", "Materials", "RealEstate", "Utilities"]


def sentiment_index_figure(sector_index: pd.DataFrame):
    # 10 sectors as overlapping lines is a rainbow spaghetti chart - no single
    # sector is traceable and the legend does more work than the plot. Small
    # multiples (one thin panel per sector, sharing one y-scale) trade a
    # single glance for the ability to actually read any one sector, and a
    # shared scale keeps them comparable at a glance.
    #
    # Plots the STANDARDISED index (z-score against each sector's own mean/
    # std), not the raw finVADER average: headlines are mildly positive on
    # average (as the course's own fear-and-greed index finds - raw scores
    # read positive on ~94% of days), so a raw index would sit above its zero
    # line almost every day and never distinguish an unusual day from a
    # normal one. Standardising is what makes "more positive/negative than
    # usual for THIS sector" readable - it does NOT make sectors comparable
    # to each other in absolute terms (a caption note says so explicitly),
    # since each is standardised against its own history, not a shared one.
    # Deliberately not called a "fear and greed" index and not labelled
    # greedy/fearful: this measures headline sentiment tone, not investor
    # psychology, which the course's own fear-and-greed lecture treats as a
    # related but distinct construct built from different inputs.
    surface, ink_primary, ink_secondary, ink_muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    gridline, baseline, blue, red = "#e1e0d9", "#c3c2b7", "#2a78d6", "#e34948"

    sectors = [s for s in SECTOR_ORDER if s in sector_index["sector"].unique()]
    smoothed_all = sector_index.groupby("sector")["sentiment_index_z"].transform(
        lambda s: s.rolling(21, min_periods=5).mean())
    y_max = max(abs(smoothed_all.min()), abs(smoothed_all.max())) * 1.1
    year_ticks = pd.date_range("2020-01-01", "2024-01-01", freq="YS")

    fig, axes = plt.subplots(2, 5, figsize=(12, 5.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(surface)
    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.055, right=0.98, hspace=0.4, wspace=0.1)

    for i, (ax, sector) in enumerate(zip(axes.flat, sectors)):
        g = sector_index.loc[sector_index["sector"] == sector].sort_values("date")
        smoothed = g["sentiment_index_z"].rolling(21, min_periods=5).mean()
        ax.set_facecolor(surface)
        ax.fill_between(g["date"], smoothed, 0, where=(smoothed >= 0), color=blue, alpha=0.35, linewidth=0, zorder=1)
        ax.fill_between(g["date"], smoothed, 0, where=(smoothed < 0), color=red, alpha=0.35, linewidth=0, zorder=1)
        ax.plot(g["date"], smoothed, color=ink_secondary, linewidth=0.7, zorder=2)
        ax.axhline(0, color=ink_muted, linewidth=1.2, alpha=0.8, zorder=3)
        ax.set_ylim(-y_max, y_max)
        ax.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01"))
        ax.set_xticks(year_ticks)
        ax.set_title(SECTOR_DISPLAY_NAMES.get(sector, sector), fontsize=10, color=ink_primary, loc="left", pad=4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(baseline)
        ax.grid(axis="y", color=gridline, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink_muted, labelsize=7.5)
        if i < 5:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.tick_params(axis="x", rotation=0)
            ax.set_xticklabels([d.strftime("%Y") for d in year_ticks])
        if i % 5 != 0:
            ax.tick_params(axis="y", labelleft=False)

    fig.text(0.055, 0.965, "Standardised Sector News Sentiment, 2020-2023", color=ink_primary, fontsize=13.5, ha="left", va="top")
    fig.text(0.055, 0.925, "21-day rolling average of sector-level finVADER sentiment z-scores. "
                           "Blue = above-average sentiment; red = below-average sentiment",
              color=ink_secondary, fontsize=9, ha="left", va="top")
    fig.text(0.055, 0.035, "Each sector is standardised against its own 2020-2023 mean/std - values are "
                           "comparable as deviations from a sector's own norm, not as sentiment levels across sectors.",
              color=ink_muted, fontsize=7.5, ha="left", va="bottom")
    fig.text(0.055, 0.005, "\"Consumer\" merges Discretionary and Staples tickers (see text).",
              color=ink_muted, fontsize=7.5, ha="left", va="bottom")

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


def save_table_image(df: pd.DataFrame, filename: str, title: str, subtitle: str = "",
                      col_widths=None, index=False):
    """Render a dataframe as a clean table image (for embedding in the report
    alongside the CSV) - same palette as the other figures. Saved under
    results/tables/, since it is a picture OF a table, not a chart."""
    display_df = df.reset_index() if index else df
    n_rows, n_cols = display_df.shape
    fig_h = 0.85 + 0.32 * (n_rows + 1)

    # Auto-size columns from actual content length instead of splitting the
    # table width evenly - an evenly-split width truncates a long "Fund"
    # name column the moment any numeric column is present alongside it.
    if col_widths is None:
        cell_text = display_df.values.tolist()
        col_labels = list(display_df.columns)
        max_len = [max([len(str(h))] + [len(str(row[i])) for row in cell_text]) for i, h in enumerate(col_labels)]
        total = sum(max_len)
        col_widths = [m / total for m in max_len]
    else:
        cell_text = display_df.values.tolist()
        col_labels = list(display_df.columns)

    fig, ax = plt.subplots(figsize=(max(7, 1.15 * n_cols), fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center",
                      colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRIDLINE)
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("#ffffff")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(SURFACE if row % 2 else "#f3f3f0")
            cell.get_text().set_color(INK_PRIMARY)
        if col == 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.03

    # Fixed INCH offsets for the title block, converted to figure-fraction
    # coordinates - a fractional offset alone (e.g. "-0.045") shrinks to a
    # handful of pixels on a short (few-row) table and the title/subtitle
    # collide, since the same fraction means far less absolute space on a
    # 2-inch-tall figure than on a 6-inch one.
    title_y = 1 - 0.22 / fig_h
    subtitle_y = 1 - 0.42 / fig_h
    table_top = 1 - (0.62 if subtitle else 0.38) / fig_h
    fig.text(0.03, title_y, title, color=INK_PRIMARY, fontsize=13, ha="left", va="top", weight="bold")
    if subtitle:
        fig.text(0.03, subtitle_y, subtitle, color=INK_SECONDARY, fontsize=9, ha="left", va="top")
    fig.subplots_adjust(top=table_top, bottom=0.03, left=0.03, right=0.97)

    fig.savefig(RESULTS / "tables" / filename, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def table_images():
    """Render every table that appears in the Part B report as a PNG image
    under results/tables/, straight from the same committed CSVs - no
    numbers are retyped or re-derived here."""
    perf = pd.read_csv(RESULTS / "tables" / "performance_metrics.csv")

    main_cols = ["fund", "annualised_return", "annualised_volatility", "sharpe_ratio",
                 "max_drawdown", "effective_n_holdings"]
    main = perf[main_cols].copy()
    main["annualised_return"] = (main["annualised_return"] * 100).round(1).astype(str) + "%"
    main["annualised_volatility"] = (main["annualised_volatility"] * 100).round(1).astype(str) + "%"
    main["max_drawdown"] = (main["max_drawdown"] * 100).round(1).astype(str) + "%"
    main["sharpe_ratio"] = main["sharpe_ratio"].round(2)
    main["effective_n_holdings"] = main["effective_n_holdings"].round(1)
    main = main[main["fund"] != "Equity Max-Sharpe + Sentiment Tilt"]
    main.columns = ["Fund", "Ann. return", "Ann. vol.", "Sharpe", "Max DD", "Effective N"]
    save_table_image(main, "table1_performance_metrics.png",
                      "Table 1. Out-of-sample performance and latest effective number of holdings",
                      "Equity/Combined annualised with 252 days; Crypto with 365. "
                      "Source: results/tables/performance_metrics.csv")

    vf = pd.read_csv(RESULTS / "tables" / "vader_vs_finvader.csv")
    vf["mean_sentiment"] = vf["mean_sentiment"].round(3)
    vf["pct_exact_zero"] = (vf["pct_exact_zero"] * 100).round(1).astype(str) + "%"
    vf.columns = ["Model", "Mean ticker-day sentiment", "Exact-zero observations"]
    save_table_image(vf, "table2_vader_vs_finvader.png",
                      "Table 2. Standard VADER versus finVADER validation comparison",
                      "Source: results/tables/vader_vs_finvader.csv")

    fc = pd.read_csv(RESULTS / "tables" / "fusion_comparison.csv")
    fc_disp = fc[["fund", "annualised_return", "annualised_volatility", "sharpe_ratio",
                  "max_drawdown", "avg_turnover", "effective_n_holdings"]].copy()
    fc_disp["annualised_return"] = (fc_disp["annualised_return"] * 100).round(2).astype(str) + "%"
    fc_disp["annualised_volatility"] = (fc_disp["annualised_volatility"] * 100).round(2).astype(str) + "%"
    fc_disp["max_drawdown"] = (fc_disp["max_drawdown"] * 100).round(2).astype(str) + "%"
    fc_disp["avg_turnover"] = (fc_disp["avg_turnover"] * 100).round(1).astype(str) + "%"
    fc_disp["sharpe_ratio"] = fc_disp["sharpe_ratio"].round(3)
    fc_disp["effective_n_holdings"] = fc_disp["effective_n_holdings"].round(2)
    fc_disp["fund"] = ["Equity Max-Sharpe", "+ Sentiment tilt"]
    fc_disp.columns = ["Fund", "Return", "Vol.", "Sharpe", "Max DD", "Turnover", "Effective N"]
    save_table_image(fc_disp, "table3_fusion_comparison.png",
                      "Table 3. Equity Max-Sharpe before and after the 0.50 sentiment tilt",
                      "Source: results/tables/fusion_comparison.csv")

    fr = pd.read_csv(RESULTS / "tables" / "fusion_robustness.csv")
    fr["annualised_return"] = (fr["annualised_return"] * 100).round(2).astype(str) + "%"
    fr["annualised_volatility"] = (fr["annualised_volatility"] * 100).round(2).astype(str) + "%"
    fr["max_drawdown"] = (fr["max_drawdown"] * 100).round(2).astype(str) + "%"
    fr["turnover"] = (fr["turnover"] * 100).round(1).astype(str) + "%"
    fr["sharpe_ratio"] = fr["sharpe_ratio"].round(3)
    fr["tilt_strength"] = fr["tilt_strength"].map(lambda x: f"{x:.2f}")
    fr.columns = ["Tilt k", "Ann. return", "Ann. vol.", "Sharpe", "Max DD", "Turnover"]
    save_table_image(fr, "table4_fusion_robustness.png",
                      "Table 4. Pre-specified sentiment-tilt robustness test",
                      "Source: results/tables/fusion_robustness.csv")

    full = perf.copy()
    for c in ["annualised_return", "annualised_volatility", "max_drawdown", "avg_turnover", "latest_max_weight"]:
        full[c] = (full[c] * 100).round(1).astype(str) + "%"
    full["sharpe_ratio"] = full["sharpe_ratio"].round(2)
    full["herfindahl_index"] = full["herfindahl_index"].round(3)
    full["effective_n_holdings"] = full["effective_n_holdings"].round(1)
    full = full[["fund", "annualised_return", "annualised_volatility", "sharpe_ratio", "max_drawdown",
                 "avg_turnover", "latest_max_weight", "herfindahl_index", "effective_n_holdings",
                 "periods_per_year"]]
    full.columns = ["Fund", "Return", "Vol.", "Sharpe", "Max DD", "Turnover", "Max wt.", "HHI", "Eff. N", "Ann."]
    save_table_image(full, "tableB1_full_diagnostics.png",
                      "Table B1. Full performance metrics, including implementation and concentration diagnostics",
                      "Source: results/tables/performance_metrics.csv")

    print("saved 5 table images to results/tables/")


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

    print("fusion tilt-strength robustness check...")
    base_fund = funds[FUSION_BASE_FUND]
    fusion_robustness_table(scores, equity_wide, base_fund["weights"])

    print("saving required CSVs...")
    save_required_csvs(funds)
    perf_table = save_performance_table(funds)
    print(perf_table.round(4).to_string(index=False))

    print("building figures...")
    growth_figure(funds)
    drawdown_figure(funds, fund_name="Combined Max-Sharpe")
    weights_over_time_figure(funds, universe="Equity")
    sharpe_barplot_figure(perf_table)
    sentiment_index_figure(sector_index)
    fusion_comparison(funds, fused_name, perf_table)

    print("rendering table images for the report...")
    table_images()

    print("\ndone - results/ populated.")


if __name__ == "__main__":
    main()
