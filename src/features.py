"""Station 2 - features: return features and headline text assembly.

Reused from my own Part A submission (the brief permits this). Feeds
Station 3: `combined_returns_panel` is the optimiser's input, and
`assemble_headline_panel` is the sentiment model's input (scored in
src/sentiment.py, not here).
"""
from __future__ import annotations

import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Simple daily returns per ticker, long format: (ticker, date, ret).

    Computed WITHIN this panel only (equity or crypto separately) - never
    across a merged calendar.
    """
    df = prices.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")[price_col].pct_change()
    return df[["ticker", "date", "ret"]].dropna(subset=["ret"]).reset_index(drop=True)


def combined_returns_panel(equity_returns: pd.DataFrame, crypto_returns: pd.DataFrame) -> pd.DataFrame:
    """Left-merge crypto returns onto the equity trading calendar.

    Both inputs must already be returns computed within their own panel - this
    intentionally drops weekend-only crypto moves a fund trading on equity
    days could not act on.
    """
    equity_dates = equity_returns[["date"]].drop_duplicates()
    equity_side = equity_returns.assign(asset_class="equity")
    crypto_on_equity_days = (crypto_returns.merge(equity_dates, on="date", how="inner")
                                            .assign(asset_class="crypto"))
    return pd.concat([equity_side, crypto_on_equity_days], ignore_index=True)


def descriptive_stats_by_asset_class(returns_panel: pd.DataFrame) -> pd.DataFrame:
    """Mean, vol, min, max, skew, kurtosis of returns by asset class."""
    g = returns_panel.groupby("asset_class")["ret"]
    return pd.DataFrame({
        "mean": g.mean(),
        "std": g.std(),
        "min": g.min(),
        "max": g.max(),
        "skew": g.skew(),
        "kurtosis": g.apply(pd.Series.kurt),
    })


def assemble_headline_panel(headlines: pd.DataFrame, trading_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Assemble the headlines into a daily panel per ticker and sector.

    Maps each headline to its equity trading day (same day if it's a trading
    day, otherwise the next one). Headline dates are UTC tz-aware; the price
    calendar is tz-naive, so we strip the timezone before aligning. Keeps raw
    headline text - no stopword stripping (VADER needs it).
    """
    df = headlines.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    calendar = pd.DatetimeIndex(sorted(pd.Series(trading_calendar).unique()))

    def _next_trading_day(d):
        idx = calendar.searchsorted(d)
        return calendar[idx] if idx < len(calendar) else pd.NaT

    df["trading_day"] = df["date"].apply(_next_trading_day)
    df = df.dropna(subset=["trading_day"])

    panel = (df.groupby(["ticker", "sector", "trading_day"])["title"]
               .apply(list)
               .reset_index()
               .rename(columns={"trading_day": "date", "title": "headlines"}))
    panel["n_headlines"] = panel["headlines"].apply(len)
    return panel
