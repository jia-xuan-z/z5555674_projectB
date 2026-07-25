"""Station 1 - ETL: load and clean the data.

Reused from my own Part A submission (the brief permits this) - same
integrity checks (duplicates, missing dates, nulls, impossible prices,
outliers), applied here to feed Part B's funds and sentiment model.
"""
from __future__ import annotations

import pandas as pd

from src import data_access


def _duplicate_report(df: pd.DataFrame, subset: list[str]) -> tuple[dict, pd.Series]:
    """Exact-duplicate check on `subset`. Returns a summary dict and a boolean
    mask of the rows to drop (keeps the first occurrence)."""
    dup_mask = df.duplicated(subset=subset, keep="first")
    return {
        "subset": subset,
        "n_duplicates": int(dup_mask.sum()),
        "sample_duplicate_keys": df.loc[dup_mask, subset].head(10).to_dict("records"),
    }, dup_mask


def _missing_dates_report(df: pd.DataFrame, reference_dates=None, freq: str | None = None) -> dict:
    """Compare each ticker's observed dates to a reference trading calendar.

    Pass `reference_dates` explicitly when you have a better source of truth
    (e.g. the union of dates observed across all tickers) - falls back to a
    naive pd.date_range(freq=...) only if none is given.
    """
    if reference_dates is None:
        if freq is None:
            raise ValueError("provide either reference_dates or freq")
        reference_dates = pd.date_range(df["date"].min(), df["date"].max(), freq=freq)
    reference_dates = set(reference_dates)

    missing_by_ticker = {}
    for ticker, g in df.groupby("ticker"):
        observed = set(g["date"])
        n_missing = len(reference_dates - observed)
        if n_missing:
            missing_by_ticker[ticker] = n_missing
    return {
        "n_reference_dates": len(reference_dates),
        "n_tickers_with_gaps": len(missing_by_ticker),
        "total_missing_dates": sum(missing_by_ticker.values()),
        "missing_by_ticker": missing_by_ticker,
    }


def _null_value_report(df: pd.DataFrame, cols: list[str]) -> dict:
    """Count missing (NaN/None) values per column. Distinct from the missing-
    DATE calendar audit - this checks for blank cells within rows that exist."""
    return {c: int(df[c].isna().sum()) for c in cols}


def _impossible_price_report(df: pd.DataFrame,
                              price_cols: tuple[str, ...] = ("open", "high", "low", "close", "adjClose")
                              ) -> dict:
    """Flag rows with a zero or negative price - a price can never be <= 0."""
    bad_mask = (df[list(price_cols)] <= 0).any(axis=1)
    return {
        "n_impossible_rows": int(bad_mask.sum()),
        "by_column": {c: int((df[c] <= 0).sum()) for c in price_cols},
        "sample_rows": df.loc[bad_mask, ["ticker", "date", *price_cols]].head(10).to_dict("records"),
    }


def _outlier_report(df: pd.DataFrame, price_col: str = "adjClose",
                     z_thresh: float = 8.0) -> dict:
    """Flag extreme daily returns (|z-score| > z_thresh) per ticker. These are
    NOT dropped - genuine extreme returns are kept and documented."""
    tmp = df.sort_values(["ticker", "date"]).copy()
    tmp["ret"] = tmp.groupby("ticker")[price_col].pct_change()
    mu, sigma = tmp["ret"].mean(), tmp["ret"].std()
    tmp["z"] = (tmp["ret"] - mu) / sigma
    outliers = tmp.loc[tmp["z"].abs() > z_thresh, ["ticker", "date", "ret", "z"]]
    return {
        "price_col": price_col,
        "z_threshold": z_thresh,
        "n_outliers": len(outliers),
        "outliers": outliers.sort_values("z", key=abs, ascending=False),
    }


def load_clean_equities() -> tuple[pd.DataFrame, dict]:
    """Load equity prices and run the Station 1 integrity checks.

    Returns (clean_df, report). Exact (ticker, date) duplicates are dropped
    (kept the first). Missing dates and return outliers are flagged and
    reported, not dropped.
    """
    df = data_access.load_equity_prices()

    dup_report, dup_mask = _duplicate_report(df, subset=["ticker", "date"])
    df = df.loc[~dup_mask].reset_index(drop=True)

    trading_calendar = pd.DatetimeIndex(sorted(df["date"].unique()))
    price_cols = ["open", "high", "low", "close", "adjClose", "volume"]

    report = {
        "n_rows": len(df),
        "n_tickers": df["ticker"].nunique(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "duplicates": dup_report,
        "missing_dates": _missing_dates_report(df, reference_dates=trading_calendar),
        "null_values": _null_value_report(df, price_cols),
        "impossible_prices": _impossible_price_report(df),
        "outliers": _outlier_report(df),
    }
    return df, report


def load_clean_crypto() -> tuple[pd.DataFrame, dict]:
    """Load crypto prices (365-day calendar) and run the Station 1 integrity checks.

    Caps the sample at 2023-12-31 (10 rows are dated 2024-01-01, outside the
    stated sample period).
    """
    df = data_access.load_crypto_prices()
    df = df.loc[df["date"] <= "2023-12-31"].reset_index(drop=True)

    dup_report, dup_mask = _duplicate_report(df, subset=["ticker", "date"])
    df = df.loc[~dup_mask].reset_index(drop=True)

    price_cols = ["open", "high", "low", "close", "adjClose", "volume"]

    report = {
        "n_rows": len(df),
        "n_tickers": df["ticker"].nunique(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "duplicates": dup_report,
        "missing_dates": _missing_dates_report(df, freq="D"),
        "null_values": _null_value_report(df, price_cols),
        "impossible_prices": _impossible_price_report(df),
        "outliers": _outlier_report(df),
    }
    return df, report


def load_clean_news() -> tuple[pd.DataFrame, dict]:
    """Load news headlines and run the Station 1 integrity checks.

    News duplicates are checked on (ticker, date, title) - NOT ticker-date
    alone, since many distinct headlines legitimately share a ticker-date.
    """
    df = data_access.load_news_headlines()

    dup_report, dup_mask = _duplicate_report(df, subset=["ticker", "date", "title"])
    df = df.loc[~dup_mask].reset_index(drop=True)

    report = {
        "n_rows": len(df),
        "n_tickers": df["ticker"].nunique(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "duplicates": dup_report,
        "null_values": _null_value_report(df, ["title"]),
        "pct_missing_publisher": float(df["publisher"].isna().mean()),
    }
    return df, report
