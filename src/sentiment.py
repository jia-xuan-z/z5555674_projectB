"""Station 3 - sentiment model and sector index from news headlines.

Scores each headline with VADER, aggregates to a per-ticker-day mean score,
then builds an equal-weight sector index. VADER needs a one-time
nltk.download('vader_lexicon') before it scores (a build step, run in
scripts/run_part_b.py - the deployed app never imports nltk).
"""
from __future__ import annotations

import pandas as pd


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score each ticker-day's headlines with VADER; mean compound score per row.

    `panel` is the output of features.assemble_headline_panel (one row per
    ticker-day with a list of raw headline strings). VADER relies on casing,
    punctuation, and negation, so the headlines are scored unmodified.
    """
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    def _mean_compound(headlines):
        scores = [sia.polarity_scores(h)["compound"] for h in headlines]
        return sum(scores) / len(scores) if scores else float("nan")

    out = panel.copy()
    out["sentiment"] = out["headlines"].apply(_mean_compound)
    return out[["ticker", "sector", "date", "sentiment", "n_headlines"]]


def sector_sentiment_index(scores: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight sector sentiment index, one row per (sector, date).

    `scores` (from score_headlines) only has rows for ticker-days that had at
    least one headline. Ticker-days with no headline forward-fill the
    ticker's last known score before averaging equal-weight across tickers
    within each sector-day: silence is not evidence of neutral sentiment
    (treating it as 0 would pull sparsely-covered sectors - Materials,
    Utilities, Real Estate - artificially toward neutral), and dropping the
    ticker that day would starve those same thin sectors of signal on most
    days. A ticker before its first-ever headline stays NaN and is excluded.
    """
    calendar = pd.DatetimeIndex(sorted(scores["date"].unique()))
    ticker_sector = scores[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]

    filled = []
    for ticker, g in scores.groupby("ticker"):
        s = g.set_index("date")["sentiment"].reindex(calendar).ffill()
        filled.append(pd.DataFrame({
            "ticker": ticker,
            "sector": ticker_sector[ticker],
            "date": calendar,
            "sentiment": s.values,
        }))
    long_filled = pd.concat(filled, ignore_index=True).dropna(subset=["sentiment"])

    index = (long_filled.groupby(["sector", "date"])["sentiment"]
                         .mean()
                         .reset_index()
                         .rename(columns={"sentiment": "sentiment_index"}))
    return index.sort_values(["sector", "date"]).reset_index(drop=True)
