"""Station 3 - sentiment model and sector index from news headlines.

Scores each headline with finVADER (VADER extended with two finance word
lists - SentiBigNomics and Henry's list, per Koraub 2023) by default, or
plain VADER for comparison, aggregates to a per-ticker-day mean score, then
builds a standardised equal-weight sector index. Both models need a
one-time nltk.download('vader_lexicon') before they score (a build step,
run in scripts/run_part_b.py - the deployed app never imports nltk or
finvader).

Plain VADER's lexicon is built for everyday/social-media text: finance
terms like "hawkish", "guidance", or "earnings beat" are absent and
contribute nothing, and words that are ordinary accounting vocabulary
(e.g. "debt", "liability", "tax") get scored as generically negative
(Loughran and McDonald, 2011). finVADER adds ~7,500 finance-specific terms
on top of VADER's own lexicon and rules, so it is the model used for the
headline scores that feed the sector index and the fusion tilt.
"""
from __future__ import annotations

import pandas as pd

MODELS = ("finvader", "vader")


def _get_scorer(model: str):
    if model == "finvader":
        # finvader.finvader() rebuilds the analyzer and re-merges its ~7,500-
        # term lexicons on every single call - fine for one headline, far too
        # slow for ~147,000 (confirmed: still running after 3 minutes on a
        # background test). Build the merged analyzer once instead, exactly
        # like finvader's own "use both word lists" branch does internally,
        # and reuse it - this is the "built once, so it is fast" pattern the
        # course's own fear_greed_tools.build_finvader() helper uses.
        from nltk.sentiment import SentimentIntensityAnalyzer
        from finvader.SentiBignomics import lexicon1
        from finvader.Henry import lexicon2

        sentibignomics = {k: v * 0.1 for k, v in lexicon1().items()}
        henry = lexicon2()
        sia = SentimentIntensityAnalyzer()
        sia.lexicon.update({**sentibignomics, **henry})
        return lambda text: sia.polarity_scores(text)["compound"]
    if model == "vader":
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return lambda text: sia.polarity_scores(text)["compound"]
    raise ValueError(f"unknown model {model!r}, choose from {MODELS}")


def score_headlines(panel: pd.DataFrame, model: str = "finvader") -> pd.DataFrame:
    """Score each ticker-day's headlines; mean compound score per row.

    `panel` is the output of features.assemble_headline_panel (one row per
    ticker-day with a list of raw headline strings). Headlines are scored
    unmodified (no casing/punctuation stripping) - both VADER and finVADER
    rely on casing, punctuation, and negation.
    """
    scorer = _get_scorer(model)

    def _mean_compound(headlines):
        scores = [scorer(h) for h in headlines]
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

    Also adds `sentiment_index_z`, the index standardised against its own
    per-sector mean and standard deviation. Raw compound/finVADER scores
    skew positive on average (headlines are mostly mildly upbeat, matching
    the mean of ~53-55 out of 100 the course's own fear-and-greed index
    finds), so a raw index reads "positive" on nearly every day and cannot
    tell an unusually positive day from a normal one. Standardising against
    the sector's own history is what makes the index informative day to
    day, not just its sign.
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
    index = index.sort_values(["sector", "date"]).reset_index(drop=True)

    stats = index.groupby("sector")["sentiment_index"].agg(["mean", "std"])
    index = index.join(stats, on="sector")
    index["sentiment_index_z"] = (index["sentiment_index"] - index["mean"]) / index["std"]
    return index.drop(columns=["mean", "std"])
