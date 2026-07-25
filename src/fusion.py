"""Station 3 (extension) - fuse sentiment into the equity funds.

Tilts an equity fund's base (unfused) weights toward names with more
positive recent sentiment, look-ahead safe: the sentiment used at a
rebalance on date t is each ticker's last known score strictly BEFORE t (at
least one trading day's lag, per the brief). Tested against the base fund -
an honest negative result, explained, still counts as valid work.
"""
from __future__ import annotations

import pandas as pd


def apply_sentiment(weights: pd.DataFrame, scores: pd.DataFrame, tilt_strength: float = 0.5) -> pd.DataFrame:
    """Tilt base weights toward positive-sentiment names at each rebalance.

    `weights`: rebalance_date x ticker (an equity fund's base weights, from
    portfolios.oos_backtest). `scores`: long (ticker, date, sentiment) from
    sentiment.score_headlines. For each rebalance date, only sentiment
    strictly before that date is used, and each ticker's LAST known score is
    carried forward (headlines are sparse, so most rebalance dates have no
    same-week headline for a given name).

    new_weight_i is proportional to base_weight_i * (1 + tilt_strength * s_i),
    floored at 0 (long-only) and renormalised to sum to 1. A ticker with no
    prior sentiment observation gets s_i = 0 (no tilt).
    """
    scores_wide = scores.pivot(index="date", columns="ticker", values="sentiment").sort_index()

    tilted_rows = {}
    for rb_date in weights.index:
        prior = scores_wide.loc[scores_wide.index < rb_date]
        latest_scores = prior.ffill().iloc[-1] if len(prior) else pd.Series(dtype=float)
        base_w = weights.loc[rb_date]
        s = latest_scores.reindex(base_w.index).fillna(0.0)
        tilted = (base_w * (1 + tilt_strength * s)).clip(lower=0.0)
        total = tilted.sum()
        tilted_rows[rb_date] = tilted / total if total > 0 else base_w
    return pd.DataFrame(tilted_rows).T


def apply_weights_to_returns(tilted_weights: pd.DataFrame, wide_returns: pd.DataFrame) -> pd.Series:
    """Realised daily returns of a rebalance-date weight schedule.

    Mirrors portfolios.oos_backtest's hold-between-rebalances logic so the
    fused fund's daily returns are directly comparable to the base fund's.
    """
    rb_dates = list(tilted_weights.index)
    cols = tilted_weights.columns
    daily_rets = []
    for i, rb_date in enumerate(rb_dates):
        period_end = rb_dates[i + 1] if i + 1 < len(rb_dates) else wide_returns.index[-1] + pd.Timedelta(days=1)
        period_returns = wide_returns.loc[(wide_returns.index >= rb_date) & (wide_returns.index < period_end), cols]
        daily_rets.append(period_returns.fillna(0.0) @ tilted_weights.loc[rb_date])
    out = pd.concat(daily_rets).sort_index()
    return out[~out.index.duplicated(keep="first")]
