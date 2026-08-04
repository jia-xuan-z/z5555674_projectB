"""Station 3 - funds: optimal portfolios + out-of-sample backtest.

Three long-only optimisation methods - minimum-variance, maximum-Sharpe
(mean-variance tangency), and risk parity - each run as a walk-forward
out-of-sample backtest: weights at a rebalance date are estimated from a
trailing window of PAST returns only, then held fixed and applied to
realised returns until the next rebalance. No look-ahead: the estimation
window for a rebalance on date t covers [t-window, t), strictly before t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

METHODS = ("min_variance", "max_sharpe", "risk_parity")


def long_to_wide(returns: pd.DataFrame) -> pd.DataFrame:
    """(ticker, date, ret) long panel -> wide date x ticker matrix."""
    return returns.pivot(index="date", columns="ticker", values="ret").sort_index()


# Daily-return covariances are tiny (~1e-4), and scipy's SLSQP default ftol
# (1e-6) is an ABSOLUTE function-value tolerance - against an objective this
# small, the solver can decide it has "converged" after 1 iteration without
# moving away from the equal-weight starting point at all, while still
# reporting success=True. Confirmed empirically: without this scaling, the
# equity min-variance fund silently returned equal weight (1/50 exactly) at
# 31 of 36 rebalances, and risk parity did so at every single rebalance
# across all three universes - the brief warns about exactly this failure
# mode. Multiplying the objective by a constant does not change the argmin,
# so this only fixes the solver's numerics, not the optimisation problem.
_OBJECTIVE_SCALE = 1e4
_SOLVER_OPTIONS = {"ftol": 1e-14, "maxiter": 500}


def _min_variance_weights(mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, bool]:
    n = cov.shape[0]
    w0 = np.repeat(1 / n, n)
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    res = minimize(lambda w: _OBJECTIVE_SCALE * (w @ cov @ w), w0, method="SLSQP",
                    bounds=bounds, constraints=cons, options=_SOLVER_OPTIONS)
    return (res.x if res.success else w0), res.success


def _max_sharpe_weights(mean: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> tuple[np.ndarray, bool]:
    n = cov.shape[0]
    w0 = np.repeat(1 / n, n)
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)

    def neg_sharpe(w):
        port_ret = w @ mean - rf
        port_vol = np.sqrt(w @ cov @ w)
        return -port_ret / port_vol if port_vol > 1e-12 else 0.0

    # Not empirically affected by the scaling issue above (the Sharpe ratio
    # is already an O(1) quantity, not an O(1e-4) one), but the tighter
    # tolerance is free insurance against the same failure mode.
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
                    options=_SOLVER_OPTIONS)
    return (res.x if res.success else w0), res.success


def _risk_parity_weights(mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, bool]:
    n = cov.shape[0]
    w0 = np.repeat(1 / n, n)
    bounds = [(1e-6, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)

    cov_scaled = cov * _OBJECTIVE_SCALE

    def objective(w):
        port_var = w @ cov_scaled @ w
        risk_contrib = w * (cov_scaled @ w)
        target = port_var / n
        return np.sum((risk_contrib - target) ** 2)

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
                    options=_SOLVER_OPTIONS)
    return (res.x if res.success else w0), res.success


_WEIGHT_FUNCS = {
    "min_variance": _min_variance_weights,
    "max_sharpe": _max_sharpe_weights,
    "risk_parity": _risk_parity_weights,
}


def oos_backtest(returns: pd.DataFrame, method: str = "min_variance",
                  window: int = 252, rebalance: str = "MS") -> dict:
    """Walk-forward out-of-sample backtest for one asset universe.

    `returns` is the long (ticker, date, ret) panel for ONE universe
    (equity-only, crypto-only, or combined - call separately per universe).
    `window` is the trailing trading-day estimation window; `rebalance` is a
    pandas offset alias (default 'MS' = first trading day on/after each
    calendar month start).

    Returns dict: method, window, first_live_date, daily_returns (pd.Series),
    weights (pd.DataFrame, rebalance_date x ticker), growth (pd.Series),
    n_convergence_failures (int - rebalances where the SLSQP solver did not
    report success; the brief warns these solvers can silently stall on
    tiny daily-return covariances, so this is checked every rebalance rather
    than assumed).
    """
    if method not in _WEIGHT_FUNCS:
        raise ValueError(f"unknown method {method!r}, choose from {METHODS}")

    wide = long_to_wide(returns)
    dates = wide.index
    if len(dates) <= window:
        raise ValueError(f"only {len(dates)} dates available, need > window={window}")

    month_marker = pd.Series(dates, index=dates).resample(rebalance).first().dropna()
    rebalance_dates = [d for d in month_marker if d >= dates[window]]
    if not rebalance_dates:
        raise ValueError("no rebalance dates after the initial estimation window")

    weight_rows = {}
    daily_rets = []
    n_failures = 0

    for i, rb_date in enumerate(rebalance_dates):
        est_end_loc = dates.get_loc(rb_date)
        est_window = wide.iloc[est_end_loc - window:est_end_loc]
        valid_cols = est_window.columns[est_window.notna().all()]
        est_window = est_window[valid_cols]

        mean = est_window.mean().to_numpy()
        cov = est_window.cov().to_numpy()
        w, converged = _WEIGHT_FUNCS[method](mean, cov)
        if not converged:
            n_failures += 1
        weights = pd.Series(w, index=valid_cols)
        weight_rows[rb_date] = weights

        period_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else dates[-1] + pd.Timedelta(days=1)
        period_returns = wide.loc[(wide.index >= rb_date) & (wide.index < period_end), valid_cols]
        port_ret = period_returns.fillna(0.0) @ weights
        daily_rets.append(port_ret)

    daily_returns = pd.concat(daily_rets).sort_index()
    daily_returns = daily_returns[~daily_returns.index.duplicated(keep="first")]
    weights_df = pd.DataFrame(weight_rows).T.fillna(0.0)
    growth = (1 + daily_returns).cumprod()

    if n_failures:
        print(f"  [warning] {method}: {n_failures}/{len(rebalance_dates)} rebalances did not converge (SLSQP)")

    return {
        "method": method,
        "window": window,
        "first_live_date": rebalance_dates[0],
        "daily_returns": daily_returns,
        "weights": weights_df,
        "growth": growth,
        "n_convergence_failures": n_failures,
    }


def average_turnover(weights: pd.DataFrame) -> float:
    """Mean one-way turnover per rebalance: 0.5 * sum(|w_t - w_t-1|), averaged
    across all rebalances after the first (the first rebalance has no prior
    weights to compare against, so it is excluded). 0.5x is the standard
    one-way convention - it counts a full swap from one name to another as
    one unit of turnover, not two (one sale + one purchase counted twice)."""
    diffs = weights.diff().abs().sum(axis=1).iloc[1:]
    return float(0.5 * diffs.mean())


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> dict:
    """Annualised return, annualised volatility, Sharpe, and max drawdown.

    `periods_per_year` must match the calendar the return series is actually
    on (252 for equity-only, 365 for crypto-only; the combined panel runs on
    equity trading days, so 252).
    """
    mean_ret = daily_returns.mean() * periods_per_year
    vol = daily_returns.std() * np.sqrt(periods_per_year)
    sharpe = (mean_ret - rf) / vol if vol > 1e-12 else float("nan")
    growth = (1 + daily_returns).cumprod()
    running_max = growth.cummax()
    drawdown = growth / running_max - 1
    return {
        "annualised_return": float(mean_ret),
        "annualised_volatility": float(vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }
