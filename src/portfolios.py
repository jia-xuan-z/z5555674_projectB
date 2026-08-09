"""Station 3 - funds: optimal portfolios + out-of-sample backtest.

Five long-only optimisation methods - equal-weight, minimum-variance,
maximum-Sharpe (mean-variance tangency), risk parity, and mean-CVaR - each
run as a walk-forward out-of-sample backtest: weights at a rebalance date
are estimated from a trailing window of PAST returns only, then held
fixed and applied to realised returns until the next rebalance. No
look-ahead: the estimation window for a rebalance on date t covers
[t-window, t), strictly before t.

Equal-weight uses neither mean nor covariance - it is the naive 1/N
benchmark that DeMiguel, Garlappi & Uppal (2009) find is hard for
"optimised" methods to beat out of sample. It is included precisely to
let the report state whether the estimated methods actually add value
over doing nothing clever, not only to compare them against each other.

Mean-CVaR replaces max-Sharpe's volatility denominator with Conditional
Value at Risk - the average loss in the worst tail of outcomes - so it
targets tail risk directly rather than the whole distribution's spread
(Rockafellar & Uryasev, 2000). Because CVaR is a sample quantile of
realised portfolio returns, not a closed-form function of mean/cov like
variance, its weight function additionally needs the raw estimation
window of returns - every _*_weights() function below takes a `window`
argument for a uniform dispatch signature, even though only mean-CVaR
uses it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

METHODS = ("equal_weight", "min_variance", "max_sharpe", "risk_parity", "mean_cvar")


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


def _equal_weight_weights(mean: np.ndarray, cov: np.ndarray, window: pd.DataFrame) -> tuple[np.ndarray, bool]:
    n = cov.shape[0]
    return np.repeat(1 / n, n), True


def _min_variance_weights(mean: np.ndarray, cov: np.ndarray, window: pd.DataFrame) -> tuple[np.ndarray, bool]:
    n = cov.shape[0]
    w0 = np.repeat(1 / n, n)
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    res = minimize(lambda w: _OBJECTIVE_SCALE * (w @ cov @ w), w0, method="SLSQP",
                    bounds=bounds, constraints=cons, options=_SOLVER_OPTIONS)
    return (res.x if res.success else w0), res.success


def _max_sharpe_weights(mean: np.ndarray, cov: np.ndarray, window: pd.DataFrame, rf: float = 0.0) -> tuple[np.ndarray, bool]:
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


def _risk_parity_weights(mean: np.ndarray, cov: np.ndarray, window: pd.DataFrame) -> tuple[np.ndarray, bool]:
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


def _mean_cvar_weights(mean: np.ndarray, cov: np.ndarray, window: pd.DataFrame,
                        alpha: float = 0.95) -> tuple[np.ndarray, bool]:
    """Minimise CVaR at confidence alpha subject to a floor on expected
    return - Rockafellar & Uryasev's (2000) linear-programming formulation,
    solved exactly with HiGHS rather than as a black-box SLSQP objective.

    A first version maximised (mean - rf) / CVaR with SLSQP, mirroring
    max-Sharpe's shape - but CVaR is a sample quantile of w-dependent
    scenario returns, not a smooth closed-form function of w like
    variance is, and SLSQP (a smooth-gradient method) left up to 13 of 36
    Combined-universe rebalances unconverged. CVaR minimisation subject to
    linear constraints, by contrast, IS exactly representable as a linear
    program via Rockafellar and Uryasev's auxiliary-variable trick, which
    has no such convergence risk - reformulating the problem, not just
    retuning the solver, is what actually fixes this (the same lesson as
    the earlier SLSQP-scaling fix: check whether the solver can even solve
    the problem as posed before trusting its reported status).

    Variables: portfolio weights w (n), a VaR estimate zeta (1), and one
    non-negative shortfall slack u_t per day in the window (T). The
    objective zeta + mean(shortfall beyond zeta) / (1-alpha) equals CVaR_
    alpha at the optimum. `target_return` (the mean constraint that makes
    this "mean-CVaR" rather than plain minimum-CVaR) defaults to the
    average individual asset's own estimated mean return in the window -
    "do not give up more expected return than a typical single asset
    would offer while minimising tail risk" - a principled floor, not an
    arbitrary tuning knob.
    """
    from scipy.optimize import linprog

    n = cov.shape[0]
    R = window.to_numpy()
    T = R.shape[0]
    target_return = float(mean.mean())

    n_vars = n + 1 + T
    c = np.zeros(n_vars)
    c[n] = 1.0
    c[n + 1:] = 1.0 / ((1 - alpha) * T)

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    A_ub = np.zeros((1 + T, n_vars))
    b_ub = np.zeros(1 + T)
    A_ub[0, :n] = -mean
    b_ub[0] = -target_return
    A_ub[1:, :n] = -R
    A_ub[1:, n] = -1.0
    A_ub[1:, n + 1:] = -np.eye(T)

    bounds = [(0.0, 1.0)] * n + [(None, None)] + [(0.0, None)] * T

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")
    if not res.success:
        return np.repeat(1 / n, n), False
    w = np.clip(res.x[:n], 0.0, None)
    total = w.sum()
    return (w / total if total > 0 else np.repeat(1 / n, n)), True


_WEIGHT_FUNCS = {
    "equal_weight": _equal_weight_weights,
    "min_variance": _min_variance_weights,
    "max_sharpe": _max_sharpe_weights,
    "risk_parity": _risk_parity_weights,
    "mean_cvar": _mean_cvar_weights,
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
        w, converged = _WEIGHT_FUNCS[method](mean, cov, est_window)
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
        print(f"  [warning] {method}: {n_failures}/{len(rebalance_dates)} rebalances did not converge")

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


def concentration_metrics(weights: pd.DataFrame) -> dict:
    """Concentration diagnostics from the most recent rebalance's weights:
    latest_max_weight (the single largest holding), herfindahl_index (HHI =
    sum(w_i^2), higher = more concentrated; 1/N for an equal-weight book),
    and effective_n_holdings (1/HHI, the number of EQUAL-sized positions
    that would give the same concentration - e.g. 2.0 means "as concentrated
    as holding exactly 2 names equally", however many names are actually
    nonzero). These describe the CURRENT book only, not a history."""
    latest = weights.iloc[-1]
    hhi = float((latest ** 2).sum())
    return {
        "latest_max_weight": float(latest.max()),
        "herfindahl_index": hhi,
        "effective_n_holdings": float(1.0 / hhi) if hhi > 0 else float("nan"),
    }


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
