# Prompt log - Station 1-2 ETL & features (reused from Part A)

## What I wanted
Bring my own Part A cleaning and feature code (src/etl.py, src/features.py)
into Part B so Station 3 (funds, sentiment) has a clean returns panel and
headline panel to build on. The brief explicitly allows reusing my own Part A
work in Part B.

## Prompt(s)
- "Start Part B - read through the whole z5555674_projectB folder" -
  asked the assistant to read the Part B starter (brief, rubric, starter
  code) before doing anything else.
- "Sounds good" - approved the assistant's proposed roadmap (ETL/features
  reuse first, then portfolios, then sentiment, then fusion, then the app),
  which is what triggered porting Part A's src/etl.py and src/features.py
  into Part B.

## What the assistant produced
Copied my own Part A etl.py (load_clean_equities, load_clean_crypto,
load_clean_news with duplicate/missing-date/null/impossible-price/outlier
checks) and features.py (daily_returns, combined_returns_panel,
descriptive_stats_by_asset_class, assemble_headline_panel) into Part B's
src/ folder unchanged, since the logic doesn't depend on anything Part-A-
specific.

## What was wrong or risky
None found on this pass - re-ran both modules against the live data and
confirmed the same row/ticker counts as my verified Part A run (equity
50,300 rows/50 tickers, crypto 14,610 rows/10 tickers, news 146,836 rows
after removing 2,847 exact duplicates, headline panel 37,962 ticker-day
rows). Because this is a straight port of code I already checked in Part A,
the main risk was a silent copy-paste error - checking the counts against
Part A's known-good numbers was the safeguard.

## What I changed and why
No logic changes. Only re-verified by running against the live data rather
than assuming the port was correct.

---

# Prompt log - Station 3 funds, sentiment index, and fusion extension

## What I wanted
Build the required minimum Station 3 work (a combined equity+crypto fund
with 2+ optimisation methods, a VADER sector sentiment index, a fusion
extension), but scoped up to the higher grade band: equity-only, crypto-only,
AND combined funds, each with 3 methods (minimum-variance, max-Sharpe,
risk parity) - 9 funds total, per the rubric's HD description for Station 3.

## Prompt(s)
- Answered the assistant's scope question ("how many funds/methods for
  Station 3?") by picking the ambitious option over the minimum-viable one:
  equity-only + crypto-only + combined, 3 methods each (min-variance,
  max-Sharpe, risk parity) - aiming for the D/HD band rather than just
  clearing the required minimum.
- Did not specify the estimation window, rebalance frequency, optimiser
  implementation, sentiment no-headline-day handling, or fusion tilt
  formula - left these design choices to the assistant, to be checked
  afterward rather than dictated up front.
- Let the assistant implement src/portfolios.py (walk-forward OOS backtest,
  3 long-only optimisers via scipy.optimize.minimize), src/sentiment.py
  (VADER scoring + sector index), src/fusion.py (sentiment tilt on an
  equity fund's weights), and scripts/run_part_b.py wiring all of it
  together and saving the required results/ files.

## What the assistant produced
oos_backtest() with a 252-day trailing estimation window, monthly
rebalancing (first trading day of the month), weights formed strictly
before the rebalance date. performance_metrics() with per-universe
annualisation (252 for equity/combined, 365 for crypto). sector_sentiment_
index() forward-fills each ticker's last known sentiment before averaging
within sector-day, rather than treating no-headline days as neutral (0) or
dropping them - I need to double-check this choice holds up for the
thinnest sectors (Materials, Utilities, Real Estate) before I defend it in
the report. fusion.apply_sentiment() tilts weights by (1 + tilt_strength *
lagged_sentiment), floored at zero and renormalised.

## What was wrong or risky
Nothing incorrect on the first pass, but I have NOT yet independently
verified: (1) that the 252-day window and monthly rebalance are actually a
good design choice versus alternatives (I accepted the assistant's default
rather than testing sensitivity), (2) that the risk-parity objective
converges reliably across all 9 backtests rather than silently stalling
(the brief explicitly warns about this) - I only checked that weights sum
to 1 and produce plausible Sharpe ratios, not that the optimiser actually
converged (res.success) at every rebalance, and (3) that a 0.5 tilt strength
for the fusion is a deliberate, motivated choice rather than an arbitrary
default. I ran the full pipeline once and the numbers look economically
sensible (crypto funds show the 2021 bull run and crash, the fusion has a
small negative effect on Sharpe which the brief says is fine to report
honestly), but I still need to read through portfolios.py myself and
understand the optimisation code before I write it up as my own analysis.

## What I changed and why
Checked point (2) directly: reproduced the rebalance loop for the equity
universe outside the main pipeline and inspected scipy's res.success for
every SLSQP call across all 36 rebalances x 3 methods - zero failures.
Then, instead of leaving this as a one-off check, changed
_min_variance_weights/_max_sharpe_weights/_risk_parity_weights in
src/portfolios.py to return (weights, converged) and added an
n_convergence_failures counter to oos_backtest()'s return dict, with a
printed warning if any rebalance fails to converge. Re-ran the full
scripts/run_part_b.py pipeline (all 9 base funds + the fused fund) and
confirmed no convergence warnings printed for any universe or method - so
this is not just a spot-check on equity, it is now a standing, automatic
check on every run, which is the more defensible thing to state in the
report than "I checked it once." Points (1) [window/rebalance sensitivity]
and (3) [tilt strength justification] are still open - I have not yet run
alternative windows/rebalance frequencies or tilt strengths to show the
0.5 tilt and 252-day/monthly choices are deliberate rather than arbitrary
defaults, and I should do that before writing the report's methodology
section.

---

# Prompt log - Streamlit app (Station 4)

## What I wanted
A working local app covering the investor journey: compare funds, read a
fact sheet, set an allocation, see the sentiment analytics - reading only
precomputed results/ artifacts (no nltk, no backtest recompute), per the
brief's Station 4 requirement.

## Prompt(s)
- "Keep going" - continued from the finished Station 3 code straight into
  building streamlit_app.py, with no further scope instructions from me;
  the fund picker / fact sheet / allocation / sentiment layout was the
  assistant's design.

## What the assistant produced
A 3-tab app (Funds, Sentiment, Build a portfolio) reading fund_returns.csv,
fund_weights.csv, sector_sentiment_index.csv, and performance_metrics.csv.
Ran it locally via `streamlit run` and checked all three tabs in a browser.

## What was wrong or risky
The "Build a portfolio" allocation tab had a real bug: it pivoted all 10
funds' daily returns to one wide table and filled missing dates with 0.0
before blending. Crypto funds start 2020-10-01 (365-day calendar) but
equity/combined funds only start 2021-01-04 (252-day calendar, after the
252-day estimation window) - so a 100%-Equity-Min-Variance allocation was
silently padded with ~3 months of fabricated "0% return" days at the start,
understating its annualised return (6.51% shown vs. the fund's real 10.26%
in performance_metrics.csv) and volatility. I caught this by checking a
100%-single-fund allocation against that fund's own row in the metrics
table - they should match exactly, and did not.

## What I changed and why
Changed the blend to select only the funds with nonzero weight, then
`.dropna()` the pivoted wide table instead of `.fillna(0.0)`, restricting
the blended return series to dates where every selected fund is actually
live. Re-tested: 100% Equity Min-Variance now reproduces 10.26% / 15.98% /
0.64 / -20.32% exactly, and a 50/50 Equity+Crypto Min-Variance blend runs
without error over the intersected date range. This is a fusion-adjacent
lesson worth restating in the report if I discuss the app: never blend
return series with different native calendars by zero-filling the gap.
