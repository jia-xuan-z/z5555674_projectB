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

---

# Prompt log - redesigning the fusion comparison figure

## What I wanted
The `fusion_comparison.png` exhibit (base equity fund vs. the sentiment-
tilted version) was hard to read - two growth-of-$1 lines that track each
other almost exactly, so it looked like a single smudged line rather than a
comparison. I wanted a version that actually shows whether the tilt did
anything.

## Prompt(s)
- can you change the visualization，of this figure - the chart's
  visualization felt bad, redesign it.
- can you adjust each line's thickness so the difference between them is more
 visible.
- still not happy, the thickened blue
  line now hides the change entirely; is there a different way to present
  this, does it have to be this kind of line chart.

## What the assistant produced
Three iterations, each addressing the previous one's complaint at face
value rather than questioning the chart form: (1) a two-panel figure adding
a spread/difference sub-panel below the original overlapping lines; (2)
asymmetric line weights - a thick pale base line behind a thin crisp dashed
tilt line - to try to make the overlap itself look intentional; (3) after
the user pushed back a third time, dropped the growth-of-$1 line overlay
entirely and rebuilt the exhibit as stat-callout comparisons (return, vol,
Sharpe, max drawdown, each shown as "base -> fused" with a signed, colour-
coded delta) sitting above a single full-size spread-over-time area chart.

## What was wrong or risky
The first two iterations were the wrong response to the underlying
problem. The base and fused funds really are within a few cents of each
other on almost every day - that IS the finding (a lagged sentiment tilt
with tilt_strength=0.5 is a small nudge, not a redesign of the portfolio),
and no line-weight or dash trick makes two genuinely near-identical series
look visually distinct without overstating a difference that isn't there.
I kept trying to solve a chart-form problem with chart-styling fixes for
two rounds before addressing the actual issue: an overlapping-line chart
is the wrong form when the whole point is a small difference between two
similar series.

## What I changed and why
Replaced the line-overlay approach with a chart form suited to a small
delta: the metrics themselves (already computed and saved in
fusion_comparison.csv) as compact before/after text stats up top, and the
signed spread (fused minus base, in cents per $1) as the one chart, coloured
diverging blue (tilt ahead) / red (tilt behind). This is also just a more
honest picture of the result: the fusion is essentially neutral (Sharpe
0.59 -> 0.58, a few basis points either way through 2021 and reversing
after mid-2022), and the redesigned figure shows that plainly instead of
implying a visual difference that isn't really there.

---

# Prompt log - redesigning the sentiment index figure

## What I wanted
Fix the sector_sentiment_index chart, which had the opposite problem from
the fusion chart: instead of two lines fighting for the same space, it had
all 10 sectors overlaid on one axis in 10 different colours - a rainbow
"spaghetti" chart where no single sector's line could actually be traced,
and the legend took up a quarter of the plot.

## Prompt(s)
-sentiment_index, this chart also looks bad, can you improve it.

## What the assistant produced
Rebuilt the figure as small multiples: a 2x5 grid, one panel per sector,
each a single thin blue line on its own axes, all sharing the same y-scale
so magnitudes stay comparable across panels, plus a shared title/subtitle
above the grid instead of a per-panel title fighting a legend.

## What was wrong or risky
This is the same underlying mistake as the fusion chart, just the opposite
failure mode: too many categorical series crammed onto one axis instead of
two series crammed too close together. Both come from defaulting to "one
line chart with everything on it" instead of asking what chart form the
number of series and the size of the differences actually support. I
should check the other multi-series figures (growth_of_dollar.png, which
overlays all 9 base funds) for the same issue before treating Part B's
figures as finished, rather than waiting for each one to be flagged
individually.

## What I changed and why
Small multiples instead of one shared axis - the standard fix once a
categorical chart passes about 6-8 series (this skill/style guide's own
rule of thumb). Re-ran the full pipeline and visually confirmed each
sector's line is now legible on its own, and the shared y-scale still
lets me compare sectors like RealEstate/Utilities (sparser news, per
DATA_GUIDE.md, and visibly spikier here) against steadier ones like Comm
or Consumer - a pattern the original overlapping chart didn't surface at
all.

---

# Prompt log - redesigning the weights-over-time figure, which surfaced a real optimiser bug

## What I wanted
Fix the weights_over_time chart, which had the opposite failure from the
sentiment chart: 8 top holdings stacked under a giant, undifferentiated
"Other" band that swallowed most of the plot area.

## Prompt(s)
- Pasted the stacked-area chart and said: "改这个吧" - fix this one.

## What the assistant produced
First pass: dropped the stacking and plotted the top-8 holdings as their
own lines (not stacked against "Other") using the validated 8-colour
categorical palette. Re-ran and looked at the result before shipping it.

## What was wrong or risky
The redesigned chart showed all 8 lines flatlining at EXACTLY 2.00% (=1/50,
equal weight) from mid-2021 onward and never moving again for the rest of
the 3-year sample - a pattern too clean to be a real result. This is
precisely the "solver silently stalls" failure the brief warns about
(Appendix / Important Points: "optimisers on tiny daily-return covariances
can silently stall... sanity-check that weights actually change across
methods"). I stopped the chart work and investigated directly: reproduced
a single rebalance (2022-06) outside the pipeline and found scipy's SLSQP
terminated after nit=1 with res.success=True and zero movement from the
equal-weight starting point - the solver's default ftol=1e-6 is an
ABSOLUTE tolerance, and daily-return covariances are ~1e-4, so the
objective's natural scale was already below the tolerance the solver used
to decide it was "done." I then scanned all 9 base backtests for
exact-equal-weight rebalances: Equity min-variance was stuck 31/36 times,
risk parity was stuck 36/36, 39/39, and 36/36 across ALL THREE universes
(100% of every risk-parity fund was silently just equal-weight), Combined/
Crypto min-variance were stuck 1 time each, and max-Sharpe was never
affected (its objective is a ratio, not a raw ~1e-4 quadratic form, so it
was never near the tolerance floor). My earlier n_convergence_failures
counter (added in the Station 3 log entry) did NOT catch this, because
scipy reports success=True for a silent stall - checking res.success is
not the same as checking that the optimiser actually moved.

## What I changed and why
In src/portfolios.py: rescaled the min-variance and risk-parity objectives
by a constant factor (1e4) before passing them to SLSQP - multiplying an
objective by a positive constant does not change its argmin, so this only
fixes the solver's numerics, not the optimisation problem - and tightened
ftol to 1e-14 with maxiter=500 on all three methods as a second line of
defence. Verified directly on the 2022-06 case (nit went from 1 to 13-24,
weights moved by up to 0.15 instead of 0.0) and then re-scanned all 9
backtests: exact-equal-weight rebalances dropped from 144 total (31+36+1+
39+36+1) to 3, and those 3 now correctly report
n_convergence_failures=1 each (a genuine occasional non-convergence,
safely caught and falling back to equal weight for just that one
rebalance, not a silent one). Re-ran the full pipeline: the performance
numbers changed materially, not just the chart - e.g. Equity Min-Variance's
annualised volatility dropped from 15.98% to 12.76% and Sharpe from 0.64 to
0.49, which now makes more economic sense (a genuine minimum-variance
solution should have LOWER volatility than the equal-weight fallback it
was silently defaulting to, not the same or higher). This means every
number in results/tables/performance_metrics.csv, fund_returns.csv, and
every figure derived from min-variance or risk-parity funds was wrong
before this fix - caught only because a chart redesign happened to make
the underlying bug visually obvious, not because I had verified solver
convergence at the level of "did it actually move," only "did it report
success." I should build that stronger check (verify against multiple
random restarts, or compare the found objective value to the equal-weight
objective value) into the standard workflow rather than relying on a
chart to reveal it next time.
