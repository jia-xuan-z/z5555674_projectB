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

# Prompt log - redesigning the weights-over-time figure

## What I wanted
Fix the weights_over_time chart, which had the opposite failure from the
sentiment chart: 8 top holdings stacked under a giant, undifferentiated
"Other" band that swallowed most of the plot area.

## Prompt(s)
- make the visualization of this figure better, plotted the top-8 holdings as their own lines

## What the assistant produced
Dropped the stacking and plotted the top-8 holdings as their own lines
(not stacked against "Other") using the validated 8-colour categorical
palette, so each holding's own weight trajectory is legible instead of
being squeezed into a thin band under one dominant block.

## What was wrong or risky
Re-running the line-per-ticker version to check the output surfaced a
separate, serious problem in the underlying data first (all 8 lines
flatlining at exactly equal weight from mid-2021 onward) - see the next
log entry ("investigating and fixing a silent SLSQP optimiser stall") for
that investigation, since it turned into a real correctness bug fix rather
than a visualisation task and is worth keeping separate. After that fix,
the regenerated line chart had its own, purely visual problems: with 8
lines sharing one axis, the in-plot legend sat on top of some lines' peaks
(WMT/KO around the top of the chart), and two of the 8 categorical colours
- GILD (red) and T (orange) - were close enough to be hard to tell apart
right where lines crossed. Both were flagged directly against a screenshot
rather than found by me.

## What I changed and why
Two rounds of chart-form changes. First: dropped the "Other" bucket
entirely and let the 8 holdings of interest use the full vertical scale as
plain lines, instead of stacking them under one dominant undifferentiated
block (the opposite problem from the sentiment chart's too-many-series
issue). Second, once that version's legend-overlap and colour-adjacency
problems were flagged: replaced the shared-axis line chart with the same
small-multiples treatment already used for the sentiment index - a 2x4
grid, one panel per top-8 ticker, sharing one y-scale for comparability.
Small multiples remove both remaining complaints at once: no legend
competing for plot area (each panel is titled with its own ticker instead),
and no colour-adjacency confusion (every panel uses the same single colour,
so no two tickers are ever competing for the same hue at a crossing point).

---

# Prompt log - investigating and fixing a silent SLSQP optimiser stall

## What I wanted
Determine whether these two optimization methods—min-variance and risk-parity—are actually optimizing portfolios or whether they are quietly degenerating into “equal-weighting.”


## Prompt(s)
The brief mentions that ‘optimizers on tiny daily-return covariances can silently stall.’ I'm concerned that the two optimizers, min_variance and risk_parity, might not actually be optimizing—they might just report success while the weights haven't really changed. Please check this for me: For each rebalance, compare the actual calculated weights with the initial equal weights (1/n) to see how much the difference is; if you find that certain methods or universes are consistently stuck near the equal weights, identify the root cause and fix it—don’t just look at the `res.success` field."


## What the assistant produced
Code level:
src/portfolios.py — the three functions _min_variance_weights, _risk_parity_weights, and _max_sharpe_weights all got two changes: scaling the objective function up by a factor of 10,000 (_OBJECTIVE_SCALE = 1e4) and tightening ftol to 1e-14, so scipy's SLSQP solver no longer thinks it has "converged" just because the raw objective values were too small to register a meaningful change.
Data level:
results/tables/performance_metrics.csv — all 9 funds' return/volatility/Sharpe/drawdown numbers recomputed using the genuinely optimised weights
results/data/fund_returns.csv, fund_weights.csv — the underlying daily returns and weights fully recalculated
Every figure under results/figures/ tied to these funds (weights_over_time, growth_of_dollar, drawdown, sharpe_barplot, fusion_comparison) redrawn from the corrected data
A concrete example of the numbers changing:
Equity Min-Variance: volatility 15.98% → 12.76%, Sharpe 0.64 → 0.49

## What was wrong or risky
Every fund built on min-variance or risk-parity across all three universes
was, before this fix, either fully or mostly just equal-weight wearing a
different label - a real methodological flaw, not a display issue, and one
the brief specifically warned could happen and specifically asked to be
sanity-checked for. My existing convergence check was insufficient because
it only asked "did scipy report success," not "did scipy actually search."

## What I changed and why
In src/portfolios.py: rescaled the min-variance and risk-parity objectives
by a constant factor (1e4) before passing them to SLSQP - multiplying an
objective by a positive constant does not change its argmin, so this only
fixes the solver's numerics, not the optimisation problem - and tightened
ftol to 1e-14 with maxiter=500 on all three methods as a second line of
defence. Verified directly on the 2022-06 case (nit went from 1 to 13-24,
weights moved by up to 0.15 instead of 0.0) and then re-scanned all 9
backtests: exact-equal-weight rebalances dropped from 144 total (31+36+1+
39+36+1) to 3, and those 3 now correctly report n_convergence_failures=1
each (a genuine occasional non-convergence, safely caught and falling back
to equal weight for just that one rebalance, not a silent one). Re-ran the
full pipeline: the performance numbers changed materially, not just the
chart - e.g. Equity Min-Variance's annualised volatility dropped from
15.98% to 12.76% and Sharpe from 0.64 to 0.49, which now makes more
economic sense (a genuine minimum-variance solution should have LOWER
volatility than the equal-weight fallback it was silently defaulting to,
not the same or higher). This means every number in
results/tables/performance_metrics.csv, fund_returns.csv, and every figure
derived from min-variance or risk-parity funds was wrong before this fix -
caught only because a chart redesign happened to make the underlying bug
visually obvious, not because I had verified solver convergence at the
level of "did it actually move," only "did it report success." I should
build that stronger check (verify against multiple random restarts, or
compare the found objective value to the equal-weight objective value)
into the standard workflow rather than relying on a chart to reveal it
next time.

---

# Prompt log - finVADER and a standardised sentiment index (course-lecture-driven)

## What I wanted
After reading the course's own week08 (VADER) and week09 (fear-and-greed
index) lecture PDFs while drafting the report, I wanted to check whether
my sentiment implementation matched what the course itself teaches as
best practice, since both lectures turned out to describe problems that
matched my own results almost exactly.

## Prompt(s)
- after reading these, do you see anything to improve.
- approved implementing
  the three concrete things identified: standardising the sentiment index,
  adding finVADER, and updating the report's explanations/citations.

## What the assistant produced
Two real findings from the lectures, not just report-wording fixes:
(1) week09 explicitly builds and then critiques a raw (non-standardised)
sentiment index - "In levels the index is almost always greedy... A
measure that returns the same answer every day carries no information
about any particular day" - and fixes it by standardising against the
index's own mean/std. My sector_sentiment_index() had exactly that flaw:
it only ever reported the raw average, never standardised. (2) week08 and
week09 both document that plain VADER under-covers finance vocabulary
(hawkish, guidance, impairment score as 0 - absent, not neutral) and that
ordinary accounting words get scored as generically negative (Loughran &
McDonald 2011); the course's own fix is finVADER (VADER + ~7,500 finance
terms from SentiBigNomics and Henry's list), a real installable package,
not something to build from scratch.

## What was wrong or risky
Implementing finVADER directly via the package's own `finvader()` function
was far too slow to run at scale (a background test against all 37,962
ticker-days did not finish in 3 minutes) - the function rebuilds and
re-merges its ~7,500-term lexicon from scratch on every single call,
which is fine for one headline (as shown in the lecture slide) but not
for scoring headlines at the scale this project needs. I would not have
caught this without actually running it end to end rather than trusting
the package's documented usage pattern.

## What I changed and why
In src/sentiment.py: replaced the per-call `finvader()` usage with a
scorer built once (merging the SentiBignomics and Henry lexicons into a
single nltk SentimentIntensityAnalyzer up front, mirroring finvader's own
"use both lists" code path) and reused across all headlines - this is the
"built once, so it is fast" pattern the course's own fear_greed_tools
helper uses. Scoring time dropped from >3 minutes (killed, never
finished) to 11.9 seconds for all 37,962 ticker-day rows. Also added
`sentiment_index_z`, the index standardised per sector against its own
mean/std, and made it - not the raw score - the one plotted in
sentiment_index_figure() and shown in the Streamlit Sentiment tab, with a
diverging red/blue fill (fearful/greedy) instead of a single-hue line.
build_sentiment() in scripts/run_part_b.py now also runs a plain-VADER
pass alongside finVADER purely for comparison, saved to
results/tables/vader_vs_finvader.csv (mean sentiment 0.113 for VADER vs.
0.072 for finVADER; exact-zero rate 23.0% vs. 5.9% - finVADER's larger
vocabulary leaves far fewer headlines with no scored words at all), as
documented evidence for the Innovation criterion rather than an
unsubstantiated claim of using a "better" model. The fusion tilt in
src/fusion.py now runs on finVADER-based sentiment automatically, since it
consumes the same `scores` dataframe build_sentiment() produces - no
separate change needed there, but worth being deliberate that this altered
the fusion result slightly (Sharpe 0.581 with plain VADER before this
change, 0.584 with finVADER now) and re-verifying the full pipeline still
runs end to end and check_handin.py still passes (22 checks, only the
pycache-cleanup reminder left) before treating this as done.

---

# Prompt log - fixing the weights-over-time exhibit to actually compare methods

## What I wanted
Fix a compliance gap flagged against the brief's own wording, not a style
complaint: the brief requires "A portfolio-weights-over-time figure
ACROSS METHODS for at least one fund" (Section 5's required exhibits list),
but the existing figure compared the top 8 holdings within a single
method (Equity Min-Variance) - it never showed more than one method, so
it did not answer what the brief actually asked for.

## Prompt(s)
- weights_over_time.png does not fully meet the requirements now, 
- can you redesign it using "cross method", Comparing the Three Methods Within the Same Asset Scope:
Equity Min-Variance
Equity Max-Sharpe
Equity Risk-Parity
## What the assistant produced
Verified the brief's exact wording first (PROJECT_BRIEF.md line 160)
rather than taking the feedback on faith, then rebuilt
weights_over_time_figure(): instead of small multiples of the top 8
holdings under one method, it is now small multiples of 6 tickers - the
ones where Min-Variance, Max-Sharpe, and Risk-Parity disagree most (by
max-minus-min average weight across the three methods) - with each panel
overlaying all three methods' weight for that one ticker as three
coloured lines, plus a shared legend.

## What was wrong or risky
This was a real requirement I had misread earlier and nobody, including
me, caught it until a later review pass. Comparing holdings within one
method and comparing methods for one holding look superficially similar
(both are "weights over time" charts) but answer different questions, and
only the second is what was actually asked for - a reminder that
"produces a plausible-looking chart" is not the same bar as "answers the
brief's actual exhibit requirement," and I should check exhibit wording
against the brief text directly rather than trusting my own memory of
what it said.

## What I changed and why
Picked tickers by cross-method disagreement (spread = max weight across
methods - min weight across methods) specifically because a ticker all
three methods treat identically has nothing to show in an "across
methods" comparison - the resulting panels (WMT, GE, MRK, PSA, KO, EA)
show genuinely different stories per method: Min-Variance favours WMT
consistently while Max-Sharpe mostly ignores it; Max-Sharpe concentrates
heavily into GE and MRK at points (up to ~50-65%) while the other two
methods barely hold them; PSA gets a large Max-Sharpe allocation for about
a year in 2021-2022 that both other methods never give it, then goes to
zero. Re-ran the full pipeline and check_handin.py (22 checks, only the
pycache reminder) to confirm nothing else broke from the signature change
(fund_name= -> universe=).

---

# Prompt log - fusion tilt-strength robustness table

## What I wanted
Address an open item I had already flagged in an earlier log entry
(Station 3 funds/sentiment/fusion) but never followed up on: tilt_strength
= 0.5 for the sentiment fusion was an unmotivated default, never tested
against alternatives, so I could not say whether the fusion's roughly-
neutral result was specific to that one number or a more general pattern.

## Prompt(s)
- Pasted a review suggestion: test pre-set tilt strengths (0.25, 0.50,
  1.00), build a robustness table of annualised return/volatility/Sharpe/
  max drawdown/turnover across them, and explicitly warned against
  picking whichever value looks best afterwards and reporting only that
  one, since that would be data snooping - report all of them as a
  robustness check instead. Asked "你觉得这个有道理吗" (does this make
  sense) before implementing; after I agreed it was sound and explained
  why (matches an already-flagged gap, correctly avoids re-optimising on
  the OOS period), replied "好的" to proceed.

## What the assistant produced
Added `average_turnover()` to src/portfolios.py (mean one-way turnover
per rebalance, 0.5 x sum of absolute weight changes, excluding the first
rebalance which has no prior weights to compare against) and
`fusion_robustness_table()` to scripts/run_part_b.py, which runs the
fusion at the base fund (tilt_strength=0, i.e. no tilt) plus three preset
strengths (0.25, 0.5, 1.0) chosen before looking at any result, and saves
all four rows to results/tables/fusion_robustness.csv. TILT_STRENGTH
(0.5) stays what every other part of the pipeline uses; this table is
reporting only, never used to silently swap in a "better" value.

## What was wrong or risky
Nothing incorrect, but worth stating plainly: this is confirmatory
evidence gathered on the same out-of-sample period the rest of the
report already evaluates the fusion on, so it demonstrates the base
result generalises across nearby parameter choices, not that it would
hold on genuinely unseen future data - I should not oversell this table
as validating the fusion approach more broadly than that.

## What I changed and why
Ran it once the code was in place: Sharpe declines monotonically as tilt
strength rises (0.5865 at no tilt -> 0.5853 -> 0.5841 -> 0.5814 at
tilt_strength=1.0) and turnover rises monotonically alongside it (0.342 ->
0.343 -> 0.345 -> 0.351), with max drawdown also monotonically worsening.
This is a cleaner, more defensible finding than the single tilt_strength
=0.5 result alone: it shows a consistent direction across the whole
tested range (more tilt = worse risk-adjusted performance and higher
turnover), not noise specific to one arbitrary parameter value. Re-ran
check_handin.py (22 checks, only the pycache reminder) to confirm nothing
broke.
