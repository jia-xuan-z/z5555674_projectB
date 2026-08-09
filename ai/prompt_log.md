# Prompt log - Station 1-2 ETL & features (reused from Part A)

## What I wanted
Bring my own Part A cleaning and feature code (src/etl.py, src/features.py)
into Part B so Station 3 (funds, sentiment) has a clean returns panel and
headline panel to build on. The brief explicitly allows reusing my own Part A
work in Part B.

## Prompt(s)
porting Part A's src/etl.py and src/features.py into Part B.

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
- using the method:
  equity-only + crypto-only + combined, 3 methods each (min-variance,
  max-Sharpe, risk parity) 
- Do not specify the estimation window, rebalance frequency, optimiser
  implementation, sentiment no-headline-day handling, or fusion tilt
  formula - left these design choices to be checked
  afterward 
- implement src/portfolios.py (walk-forward OOS backtest,
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
honestly), but I still need double check whether it is proper.

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
confirmed no convergence warnings printed for any universe or method 

---

# Prompt log - Streamlit app (Station 4)

## What I wanted
A working local app covering the investor journey: compare funds, read a
fact sheet, set an allocation, see the sentiment analytics - reading only
precomputed results/ artifacts (no nltk, no backtest recompute), per the
brief's Station 4 requirement.

## Prompt(s)
- continued from the finished Station 3 code straight into
  building streamlit_app.py
  

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
without error over the intersected date range. 
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
- make the visualization of this figure better, 
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
- test pre-set tilt strengths (0.25, 0.50,
  1.00), build a robustness table of annualised return/volatility/Sharpe/
  max drawdown/turnover across them, and explicitly warned against
  picking whichever value looks best afterwards and reporting only that
  one, since that would be data snooping 

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

---

# Prompt log - concentration and turnover diagnostics for the performance table

## What I wanted
Respond to a review comment flagging that Combined Max-Sharpe's most
recent holdings are ~64% concentrated in just two names (GE 45.3%, NVDA
18.7%), with a documented, cited critique (naive mean-variance/max-Sharpe
optimisation is known to be highly sensitive to noisy historical mean-
return estimates and prone to corner solutions under long-only
constraints - Michaud 1989) and a request to add diagnostics (turnover,
latest max weight, effective N holdings, Herfindahl index) to the
performance table.

## Prompt(s)
- You should include turnover or concentration metrics. the current Combined Max-Sharpe's latest weightings are highly concentrated.
GE：45.32%
NVDA：18.71%
these two equity has already account for about 64%, which poses a significant risk for a product aimed at retail investors.
you should add: 
avg_turnover
latest_max_weight
herfindahl_index
effective_n_holdings 
## What the assistant produced
Added `concentration_metrics()` to src/portfolios.py (latest_max_weight,
herfindahl_index = sum(w_i^2) from the most recent rebalance, and
effective_n_holdings = 1/HHI), and wired both it and the already-built
`average_turnover()` into save_performance_table() in
scripts/run_part_b.py, so every fund's row in
results/tables/performance_metrics.csv now also reports these four
diagnostics.

## What was wrong or risky
None - this only reads already-computed weights and adds columns; it does
not change any fund's actual weights, returns, or backtest results, so
there was no risk of silently altering numbers already used elsewhere in
the report.

## What I changed and why
Ran the full pipeline and confirmed the diagnostics tell a clear, honest
story that matches the review's original concern: effective N holdings is
1.4-3.8 for every Max-Sharpe fund (crypto is the most extreme, 1.43 - the
book is effectively betting on barely more than one name) versus 8.8-9.6
for Min-Variance funds and 44.6-51.1 for Risk-Parity funds, which are
genuinely well-diversified. This is exactly the concentration-risk
evidence the report needs, gathered without touching any fund's actual
construction - the weight-cap/shrinkage idea itself is deferred to
Section 6 as a recommendation rather than implemented, per the scope
decision above. check_handin.py still passes (22 checks).

---

# Prompt log - splitting growth-of-$1 into one panel per universe

## What I wanted
Fix a real readability problem in growth_of_dollar.png flagged by review:
all 9 base funds sharing one linear y-axis meant crypto's much larger
growth (up to ~10x) compressed equity and combined (both under ~2x) into
a flat band near the bottom of the chart, on top of a 9-way legend being
hard to parse.

## Prompt(s)
- may be difficult to read due to the clutter, and the high growth of crypto may obscure the differences between equity and combined funds.
can you improve

## What the assistant produced
Chose the three-panel option over log scale (log would need justifying
in the report and still visually favours whichever series has the
smoothest compounding, rather than just fixing the scale mismatch) and
rebuilt growth_figure() to match the by-universe/by-method pattern
already used for weights_over_time_figure(): one panel per universe
(Equity, Crypto, Combined), each with its own y-axis (deliberately NOT
shared, since forcing one scale across panels would just recreate the
same crypto-dominates problem one level up), three method-coloured lines
per panel, one shared legend at the top instead of three separate
in-plot ones.

## What was wrong or risky
None found - this is a pure presentation change, the underlying growth
series are unchanged.

## What I changed and why
Splitting by universe with independent y-scales let each panel actually
be read: Equity shows the three methods staying fairly close together
throughout; Crypto shows Max-Sharpe surging hardest through the 2021
peak then crashing furthest and never recovering (ending near 1.0,
essentially flat over 3+ years) while Min-Variance and Risk-Parity both
recover strongly into 2023; Combined shows Max-Sharpe clearly ahead of
the other two for almost the entire sample. None of this was visible in
the single shared-axis version. Re-ran the full pipeline and
check_handin.py (22 checks) to confirm nothing else broke.

---

# Prompt log - app risk disclosures and fact-sheet methodology fields

## What I wanted
Add standard investment-product disclosures and extra methodology fields
to the Streamlit fact sheet, per a review suggestion, without recomputing
anything - the app already has every number needed (first live date is
the growth series' own start date, concentration is the diagnostics just
added to performance_metrics.csv, window/rebalance are pipeline
constants).

## Prompt(s)
- current app covers the full investor journey
  (compare/fact sheet/allocation/sentiment) but as an investment product
  should add short disclosures (backtested performance is not actual performance, past
  performance doesn't guarantee future results, costs/taxes/fees
  excluded, crypto materially riskier, sentiment is headline-only) and
  extra fact-sheet fields (first live backtest date, estimation window,
  rebalance frequency, current concentration, asset-class label).

## What the assistant produced
A collapsed `st.expander("Important disclosures")` under the title with
the five suggested bullet points, and one extra caption line on the fact
sheet showing asset class (parsed from the fund name), first live date
(the growth series' own earliest date), estimation window and rebalance
frequency (pipeline constants, same for every fund), and current
concentration (largest holding % and effective N holdings, both already
computed by the concentration diagnostics added earlier).

## What was wrong or risky
Testing this locally hit a flaky repeat of an earlier issue: the first
browser tab's `get_page_text` kept returning an empty body for the outer
Streamlit Cloud-style wrapper page even after 20+ seconds of waiting,
despite all JS/CSS assets loading with 200s and no console errors. Opening
a completely fresh browser tab and navigating there resolved it
immediately. Not a bug in the app - a rendering/connection quirk in the
first tab - but worth remembering rather than assuming the app itself was
broken.

## What I changed and why
No logic changes to any fund's data - purely additive UI/text. Verified
in the fresh tab: the disclosures expander opens and shows all five
bullets, and the fact sheet caption correctly shows, e.g. for Combined
Max-Sharpe: "Asset class: Combined | First live date: 2021-01-04 |
Estimation window: 252 trading days, rolled forward | Rebalance: monthly
| Current concentration: largest holding 45.3%, effective 3.8 equal-sized
holdings" - matching the numbers already verified in
performance_metrics.csv. check_handin.py still passes (22 checks).

---

# Prompt log - formatting the new diagnostic columns in the app table

## What I wanted
Fix a display inconsistency the user spotted after deploying and
rebooting the app: the "Compare funds" table's newer columns
(avg_turnover, latest_max_weight, herfindahl_index, effective_n_holdings)
showed their raw snake_case CSV column names, while the original six
columns had been renamed to readable labels ("Ann. return (%)", "Sharpe",
etc.) - inconsistent formatting in a table meant for a non-technical
investor reader.

## Prompt(s)
- The names of these four newly added column headers are still the original code variable names (with underscore-based naming conventions like “avg_turnover”), which don't match the formatted column names next to them, such as “Ann. return (%)” and “Sharpe.” They look a bit out of place. can you fix this

## What the assistant produced
In streamlit_app.py: scaled avg_turnover and latest_max_weight to
percentages (rounded to 1dp) and effective_n_holdings to 1dp, dropped
herfindahl_index from the displayed table entirely (effective_n_holdings
is the same information in a more directly interpretable form - showing
both is redundant for a reader who is not going to compute with the raw
HHI), renamed the three kept columns to "Avg turnover (%)", "Largest
holding (%)", "Effective N holdings", and added a one-line caption
explaining what each means.

## What was wrong or risky
None - display-only change, verified the formatting logic against the
real CSV data before committing (all 10 funds' rows print correctly,
e.g. Crypto Max-Sharpe shows 32.5% turnover, 82.1% largest holding, 1.4
effective holdings, matching the raw numbers already verified earlier).

## What I changed and why
Purely cosmetic/readability - no underlying data changed. The point was
consistency: a reader scanning the table should not hit a wall of
unlabelled snake_case columns partway through an otherwise clean table.

---

# Prompt log - sentiment index figure: terminology, labelling, and readability pass

## What I wanted
make sentiment_index.png look better

## Prompt(s)
- for sentiment_index.png, shorten the subtitle;make it less crowded, and accpet the changes that what you think can make it better

## What the assistant produced 
checking point 3 against context/DATA_GUIDE.md
before acting (Comm/RealEstate/Tech map 1:1 to GICS names, but "Consumer"
mixes Discretionary tickers - DIS, NKE, SBUX - with Staples tickers - WMT,
KO - so relabelling it as either specific GICS sector would misstate the
data; kept "Consumer" and added a caption note about the merge instead),
kept the blue/red diverging pair since it is the
validated diverging pair from the colour system already used consistently
elsewhere in this report (blue/orange is a categorical, not diverging,
pairing in that system), but did reduce fill opacity as suggested.
Rebuilt sentiment_index_figure() with all the agreed changes: renamed
sector titles, GICS-ordered panels, date ticks only on the bottom row
(yearly, not rotated), value ticks only on the left column, a thicker/more
opaque zero line, lower fill alpha, a shorter title/subtitle, and a
two-line footnote (the standardisation caveat and the Consumer-merge note,
split onto two lines after the first version overflowed the figure width).

## What was wrong or risky
The first version of the added footnote text was too long for the figure
width at the given font size and got silently truncated off the right
edge - caught by looking at the actual rendered PNG rather than assuming
the fig.text() call worked because it didn't error. Fixed by splitting
into two shorter fig.text() calls and increasing the bottom margin to fit
both lines.

## What I changed and why
Every change here is presentation/labelling only - no underlying sentiment
data, scores, or index values changed. Re-ran the full pipeline and
check_handin.py (22 checks) to confirm nothing else broke from the
function rewrite.

---

# Prompt log - table images for the report

## What I wanted
the figure of each table

## Prompt(s)
- generate image versions of every table, saved to results/tables/.

## What the assistant produced
A `save_table_image()` helper in scripts/run_part_b.py (matplotlib table
render, same palette as the other figures) and a `table_images()` function
that builds all 5 images directly from the already-committed CSVs
(performance_metrics.csv, vader_vs_finvader.csv, fusion_comparison.csv,
fusion_robustness.csv), with no numbers retyped or re-derived - wired into
main() so it stays part of the one-command reproducible pipeline.

## What was wrong or risky
Two rendering bugs caught by actually looking at the output PNGs rather
than trusting the code ran without error: (1) equal-width columns
truncated the "Fund" column's longer names (e.g. "Equity Min-Varia...")
the moment any numeric column sat next to it; (2) the title/subtitle used
a FIXED FRACTIONAL offset from the top of the figure, which meant the
literal pixel gap between them shrank on short (2-3 row) tables until the
two lines of text overlapped and became unreadable (Table 3, only 2 data
rows, was the clearest case).

## What I changed and why
(1) Column widths are now computed from actual max text length per column
(header vs. every cell) instead of splitting the width evenly, so a long
first column gets proportionally more space. (2) Title/subtitle vertical
offsets are computed as fixed INCH amounts (converted to a figure-fraction
using the actual figure height) rather than a flat fraction, so the gap
between them stays legible regardless of how few rows the table has, and
the base figure-height constant was increased to give short tables more
headroom generally. Re-rendered all 5 and visually confirmed each one: no
truncated text, no overlapping title/subtitle. Re-ran the full pipeline
and check_handin.py (21 checks - the report.pdf reminder is expected,
since the actual editable report lives outside this repo while it's being
drafted) to confirm nothing else broke.

---

---

# Prompt log - fixing the allocation tab's fixed-252 annualisation bug

## What I wanted
Fix a real, if minor, bug flagged by review: the "Build a portfolio" tab
always annualised the blended return/volatility with 252, regardless of
which funds were selected, so a 100%-crypto allocation (whose dates run on
the 365-day calendar) had its annualised numbers understated.

## Prompt(s)
- Pasted two review points together: (1) ai/prompt_log.md had uncommitted
  changes and needed pushing since it's graded, (2) the fixed-252
  annualisation issue in streamlit_app.py:177, both labelled clearly
  (point 2 as "Minor display issue only"). Asked what point 2 meant before
  acting; after I explained both, replied "好的" to fix them.

## What the assistant produced
Committed the outstanding prompt_log.md changes first. Then, in
streamlit_app.py's allocation tab, replaced the hardcoded `* 252` /
`sqrt(252)` with a check: if every selected fund's name starts with
"Crypto", annualise with 365; otherwise 252. This follows from how the
blend's date range is actually built - `dropna()` restricts it to the
INTERSECTION of every selected fund's live dates, and an equity or
combined fund only has data on the ~252-day equity calendar, so including
even one such fund caps the whole blend at 252 regardless of what else is
mixed in. Only an all-crypto blend actually runs on the 365-day calendar.

## What was wrong or risky
None in the fix itself, but worth noting the reasoning isn't obvious from
the symptom alone - a naive fix might have tried to average or weight the
252/365 factors by allocation percentage, which would be wrong; the
correct behaviour depends on which calendar the INTERSECTED dates actually
fall on, not on the crypto allocation share.

## What I changed and why
Verified directly against real data rather than trusting the logic alone:
a 100%-Crypto-Min-Variance blend now reproduces 81.97%/65.26% (365-day
annualisation), matching performance_metrics.csv's 82.0%/65.3% row
exactly, versus the old always-252 code understating it. A 50/50 Equity+
Crypto blend correctly falls back to 252 with 753 dates (the equity
calendar length), confirming the intersection logic. check_handin.py
still passes (21 checks).

---

# Prompt log - decluttering the sentiment index small multiples

## What I wanted
sentiment_index.png (the 2x5 small-multiples sector figure) still felt
too crowded even after the earlier GICS-ordering/labelling pass, and I
wanted to know whether it was genuinely a design problem or whether I was
just looking at a stale file that didn't match the current plotting code.

## Prompt(s)
-  sentiment_index.png and asked whether it could be
  redesigned to feel less crowded.

## What the assistant produced
First checked whether the on-disk PNG matched the current
sentiment_index_figure() logic in scripts/run_part_b.py, since the panel
order looked like it could be stale relative to an earlier redesign
described in this log. Regenerated the figure directly from the
committed results/data/sector_sentiment_index.csv (no need to rerun ETL
or the backtests) to compare. The first regeneration attempt produced a
byte-identical file and print a success message, which read as "the code
already matches, nothing to fix" - but this conclusion was wrong, caught
on a second look (see below). Once confirmed the code path was actually
being exercised, made three changes to sentiment_index_figure(): enlarged
the figure (12x5.4in -> 13.5x6.6in) and widened the row/column spacing
(hspace 0.4->0.55, wspace 0.1->0.16) for more room per panel, and removed
the dark ink_secondary outline line that was drawn on top of each fill,
raising the fill alpha from 0.35 to 0.5 so the shape stays legible
without it - with 10 panels already competing for attention, that second
line layer was adding visual noise without adding information.

## What was wrong or risky
The first regeneration attempt was silently writing to the wrong
location. The script used `exec()` on the file's source text (everything
before `def main():`) to reuse sentiment_index_figure() without a full
pipeline run, but that source text includes
`ROOT = pathlib.Path(__file__).resolve().parent.parent` - and __file__
is not defined when code runs from a heredoc/stdin, not a real script
file. Because exec() ran in the same namespace as my own driver script, a
NameError there would have been obvious, but instead __file__ resolved
to some Python-internal placeholder, so ROOT and then RESULTS resolved to
a path that was not the project's actual results/ folder, and
fig.savefig() wrote there without error. The regenerated PNG's dimensions
(1800x810, matching the OLD figsize) versus what the new figsize should
have produced (2025x990) was the tell - the file in results/figures/ had
simply never been touched by either "successful" regeneration run, and a
size check was what caught it, not the absence of an exception.

## What I changed and why
Set `__file__` explicitly to the real scripts/run_part_b.py path before
exec()'ing its source, so ROOT and RESULTS resolve to the project's real
results/ folder. Verified the fix by checking the output file's
dimensions (2025x990) and MD5 hash changed before treating the figure as
regenerated. This is a narrow case, but the general lesson - a script
that prints a success message and produces a plausible-sized file is not
proof it wrote where I think it did - matches the standing instruction to
verify actual output rather than trust that code ran correctly, the same
principle behind the earlier SLSQP convergence check.

---

# Prompt log - adding an Equal-Weight (1/N) benchmark

## What I wanted
After reading the course's own week10 revision slides, I noticed the
instructor's reference project runs four methods (equal-weight, minimum-
variance, maximum-Sharpe, risk parity), not three, and explicitly frames
equal-weight as a benchmark that "clever" methods often fail to beat out
of sample (citing DeMiguel, Garlappi & Uppal 2009). My own funds only had
the three estimated methods, so I could not say whether they actually add
value over doing nothing clever - I wanted that comparison.

## Prompt(s)
- can you add Equal-Weight as a fourth method, 

## What the assistant produced
Added `_equal_weight_weights()` to src/portfolios.py (returns 1/N,
trivially converged) and "equal_weight" to portfolios.METHODS, so
build_funds() in scripts/run_part_b.py automatically backtests it across
all three universes - 12 base funds instead of 9. Updated
METHOD_LABELS, growth_figure()'s method_colors, and
sharpe_barplot_figure()'s figure width to include the new fund. Deliberately
did NOT add Equal-Weight to weights_over_time_figure(): that exhibit
compares how the ESTIMATED methods disagree with each other ticker by
ticker, and Equal-Weight is always exactly 1/N on every ticker, so
including it would just measure "distance from a flat line" rather than
genuine three-way disagreement - added an ESTIMATED_METHOD_LABELS
constant to keep that figure scoped to the original three methods
deliberately, not by accident.

## What was wrong or risky
Nothing incorrect, but this changes fund counts in several places at
once (results/data/fund_returns.csv, fund_weights.csv,
performance_metrics.csv, the app's fund list, and every figure built from
funds), so a partial edit could easily have left one exhibit showing 9
funds and another showing 12. Re-ran the full pipeline rather than
patching individual outputs, and checked growth_of_dollar.png,
sharpe_barplot.png, weights_over_time.png, and table1_performance_metrics.png
by eye to confirm each fund count matches what that specific exhibit
should show (12 in three of them, 3 methods only in weights-over-time, as
designed).

## What I changed and why
The result is more informative than I expected: Equity Equal-Weight
posts a Sharpe ratio of 0.82, HIGHER than all three estimated equity
methods (Risk-Parity 0.72, Max-Sharpe 0.59, Min-Variance 0.49). In
Crypto and Combined, Equal-Weight sits mid-pack rather than on top
(Crypto: Min-Variance 1.26 > Risk-Parity 0.98 > Equal-Weight 0.93 >
Max-Sharpe 0.29; Combined: Max-Sharpe 1.05 > Risk-Parity 0.90 >
Equal-Weight 0.76 > Min-Variance 0.52). This is a genuine, unplanned
finding, not something I picked because it looked good - the equity
result in particular directly replicates the pattern the course's own
slides describe (Michaud 1989's estimation-error critique of mean-
variance, and DeMiguel et al. 2009's finding that naive diversification
is hard to beat), using my own data rather than citing the lecture's
numbers. I still need to update the report text (Section 2, the
performance table discussion, and the fact sheet count) to describe 12
funds instead of 9 and discuss this finding - not done yet, since the
report is a separate file I do not edit directly.

---

# Prompt log - building a self-built VADER lexicon extension

## What I wanted
The week10 revision slides say "I strongly encourage you to use your AI
systems and build your own lexicon or your own VADER model" and
separately suggest "extend the VADER lexicon with finance terms, then
have your AI agent rate them and keep the ones raters agree on." I
realised what I actually had was finVADER, someone else's already-built
package (Koraub, 2023) - a legitimate, cited tool, but not something I
built myself. I wanted to actually do the exercise the slides describe,
not just cite a third-party library and call it done.

## Prompt(s)
- asked whether our finVADER setup counted as "building our own VADER
  model" per the lecture's suggestion
- agreed to actually build one: propose candidate finance terms, rate
  them twice independently, keep only the ones that agree

## What the assistant produced
First checked which candidate finance terms were actually missing from
the combined VADER+finVADER vocabulary (13,324 words) rather than
guessing - several gaps turned out to be asymmetric (finVADER has
"hawkish" but not "dovish", "headwind" but not "tailwind", "overbought"
but not "oversold"), which is itself a real, checkable finding, not an
assumption. Narrowed to 13 genuinely absent candidates, rated each twice
independently in a financial-news-headline context, and kept only the 11
where both passes agreed in sign and were within 0.25 of each other on a
-1..+1 scale. Two terms - "deleveraging" and "derisking" - failed the
check (the two passes disagreed on sign, since both can read as prudent
discipline or as a symptom of distress depending on context) and were
dropped rather than resolved by picking whichever score looked better.
Added CUSTOM_LEXICON and a third "custom" model to src/sentiment.py
(plain VADER + these 11 terms only, kept separate from finVADER's lexicon
so the three-way comparison stays clean), and extended
build_sentiment() in scripts/run_part_b.py to score headlines with all
three models and save a three-row comparison to
results/tables/vader_vs_finvader.csv and table2_vader_vs_finvader.png.

## What was wrong or risky
None found in the build itself, but the result needed an honest read
rather than a favourable spin: the custom lexicon barely moves the
aggregate statistics (mean ticker-day sentiment 0.114 vs plain VADER's
0.113, exact-zero rate 22.8% vs 23.0%), nowhere near finVADER's shift
(0.072, 5.9%). This is expected, not a bug - 11 words cannot compete with
finVADER's ~7,500-term extension in aggregate, and the point of the
exercise was the disciplined rating process, not beating finVADER on
these summary numbers. Reporting the small effect size honestly matters
more here than claiming a win.

## What I changed and why
Deliberately kept finVADER, not the new custom lexicon, as the production
model for the sector index and the fusion tilt (src/sentiment.py's
score_headlines default and build_sentiment()'s main scores variable are
unchanged) - finVADER is a validated, published extension with its own
accuracy testing (Koraub, 2023), while the custom lexicon is 11 terms
checked only by this project's own two-pass process. Ran the full
scripts/run_part_b.py pipeline and confirmed every fund's numbers in
performance_metrics.csv are byte-identical to before this change, since
the custom lexicon never touches the production scoring path - only
Table 2 and its underlying CSV changed, from a two-row VADER/finVADER
comparison to a three-row comparison that also reports the self-built
lexicon. check_handin.py still passes (21 checks). The report needs a
matching update - Table 2 is now three rows, Section 3.1's methodology
paragraph should mention the self-built lexicon and its two-pass rating
process, and Section 4's innovation summary should describe the finance
lexicon extension as self-built rather than only "finance-adapted" - not
done yet, since the report is a separate file I do not edit directly.

---

# Prompt log - reinstating fear-and-greed terminology

## What I wanted
A much earlier session decided NOT to call sentiment_index.png a
"fear and greed" index, reasoning that the course's own fear-and-greed
index was "a related but distinct construct built from different
inputs." I had never actually checked that claim against the course's
own material. Looking directly at a screenshot of the week10 revision
slide "A Fear and Greed Index," the course's own index is built by
averaging finVADER headline sentiment across all 50 stocks, rescaling to
0-100, and standardising it - the same construction as this project's
sector index, just aggregated market-wide instead of per sector, and
from the identical input (finVADER headline sentiment, not options data,
put/call ratios, or any of the other inputs a general-purpose fear-and-
greed index like CNN's actually uses). The earlier reasoning was an
unverified assumption, not a checked fact.

## Prompt(s)
- shared the week10 slide screenshot and asked whether the earlier
  "don't call it greedy/fearful" advice still held up
- agreed with the correction and asked to (1) reinstate fear-and-greed
  terminology in the figure and (2) go back and fix the original
  prompt-log entry that gave the wrong advice, rather than leaving it
  uncorrected

## What the assistant produced
In scripts/run_part_b.py's sentiment_index_figure(): replaced the title
"Standardised Sector News Sentiment, 2020-2023" with "Sector Fear &
Greed, from the News, 2020-2023", and the subtitle's "Blue = above-
average sentiment; red = below-average sentiment" with "Blue = greedy
(above-average sentiment); red = fearful (below-average sentiment)".
Updated the in-code comment explaining the design choice to record the
correction rather than silently changing the reasoning. Regenerated the
figure and confirmed it renders correctly. Also added a "Correction"
note to the original prompt-log entry ("sentiment index figure:
terminology, labelling, and readability pass") rather than editing that
entry's original content, so the log still shows what was actually
decided at the time and why it changed later, not a rewritten history.

## What was wrong or risky
Worth noting for next time: streamlit_app.py's Sentiment tab caption was
ALREADY using "relatively greedy" / "relatively fearful" wording (added
in a separate, later session that apparently didn't cross-check this
figure's own avoid-the-term decision) - the app and the report figure
had drifted to opposite terminology choices without either session
noticing the inconsistency. This fix makes them consistent, but the root
cause was the same one flagged in the sentiment-index-decluttering log
entry: a claim should be checked against the actual source before being
used as the reason for a design decision, and once a design decision is
made in one place, related surfaces (app text, figure text, report
prose) should be checked for drift, not assumed to still agree.

## What I changed and why
Regenerated results/figures/sentiment_index.png with the new title and
subtitle text; no underlying sentiment data, scores, or index values
changed, so this is presentation-only, matching the scope of the
original redesign entry. Did not add a market-wide aggregate gauge to
match the course slide exactly (only relabelled the existing per-sector
figure) - the user confirmed they only wanted the terminology fixed, not
a new exhibit.

---

# Prompt log - adding Mean-CVaR as a fifth optimisation method

## What I wanted
build mean——Cvar

## Prompt(s)
I want you to extend my existing Python portfolio project, by adding a long-only, fully invested Mean-CVaR portfolio optimisation method.


## What the assistant produced
Matched "mean-[something]" to mean-CVaR based on the week10 revision
slides' own "Ideas, Structured" list, which explicitly suggests it:
"Add a tail-aware objective such as mean-CVaR, which targets the worst
outcomes in the tail of the returns." Added it to src/portfolios.py as a
fifth method (METHODS, _WEIGHT_FUNCS), changed every _*_weights()
function's signature to also take the raw estimation-window returns
(needed for CVaR, which is a sample quantile of scenario returns, not a
closed-form function of mean/cov like variance is), and updated
scripts/run_part_b.py's method labels, colours (a 5th categorical hue),
and legend/figure sizing for 15 base funds and 4-method comparisons
instead of 12 and 3.

## What was wrong or risky
The first implementation maximised (mean - rf) / CVaR with SLSQP,
mirroring max-Sharpe's shape. This converged badly: 8/36 Equity, 2/39
Crypto, and 13/36 Combined rebalances failed (res.success == False),
worse than the SLSQP-scaling issue documented earlier in this log for
min-variance/risk-parity. The root cause is different this time and not
fixable the same way (objective rescaling only helps when the solver
under-searches a well-behaved objective; here the objective itself is
non-smooth) - CVaR, defined via an empirical quantile of w-dependent
scenario returns, has a kink in w wherever the identity of the worst-tail
scenario changes, and SLSQP assumes a smooth, differentiable objective.
Retuning tolerances would not have fixed this; the problem needed a
different formulation, not a better-tuned solver.

## What I changed and why
Replaced the SLSQP ratio-maximisation with Rockafellar & Uryasev's (2000)
linear-programming formulation of CVaR minimisation (auxiliary VaR
variable zeta plus one non-negative shortfall slack per day in the
window, solved with scipy.optimize.linprog's HiGHS backend), minimising
CVaR subject to a floor on expected return (defaulted to the average
individual asset's own mean return in the window, so the mean constraint
is principled rather than an arbitrary knob) rather than maximising a
mean/CVaR ratio. This is not just a numerical fix: CVaR minimisation
under linear constraints is an exact linear program, so there is no
convergence risk of the kind SLSQP hit, and it also matches the course
slide's own framing ("targets the worst outcomes in the tail") more
directly than a Sharpe-style ratio would have. Re-ran the full pipeline:
zero convergence failures across all three Mean-CVaR backtests (down
from 8/2/13), and the only remaining non-convergence warnings are the
pre-existing, already-documented 1-rebalance cases for min-variance
(Crypto) and max-sharpe (Combined) that this change did not touch.
check_handin.py still passes (21 checks). Results are economically
plausible: Mean-CVaR sits mid-pack on Sharpe in all three universes
(0.23 Equity, 0.84 Crypto, 0.31 Combined) with diversification
comparable to Min-Variance (effective N around 9 in Equity/Combined),
consistent with a method that trades away some return for a floor
against tail losses rather than chasing the highest available Sharpe
ratio.

The second method the lecturer mentioned is still unconfirmed - the
report's fund counts (now fifteen base funds, five methods) and the
performance-table discussion need updating for Mean-CVaR once the report
text is next revised, and again if/when the second method is added.

---

# Prompt log - independent y-scales for weights_over_time.png

## What I wanted
Shared a screenshot of one panel of weights_over_time.png: one method's
line was clearly visible (a sharp rise), but the other two-to-three
lines were flattened into an unreadable band near zero.

## Prompt(s)
- Portfolio weights over time. Look at my screenshot. There is one line that is quite obvious. That's good. But there are two lines below that are completely indiscernible. Do you have any suggestions on how to improve this issue?

## What the assistant produced
The panels shared one y-axis (sharey=True) sized to the single largest
weight across ALL six tickers and all four methods - GE's Max-Sharpe
weight spiking to ~50% forced every other panel, and every other line
within GE's own panel, onto that same 0-50%+ scale, flattening anything
under ~10%. Removed sharey and the single global y_max in
weights_over_time_figure() (scripts/run_part_b.py), replaced with a
per-panel y-limit computed from only that ticker's own four methods.
This is the same fix already applied to growth_figure() for the
equivalent problem one level up (universes there, tickers here, both
documented in-code as "one dominant series drowns the rest").

## What was wrong or risky
None found - regenerated the figure and confirmed each panel's smaller
lines (e.g. WMT's Min-Variance and Mean-CVaR, both under 20%) are now
clearly readable with their own visible shape, not flattened by
whichever ticker happened to have the largest spike elsewhere in the
grid. The subtitle text also needed splitting across two fig.text()
calls after the first rewrite ran off the right edge of the figure at
12 inches wide - caught by checking the rendered PNG's actual pixel
width, the same check this log has needed for text overflow before.

## What I changed and why
Per-panel y-scaling only; no underlying weights data changed, so this
is presentation-only like the earlier redesigns. check_handin.py still
passes (21 checks).

## Follow-up: per-panel y-limits were not enough for every panel
Shared a second screenshot showing the GE panel specifically still had
three of its four lines pinned flat near zero. The per-panel fix above
solves disagreement BETWEEN panels, but GE has the same 50x-plus spread
WITHIN one panel (three methods under ~10% all period, Max-Sharpe alone
spiking past 50%) - no linear y-limit, panel-specific or not, can make a
1% line and a 50% line both legible on the same linear axis. Switched
the y-axis to symlog (linear below a 2-percentage-point threshold,
logarithmic above it) - chosen specifically because it handles genuine
0% weights correctly, which plain log cannot (log(0) is undefined).
Confirmed by inspecting the regenerated PNG: GE's previously-invisible
Mean-CVaR bump (~4-10% through 2021-2022) is now clearly readable
against Max-Sharpe's later spike to 50%+. Traded a real cost for this -
symlog makes a line's drop to zero look like a sharp spike (compressed
log space above 2% versus linear space below it), and the axis reads
more "technical" (10^1/10^0-style ticks) than the project's usual
non-technical-reader style. Flagged this trade-off to the user rather
than silently shipping it. Also needed two rounds of shortening the new
subtitle text after it ran off the figure's right edge twice - same
render-and-look check this log keeps needing for text overflow.
check_handin.py still passes (21 checks).

---

# Prompt log - redesigning sentiment_index.png for the Word page, not the screen

## What I wanted
the sentiment index image overflowed the page's right margin. 

## Prompt(s)
- shared the Word overflow screenshot, the sentiment index image overflowed the page's right margin. how to fix it

## What the assistant produced
The figure was designed at 13.5x6.6in - sized for full-screen viewing,
never for a Word page. Inserted at a typical ~6.5in content width, that
is roughly a 48% shrink, taking every font size down with it until axis
labels became unreadable - resizing in Word cannot fix a figure that was
never designed for the container it is going into. Rewrote
sentiment_index_figure() in scripts/run_part_b.py from a 2-row x 5-column
grid to 5 rows x 2 columns, sized at 6.6 x 10.0in - close to a Word
page's actual content width, so it drops in near its native size rather
than needing a crushing shrink. Recomputed all fig.text() positions and
font sizes for the new, much taller aspect ratio.

## What was wrong or risky
Two rounds of text overflow on the redesign, caught by inspecting the
regenerated PNG each time rather than assuming the fig.text() calls
worked: the footnote ran off the new figure's narrower 6.6in width and
also collided with the bottom row's x-axis tick labels, since the first
attempt kept the old figure's tight bottom margin. Needed both a larger
bottom subplots_adjust margin (0.05 -> 0.10, figure height 9.4 -> 10.0in)
and shorter footnote wording (twice) before it stopped clipping.

## What I changed and why
Presentation/layout only - no sentiment data or values changed, same
scope as every other figure redesign in this log. The output is a
1500x990px (6.6x10in at 150dpi) portrait image intended to be inserted
at close to its native width in a Word document, not scaled down from a
screen-sized original. check_handin.py still passes (21 checks). Worth
generalising: any exhibit meant for the Word report, not just on-screen
viewing, should probably be designed at report-page proportions (tall,
~6.5in wide) from the start, rather than at presentation/screen
proportions (wide, short) and resized down later - this was the second
figure (after weights_over_time.png's symlog fix) where a screen-first
design choice caused a problem only visible once actually placed in the
report.

---

# Prompt log - decluttering growth_of_dollar.png's legend

## What I wanted
Shared a screenshot of growth_of_dollar.png (now 5 methods after adding
Equal-Weight and Mean-CVaR) and flagged the top-right legend as crowded
again.

## Prompt(s)
- shared the screenshot, fix it

## What the assistant produced
The legend shared its row with the title text (loc="upper right",
anchored near the title), which worked with 3-4 methods but left no
room to breathe once a 5th (Mean-CVaR) was added - it also, by design,
had no dedicated vertical space of its own. Gave it a dedicated centred
row in growth_figure() (scripts/run_part_b.py), below the subtitle and
above the panel titles, with added columnspacing/handlelength/
handletextpad so the five entries do not run together.

## What was wrong or risky
The first attempt at the new row (bbox_to_anchor y=0.80) collided with
the panel titles just below it - caught by looking at the regenerated
PNG, where "Crypto" and "Combined" overlapped the legend text. Fixed by
increasing the figure height (4.6 -> 5.2in) and widening the gap between
the legend's row and the axes area (subplots_adjust top 0.78 -> 0.68),
not just nudging the legend's own y-position, since the panel titles are
drawn just above the axes boundary and need their own clearance too.

## What I changed and why
Presentation-only, same scope as the other figure redesigns in this
log - no underlying growth data changed. check_handin.py still passes
(21 checks). This figure is already embedded in the version of
report.docx sent to the user; they still need to re-insert this
regenerated PNG (or receive a re-sent docx) for the report to show the
fixed version.

---


# Prompt log - adding formula to this report
## What I wanted
The formula used in the report should all be added to the place where I used them

## Prompt(s)
- shared the screenshot, this is the format of how you should add the formula, carefullly read my report and add all the formula being used under sentence I used them

## What the assistant produced
Confirmed I only wanted the
formula FORMATTING (typeset, numbered, one per relevant sentence), not
those specific equations copied in - reproducing another paper's
content into this report would have been both irrelevant and an
unauthorised copy of someone else's work. With that confirmed, wrote
the actual formulas this project uses - the four (later five) portfolio
objectives (equal-weight, minimum-variance, maximum-Sharpe, risk parity,
later mean-CVaR), the Sharpe ratio and max-drawdown definitions, the
sentiment scoring/sector-index/standardisation formulas, and the
sentiment-tilt formula - as native, editable Word equations (OOXML math
markup, the same format Word's own equation editor produces), not as
inserted images.

## What was wrong or risky
Several rounds of malformed equations, all caught by re-reading the
generated XML or the schema validator rather than assuming the first
version was right: subscripts/superscripts built with an empty base
(e.g. a floating "_f" not actually attached to "r"), and sentences that
trailed off mid-clause into an equation and never grammatically
finished ("using the assumed risk-free rate r" followed by an unrelated
equation). Each was rewritten so the equation completes the sentence it
sits in, per the project's own grammar rule that displayed equations are
punctuated as part of the surrounding sentence, not floating fragments.

## What I changed and why
Roughly ten equations across the report, each placed directly under the
sentence introducing it rather than collected separately, so a reader
never has to jump elsewhere to find what a symbol means. Verified with
the docx skill's schema validator (PYTHONUTF8=1 python validate.py
report.docx) that every equation is valid OOXML math before treating any
of it as done, not just that python-docx ran without raising an
exception - a script completing without an error is not proof the XML
it wrote is well-formed.
