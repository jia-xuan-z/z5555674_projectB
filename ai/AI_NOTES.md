# AI notes (first draft - rewrite in your own words before submission)

This is a starting draft summarising the actual AI workflow used across Part B,
built from the real session history. It is factual and accurate, but it still
needs to go through in your own voice and judgement before submission - the
brief specifically wants your own account, not an AI-generated one describing
itself. Treat every sentence below as something to confirm, adjust, or replace
with how you'd actually describe it.

## How I directed the AI agent

I worked through Part B roughly station by station: reused my own verified
Part A ETL/features code first, then built the portfolio backtests, then the
sentiment model, then the fusion extension, then the app. At each stage I
either gave a specific instruction (e.g. approving a scope decision like
building 9 funds instead of the required minimum) or left a design choice open
(e.g. the estimation window, rebalance frequency, tilt formula) and reviewed
what came back rather than specifying every parameter up front. See
`ai/prompt_log.md` for the dated, task-by-task record - 15 entries covering
ETL reuse, the funds/sentiment/fusion build, the Streamlit app, and a series
of figure and methodology fixes.

## How I checked the output

- Ran the actual pipeline (`scripts/run_part_b.py`) end to end repeatedly
  rather than trusting that code "looked right", and checked real numbers
  against expectations each time.
- Noticed a suspicious pattern in a generated chart (portfolio weights sitting
  at exactly equal-weight for months) and treated that as a signal to
  investigate rather than accept - this led to finding and fixing a real bug
  where scipy's optimiser was silently failing to converge on ~150 of 333
  rebalances across the min-variance and risk-parity funds.
- Cross-checked a Streamlit app calculation (blended portfolio allocation)
  against a fund's own reported metrics and caught a data-handling bug
  (zero-filling a fund's pre-launch dates) from the mismatch.
- Verified the brief's exact wording for required exhibits against what the
  code actually produced, rather than assuming a plausible-looking chart
  satisfied the requirement - this caught one exhibit (weights over time) that
  compared holdings within one method when the brief asked for a comparison
  across methods.
- Pushed back on AI-suggested changes I judged out of scope or not clearly
  beneficial (e.g. declined to add a portfolio weight cap / covariance
  shrinkage this late in the process, after weighing whether it would
  actually improve results against the risk of re-changing every number
  again close to submission).

## Where I made my own calls versus followed AI defaults

I made the scope and judgement calls: how many funds/methods to build, which
robustness checks were worth running, which suggested changes to accept versus
defer to the report's reflection section instead of implementing. I left
lower-level implementation choices (exact solver tolerances, specific chart
layout) to the assistant and checked the results rather than specifying them
myself.

## Where AI got things wrong

The clearest case is the optimiser convergence bug described above - the
assistant's first implementation of the walk-forward backtest silently
produced equal-weight portfolios for min-variance and risk-parity funds most
of the time, while reporting success. It was not caught by the assistant's
own first-pass verification (checking `res.success`); it was caught because
a chart looked too clean to be real. A second case: an early version of the
Streamlit allocation blend understated a fund's true performance by
zero-filling dates before that fund existed. Both are documented with the
full diagnosis and fix in `ai/prompt_log.md`.
