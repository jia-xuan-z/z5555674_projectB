# CLAUDE.md - SentiFolio Part B project instructions

## Project context

"SentiFolio", a prototype investment app offering systematically managed funds
(equity, crypto, combined) plus a news-sentiment index across equity sectors.
Full brief: PROJECT_BRIEF.md. Context: context/project_context.md,
context/DATA_GUIDE.md.

Part B = Station 3 (out-of-sample portfolio optimisation, sentiment model,
sentiment-fusion extension) + Station 4 (deployed Streamlit app). Part A
(ETL + return/text features) is already built and reused here unchanged.

## Data & feature rules (carried over from Part A)

- Always load data through src/data_access.py. Never scrape or hit the network
  directly.
- Never commit raw .parquet files or the full dataset. Precomputed app
  artifacts under results/ ARE committed (the deployed app reads them).
- Compute returns WITHIN each panel first (equity, then crypto separately),
  then left-merge crypto returns onto the equity trading calendar.
- News dates are UTC/timezone-aware; price dates are not. Normalise both to
  the same tz-naive dtype before any merge or date-alignment.
- Map each headline to its equity trading day: same day if it's a trading
  day, otherwise the next trading day.

## Station 3 rules - no look-ahead bias

- Portfolio weights at time t are formed ONLY from data available before t
  (a trailing estimation window). Walk-forward, rebalance monthly or less
  often. State the first live backtest date and the estimation window length.
- Sentiment signal used at day t must come from day t-1 or earlier - lag by
  at least one trading day. A Saturday/Monday headline (both aligned to
  Monday) is first usable for Tuesday's trade.
- Annualise equity-only quantities with sqrt(252)/252, crypto-only or
  combined-with-crypto-days quantities need the right factor for whichever
  calendar the return series is actually on - state the choice.
- Flag explicitly whether any Station 3 code could introduce look-ahead bias
  or a merge-order bug, even in draft form.

## Coding conventions

- Code lives in src/ (etl.py, features.py already built; portfolios.py,
  sentiment.py, fusion.py are Part B's new work; data_access.py is provided,
  do not edit it). Runnable entry point: scripts/run_part_b.py.
- Keep functions small and testable; prefer pandas/numpy vectorised ops.
- Every function that produces a required exhibit should be callable from
  scripts/run_part_b.py so results are reproducible with one command.
- The deployed Streamlit app (streamlit_app.py) must NOT import nltk or
  recompute backtests/sentiment at runtime - it only reads precomputed
  results/ artifacts. Precomputation happens in scripts/run_part_b.py.

## Required output filenames (exact - the app and markers rely on these)

- results/data/fund_returns.csv
- results/data/fund_weights.csv
- results/data/sector_sentiment_index.csv
- results/tables/performance_metrics.csv

## How I want you (the assistant) to work

- Do not invent backtest results, metrics, or row counts - always run the
  code and report actual output.
- When drafting report prose, write a first pass only; I will rewrite it in
  my own words before submission.
- Log significant prompts/outputs/corrections into ai/prompt_log.md as we go,
  not retroactively.
- Never invent a citation, a statistic, or a source. Flag any claim you
  cannot verify instead of stating it confidently.

## Verification

- Before treating any exhibit as final, run scripts/run_part_b.py end-to-end
  from a clean state and confirm the output files exist under results/.
- Run streamlit run streamlit_app.py locally before considering Station 4 done.
- Run scripts/check_handin.py before considering Part B submission-ready.
