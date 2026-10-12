# Project: Budget-constrained promotion targeting from an RCT

## Goal
A public, reproducible research repo showing the full causal decision pipeline:
analyse a randomised experiment -> estimate heterogeneous treatment effects (uplift)
-> allocate promotions under a budget with constrained optimisation -> evaluate the
resulting policy offline. Audience: ML/causal-inference reviewers at a tech company.
Quality bar: rigorous, honest about negative results, runs end-to-end with one command.

## Data

### Primary: Hillstrom email RCT
- 64,000 customers who purchased in the last 12 months, randomised in equal thirds to
  `Mens E-Mail`, `Womens E-Mail` or `No E-Mail`; outcomes tracked for 2 weeks after the email.
- Covariates: `recency` (months since last purchase), `history_segment` (binned past-year spend),
  `history` (past-year $ spend), `mens`, `womens` (1/0 bought that category last year),
  `zip_code` (Urban/Suburban/Rural), `newbie` (1/0 new customer), `channel` (Phone/Web/Multichannel).
- Outcomes: `visit` (1/0), `conversion` (1/0), `spend` ($, very skewed, mostly 0).
- Treatment column: `segment`. Approximate base rates: visit ~ 0.15, conversion ~ 0.009.
- Randomisation is uniform, so the true propensity of each arm is 1/3. Use it; do not estimate it
  except in the deliberately confounded extension.

### Primary: how to load
```python
from sklift.datasets import fetch_hillstrom

bunch = fetch_hillstrom(target_col="all", data_home="data/raw")  # downloads once, then caches
X = bunch.data            # DataFrame of covariates
Y = bunch.target          # DataFrame: visit, conversion, spend
T = bunch.treatment       # Series: 'Mens E-Mail' / 'Womens E-Mail' / 'No E-Mail'
```
`src/uplift_alloc/data.py` must:
1. Call the loader above (package `scikit-uplift`, imported as `sklift`).
2. Combine into one DataFrame and save `data/processed/hillstrom.parquet`.
3. Assert: 64,000 rows; no missing values; exactly the three arm labels, each ~1/3 of rows;
   `visit`/`conversion` in {0,1}; `spend` >= 0; `conversion == 1` implies `visit == 1` (report violations, don't drop).
4. Compute the SHA-256 of the raw CSV and write it with the download date to `data/DATA_VERSION.txt`;
   on later runs, fail loudly if the hash changes.
5. Encode: arm -> {0: No E-Mail, 1: Mens E-Mail, 2: Womens E-Mail}; one-hot `zip_code` and `channel`;
   drop `history_segment` from model features (it is a binned copy of `history`) but keep it for reporting.
6. Expose `load_hillstrom() -> (X: DataFrame, T: ndarray[int], Y: DataFrame)`.

If `scikit-uplift` fails to install against current numpy/scikit-learn (its last release is 0.5.1),
do NOT pin old versions of the whole stack. Instead, read the download URL from the
`fetch_hillstrom` source in the sklift GitHub repo (`sklift/datasets/datasets.py`), download the CSV
directly into `data/raw/`, and parse it with pandas. Same checks apply.

### Optional scale check: Criteo uplift
```python
from sklift.datasets import fetch_criteo

bunch = fetch_criteo(target_col="all", treatment_col="all", percent10=True, data_home="data/raw")
```
- Full set: ~14M rows, 3.2 GB uncompressed. Start with `percent10=True` (10% sample); go to full only if memory allows.
- Features `f0`-`f11` (anonymised floats); targets `visit`, `conversion`; treatment `treatment`
  (randomised, ~85% treated) and `exposure` (actually shown the ad, NOT randomised).
- Use `treatment` as the treatment and propensity 0.85. Never use `exposure` as a treatment
  without a non-compliance/IV argument; mention it as a limitation.
- Binary treatment and no cost data: set a cost per treated user in config.

### Data hygiene
- `data/` is git-ignored; the repo stays reproducible through the loaders and `DATA_VERSION.txt`.
- The train/test split is made once, stratified by arm, saved as row indices in
  `data/processed/split.json`, and reused everywhere.

## Stack
Python 3.11+, pandas, numpy, scikit-learn, lightgbm, econml, scikit-uplift, statsmodels,
OR-Tools (or PuLP), matplotlib, pytest. Pin versions in `pyproject.toml`.

## Rules
- All analysis choices live in `config/config.yaml` (seeds, train/test split, email costs,
  budget grid, bootstrap reps). Fix the config BEFORE looking at test-set results, and never tune on
  the test split. Log any later config change with a reason in `DECISIONS.md`.
- Library code in `src/uplift_alloc/`, thin scripts in `scripts/`, tests in `tests/`.
  No logic in notebooks; notebooks only for exploration if at all.
- Every estimator gets a unit test on simulated data with a known true effect.
- Every reported number comes with an uncertainty interval (bootstrap or analytic).
- Report negative results (e.g. a learner that fails to beat a baseline) in the README, not hidden.
- Figures saved to `results/figures/`, tables to `results/tables/` as CSV.
- `make all` (or `python -m uplift_alloc.run_all`) reproduces every result from scratch.
- Keep functions small and typed; docstrings state assumptions (e.g. SUTVA, known propensities).

Work through `TODO.md` in order. Tick items off as you finish them. Stop and ask before
changing the project scope.
