# TODO — work top to bottom, tick items when done

## 0. Setup
- [x] Create repo structure: `src/uplift_alloc/`, `scripts/`, `tests/`, `config/`, `results/{figures,tables}/`
- [x] `pyproject.toml` with pinned dependencies; `Makefile` with `install`, `test`, `all` targets
- [x] `config/config.yaml`: seed, 70/30 stratified train/test split by arm, email cost per arm,
      budget grid (e.g. 0–100% of customers in 10 steps), bootstrap reps (≥ 500)
- [x] Add `data/` to `.gitignore`
- [x] `src/uplift_alloc/data.py`: implement exactly the 6 steps under "Primary: how to load" in CLAUDE.md
      (fetch → parquet → assertions → SHA-256 in `DATA_VERSION.txt` → encoding → `load_hillstrom()`)
- [x] If `scikit-uplift` won't install, use the direct-CSV fallback described in CLAUDE.md — not needed: scikit-uplift 0.5.1 installs and downloads with numpy 2.5 / scikit-learn 1.9 (see DECISIONS.md)
- [x] `scripts/download_data.py` + `make data` target that runs the loader on a fresh clone
- [x] Create and save the stratified train/test split to `data/processed/split.json`
- [x] Tests (`tests/test_data.py`): row count, arm labels and ~1/3 shares, value ranges, encoding round-trip, split has no overlap
- [x] `DATA_CARD.md`: rows per arm, outcome base rates, covariate descriptions
- [x] GitHub Actions workflow running `pytest` on push

## 1. RCT analysis — `src/uplift_alloc/rct.py`
- [x] Balance table: standardised mean difference of each covariate, each treatment arm vs control; flag |SMD| > 0.1
- [x] ATE per arm vs control for visit, conversion, spend: difference in means with 95% CIs
- [x] Regression-adjusted ATE (CUPED-style, using pre-period `history` and other covariates);
      report CI width reduction vs the unadjusted estimate
- [x] Power / MDE: sample size needed to detect the observed spend effect at 80% power, α = 0.05
- [x] Output: `results/tables/rct_summary.csv`, `results/figures/balance.png`
- [x] Tests: on simulated RCT data, difference-in-means and regression adjustment both recover the true ATE

## 2. CATE / uplift models — `src/uplift_alloc/cate.py`
- [x] Multi-arm S-, T-, X-learners with LightGBM base models (outcome: spend; also run on conversion)
- [x] DR-learner and causal forest via EconML
- [x] Hyperparameters chosen by CV on train only
- [x] Evaluate on test: Qini and AUUC per arm, with bootstrap 95% intervals; random-targeting baseline
- [x] Output: `results/figures/qini_curves.png`, `results/tables/learner_comparison.csv`
- [x] Tests: on simulated data with known heterogeneous effects, each learner beats random on Qini

## 3. Budget-constrained allocation — `src/uplift_alloc/allocate.py`
- [x] Integer program: x[i,a] ∈ {0,1}, each customer gets at most one email,
      Σ cost[a]·x[i,a] ≤ B, maximise Σ τ̂[i,a]·x[i,a] (OR-Tools CP-SAT or PuLP/CBC) — solved with SCIP via OR-Tools; CP-SAT also reported (DECISIONS D14)
- [x] Lagrangian heuristic: a*(i) = argmax_a (τ̂[i,a] − λ·cost[a]) with 0 for "no email";
      binary-search λ until spend ≤ B
- [x] Compare objective value and runtime of both across the budget grid and for n ∈ {1k, 10k, 64k}
- [x] Output: `results/tables/allocation_comparison.csv`, `results/figures/runtime.png`
- [x] Tests: on a tiny instance, IP matches brute force; Lagrangian gap to IP is reported and small

## 4. Off-policy evaluation — `src/uplift_alloc/ope.py`
- [x] IPW and doubly-robust policy value estimators using the known propensity 1/3
- [x] Evaluate on the test split, for every budget: our policy, random targeting at the same budget,
      "send everyone the single best arm", "send nobody"
- [x] Bootstrap 95% CIs for each policy value
- [x] Headline figure: incremental spend vs budget, all policies with CI bands → `results/figures/policy_value_vs_budget.png`
- [x] Tests: on simulated data, IPW and DR recover the true value of a fixed policy

## 5. Extensions (in order; each in its own module)
- [x] `confounding.py`: make the data observational by keeping treated customers with probability rising in `history`;
      draw the assumed DAG (`results/figures/dag.png`); show the naive ATE is biased;
      recover it with estimated-propensity IPW and DR; compare against the RCT ATE
- [x] `robust_allocate.py`: allocate on a bootstrap lower confidence bound of τ̂ instead of the point estimate;
      compare mean and 5th-percentile policy value against the point-estimate policy
- [x] `llm_features.py`: build a one-sentence text description per customer, embed with an open
      sentence-embedding model, reduce with PCA, add to the best learner; report Qini with and without (keep it even if no gain)
- [ ] ~~Optional `scale_criteo.py`~~ NOT RUN — every Criteo download URL returns 404 (DECISIONS D21);: rerun steps 2–4 on Criteo; report runtime and whether conclusions hold

## 6. Write-up
- [x] `README.md`: question → data → method → headline figure → results table → negative results →
      limitations (SUTVA, single-period effects, cost assumptions) → how to reproduce
- [x] `DECISIONS.md` complete and dated
- [x] Fresh-clone check: `make install && make all` reproduces every figure; `make test` passes
- [x] Draft a one-line CV bullet with the real headline numbers at the bottom of the README

## 7. Semi-synthetic positive control + reviewer polish
- [x] Add a `positive_control` block to `config/config.yaml` (segment, baselines, effects, seed) and freeze it
      in a commit BEFORE running anything on it (log in `DECISIONS.md`)
- [x] `src/uplift_alloc/positive_control.py`: keep real Hillstrom covariates, arm assignment and split;
      simulate `visit` and `spend` with a known heterogeneous effect (one arm helps only a defined segment)
- [x] Rerun all CATE learners on it: Qini vs the permutation null and correlation of τ̂ with the true τ, with CIs
- [x] Rerun allocation + OPE on it at every budget: our policy vs random (paired CIs), compared against the
      known true policy value; include an oracle policy (IP on the true τ)
- [x] Tests: simulator returns the configured true τ; segment-only effect is zero outside the segment
- [x] Real data: report CATE and allocation/OPE for `visit` as well as `spend`
- [x] `make quick`: whole pipeline with 50 bootstrap reps (and lighter solver benchmark) into `results/quick/`
- [x] README story: real data shows no targeting signal; the positive control shows the pipeline would have found it
- [ ] Commit (no data files), push to a new public GitHub repo `budgeted-uplift`, confirm CI passes
