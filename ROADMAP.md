# Roadmap

Milestones are versions. Each release keeps the v1.0 rules: analysis choices are frozen in
`config/config.yaml` and committed before anything runs on them, later changes are logged in
`DECISIONS.md`, every number carries an interval, and negative results stay in the README.

## v1.0 — pre-registered pipeline (done)

Goal: RCT analysis → uplift models → budget-constrained allocation → off-policy evaluation on
Hillstrom, with a semi-synthetic positive control.

- [x] Data loading with hash check, stratified split, data card
- [x] RCT balance, ATEs, regression adjustment, power
- [x] Multi-arm S/T/X, DR-learner and causal forest; Qini/AUUC with permutation null
- [x] Exact IP (SCIP) and Lagrangian allocation
- [x] IPW and doubly robust policy evaluation
- [x] Positive control, confounding, robust allocation and LLM-feature extensions

## v1.1 — when does targeting pay off?

Goal: sweep effect heterogeneity × sample size on the semi-synthetic generator, and map where
model-based targeting beats random targeting, with intervals.

- [ ] Sweep grid (heterogeneity levels, sample sizes, replications) frozen in config before running
- [ ] `sweep.py` reusing the existing learners, allocation and OPE code; tests on a tiny grid
- [ ] Heatmap of policy value minus random-targeting value, cells marked where the 95% interval excludes 0, plus CSV
- [ ] `make sweep` and `make sweep-quick`
- [ ] README section: the question, the heatmap, and what it means for the Hillstrom null result

## v1.2 — profit objective

Goal: optimise profit instead of revenue and find the profit-optimal budget.

- [ ] Margin assumption and cost per e-mail in config
- [ ] Profit objective in allocation and evaluation
- [ ] Profit-optimal budget with an interval
- [ ] Cross-fitted intervals for policy values (include model-refit variance)

## v1.3 — direct policy learning

Goal: compare doubly robust policy learning against estimate-then-optimise.

- [ ] Doubly robust policy learning under the budget constraint
- [ ] Head-to-head with estimate-then-optimise on real and semi-synthetic data
- [ ] Chance-constrained budget for random customer uptake

## v1.4 — sequential allocation

Goal: compare a budgeted contextual bandit with one-shot "experiment, then target".

- [ ] Semi-synthetic sequential environment built on the v1.0 generator
- [ ] Budgeted contextual bandit
- [ ] Comparison with "experiment, then target" at equal total budget, with intervals
