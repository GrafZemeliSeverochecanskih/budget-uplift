# Roadmap

Milestones are versions. Each release keeps the v1.0 rules: analysis choices are frozen in
`config/config.yaml` and committed before anything runs on them, later changes are logged in
`DECISIONS.md`, every number carries an interval, and negative results stay in the README.

## v1.0 - pre-registered pipeline (done)

Goal: RCT analysis -> uplift models -> budget-constrained allocation -> off-policy evaluation on
Hillstrom, with a semi-synthetic positive control.

- [x] Data loading with hash check, stratified split, data card
- [x] RCT balance, ATEs, regression adjustment, power
- [x] Multi-arm S/T/X, DR-learner and causal forest; Qini/AUUC with permutation null
- [x] Exact IP (SCIP) and Lagrangian allocation
- [x] IPW and doubly robust policy evaluation
- [x] Positive control, confounding, robust allocation and LLM-feature extensions

## v1.1 - when does targeting pay off? (done)

Goal: sweep effect heterogeneity x sample size on the semi-synthetic generator, and map where
model-based targeting beats random targeting, with intervals.

- [x] Sweep grid (heterogeneity levels, sample sizes, replications) frozen in config before running
- [x] `sweep.py` reusing the existing learners, allocation and OPE code; tests on a tiny grid
- [x] Heatmap of policy value minus random-targeting value, cells marked where the 95% interval excludes 0, plus CSV
- [x] `make sweep` and `make sweep-quick`
- [x] README section: the question, the heatmap, and what it means for the Hillstrom null result

## v1.2 - profit objective (done)

Goal: optimise profit instead of revenue and find the profit-optimal budget.

- [x] Margin assumption and cost per e-mail in config
- [x] Profit objective in allocation and evaluation
- [x] Profit-optimal budget with an interval
- [x] Cross-fitted intervals for policy values (include model-refit variance)

## v1.3 - direct policy learning (done)

Goal: compare doubly robust policy learning against estimate-then-optimise.

- [x] Doubly robust policy learning under the budget constraint
- [x] Head-to-head with estimate-then-optimise on real and semi-synthetic data
- [x] Chance-constrained budget for random customer uptake

## v1.4 - sequential allocation (done)

Goal: compare a budgeted contextual bandit with one-shot "experiment, then target".

- [x] Semi-synthetic sequential environment built on the v1.0 generator
- [x] Budgeted contextual bandit
- [x] Comparison with "experiment, then target" at equal total budget, with intervals

## v1.5 - selection-aware chance constraint (done)

Goal: make the v1.3 chance constraint deliver its promised overrun probability by correcting the
winner's curse in estimated uptake among the customers the allocator selects.

- [x] Recalibration design frozen in config before running
- [x] Selection-aware recalibration of uptake (bias factor estimated on held-out selected customers)
- [x] Conservative alternative: bootstrap upper quantile of uptake
- [x] Head-to-head with the v1.3 mean-budget and plug-in chance policies on the same campaigns, true overrun on the positive control
- [x] README section, `make chance15` / `make chance15-quick`, tests
