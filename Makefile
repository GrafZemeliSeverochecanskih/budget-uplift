# On Windows without GNU make, use `mingw32-make` or run the commands directly.
PYTHON ?= python

.PHONY: install data test all quick sweep sweep-quick profit profit-quick policy policy-quick sequential sequential-quick clean

install:
	$(PYTHON) -m pip install -e ".[dev,llm]"

data:
	$(PYTHON) scripts/download_data.py

test:
	$(PYTHON) -m pytest -q

all: data
	$(PYTHON) -m uplift_alloc.run_all

# Same pipeline with 50 bootstrap reps and a lighter solver benchmark; writes results/quick/.
quick: data
	$(PYTHON) -m uplift_alloc.run_all --quick

# v1.1 heterogeneity x sample-size sweep (also part of `make all`); sweep-quick = 2x2 grid, 3 replications.
sweep: data
	$(PYTHON) -m uplift_alloc.run_all --only sweep

sweep-quick: data
	$(PYTHON) -m uplift_alloc.run_all --quick --only sweep

# v1.2 profit objective with cross-fitted evaluation (also part of `make all`).
profit: data
	$(PYTHON) -m uplift_alloc.run_all --only profit

profit-quick: data
	$(PYTHON) -m uplift_alloc.run_all --quick --only profit

# v1.3 direct policy learning + chance-constrained budget (also part of `make all`).
policy: data
	$(PYTHON) -m uplift_alloc.run_all --only policy_learning chance

policy-quick: data
	$(PYTHON) -m uplift_alloc.run_all --quick --only policy_learning chance

# v1.4 budgeted bandit vs experiment-then-target (also part of `make all`).
sequential: data
	$(PYTHON) -m uplift_alloc.run_all --only sequential

sequential-quick: data
	$(PYTHON) -m uplift_alloc.run_all --quick --only sequential

clean:
	rm -rf results/figures/*.png results/tables/*.csv results/quick
