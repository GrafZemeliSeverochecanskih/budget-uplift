# On Windows without GNU make, use `mingw32-make` or run the commands directly.
PYTHON ?= python

.PHONY: install data test all quick sweep sweep-quick profit profit-quick clean

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

clean:
	rm -rf results/figures/*.png results/tables/*.csv results/quick
