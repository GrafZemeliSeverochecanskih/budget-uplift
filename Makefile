# On Windows without GNU make, use `mingw32-make` or run the commands directly.
PYTHON ?= python

.PHONY: install data test all quick clean

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

clean:
	rm -rf results/figures/*.png results/tables/*.csv results/quick
