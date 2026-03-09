MAKE_HELP_LEFT_COLUMN_WIDTH:=14
PYTHON = python3

.PHONY: help build
help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-$(MAKE_HELP_LEFT_COLUMN_WIDTH)s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

fmt: ## Format Python code
	ruff check --select I --fix . && black .;

fmt-check: ## Check Python formatting
	black --check .;

benchmark-matmul: ## Run matmul benchmark
	PYTHONPATH=. $(PYTHON) benchmarks/matmul/benchmark_matmul_guide.py --save-plots --plot-dir=docs/plots/matmul/

benchmark-softmax: ## Run softmax benchmark
	PYTHONPATH=. $(PYTHON) benchmarks/attention/benchmark_softmax_causal.py --save-plots --plot-dir=docs/plots/attention/

benchmark-rms-norm: ## Run RMSNorm benchmark
	PYTHONPATH=. $(PYTHON) benchmarks/normalization/benchmark_rms_norm.py --save-plots --plot-dir=docs/plots/normalization/
