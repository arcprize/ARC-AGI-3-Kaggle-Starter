# ARC Prize 2026 — ARC-AGI-3 local dev workflow.
#
# Five commands cover the whole loop:
#   make setup        # one-time: venv + arc-agi + clone framework
#   make play-local   # fast inner loop: run agent/my_agent.py on a real game
#   make pull-sample  # fetch the official Stochastic Goose sample for reference
#   make submit       # build notebook from agent/my_agent.py + push to Kaggle
#   make status       # tail the latest Kaggle run

PYTHON          ?= python3.12
VENV            := .venv
VENV_PY         := $(VENV)/bin/python
VENV_PIP        := $(VENV)/bin/pip
# Read the project-local token at recipe time and expose it as KAGGLE_API_TOKEN
# (the only env var the modern Kaggle CLI honours for token auth).
KAGGLE          := KAGGLE_API_TOKEN=$$(cat .kaggle/access_token) $(VENV)/bin/kaggle
FRAMEWORK_REPO  := https://github.com/arcprize/ARC-AGI-3-Agents.git
FRAMEWORK_DIR   := vendor/ARC-AGI-3-Agents
COMP_SLUG       := arc-prize-2026-arc-agi-3
GAME            ?=
STEPS           ?= 200

.PHONY: help setup test format play-local pull-sample notebook submit status verify-local clean _check-kaggle

_check-kaggle:
	@if [ ! -s .kaggle/access_token ]; then \
	    echo "ERROR: .kaggle/access_token is missing or empty."; \
	    echo "       Generate a token at https://www.kaggle.com/settings (API → Create New Token)"; \
	    echo "       and save it as a one-line file at: $(PWD)/.kaggle/access_token"; \
	    exit 1; \
	fi

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-15s %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Vars: PYTHON=$(PYTHON)  GAME=$(GAME)  STEPS=$(STEPS)"

setup: ## One-time install: venv, arc-agi, kaggle CLI, clone framework
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install "arc-agi>=0.9.6" "kaggle>=2.2" python-dotenv pandas pyarrow pytest ruff
	@if [ ! -d "$(FRAMEWORK_DIR)/.git" ]; then \
	    mkdir -p vendor && git clone --depth 1 $(FRAMEWORK_REPO) $(FRAMEWORK_DIR); \
	else \
	    git -C $(FRAMEWORK_DIR) pull --ff-only; \
	fi
	@# Slim agents/__init__.py so we don't need langgraph/langsmith/smolagents/etc.
	@# (Same trick the official Stochastic Goose sample uses on Kaggle.)
	@$(VENV_PY) scripts/slim_framework.py
	@echo ""
	@echo "Setup complete. Try:  make play-local"

test: ## Run HypothesisWorld's synthetic unit tests
	$(VENV_PY) -m pytest

format: ## Format and lint the research agent and tests
	$(VENV_PY) -m ruff format agent tests scripts
	$(VENV_PY) -m ruff check --fix agent tests scripts

play-local: ## Run agent/my_agent.py against ALL games (or GAME=ls20 for a single one)
	$(VENV_PY) scripts/play_local.py $(if $(GAME),--game $(GAME)) --max-steps $(STEPS)

verify-local: ## Quick smoke test: 50 steps on ls20 + vc33 only
	$(VENV_PY) scripts/play_local.py --game ls20,vc33 --max-steps 50

list-games: ## Show all available games
	$(VENV_PY) scripts/play_local.py --list

pull-sample: _check-kaggle ## Download the official Stochastic Goose sample notebook for reference
	mkdir -p reference/stochastic-goose
	$(KAGGLE) kernels pull inversion/arc3-sample-submission-stochastic-goose \
	    -p reference/stochastic-goose -m
	@echo "Open reference/stochastic-goose/*.ipynb for the canonical pattern."

notebook: ## Splice agent/my_agent.py into notebooks/submission.ipynb
	$(VENV_PY) scripts/build_notebook.py

submit: notebook _check-kaggle ## Build notebook and push to Kaggle (one-line submission)
	@grep -q REPLACE_WITH_YOUR_USERNAME notebooks/kernel-metadata.json && { \
	    echo "ERROR: edit notebooks/kernel-metadata.json and replace REPLACE_WITH_YOUR_USERNAME"; \
	    exit 1; } || true
	$(KAGGLE) kernels push -p notebooks/
	@echo ""
	@echo "Pushed. Track it with:  make status"

status: _check-kaggle ## Show the status of your most recent Kaggle kernel run
	@KERNEL_ID=$$(python3 -c "import json; print(json.load(open('notebooks/kernel-metadata.json'))['id'])"); \
	$(KAGGLE) kernels status $$KERNEL_ID

clean: ## Remove generated artefacts (venv, downloaded games, vendored repos)
	rm -rf $(VENV) vendor environment_files recordings notebooks/submission.ipynb \
	       reference logs.log __pycache__ .pytest_cache
