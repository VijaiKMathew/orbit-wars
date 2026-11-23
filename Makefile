# ==============================================================================
# Orbit Wars: Autonomous Agent Solution Pipeline Makefile
# Powered by Astral 'uv' and the official Kaggle Simulation Engine
# ==============================================================================

SHELL := /bin/bash
PYTHON := python3

.PHONY: help setup verify test report clean all

help:
	@echo "================================================================================"
	@echo " Orbit Wars: Autonomous Agent Pipeline Automation"
	@echo "================================================================================"
	@echo "  make setup     - Install minimal dependencies via pip / uv"
	@echo "  make verify    - Run 1-command verification suite (bash run.sh)"
	@echo "  make test      - Test standalone agent submission syntax and compilation"
	@echo "  make report    - Open interactive HTML solution preview"
	@echo "  make clean     - Remove temporary caches and pycache"
	@echo "  make all       - Run complete verification pipeline"
	@echo "================================================================================"

setup:
	@echo "==> Setting up environment..."
	pip install -r requirements.txt || uv pip install -r requirements.txt

verify:
	@echo "==> Running 1-command verification..."
	bash run.sh

test:
	@echo "==> Testing submission agent syntax..."
	$(PYTHON) -c "import ast; ast.parse(open('submission.py').read()); print('✓ submission.py AST verified successfully.')"

report:
	@echo "==> Opening interactive HTML report preview..."
	open reports/solution_preview.html || xdg-open reports/solution_preview.html || echo "Preview located at reports/solution_preview.html"

clean:
	@echo "==> Cleaning transient files and pycache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "==> Clean complete."

all: verify test
	@echo "==> Orbit Wars pipeline verification complete!"
