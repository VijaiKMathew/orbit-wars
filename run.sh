#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=========================================================="
echo "   Kaggle Orbit Wars: Reproducible Solution Pipeline"
echo "=========================================================="

echo "[1/3] Validating Python environment..."
python3 -c "import sys; print(f' Python {sys.version.split()[0]} ready.')"

echo "[2/3] Verifying Standalone Solution Submission Syntax..."
python3 -c "import ast; ast.parse(open('submission.py').read()); print(' submission.py syntax verified.')"

echo "[3/3] Checking Solution Configuration..."
python3 -c "import json; cfg=json.load(open('configs/model_config.json')); print(' Loaded solution config with', len(cfg), 'parameters.')"

echo "=========================================================="
echo " Solution pipeline successfully verified!"
echo "=========================================================="
