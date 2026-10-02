#!/usr/bin/env bash
# Offline verification only: temporary fixtures and mocks; no publication or NHL requests.
set -euo pipefail
cd "$(dirname "$0")/.."
HENACHEL_PYTHON="${HENACHEL_PYTHON:-.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
git status --short
"$HENACHEL_PYTHON" -m pytest -q -W error
"$HENACHEL_PYTHON" -m pytest tests/test_feature_parity.py tests/test_pp_standings.py -q -W error
"$HENACHEL_PYTHON" -m pytest tests/test_calibration.py tests/test_science.py tests/test_model_entrypoints.py -q -W error
"$HENACHEL_PYTHON" -m pytest tests/test_matching.py -q -W error
"$HENACHEL_PYTHON" -m pytest tests/test_settlement.py tests/test_history.py -q -W error
"$HENACHEL_PYTHON" -m pytest tests/test_publication.py tests/test_runtime.py tests/test_empty_pipeline.py -q -W error
"$HENACHEL_PYTHON" -m pip --disable-pip-version-check --no-cache-dir check
git diff --check
git status --short
git log -6 --oneline
