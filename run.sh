#!/usr/bin/env bash
# One command to go from a fresh clone to a running API (macOS / Linux).
#
#   ./run.sh            set up, train, test, then start the API
#   ./run.sh --no-serve same, but stop after the tests
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
step() { printf '\n== %s\n' "$1"; }

if [ ! -x "$PY" ]; then
  step "Create virtual environment (.venv)"
  python3 -m venv .venv
fi
step "Install packages";               "$PY" -m pip install --quiet --upgrade pip && "$PY" -m pip install --quiet -r requirements.txt -e .
step "Download FD001 data";            "$PY" scripts/download_data.py
step "Train and compare models";       "$PY" -m turbofan_rul.train
step "Build website data";             "$PY" scripts/build_site_data.py
step "Run tests";                      "$PY" -m pytest

echo
echo "Done. See your runs with:  .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db"

if [ "${1:-}" != "--no-serve" ]; then
  echo "Starting the API. Open http://127.0.0.1:8000/docs  (Ctrl+C to stop)"
  "$PY" -m uvicorn turbofan_rul.api:app --port 8000
fi
