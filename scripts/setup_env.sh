#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo "Environment ready at ${ENV_DIR}"
echo "Activate with: source ${ENV_DIR}/bin/activate"

