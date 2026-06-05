#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "${PY_VERSION}" in
  3.10|3.11|3.12)
    ;;
  *)
    echo "Python ${PY_VERSION} detected. PyTorch/G3T training is expected to work best with Python 3.10 or 3.11." >&2
    echo "Set PYTHON_BIN to a compatible interpreter, for example: PYTHON_BIN=python3.10 bash scripts/setup_env.sh .venv" >&2
    exit 1
    ;;
esac

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"
mkdir -p "${ENV_DIR}/.matplotlib"
export MPLCONFIGDIR="${ENV_DIR}/.matplotlib"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo "Environment ready at ${ENV_DIR}"
echo "Activate with: source ${ENV_DIR}/bin/activate"
echo "Recommended: export MPLCONFIGDIR=${ENV_DIR}/.matplotlib"
