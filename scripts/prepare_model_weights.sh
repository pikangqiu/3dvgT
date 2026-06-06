#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_WEIGHTS_ROOT="${MODEL_WEIGHTS_ROOT:-checkpoints/g3t}"
MODEL_REPO_ID="${MODEL_REPO_ID:-thatbrguy/g3t}"
MODEL_REVISION="${MODEL_REVISION:-}"
ALLOW_PATTERNS=("${ALLOW_PATTERNS:-*.pt}" "*.bin")
DOWNLOAD="false"

usage() {
  cat <<EOF
Usage: bash scripts/prepare_model_weights.sh [--download]

Environment:
  PYTHON_BIN          Python executable. Default: python3
  MODEL_WEIGHTS_ROOT  Target checkpoint directory. Default: checkpoints/g3t
  MODEL_REPO_ID       Hugging Face repo id. Default: thatbrguy/g3t
  MODEL_REVISION      Optional Hugging Face revision.
  ALLOW_PATTERNS      First allow pattern. Default: *.pt; *.bin is always also requested.

Default mode is a dry-run that creates the target directory and prints the download,
inspection, and config-wiring steps. Use --download only when network access and
model-license/access requirements are satisfied.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download)
      DOWNLOAD="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${MODEL_WEIGHTS_ROOT}"

download_args=(
  "scripts/download_weights.py"
  "--repo-id" "${MODEL_REPO_ID}"
  "--output-dir" "${MODEL_WEIGHTS_ROOT}"
)
if [[ -n "${MODEL_REVISION}" ]]; then
  download_args+=("--revision" "${MODEL_REVISION}")
fi
for pattern in "${ALLOW_PATTERNS[@]}"; do
  download_args+=("--allow-pattern" "${pattern}")
done
if [[ "${DOWNLOAD}" != "true" ]]; then
  download_args+=("--dry-run")
fi

echo "Model weights root prepared at: ${MODEL_WEIGHTS_ROOT}"
echo "download_command: ${PYTHON_BIN} ${download_args[*]}"
"${PYTHON_BIN}" "${download_args[@]}"

cat <<EOF

Next weight wiring steps:
  1. Inspect candidates:
     PYTHONPATH=src ${PYTHON_BIN} scripts/inspect_checkpoint.py ${MODEL_WEIGHTS_ROOT}
     PYTHONPATH=src ${PYTHON_BIN} scripts/inspect_checkpoint.py ${MODEL_WEIGHTS_ROOT} --inspect-all --sample-limit 30
  2. Set runtime.model.weights_path to one concrete .pt/.pth/.bin checkpoint file.
     Do not set runtime.model.weights_path to the directory itself.
     PYTHONPATH=src ${PYTHON_BIN} scripts/configure_model_weights.py --config configs/reconstruction_first.json --weights-path <checkpoint_file> --output configs/reconstruction_first.weights.json
  3. Run adapter/readiness checks:
     PYTHONPATH=src ${PYTHON_BIN} scripts/check_training_readiness.py --config configs/reconstruction_first.weights.json
     PYTHONPATH=src ${PYTHON_BIN} scripts/check_model_adapter.py --config configs/reconstruction_first.weights.json

Explicit real download:
  bash scripts/prepare_model_weights.sh --download
EOF
