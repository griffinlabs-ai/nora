#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [options]

Sets up the LeRobot training environment and optionally downloads datasets.

Options:
  --python-version <version>  Python version for uv sync. Default: 3.12
  --hf-token <token>          Hugging Face token for non-interactive login.
  --download-profile <name>   Dataset profile: none|sample|full. Default: none
  --uv-sync-args "<args>"     Extra args passed to "uv sync".
  -h, --help                  Show this help message.
EOF
}

PYTHON_VERSION="3.12"
HF_TOKEN=""
DOWNLOAD_PROFILE="none"
UV_SYNC_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-version)
      PYTHON_VERSION="${2:-}"
      shift 2
      ;;
    --hf-token)
      HF_TOKEN="${2:-}"
      shift 2
      ;;
    --download-profile)
      DOWNLOAD_PROFILE="${2:-}"
      shift 2
      ;;
    --uv-sync-args)
      UV_SYNC_ARGS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${DOWNLOAD_PROFILE}" in
  none|sample|full) ;;
  *)
    echo "Invalid --download-profile '${DOWNLOAD_PROFILE}'. Use none|sample|full." >&2
    exit 1
    ;;
esac

cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not found. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [[ -n "${HF_TOKEN}" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "hf CLI is required for Hugging Face login/downloads but not found." >&2
    exit 1
  fi
  hf auth login --token "${HF_TOKEN}"
fi

echo "Syncing environment with uv (python=${PYTHON_VERSION})..."
if [[ -n "${UV_SYNC_ARGS}" ]]; then
  # shellcheck disable=SC2086
  uv sync --python "${PYTHON_VERSION}" ${UV_SYNC_ARGS}
else
  uv sync --python "${PYTHON_VERSION}"
fi

case "${DOWNLOAD_PROFILE}" in
  none)
    echo "Skipping dataset download (profile=none)."
    ;;
  sample)
    echo "Downloading sample datasets..."
    bash "${REPO_ROOT}/scripts/download-all-datasets-sample.sh"
    ;;
  full)
    echo "Downloading full datasets..."
    bash "${REPO_ROOT}/scripts/download-agibot-world.sh"
    bash "${REPO_ROOT}/scripts/download-galaxea-open-world-dataset.sh"
    bash "${REPO_ROOT}/scripts/download-interndata-a1.sh"
    ;;
esac

echo "Setup complete."
