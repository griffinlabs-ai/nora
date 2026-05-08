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
  --skip-system-deps          Skip system package installation (ffmpeg/build deps).
  --uv-sync-args "<args>"     Extra args passed to "uv sync".
  -h, --help                  Show this help message.
EOF
}

INHERITED_HF_TOKEN="${HF_TOKEN:-}"
PYTHON_VERSION="3.12"
HF_TOKEN=""
DOWNLOAD_PROFILE="none"
SKIP_SYSTEM_DEPS=false
UV_SYNC_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-version)
      PYTHON_VERSION="${2:-}"
      shift 2
      ;;
    --python-version=*)
      PYTHON_VERSION="${1#*=}"
      shift
      ;;
    --hf-token)
      HF_TOKEN="${2:-}"
      shift 2
      ;;
    --hf-token=*)
      HF_TOKEN="${1#*=}"
      shift
      ;;
    --download-profile)
      DOWNLOAD_PROFILE="${2:-}"
      shift 2
      ;;
    --download-profile=*)
      DOWNLOAD_PROFILE="${1#*=}"
      shift
      ;;
    --skip-system-deps)
      SKIP_SYSTEM_DEPS=true
      shift
      ;;
    --uv-sync-args)
      UV_SYNC_ARGS="${2:-}"
      shift 2
      ;;
    --uv-sync-args=*)
      UV_SYNC_ARGS="${1#*=}"
      shift
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

SETUP_HF_TOKEN="${HF_TOKEN:-${INHERITED_HF_TOKEN}}"
unset HF_TOKEN

cd "${REPO_ROOT}"

run_privileged() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Need root privileges to run: $*" >&2
    return 1
  fi
}

export PATH="${HOME:-}/.local/bin:${REPO_ROOT}/.venv/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  if ! command -v curl >/dev/null 2>&1; then
    echo "Unable to install uv automatically (missing curl)." >&2
    echo "Install uv manually: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation completed but uv is still not on PATH." >&2
  echo "Try restarting the shell or adding ~/.local/bin to PATH." >&2
  exit 1
fi

echo "Installing Python ${PYTHON_VERSION} with uv..."
uv python install "${PYTHON_VERSION}"

echo "Syncing environment with uv (python=${PYTHON_VERSION})..."
if [[ -n "${UV_SYNC_ARGS}" ]]; then
  # shellcheck disable=SC2086
  uv sync --python "${PYTHON_VERSION}" ${UV_SYNC_ARGS}
else
  uv sync --python "${PYTHON_VERSION}"
fi

echo "Installing Hugging Face Hub CLI with pip..."
"${REPO_ROOT}/.venv/bin/python" -m ensurepip --upgrade
"${REPO_ROOT}/.venv/bin/python" -m pip install --upgrade "huggingface_hub[cli]"

if [[ "${SKIP_SYSTEM_DEPS}" == "true" ]]; then
  echo "Skipping system dependency installation."
elif command -v apt-get >/dev/null 2>&1; then
  echo "Installing apt packages..."
  run_privileged apt-get update
  run_privileged apt-get install -y \
    ffmpeg \
    tmux \
    cmake \
    build-essential \
    python3-dev \
    pkg-config \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev
elif command -v brew >/dev/null 2>&1; then
  echo "Installing Homebrew packages..."
  brew install ffmpeg
else
  echo "Skipping system dependency installation (no supported package manager detected)." >&2
fi

if [[ "${DOWNLOAD_PROFILE}" != "none" || -n "${SETUP_HF_TOKEN}" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "hf CLI is required for Hugging Face login/downloads but not found after pip install." >&2
    exit 1
  fi
fi

if [[ -n "${SETUP_HF_TOKEN}" ]]; then
  env -u HF_TOKEN hf auth login --token "${SETUP_HF_TOKEN}"
  export HF_TOKEN="${SETUP_HF_TOKEN}"
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
