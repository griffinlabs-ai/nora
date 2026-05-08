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
    --hf-token)
      HF_TOKEN="${2:-}"
      shift 2
      ;;
    --download-profile)
      DOWNLOAD_PROFILE="${2:-}"
      shift 2
      ;;
    --skip-system-deps)
      SKIP_SYSTEM_DEPS=true
      shift
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

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  echo "Installing uv..."
  if command -v brew >/dev/null 2>&1; then
    brew install uv
  elif command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  else
    echo "Unable to install uv automatically (missing brew/curl)." >&2
    echo "Install uv manually: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
  fi

  if ! command -v uv >/dev/null 2>&1; then
    if [[ -x "${HOME}/.local/bin/uv" ]]; then
      export PATH="${HOME}/.local/bin:${PATH}"
    fi
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "uv installation completed but uv is still not on PATH." >&2
    echo "Try restarting the shell or adding ~/.local/bin to PATH." >&2
    exit 1
  fi
}

install_system_deps() {
  if [[ "${SKIP_SYSTEM_DEPS}" == "true" ]]; then
    echo "Skipping system dependency installation."
    return 0
  fi

  if command -v apt-get >/dev/null 2>&1; then
    local apt_pkgs=(
      ffmpeg
      tmux
      cmake
      build-essential
      python3-dev
      pkg-config
      libavcodec-dev
      libavdevice-dev
      libavfilter-dev
      libavformat-dev
      libavutil-dev
      libswresample-dev
      libswscale-dev
    )

    local missing_pkgs=()
    local pkg
    for pkg in "${apt_pkgs[@]}"; do
      if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
        missing_pkgs+=("${pkg}")
      fi
    done

    if (( ${#missing_pkgs[@]} > 0 )); then
      echo "Installing apt packages: ${missing_pkgs[*]}"
      run_privileged apt-get update
      run_privileged apt-get install -y "${missing_pkgs[@]}"
    else
      echo "System dependencies already installed."
    fi
  elif command -v brew >/dev/null 2>&1; then
    if ! command -v ffmpeg >/dev/null 2>&1; then
      echo "Installing ffmpeg via Homebrew..."
      brew install ffmpeg
    else
      echo "ffmpeg already installed."
    fi
  else
    echo "Skipping system dependency installation (no supported package manager detected)." >&2
  fi
}

if ! command -v uv >/dev/null 2>&1; then
  install_uv
fi

install_system_deps

if [[ -n "${HF_TOKEN}" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "hf CLI is required for Hugging Face login/downloads but not found." >&2
    exit 1
  fi
  hf auth login --token "${HF_TOKEN}"
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
