#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage: ./train.sh [options]

Launches LeRobot training with configurable flags for cloud compute.

Core options:
  --num-processes <n>               Accelerate processes (typically #GPUs). Default: 1
  --main-process-port <port>        Accelerate main process port. Default: 29500
  --output-dir <path>               Checkpoint output dir.
  --resume-from-checkpoint <path>   Resume with accelerate state directory.
  --load-model-weights <path>       Load safetensors weights before training.

Dataset options:
  --agibot-world-root <path>        Default: data/agibot-world/tasks
  --galaxea-root <path>             Default: data/galaxea-open-world-dataset
  --interndata-root <path>          Default: data/interndata-a1

Training hyperparameters:
  --per-device-batch-size <n>       Default: 4
  --learning-rate <float>           Default: 5e-5
  --gradient-accumulation-steps <n> Default: 64
  --num-warmup-steps <n>            Default: 64000
  --max-epochs <n>                  Default: 1
  --checkpoint-save-frequency <n>   Default: 320000
  --logging-frequency <n>           Default: 1
  --dataloader-num-workers <n>      Default: 8
  --action-chunk-size <n>           Default: 50
  --model-id <id>                   Default: google/gemma-4-E4B-it
  --action-vocab-size <n>           Default: 2048
  --max-tokens-per-image <n>        Default: 70
  --num-frames <n>                  Default: 1
  --wandb-project-name <name>       Default: Griffin Alpha
  --gradient-clipping <float>       Optional (unset by default)

Other:
  --dry-run                         Print resolved config and exit.
  -h, --help                        Show this help message.
EOF
}

# Defaults aligned with TrainingConfig in lerobot_training/lerobot_training.py
NUM_PROCESSES=1
MAIN_PROCESS_PORT=29500
OUTPUT_DIR="./griffin_alpha_finetune_object"
RESUME_FROM_CHECKPOINT=""
LOAD_MODEL_WEIGHTS=""
AGIBOT_WORLD_ROOT="data/agibot-world/tasks"
GALAXEA_ROOT="data/galaxea-open-world-dataset"
INTERNDATA_ROOT="data/interndata-a1"
PER_DEVICE_BATCH_SIZE=4
LEARNING_RATE="5e-5"
GRADIENT_ACCUMULATION_STEPS=64
NUM_WARMUP_STEPS=64000
MAX_EPOCHS=1
CHECKPOINT_SAVE_FREQUENCY=320000
LOGGING_FREQUENCY=1
DATALOADER_NUM_WORKERS=8
ACTION_CHUNK_SIZE=50
MODEL_ID="google/gemma-4-E4B-it"
ACTION_VOCAB_SIZE=2048
MAX_TOKENS_PER_IMAGE=70
NUM_FRAMES=1
WANDB_PROJECT_NAME="Griffin Alpha"
GRADIENT_CLIPPING=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-processes) NUM_PROCESSES="${2:-}"; shift 2 ;;
    --num-processes=*) NUM_PROCESSES="${1#*=}"; shift ;;
    --main-process-port) MAIN_PROCESS_PORT="${2:-}"; shift 2 ;;
    --main-process-port=*) MAIN_PROCESS_PORT="${1#*=}"; shift ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
    --resume-from-checkpoint) RESUME_FROM_CHECKPOINT="${2:-}"; shift 2 ;;
    --resume-from-checkpoint=*) RESUME_FROM_CHECKPOINT="${1#*=}"; shift ;;
    --load-model-weights) LOAD_MODEL_WEIGHTS="${2:-}"; shift 2 ;;
    --load-model-weights=*) LOAD_MODEL_WEIGHTS="${1#*=}"; shift ;;
    --agibot-world-root) AGIBOT_WORLD_ROOT="${2:-}"; shift 2 ;;
    --agibot-world-root=*) AGIBOT_WORLD_ROOT="${1#*=}"; shift ;;
    --galaxea-root) GALAXEA_ROOT="${2:-}"; shift 2 ;;
    --galaxea-root=*) GALAXEA_ROOT="${1#*=}"; shift ;;
    --interndata-root) INTERNDATA_ROOT="${2:-}"; shift 2 ;;
    --interndata-root=*) INTERNDATA_ROOT="${1#*=}"; shift ;;
    --per-device-batch-size) PER_DEVICE_BATCH_SIZE="${2:-}"; shift 2 ;;
    --per-device-batch-size=*) PER_DEVICE_BATCH_SIZE="${1#*=}"; shift ;;
    --learning-rate) LEARNING_RATE="${2:-}"; shift 2 ;;
    --learning-rate=*) LEARNING_RATE="${1#*=}"; shift ;;
    --gradient-accumulation-steps) GRADIENT_ACCUMULATION_STEPS="${2:-}"; shift 2 ;;
    --gradient-accumulation-steps=*) GRADIENT_ACCUMULATION_STEPS="${1#*=}"; shift ;;
    --num-warmup-steps) NUM_WARMUP_STEPS="${2:-}"; shift 2 ;;
    --num-warmup-steps=*) NUM_WARMUP_STEPS="${1#*=}"; shift ;;
    --max-epochs) MAX_EPOCHS="${2:-}"; shift 2 ;;
    --max-epochs=*) MAX_EPOCHS="${1#*=}"; shift ;;
    --checkpoint-save-frequency) CHECKPOINT_SAVE_FREQUENCY="${2:-}"; shift 2 ;;
    --checkpoint-save-frequency=*) CHECKPOINT_SAVE_FREQUENCY="${1#*=}"; shift ;;
    --logging-frequency) LOGGING_FREQUENCY="${2:-}"; shift 2 ;;
    --logging-frequency=*) LOGGING_FREQUENCY="${1#*=}"; shift ;;
    --dataloader-num-workers) DATALOADER_NUM_WORKERS="${2:-}"; shift 2 ;;
    --dataloader-num-workers=*) DATALOADER_NUM_WORKERS="${1#*=}"; shift ;;
    --action-chunk-size) ACTION_CHUNK_SIZE="${2:-}"; shift 2 ;;
    --action-chunk-size=*) ACTION_CHUNK_SIZE="${1#*=}"; shift ;;
    --model-id) MODEL_ID="${2:-}"; shift 2 ;;
    --model-id=*) MODEL_ID="${1#*=}"; shift ;;
    --action-vocab-size) ACTION_VOCAB_SIZE="${2:-}"; shift 2 ;;
    --action-vocab-size=*) ACTION_VOCAB_SIZE="${1#*=}"; shift ;;
    --max-tokens-per-image) MAX_TOKENS_PER_IMAGE="${2:-}"; shift 2 ;;
    --max-tokens-per-image=*) MAX_TOKENS_PER_IMAGE="${1#*=}"; shift ;;
    --num-frames) NUM_FRAMES="${2:-}"; shift 2 ;;
    --num-frames=*) NUM_FRAMES="${1#*=}"; shift ;;
    --wandb-project-name) WANDB_PROJECT_NAME="${2:-}"; shift 2 ;;
    --wandb-project-name=*) WANDB_PROJECT_NAME="${1#*=}"; shift ;;
    --gradient-clipping) GRADIENT_CLIPPING="${2:-}"; shift 2 ;;
    --gradient-clipping=*) GRADIENT_CLIPPING="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not found." >&2
  exit 1
fi

export OUTPUT_DIR
export RESUME_FROM_CHECKPOINT
export LOAD_MODEL_WEIGHTS
export AGIBOT_WORLD_ROOT
export GALAXEA_ROOT
export INTERNDATA_ROOT
export PER_DEVICE_BATCH_SIZE
export LEARNING_RATE
export GRADIENT_ACCUMULATION_STEPS
export NUM_WARMUP_STEPS
export MAX_EPOCHS
export CHECKPOINT_SAVE_FREQUENCY
export LOGGING_FREQUENCY
export DATALOADER_NUM_WORKERS
export ACTION_CHUNK_SIZE
export MODEL_ID
export ACTION_VOCAB_SIZE
export MAX_TOKENS_PER_IMAGE
export NUM_FRAMES
export WANDB_PROJECT_NAME
export GRADIENT_CLIPPING

TRAIN_CFG_JSON="$(python3 - <<'PY'
import json
import os

cfg = {
    "per_device_batch_size": int(os.environ["PER_DEVICE_BATCH_SIZE"]),
    "learning_rate": float(os.environ["LEARNING_RATE"]),
    "gradient_accumulation_steps": int(os.environ["GRADIENT_ACCUMULATION_STEPS"]),
    "num_warmup_steps": int(os.environ["NUM_WARMUP_STEPS"]),
    "max_epochs": int(os.environ["MAX_EPOCHS"]),
    "output_dir": os.environ["OUTPUT_DIR"],
    "resume_from_checkpoint": os.environ["RESUME_FROM_CHECKPOINT"],
    "load_model_weights": os.environ["LOAD_MODEL_WEIGHTS"] or None,
    "agibot_world_root": os.environ["AGIBOT_WORLD_ROOT"],
    "galaxea_open_world_ds_root": os.environ["GALAXEA_ROOT"],
    "interndata_a1_root": os.environ["INTERNDATA_ROOT"],
    "wandb_project_name": os.environ["WANDB_PROJECT_NAME"],
    "checkpoint_save_frequency": int(os.environ["CHECKPOINT_SAVE_FREQUENCY"]),
    "logging_frequency": int(os.environ["LOGGING_FREQUENCY"]),
    "dataloader_num_workers": int(os.environ["DATALOADER_NUM_WORKERS"]),
    "action_chunk_size": int(os.environ["ACTION_CHUNK_SIZE"]),
    "model_id": os.environ["MODEL_ID"],
    "action_vocab_size": int(os.environ["ACTION_VOCAB_SIZE"]),
    "max_tokens_per_image": int(os.environ["MAX_TOKENS_PER_IMAGE"]),
    "num_frames": int(os.environ["NUM_FRAMES"]),
}

gradient_clipping = os.environ["GRADIENT_CLIPPING"].strip()
if gradient_clipping:
    cfg["gradient_clipping"] = float(gradient_clipping)

print(json.dumps(cfg))
PY
)"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Resolved training config:"
  echo "${TRAIN_CFG_JSON}"
  echo "Resolved accelerate options:"
  echo "{\"num_processes\": ${NUM_PROCESSES}, \"main_process_port\": ${MAIN_PROCESS_PORT}}"
  exit 0
fi

echo "Launching training..."
uv run accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  -m lerobot_training.lerobot_training \
  --config-json "${TRAIN_CFG_JSON}"