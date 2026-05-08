#!/usr/bin/env bash
# Example train launch script
# shell expands ${HF_TOKEN} below — load .env first (uv's --env-file does not set parent-shell vars)

cd "$(dirname "$0")"
set -a
[ -f .env ] && . ./.env
set +a

uv run --env-file .env model-trainer \
  --experiment_name=bryce-test-griffin-alpha-gemma4e4b \
  --github_repo_url=https://github.com/griffinlabs-ai/nora.git \
  --github_branch=feat/deployment-tool \
  --gpu_type="NVIDIA RTX A5000" \
  --gpu_count=1 \
  --container_disk_gb=80 \
  --volume_gb=500 \
  --stop_after_training \
  --train_args.output-dir=/workspace/checkpoints/griffin_alpha_finetune_object \
  --train_args.wandb-project-name="Bryce Test Griffin Alpha"
