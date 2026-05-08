# Example train launch script

uv run --env-file .env model-trainer \
  --experiment_name=griffin-alpha-gemma4e4b \
  --github_repo_url=https://github.com/griffinlabs-ai/nora.git \
  --github_branch=main \
  --gpu_type="NVIDIA RTX A5000" \
  --gpu_count=1 \
  --container_disk_gb=80 \
  --volume_gb=500 \
  --stop_after_training \
  --setup_args.python-version=3.12 \
  --setup_args.download-profile=none \
  --setup_args.hf-token="${HF_TOKEN}" \
  --train_args.output-dir=/workspace/checkpoints/griffin_alpha_finetune_object \
  --train_args.wandb-project-name="Griffin Alpha"