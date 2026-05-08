# Launching a Training Job

```bash
cp .env.example .env
uv run model-trainer
```

The CLI is fully flag-driven (no interactive prompts). Build and review the
full command before running it.

After the pod reaches `RUNNING` the launcher prints the RunPod dashboard URL,
an `ssh` command (when the SSH port mapping is available), and a `tmux attach`
command. The launcher then exits — training continues unattended in tmux on
the pod.

To manage an existing session:

```bash
# Attach to the latest running session (or pick one if multiple)
uv run model-trainer-attach

# Stop the latest running session's pod (asks for confirmation)
uv run model-trainer-stop

# Or target a specific session ID directly
uv run model-trainer-stop <session_id> --force
```

## Credentials (.env)

Three credentials are required at startup. `GIT_TOKEN` is required when
launching GitHub repo mode.

| Key | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face dataset access |
| `WANDB_API_KEY` | Weights & Biases experiment logging |
| `RUNPOD_API_KEY` | RunPod REST API for pod creation and optional auto-stop |
| `GIT_TOKEN` | GitHub token for cloning private training repos |

## Job flags

| Flag | Required | Description | Example |
|---|---|---|---|
| `--experiment_name` | Yes | Human-friendly experiment label. Used to build run label `<experiment_name>-<session_id>` for pod/tmux naming. | `--experiment_name=act_B017_kevin_special` |
| `--train_args.dataset.repo_id` | LeRobot only | Hugging Face dataset repo ID. Forwarded to `lerobot-train` as `--dataset.repo_id`. Required when `--model_type` is set. | `--train_args.dataset.repo_id=myorg/B017_D0-D4` |
| `--github_repo_url` | One of ↓ | Full HTTPS GitHub URL for a custom training repo | `--github_repo_url=https://github.com/org/repo` |
| `--model_type` | One of ↑ | Lerobot off-the-shelf policy type | `--model_type=act` |
| `--github_branch` | If repo set | Branch to clone | `--github_branch=main` |
| `--setup_args.<key>=<value>` | No | GitHub mode only. Additional args appended to `setup.sh`. The launcher strips the `setup_args.` prefix and forwards as `--<key>=<value>`. | `--setup_args.install_extras=vision` |
| `--train_args.<key>=<value>` | No* | Additional args appended to the training command. The launcher strips the `train_args.` prefix and forwards as `--<key>=<value>`. In LeRobot mode, `dataset.repo_id` is required (see row above); other keys are policy-specific. | `--train_args.policy.device=cuda` |
| `--stop_after_training` | No | Ask the pod to stop itself via the RunPod API after the training command exits | `--stop_after_training` |
| `--gpu_type` | Yes | Exact RunPod GPU type ID/name | `--gpu_type="NVIDIA A100 80GB PCIe"` |
| `--gpu_count` | Yes | Number of GPUs | `--gpu_count=1` |
| `--container_disk_gb` | Yes | Ephemeral disk on pod (lost on stop) | `--container_disk_gb=50` |
| `--volume_gb` | Yes | Persistent network volume mounted at `/workspace` | `--volume_gb=500` |
| `--template_id` | No | RunPod template that boots SSH, Jupyter, and nginx | `--template_id=8u8g3zo2jg` |

`--github_repo_url` and `--model_type` are **mutually exclusive** — set exactly one.

\*In **LeRobot mode**, `--train_args.dataset.repo_id` is required; all other `train_args` keys remain optional unless your policy needs them.

Run label behavior:

- Pod name: `<normalized_experiment_name>-<session_id>`
- tmux session name: `<normalized_experiment_name>-<session_id>`
- If you want a job name, pass it explicitly as `--train_args.job_name=<value>`.
- The launcher does **not** inject or rename `wandb` run names.

## Modes

The launcher supports two modes, selected by which of `--github_repo_url` /
`--model_type` is set.

### GitHub repo mode

Requires `GIT_TOKEN` in `.env`. The pod clones the repo into `/workspace/repo`
using `https://x-access-token:$GIT_TOKEN@…` auth, then runs:

```bash
bash setup.sh <setup_args>
bash train.sh <train_args>
```

The repo **must contain `setup.sh` and `train.sh`** at its root (executable
optional when invoked via `bash`). `setup.sh` receives `setup_args` and
`train.sh` receives `train_args` (both from `--*_args.<key>=<value>` flags,
formatted as `--<key>=<value>`). `train.sh` is responsible for downloading the dataset (e.g.
through the Hugging Face Hub APIs using `HF_TOKEN`) and handling output paths.

**Migration:** Repos that previously used `setup.py` / `train.py` should add
thin wrappers at the repo root, e.g. `setup.sh` that runs `python setup.py`
and `train.sh` that runs `python train.py "$@"`.

### Lerobot off-the-shelf mode

The pod clones [`huggingface/lerobot`](https://github.com/huggingface/lerobot)
at the commit pinned in [`pod.py`](../src/model_trainer/pod.py) (`LEROBOT_COMMIT`),
installs Python 3.12 via `uv python install 3.12`, creates a venv with that
version, installs Lerobot into it, and runs:

```bash
lerobot-train \
  --policy.type=<model_type> \
  <train_args>
```

`--policy.type` is injected automatically. Dataset selection is supplied via
`--train_args.dataset.repo_id=<hf_repo>`, which forwards as `--dataset.repo_id=<hf_repo>` to `lerobot-train`.
Lerobot downloads the dataset from the Hub itself into `HF_HOME`.

Any other args required by the chosen policy (e.g. `--policy.device`,
`--wandb.enable`, `--policy.repo_id` for ACT) must be
supplied through additional `--train_args.<key>=<value>` flags. They are policy-specific
and not validated by the launcher.

## Cloud type

Pods are always created in **SECURE** cloud. Community cloud is not used. This
is hard-coded in [`pod.py`](../src/model_trainer/pod.py) and is not
user-configurable per job. If a Secure pod cannot be allocated for the chosen
GPU, retry with a different GPU rather than falling back to Community.

## What the pod runs

The launcher creates the pod through the RunPod REST API and passes a
`dockerStartCmd` that writes `/post_start.sh`, makes it executable, and then
execs the template's `/start.sh`. The template starts SSH/Jupyter/nginx and
runs `/post_start.sh`, which:

1. Installs apt packages: `tmux`, `cmake`, `build-essential`, `python3-dev`,
   `pkg-config`, and the ffmpeg `libav*` headers required by the `av` Python
   package.
2. Writes `/workspace/train_workflow.sh` and starts it in a detached tmux
   session named `<normalized_experiment_name>-<session_id>`.

`train_workflow.sh` then:

1. Exports the credentials and `HF_HOME=/workspace/.cache/huggingface`.
2. `mkdir -p` the cache directory.
3. `cd /workspace`.
4. Runs the GitHub-repo or Lerobot training command (see [Modes](#modes)).
5. If `stop_after_training` is enabled, POSTs to
   `https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop` after the training
   command exits.
6. Exits with the training command's exit code.

The launcher only waits for the pod to report `RUNNING`; it does not SSH into
the pod or poll training completion.

## Env vars available to the training process

The pod is created with the following environment, all of which are
re-exported inside `train_workflow.sh` so they are visible to `train.sh` /
`lerobot-train`:

| Var | Set when | Value |
|---|---|---|
| `HF_TOKEN` | Always | From local `.env` |
| `WANDB_API_KEY` | Always | From local `.env` |
| `VOLUME_MOUNT_PATH` | Always | `/workspace` |
| `HF_HOME` | Always | `/workspace/.cache/huggingface` |
| `GIT_TOKEN` | If set in local `.env` | From local `.env` |
| `RUNPOD_API_KEY` | If `stop_after_training` is true | From local `.env` |

## Filesystem layout on the pod

The training repo and Hugging Face cache live under
`/workspace`, the persistent network volume mount. Other tool caches (apt,
pip, torch, uv, wandb) use their template/container defaults.

| Path | Purpose |
|---|---|
| `/workspace/repo` | Cloned training repo (Lerobot or GitHub mode) |
| `/workspace/.cache/huggingface` | `HF_HOME` (HF Hub download cache, including downloaded datasets) |
| `/workspace/train_workflow.sh` | Generated training script run inside tmux |

## Example: ACT training (launch → delete → relaunch)

End-to-end walkthrough for launching an ACT policy training job on an
NVIDIA RTX A5000, deleting the pod, and relaunching.

### 1. Launch

```bash
uv run model-trainer \
  --experiment_name=act_B017_kevin_special \
  --model_type=act \
  --gpu_type="NVIDIA RTX A5000" \
  --gpu_count=1 \
  --container_disk_gb=50 \
  --volume_gb=500 \
  --template_id=8u8g3zo2jg \
  --train_args.dataset.repo_id=griffinlabs/B017_D0-D4_S2-S7_kevin_special \
  --train_args.output_dir=/workspace/checkpoints \
  --train_args.job_name=act_B017_kevin_special-<session_id> \
  --train_args.policy.device=cuda \
  --train_args.wandb.enable=true \
  --train_args.policy.repo_id=griffinlabs/act_B017_kevin_special \
  --train_args.tolerance_s=0.001
```

The CLI creates the pod and prints:

```
Session created: sessions/<YYYY-MM-DD>/<YYYY-MM-DD_HH-MM-SS_experiment_sessionid>.json

Creating RunPod pod (NVIDIA RTX A5000 x1)...
  → Pod ID: <pod_id>
Waiting for pod to start...
  → Pod status: RUNNING

Training is running in tmux on the pod.
  Dashboard: https://www.runpod.io/console/pods/<pod_id>
  Tmux: tmux attach -t <normalized_experiment_name>-<session_id>
```

The generated `lerobot-train` command on the pod will be:

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=griffinlabs/B017_D0-D4_S2-S7_kevin_special \
  --output_dir=/workspace/checkpoints \
  --job_name=act_B017_kevin_special-<session_id> \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=griffinlabs/act_B017_kevin_special \
  --tolerance_s=0.001
```

`--policy.type` is injected by the launcher. The other args above (including
`--dataset.repo_id`, `--output_dir`, and `--job_name`) are supplied through
`--train_args.<key>=<value>` flags.

### 2. Delete the pod

If the job fails or you need to change config, stop the pod by session:

```bash
uv run model-trainer-stop <session_id>
```

You can still stop or delete directly via the RunPod REST API:

```bash
curl -s -X DELETE \
  "https://rest.runpod.io/v1/pods/<pod_id>" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

Or stop it first (preserves the volume) and delete later:

```bash
# Stop (keeps volume data)
curl -s -X POST \
  "https://rest.runpod.io/v1/pods/<pod_id>/stop" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"

# Delete (removes everything)
curl -s -X DELETE \
  "https://rest.runpod.io/v1/pods/<pod_id>" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

### 3. Relaunch

Run `uv run model-trainer` again with the same (or updated) flags. Each
launch creates a fresh session ID, pod, and tmux session — there is no
"restart" command. Previous session JSON files remain under date folders in
`sessions/` for reference.

## Troubleshooting

### GitHub mode: `setup.sh` or `train.sh` not found

GitHub mode always runs `bash setup.sh <setup_args>` and `bash train.sh <train_args>`
from the repo root after `cd /workspace/repo`. If either file is missing or
not in the root, the shell will fail and the non-zero exit surfaces in the tmux pane running
`train_workflow.sh` (attach with the printed `tmux attach` command and scroll
up for the error).

### Lerobot `FrameTimestampError` (frame timestamp tolerance)

Lerobot validates that the closest decoded video frame for each requested
timestamp is within `tolerance_s` of the queried timestamp. The default
(`tolerance_s = 0.0001` s, i.e. 0.1 ms) is extremely tight and can fail on
real-world datasets even when sync is fine, e.g.:

```
lerobot.datasets.video_utils.FrameTimestampError: One or several query timestamps unexpectedly violate the tolerance (tensor([0.0001]) > tolerance_s=0.0001).
queried timestamps: tensor([1044.2666])
loaded timestamps:  tensor([1044.2667])
```

The drift here is exactly equal to the tolerance — floating-point comparison
just tips it over.

**Fix:** raise `tolerance_s` via `--train_args.tolerance_s=<value>`. Lerobot exposed it as a
top-level training arg in
[huggingface/lerobot#2653](https://github.com/huggingface/lerobot/pull/2653),
so:

```text
--tolerance_s=0.001
```

Rule of thumb: `tolerance_s ≈ 1 / (2 * fps)` is a safe ceiling. Suggested
defaults by capture rate:

| fps | suggested `tolerance_s` |
|---|---|
| 30 | `0.0166` |
| 60 | `0.0083` |
| 100+ | `0.001`–`0.005` |

Keep it as small as you can while still passing — a too-large value masks
genuine sync issues during data collection.
