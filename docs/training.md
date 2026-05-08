# NORA Training & Data Pipeline

NORA ships with **two independent training stacks**, picked by the dataset format you have:

| Path | Trainer entrypoint | Dataset format | Backbone | Use case |
|---|---|---|---|---|
| **RLDS** (OpenVLA-style) | [training/train.py](../training/train.py) | TFDS / RLDS (OXE mixtures) | Qwen 2.5 VL | Pretraining and OXE-mix finetuning |
| **LeRobot** | [lerobot_training/lerobot_training.py](../lerobot_training/lerobot_training.py) | `LeRobotDataset` v0.5.x | `google/gemma-4-E4B-it` (image-text-to-text) | Multi-embodiment finetuning on AgiBot World / Galaxea / InternData-A1 |

Both stacks share the same core idea: render an `(image, instruction) → <robot_action_i>` chat-template example and let the LM head learn the action token sequence.

---

## 1. RLDS path

### 1.1 `TrainingConfig` ([training/train.py:25-60](../training/train.py#L25-L60))

| Field | Default | Notes |
|---|---|---|
| `per_device_batch_size` | 16 | |
| `learning_rate` | 5e-5 | |
| `gradient_accumulation_steps` | 2 | |
| `num_warmup_steps` | 1000 | cosine warmup |
| `max_train_steps` | 100000 | |
| `output_dir` | `'/your_output'` | must be set |
| `resume_from_checkpoint` | `''` | empty = fresh run |
| `load_model_weights` | `None` | optional safetensors path |
| `data_root_dir` | `'/your_data_root_dir'` | RLDS root |
| `data_mix` | `'libero_10_no_noops'` | OXE mixture name |
| `resize_resolution` | `(224, 224)` | |
| `shuffle_buffer_size` | 256000 | TFDS shuffle |
| `wandb_project_name` | `'Nora VLA'` | |
| `checkpoint_save_frequency` | 20000 | steps |
| `logging_frequency` | 100 | steps |
| `gradient_clipping` | `None` | optional max-norm |

### 1.2 Model & processor ([training/train.py:143-165](../training/train.py#L143-L165))

```python
processor      = AutoProcessor.from_pretrained('declare-lab/nora')        # padding_side = 'left'
model          = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    'declare-lab/nora',
                    torch_dtype=torch.bfloat16,
                    attn_implementation='flash_attention_2')
fast_tokenizer = AutoProcessor.from_pretrained('physical-intelligence/fast', trust_remote_code=True)
```

Optional pretrained weights are loaded with `safe_open` and `model.load_state_dict(..., strict=False)` if `config.load_model_weights` is set.

### 1.3 Dataset wiring ([training/datasets/datasets.py:98-179](../training/datasets/datasets.py#L98-L179))

`RLDSDataset` is an `IterableDataset` that delegates to OpenVLA's `make_interleaved_dataset`. Key kwargs it sets:

- `load_camera_views=("primary",)`, `load_depth=False`, `load_proprio=True`, `load_language=True`
- `action_proprio_normalization_type=NormalizationType.BOUNDS_Q99` — q01/q99 quantile normalization mapped into `[-1, 1]`
- **Trajectory transforms**: `window_size=1`, `future_action_window_size=4`, `skip_unlabeled=True`, `goal_relabeling_strategy="uniform"` (see [datasets.py:127-135](../training/datasets/datasets.py#L127-L135)).
- **Frame transforms**: `resize_size=resize_resolution`, `num_parallel_calls=16`.
- Optional image augmentation (off by default): `random_resized_crop`, `random_brightness`, `random_contrast`, `random_saturation`, `random_hue`.

Mixture definitions (e.g. `bridge`, `rtx`, `oxe_magic_soup_plus`, `libero_10_no_noops`) live in [training/datasets/rlds/oxe/mixtures.py](../training/datasets/rlds/oxe/mixtures.py); per-dataset image keys / state keys / action encodings are in [training/datasets/rlds/oxe/configs.py](../training/datasets/rlds/oxe/configs.py).

`RLDSBatchTransform` ([datasets.py:57-95](../training/datasets/datasets.py#L57-L95)) reduces each RLDS sample to:

```python
{
  "image":   PIL.Image,                        # observation.image_primary[0]
  "action":  np.ndarray,                       # raw action chunk (window+future)
  "lang":    str,                              # decoded, lowercased instruction
  "proprio": np.ndarray,
  "dataset_name": str,
}
```

### 1.4 Action tokenization & message construction ([training/train.py:74-103](../training/train.py#L74-L103))

```python
def map_fast_token_to_vlm_action(tokens):
    return ''.join(f"<robot_action_{t}>" for t in tokens)

fast_tokens = fast_tokenizer(action)
vlm_action  = map_fast_token_to_vlm_action(fast_tokens[0])

messages = [
  {"role": "user", "content": [
      {"type": "image", "image": pixel_values},
      {"type": "text",  "text":  lang}]},
  {"role": "assistant", "content": [
      {"type": "text", "text": vlm_action}]},
]
```

### 1.5 Collation & label masking ([training/train.py:105-140](../training/train.py#L105-L140))

The collate function applies the chat template (`add_generation_prompt=False`), runs `process_vision_info`, calls the processor with `padding=True, return_tensors='pt'`, then builds `labels` such that loss is computed **only** on action tokens:

1. Clone `input_ids` into `labels`.
2. For each sequence, find the first index whose token id ∈ `[151665, 153712]`.
3. Set every label before that index to `-100`.
4. Set every PAD label to `-100`.

If a sequence contains no action token, the entire sequence is masked.

### 1.6 Optimizer, schedule, training loop ([training/train.py:168-301](../training/train.py#L168-L301))

- `AdamW(lr=5e-5, betas=(0.9, 0.95), weight_decay=1e-8, eps=1e-8)`.
- `get_scheduler(name="cosine", num_warmup_steps=..., num_training_steps=max_train_steps)`.
- `Accelerator(gradient_accumulation_steps=...)` with `dispatch_batches=False`.
- Optional `accelerator.clip_grad_norm_(..., config.gradient_clipping)` per optimizer step.
- `wandb.init(project=config.wandb_project_name)` on the main process; logs `train_loss`, `learning_rate`, and a manually computed L2 grad norm every `logging_frequency` steps.
- Checkpoints written via `accelerator.save_state(f"{output_dir}/steps_{step}")` every `checkpoint_save_frequency` steps; per-checkpoint summary appended to `{output_dir}/summary.jsonl`.
- Resume via `accelerator.load_state(config.resume_from_checkpoint)`.

### 1.7 Distributed config ([training/accelerator_config.yaml](../training/accelerator_config.yaml))

```yaml
distributed_type: MULTI_GPU
mixed_precision: bf16
num_machines: 1
num_processes: 8
gpu_ids: "0,1,2,3,4,5,6,7"
main_process_port: 29512
rdzv_backend: static
```

### 1.8 Running it

```bash
cd training
conda create -n nora_train python=3.10 -y && conda activate nora_train
pip install -r requirements.txt
accelerate launch --config_file=accelerator_config.yaml train.py
```

Edit `TrainingConfig` defaults in [train.py:25-60](../training/train.py#L25-L60) before launching (the script does not parse CLI args).

> ⚠️ To change the action-chunk horizon (NORA ↔ NORA-LONG), edit `future_action_window_size` at [training/datasets/datasets.py:132](../training/datasets/datasets.py#L132).

---

## 2. LeRobot path

This trainer targets a different model family (Gemma-4 image-text-to-text) and the lerobot 0.5.x dataset format. It also handles three multi-embodiment datasets concatenated together.

### 2.1 `TrainingConfig` ([lerobot_training/lerobot_training.py:41-64](../lerobot_training/lerobot_training.py#L41-L64))

| Field | Default | Notes |
|---|---|---|
| `per_device_batch_size` | 2 | very small — large grad accum compensates |
| `learning_rate` | 5e-5 | |
| `gradient_accumulation_steps` | 128 | |
| `num_warmup_steps` | 128000 | divided by accum at scheduler creation |
| `max_epochs` | 1 | |
| `output_dir` | `'./griffin_alpha_finetune_object'` | |
| `agibot_world_root` | `'data/agibot-world/tasks'` | |
| `galaxea_open_world_ds_root` | `'data/galaxea-open-world-dataset'` | |
| `interndata_a1_root` | `'data/interndata-a1/'` | |
| `wandb_project_name` | `'Griffin Alpha'` | |
| `checkpoint_save_frequency` | 640000 | |
| `logging_frequency` | 1 | every step |
| `dataloader_num_workers` | 4 | |
| `action_chunk_size` | 50 | |
| `model_id` | `'google/gemma-4-E4B-it'` | |
| `action_vocab_size` | 2048 | added as special tokens |
| `max_tokens_per_image` | 70 | Gemma-4 image token budget |
| `num_frames` | 3 | image history per sample |

### 2.2 Model & action vocab injection ([lerobot_training.py:236-310](../lerobot_training/lerobot_training.py#L236-L310))

```python
with accelerator.main_process_first():
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        max_soft_tokens=config.max_tokens_per_image,
        image_seq_length=config.max_tokens_per_image,
    )
    processor.tokenizer.padding_side = 'left'
    processor.tokenizer.add_tokens(
        [f"<robot_action_{i}>" for i in range(config.action_vocab_size)],
        special_tokens=True,
    )

with accelerator.main_process_first():
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id, torch_dtype=torch.bfloat16
    )

accelerator.wait_for_everyone()
model.resize_token_embeddings(new_vocab_size)
model.config.vocab_size = new_vocab_size
```

A subsequent loop walks `model.named_modules()` looking for any `nn.Embedding` whose weight still has the old vocab size and resizes/re-initializes it (keeping copied rows for the original tokens, sampling new rows from `N(mean, std)` of the originals). This is needed because Gemma-4 has nested embedding modules that `resize_token_embeddings` does not catch.

### 2.3 Action FAST tokenizer ([lerobot_training.py:120-135](../lerobot_training/lerobot_training.py#L120-L135))

The LeRobot path uses `lerobot/fast-action-tokenizer` (not `physical-intelligence/fast`). Min/max action token IDs are **discovered dynamically** by scanning the Gemma vocab for `<robot_action_*>` strings, so the hard-coded `[151665, 153712]` range from the RLDS path does not apply here.

### 2.4 Datasets ([lerobot_training/load_datasets.py](../lerobot_training/load_datasets.py))

Three loaders are concatenated via `torch.utils.data.ConcatDataset`:

| Loader | Embodiment prompt | Action layout |
|---|---|---|
| `load_agibot_world_dataset` | "AgiBot G1 with 2 grippers" | dual-arm 7-DoF, 16 dims; gripper inverted (`1 - x`) |
| `load_galaxea_dataset` | "Galaxea R1 Lite" | dual-arm 7-DoF (gripper reshaped) |
| `load_interndata_a1_dataset` | Mixed: Franka (single 7-DoF), Genie1 (dual 7-DoF), Lift2 (dual 6-DoF), Aloha-split (dual 6-DoF) | varies |

Action-dim "is-pad" masks at [load_datasets.py:10-17](../lerobot_training/load_datasets.py#L10-L17):

```python
ACTION_DIM_IS_PAD = {
    'dual_arm_7dof':  zeros(16, bool),
    'dual_arm_6dof':  [False]*6 + [True] + [False]*8,    # 7th dim (left wrist) padded
    'single_arm_7dof':[False]*8 + [True]*8,              # whole right arm padded
}
```

Image keys are standardized to:

```python
('observation.images.head', 'observation.images.hand_left', 'observation.images.hand_right')
```

### 2.5 Per-frame processing ([utils/data_loading.py:200-311](../utils/data_loading.py#L200-L311))

For each subset:

1. **`SkipEpisodesLeRobotDataset`** ([utils/data_loading.py:32-80](../utils/data_loading.py#L32-L80)) wraps `LeRobotDataset`, reads `meta/removed_episodes.json`, and remaps indices so dirty episodes are invisible without reloading the dataset. Recently introduced for performance.
2. **`delta_timestamps`** are generated per dataset:
   - actions: `[i / raw_fps for i in range(load_action_chunk_size)]`
   - images: `[float(i - num_frames + 1) for i in range(num_frames)]` (past → current)
3. **Normalization stats** loaded from `delta_norm_stats.json` at the dataset root, then merged via a per-embodiment `norm_stats_transform`.
4. **Processor pipeline** (`PolicyProcessorPipeline`) chains:
   - [`ResampleActionProcessorStep`](../utils/data_loading.py#L82-L139) — cubic-spline resampling of the action chunk (or simple stride decimation when target divides source).
   - [`Abs2DeltaActionProcessorStep`](../utils/data_loading.py#L142-L172) — converts absolute actions to deltas using the current proprio state, except on gripper dims (mask = `[True]*7 + [False]` per arm).
   - `NormalizerProcessorStep` (lerobot built-in) — quantile normalization to `[-1, 1]`.
   - A `to_transition` lambda that calls the embodiment-specific instance transform.
5. The pipeline output is wrapped in `PreprocessedDataset` so the heavy work runs inside `__getitem__`, parallelized by the DataLoader workers.

> ⚠️ **Lambda capture pitfall** (fixed in commit 5c9e0e3): the per-subset lambda binds `inst_transform` as a default argument — `lambda b, inst_transform=subset_inst_transform: ...` — so each subset keeps its own transform instead of all closing over the last one. Don't revert this. See [utils/data_loading.py:305](../utils/data_loading.py#L305).

### 2.6 Collation ([utils/data_loading.py:313-336](../utils/data_loading.py#L313-L336))

`collate_with_observation_image_lists` keeps `observation.images.*` fields as **Python lists** (one PIL/Tensor per sample) rather than stacked tensors, so heterogeneous resolutions across embodiments are tolerated. Other fields go through `default_collate`.

### 2.7 Message construction & label mask ([lerobot_training.py:160-216](../lerobot_training/lerobot_training.py#L160-L216))

```python
content = [
  {"type": "text", "text": f"[embodiment: {embodiment}] "},
  *[{"type": "image"} for _ in imgs],
  {"type": "text", "text": f"{task}\npredict subtask: {bool}"},
]
messages = [
  {"role": "user",      "content": content},
  {"role": "assistant", "content": [{"type": "text",
       "text": f"{subtask_segment}action: {vlm_action}"}]},
]
```

Label masking finds the **last** start-of-turn token (start of the assistant turn) and `-100`s everything before it, plus all PAD tokens. Loss therefore covers the full assistant response (subtask text + action tokens).

### 2.8 Optimizer, schedule, distributed setup ([lerobot_training.py:321-442](../lerobot_training/lerobot_training.py#L321-L442))

```python
accelerator = Accelerator(
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="wandb",
    kwargs_handlers=[
        InitProcessGroupKwargs(timeout=timedelta(seconds=1800)),       # 30-min NCCL timeout
        DistributedDataParallelKwargs(find_unused_parameters=True),    # required by Gemma-4
    ],
)
accelerator.dataloader_config.dispatch_batches = False
```

- Optimizer: `AdamW(lr=5e-5, betas=(0.9, 0.95), weight_decay=1e-8, eps=1e-8)`.
- Scheduler: cosine with `num_warmup_steps = ceil(num_warmup_steps / accum)` and `num_training_steps = ceil(len(loader)/accum) * max_epochs`.
- Gradient clipping (when set) and optimizer/scheduler steps fire only on sync steps (`accelerator.sync_gradients`).
- `accelerator.init_trackers(wandb_project_name, config=config)` on the main process; logs `train_loss`, `learning_rate`, `grad_norm` every `logging_frequency` step (default: every step).
- Checkpoints: `accelerator.save_state(f"{output_dir}/steps_{step}")` every `checkpoint_save_frequency` steps, plus a final save after the loop. Resume with `accelerator.load_state(config.resume_from_checkpoint)`.

### 2.9 Running it

```bash
cd lerobot_training
conda create -n nora_lerobot python=3.10 -y && conda activate nora_lerobot
pip install -r lerobot_requirements.txt
accelerate launch lerobot_training.py
```

Edit `TrainingConfig` defaults in [lerobot_training.py:41-64](../lerobot_training/lerobot_training.py#L41-L64) before launching.

Dataset download helpers live in [scripts/](../scripts/):

| Script | Purpose |
|---|---|
| [download-agibot-world.sh](../scripts/download-agibot-world.sh) | full AgiBot World |
| [download-galaxea-open-world-dataset.sh](../scripts/download-galaxea-open-world-dataset.sh) | full Galaxea OW |
| [download-interndata-a1.sh](../scripts/download-interndata-a1.sh) | full InternData-A1 |
| [download-all-datasets-sample.sh](../scripts/download-all-datasets-sample.sh) | small sample of all three |
| [remove-video-features.py](../scripts/remove-video-features.py) | strip raw video columns from a LeRobotDataset to save disk |

---

## 3. Side-by-side cheatsheet

| | RLDS path | LeRobot path |
|---|---|---|
| Dataset format | RLDS / TFDS (OXE) | `LeRobotDataset` v0.5 |
| Backbone | `Qwen2_5_VLForConditionalGeneration` | `AutoModelForImageTextToText` (Gemma-4) |
| Action FAST tokenizer | `physical-intelligence/fast` | `lerobot/fast-action-tokenizer` |
| Action token range | hard-coded `[151665, 153712]` | discovered from Gemma vocab |
| Action chunk | `future_action_window_size=4` | `action_chunk_size=50`, cubic-spline resampled |
| Action space | normalized `[-1, 1]` (q01/q99) | delta + quantile-normalized `[-1, 1]` |
| Cameras | primary only | head / hand_left / hand_right |
| Image history | 1 frame | `num_frames` frames (default 3) |
| Per-device batch | 16 | 2 |
| Grad accumulation | 2 | 128 |
| Loss mask | only action tokens | full assistant turn (subtask + actions) |
| Distributed config | YAML (`accelerator_config.yaml`) | inline `Accelerator(...)` kwargs |
| W&B project | `Nora VLA` | `Griffin Alpha` |

---

## 4. Dependencies

### RLDS — [training/requirements.txt](../training/requirements.txt)

```
tensorflow==2.15.0          tensorflow_datasets==4.9.3
tensorflow_graphics==2021.12.3
dlimp @ git+https://github.com/moojink/dlimp_openvla
torch==2.4.0                torchvision==0.19.0
transformers==4.50.0        qwen_vl_utils
accelerate==1.5.2           flash-attn==2.6.1
wandb  scipy  pillow  safetensors  numpy==1.26.4
```

### LeRobot — [lerobot_training/lerobot_requirements.txt](../lerobot_training/lerobot_requirements.txt)

```
lerobot==0.5.1
transformers>=5.5
torchvision  accelerate  wandb  wheel  scipy  pillow  safetensors  tqdm
```

The LeRobot stack pins newer torchvision (≥0.21) than the RLDS stack, which is why the README recommends a separate conda env per path.

---

## 5. Recent changes worth knowing

- **Lambda scope fix** (5c9e0e3) — per-subset instance transforms in [utils/data_loading.py:305](../utils/data_loading.py#L305) bind `inst_transform` as a default arg. Don't simplify back to a closure.
- **lerobot 0.5 API migration** (88b43c3) — `lerobot.datasets.io_utils.load_info()` (was `lerobot.datasets.utils.load_info`); also added the 30-minute `InitProcessGroupKwargs` timeout for NCCL.
- **Speed-up via `SkipEpisodesLeRobotDataset`** (c89d439) — wrap `LeRobotDataset` with index remapping instead of filtering inside `__getitem__`.
- **Accelerate tweaks** (6272205, d7c9d58) — `main_process_first()` now wraps **only** processor + model loading; dataset loading happens outside so worker ranks don't sit idle long enough to trip the timeout.

---

## 6. File map

| Concern | File |
|---|---|
| RLDS trainer | [training/train.py](../training/train.py) |
| RLDS dataset wrapper | [training/datasets/datasets.py](../training/datasets/datasets.py) |
| RLDS core (interleaving, transforms) | [training/datasets/rlds/dataset.py](../training/datasets/rlds/dataset.py) |
| Trajectory transforms | [training/datasets/rlds/utils/traj_transforms.py](../training/datasets/rlds/utils/traj_transforms.py) |
| Frame transforms | [training/datasets/rlds/utils/obs_transforms.py](../training/datasets/rlds/utils/obs_transforms.py) |
| Goal relabeling | [training/datasets/rlds/utils/goal_relabeling.py](../training/datasets/rlds/utils/goal_relabeling.py) |
| OXE mixtures | [training/datasets/rlds/oxe/mixtures.py](../training/datasets/rlds/oxe/mixtures.py) |
| OXE per-dataset config | [training/datasets/rlds/oxe/configs.py](../training/datasets/rlds/oxe/configs.py) |
| Accelerate config (RLDS) | [training/accelerator_config.yaml](../training/accelerator_config.yaml) |
| LeRobot trainer | [lerobot_training/lerobot_training.py](../lerobot_training/lerobot_training.py) |
| LeRobot dataset loaders | [lerobot_training/load_datasets.py](../lerobot_training/load_datasets.py) |
| Shared LeRobot processing | [utils/data_loading.py](../utils/data_loading.py) |
| Dataset download scripts | [scripts/](../scripts/) |
