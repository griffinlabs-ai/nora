# NORA Model Architecture

NORA (Neural Orchestrator for Robotics Autonomy) is a Vision-Language-Action (VLA) model that maps an RGB image and a natural-language instruction to a sequence of robot actions. Architecturally, NORA is a thin wrapper around a Qwen VL conditional-generation model: there is **no separate action head** — actions are emitted as special tokens through the LLM's standard language-modeling head and decoded back into continuous values by the [`fast`](https://huggingface.co/physical-intelligence/fast) tokenizer.

This document describes the model itself. For training/finetuning instructions see the project [README](../README.md).

## High-level flow

```
RGB image (224×224)  +  Instruction text
            │
            ▼
   Qwen VL processor (chat template + vision processing)
            │
            ▼
   Qwen{2.5,3} VL ForConditionalGeneration
   (ViT vision encoder → multimodal transformer → LM head)
            │
            ▼
   Token stream containing <robot_action_i> tokens
   (vocabulary range [151665, 153712], 2048 tokens)
            │
            ▼
   FAST tokenizer decode  →  normalized actions in [-1, 1]
            │
            ▼
   Per-dataset min/max denormalization (q01, q99 from norm_stats.json)
            │
            ▼
   7-DoF robot action  (6-DoF arm + 1-DoF gripper)
```

## Backbone

| Variant | Backbone class | Used in |
|---|---|---|
| Pretraining / Bridge | `Qwen2_5_VLForConditionalGeneration` | [training/train.py:147](../training/train.py#L147), [experiments/bridge/nora_utils.py](../experiments/bridge/nora_utils.py) |
| LIBERO / current inference | `Qwen3VLForConditionalGeneration` | [inference/nora.py:4](../inference/nora.py#L4), [experiments/libero/nora_utils.py](../experiments/libero/nora_utils.py) |

The backbone is loaded straight from HuggingFace (`declare-lab/nora`) with `torch_dtype=bfloat16` and (optionally) flash-attention 2. The associated `AutoProcessor` handles chat templating, image preprocessing, and tokenization.

Key class: [`Nora`](../inference/nora.py#L38) in [inference/nora.py:38-270](../inference/nora.py#L38-L270).

## Action representation

NORA does not regress continuous actions. Instead it treats actions as a discrete token language sharing the LLM vocabulary.

### Token vocabulary

- **Action token range:** `151665` … `153712` (2048 tokens) — see [inference/nora.py:42-43](../inference/nora.py#L42-L43).
- Each action token is rendered as `<robot_action_i>` (i ∈ [0, 2047]) when written into the chat-template assistant turn — see [training/train.py:74-78](../training/train.py#L74-L78).
- Continuous ↔ discrete conversion is done by the [`physical-intelligence/fast`](https://huggingface.co/physical-intelligence/fast) AutoProcessor, loaded alongside the main processor.

### Action shape

| Attribute | Default | Where set |
|---|---|---|
| `action_dim` | 16 internally; the runtime returns the leading 7 dims for Bridge / WidowX | [inference/nora.py:88](../inference/nora.py#L88) |
| `time_horizon` | 1 (single-step). NORA-LONG variant uses 5 (action chunks). | [inference/nora.py:91](../inference/nora.py#L91) |
| Output range (normalized) | [-1, 1] | FAST decoder |
| Output range (final) | dataset-specific via `q01` / `q99` | [inference/nora.py:230-248](../inference/nora.py#L230-L248) |

### Denormalization

Per-dataset action statistics live in `norm_stats.json`, fetched from the model repo on Hub. The unnormalization formula is:

```
action_unnorm = 0.5 * (action_norm + 1) * (q99 - q01) + q01
```

Implemented at [inference/nora.py:236-248](../inference/nora.py#L236-L248).

### Gripper convention

NORA pretraining standardizes the gripper to `[0, 1]` (0 = closed, 1 = open). Some downstream environments (e.g., LIBERO) use `[-1, +1]` with the opposite sign. The helpers [`normalize_gripper_action`](../inference/nora.py#L10) and [`invert_gripper_action`](../inference/nora.py#L29) bridge the two conventions.

## Inputs

**Vision.** A single RGB image, resized to `224×224`, processed by Qwen VL's ViT vision encoder. Multi-camera setups (e.g., LeRobot `head` / `hand_left` / `hand_right`) are supported in the LeRobot training path.

**Language.** A natural-language instruction string. Image and text are combined into a Qwen chat-template message:

```python
[
  {"role": "user", "content": [
      {"type": "image", "image": pil_image},
      {"type": "text",  "text":  instruction}]},
]
```

See [inference/nora.py:159-173](../inference/nora.py#L159-L173).

**Optional state.** Proprioceptive state (joint positions, etc.) is available in the RLDS pipeline but is not consumed by the base Qwen-VL pretraining recipe; it is used only for normalization references.

## Training

The training script is [training/train.py](../training/train.py). Hyperparameters live in the `TrainingConfig` dataclass at [training/train.py:25-60](../training/train.py#L25-L60).

### Loss

Standard causal-LM cross-entropy on the assistant turn, but with a custom label mask so that loss is computed **only over action tokens**:

1. Clone `input_ids` into `labels`.
2. For each sequence, find the first token whose ID is in `[151665, 153712]`. Set every label before that index to `-100`.
3. Set every PAD token's label to `-100`.

Implementation: [training/train.py:121-140](../training/train.py#L121-L140).

This means image patches, system / user prompt tokens, and chat-template scaffolding never contribute to the loss — only the predicted action tokens do.

### Optimizer & schedule

| Component | Value | Source |
|---|---|---|
| Optimizer | `AdamW(betas=(0.9, 0.95), weight_decay=1e-8, eps=1e-8)` | [train.py:193-199](../training/train.py#L193-L199) |
| LR | `5e-5` | `TrainingConfig` |
| Schedule | Cosine with warmup | [train.py:201-208](../training/train.py#L201-L208) |
| Warmup steps | 1000 | `TrainingConfig` |
| Max steps | 100000 | `TrainingConfig` |
| Per-device batch | 16 | `TrainingConfig` |
| Grad accumulation | 2 | `TrainingConfig` |
| Precision | bfloat16 | accelerator config |
| Distributed | 8-process MULTI_GPU (Accelerate) | [training/accelerator_config.yaml](../training/accelerator_config.yaml) |

### Data path (RLDS)

NORA reuses OpenVLA's RLDS pipeline:

- [`RLDSDataset`](../training/datasets/datasets.py#L98) wraps an OXE-mixture dataset with the following per-frame transforms (see [datasets.py:127-139](../training/datasets/datasets.py#L127-L139)):
  - `window_size=1`, `future_action_window_size=4` → predict 4-step action chunks.
  - `resize_size=(224, 224)`.
  - `goal_relabeling_strategy="uniform"`, `skip_unlabeled=True`.
  - q99 action / proprio normalization.
- [`RLDSBatchTransform`](../training/datasets/datasets.py#L57-L95) extracts the primary image, decodes the language instruction, and packages an example dict.
- [`collate_fn`](../training/train.py#L105) builds chat messages, runs them through the Qwen processor, and applies the action-token loss mask described above.

OXE mixtures (`bridge`, `rtx`, `oxe_magic_soup_plus`, `libero_10_no_noops`, …) are defined in [training/datasets/rlds/oxe/mixtures.py](../training/datasets/rlds/oxe/mixtures.py).

### LeRobot data path

An alternative trainer at [lerobot_training/lerobot_training.py](../lerobot_training/lerobot_training.py) consumes `LeRobotDataset` (lerobot 0.5.x). It supports `delta_timestamps` for short image histories (`num_frames=3` by default), action chunks (`action_chunk_size=50`), and dual-arm action vectors padded to 16 dims. The action conversion / normalization steps are in [utils/data_loading.py](../utils/data_loading.py).

## Inference

Entry point: [`Nora.inference`](../inference/nora.py#L140) at [inference/nora.py:140-248](../inference/nora.py#L140-L248).

```python
from inference.nora import Nora

nora = Nora(device="cuda")
action = nora.inference(
    image=pil_or_np_image,
    instruction="pick up the red cube",
    unnorm_key="bridge_orig",
)
# action: np.ndarray, shape (7,)
```

Steps:

1. Coerce image to PIL.
2. Build the chat-template prompt with `add_generation_prompt=True`.
3. Run `process_vision_info` then the main processor → tensorized inputs.
4. `model.generate(..., do_sample=False)` (deterministic).
5. Slice the generated IDs to the suffix after the prompt and keep tokens within `[151665, 153712]`.
6. `fast_tokenizer.decode(...)` → action vector in `[-1, 1]`.
7. Min-max denormalize using the `unnorm_key` slice of `norm_stats.json`.
8. Return the leading 7 dimensions as a NumPy array.

## Variants

| Variant | Backbone | Action dim | Time horizon | Notes |
|---|---|---|---|---|
| `declare-lab/nora` | Qwen2.5 VL | 7 | 1 | Base model, OXE pretraining |
| `declare-lab/nora-long` | Qwen2.5 VL | 7 | 5 | Action-chunk variant |
| LIBERO finetune | Qwen3 VL | 16 (dual-arm) | 1 | See [experiments/libero/](../experiments/libero/) |
| Bridge / WidowX eval | Qwen2.5 VL | 7 | 1 | See [experiments/bridge/run_widowx.py](../experiments/bridge/run_widowx.py) |

## File map

| Concern | File |
|---|---|
| Inference class | [inference/nora.py](../inference/nora.py) |
| Training loop | [training/train.py](../training/train.py) |
| RLDS dataset wrapper | [training/datasets/datasets.py](../training/datasets/datasets.py) |
| OXE mixtures | [training/datasets/rlds/oxe/mixtures.py](../training/datasets/rlds/oxe/mixtures.py) |
| LeRobot trainer | [lerobot_training/lerobot_training.py](../lerobot_training/lerobot_training.py) |
| LeRobot data loading | [utils/data_loading.py](../utils/data_loading.py) |
| Bridge / WidowX eval | [experiments/bridge/run_widowx.py](../experiments/bridge/run_widowx.py) |
| LIBERO eval | [experiments/libero/](../experiments/libero/) |
| Accelerate config | [training/accelerator_config.yaml](../training/accelerator_config.yaml) |

## References

- Paper: [NORA: A Small Open-Sourced Generalist VLA Model](https://www.arxiv.org/abs/2504.19854)
- Project page: <https://declare-lab.github.io/nora>
- Built on: [OpenVLA](https://github.com/openvla/openvla), [Open X-Embodiment](https://robotics-transformer-x.github.io/), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), HuggingFace Transformers + Accelerate.
