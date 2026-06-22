import argparse
import csv
import importlib
import json
import math
import pathlib
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText as AutoModelClass
from transformers import AutoProcessor

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
for _path in (_ROOT, _THIS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import load_datasets
from lerobot_training import TrainingConfig, install_subtask_complementary_key, make_policy_processor
from utils.data_loading import collate_with_observation_image_lists


DATASET_LOADERS = {
    "agibot_world": ("AgiBotWorld-Beta", "agibot_world_root", load_datasets.load_agibot_world_dataset),
    "galaxea": ("Galaxea A1", "galaxea_open_world_ds_root", load_datasets.load_galaxea_dataset),
    "interndata_a1": ("InternVLA simulation", "interndata_a1_root", load_datasets.load_interndata_a1_dataset),
    "droid": ("DROID", "droid_root", load_datasets.load_droid_dataset),
}


@dataclass
class EvalConfig:
    backend: str = "vlm"
    policy_class: str | None = None
    policy_pretrained_path: str | None = None
    model_id: str = "google/gemma-4-E4B-it"
    load_model_weights: str | None = None
    output_dir: str = "./eval_results"
    run_name: str | None = None
    datasets: tuple[str, ...] = ("agibot_world", "galaxea", "interndata_a1", "droid")
    split: str = "val"
    val_fraction: float = 0.05
    per_device_batch_size: int = 4
    dataloader_num_workers: int = 0
    max_eval_samples: int | None = 2048
    max_samples_per_dataset: int | None = None
    seed: int = 42
    action_chunk_size: int = 50
    num_frames: int = 1
    action_vocab_size: int = 2048
    proprio_vocab_size: int = 256
    max_tokens_per_image: int = 70
    max_sequence_length: int = 500
    device: str | None = None
    dtype: str = "bfloat16"
    use_wandb: bool = False
    wandb_project: str = "Griffin Alpha Eval"
    wandb_entity: str | None = None
    agibot_world_root: str = "data/agibot-world/tasks"
    galaxea_open_world_ds_root: str = "data/galaxea-open-world-dataset"
    interndata_a1_root: str = "data/interndata-a1/"
    droid_root: str = "data/droid_1.0.1"


class NamedDataset(Dataset):
    def __init__(self, dataset: Dataset, name: str):
        self.dataset = dataset
        self.name = name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        item = dict(item)
        info = dict(item.get("info") or {})
        info["eval_dataset_name"] = self.name
        item["info"] = info
        return item


class MetricAccumulator:
    def __init__(self):
        self.values = defaultdict(float)

    def add(self, key: str, value: float, n: float = 1.0) -> None:
        self.values[f"{key}_sum"] += float(value)
        self.values[f"{key}_count"] += float(n)

    def add_raw(self, key: str, value: float) -> None:
        self.values[key] += float(value)

    def mean(self, key: str) -> float | None:
        count = self.values.get(f"{key}_count", 0.0)
        if count <= 0:
            return None
        return self.values[f"{key}_sum"] / count

    def to_dict(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key in sorted(self.values):
            if not key.endswith("_sum"):
                continue
            metric_name = key.removesuffix("_sum")
            mean_value = self.mean(metric_name)
            if mean_value is not None:
                metrics[metric_name] = mean_value
        for key in sorted(self.values):
            if not (key.endswith("_sum") or key.endswith("_count")):
                metrics[key] = self.values[key]
        return metrics


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate Nora/LeRobot policies on held-out LeRobot samples.")
    parser.add_argument("--backend", choices=("vlm", "policy"), default=EvalConfig.backend)
    parser.add_argument("--policy-class", default=None, help="Import path, e.g. package.module:PolicyClass.")
    parser.add_argument("--policy-pretrained-path", default=None)
    parser.add_argument("--model-id", default=EvalConfig.model_id)
    parser.add_argument("--load-model-weights", default=None, help="Optional single safetensors state dict.")
    parser.add_argument("--output-dir", default=EvalConfig.output_dir)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--datasets", default=",".join(EvalConfig.datasets), help="Comma-separated dataset keys.")
    parser.add_argument("--split", choices=("all", "train", "val"), default=EvalConfig.split)
    parser.add_argument("--val-fraction", type=float, default=EvalConfig.val_fraction)
    parser.add_argument("--per-device-batch-size", type=int, default=EvalConfig.per_device_batch_size)
    parser.add_argument("--dataloader-num-workers", type=int, default=EvalConfig.dataloader_num_workers)
    parser.add_argument("--max-eval-samples", type=int, default=EvalConfig.max_eval_samples)
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=EvalConfig.seed)
    parser.add_argument("--action-chunk-size", type=int, default=EvalConfig.action_chunk_size)
    parser.add_argument("--num-frames", type=int, default=EvalConfig.num_frames)
    parser.add_argument("--action-vocab-size", type=int, default=EvalConfig.action_vocab_size)
    parser.add_argument("--proprio-vocab-size", type=int, default=EvalConfig.proprio_vocab_size)
    parser.add_argument("--max-tokens-per-image", type=int, default=EvalConfig.max_tokens_per_image)
    parser.add_argument("--max-sequence-length", type=int, default=EvalConfig.max_sequence_length)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=EvalConfig.dtype)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=EvalConfig.wandb_project)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--agibot-world-root", default=EvalConfig.agibot_world_root)
    parser.add_argument("--galaxea-open-world-ds-root", default=EvalConfig.galaxea_open_world_ds_root)
    parser.add_argument("--interndata-a1-root", default=EvalConfig.interndata_a1_root)
    parser.add_argument("--droid-root", default=EvalConfig.droid_root)
    args = parser.parse_args()
    cfg = EvalConfig(**vars(args))
    cfg.datasets = tuple(dataset.strip() for dataset in args.datasets.split(",") if dataset.strip())
    if not 0 < cfg.val_fraction < 1:
        raise ValueError(f"--val-fraction must be in (0, 1), got {cfg.val_fraction}")
    if cfg.max_eval_samples is not None and cfg.max_eval_samples <= 0:
        cfg.max_eval_samples = None
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: EvalConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype(cfg: EvalConfig) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[cfg.dtype]


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    return value


def get_dataset_names(raw_batch: Mapping[str, Any]) -> list[str]:
    info = raw_batch.get("info") or {}
    names = info.get("eval_dataset_name")
    if names is None:
        batch_size = raw_batch["action"].shape[0]
        return ["unknown"] * batch_size
    if isinstance(names, str):
        return [names]
    return list(names)


def get_action_dim_mask(raw_batch: Mapping[str, Any], target: torch.Tensor) -> torch.Tensor:
    batch_size = target.shape[0]
    action_dim = target.shape[-1]
    info = raw_batch.get("info") or {}
    n_action_dims = info.get("n_action_dims")
    if n_action_dims is None:
        return torch.ones(batch_size, action_dim, dtype=torch.bool, device=target.device)
    if torch.is_tensor(n_action_dims):
        dims = n_action_dims.to(target.device).long().view(-1)
    else:
        dims = torch.as_tensor(n_action_dims, device=target.device).long().view(-1)
    return torch.arange(action_dim, device=target.device).unsqueeze(0) < dims.unsqueeze(1)


def sample_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def split_dataset(dataset: Dataset, split: str, val_fraction: float, seed: int) -> Dataset:
    if split == "all":
        return dataset

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_count = max(1, int(round(len(indices) * val_fraction)))
    if split == "val":
        selected = indices[:val_count]
    elif split == "train":
        selected = indices[val_count:]
    else:
        raise ValueError(f"Unknown split: {split}")
    if not selected:
        raise ValueError(f"Split {split!r} is empty for dataset of length {len(dataset)}.")
    return Subset(dataset, selected)


def build_eval_dataset(cfg: EvalConfig) -> Dataset:
    unknown = sorted(set(cfg.datasets) - set(DATASET_LOADERS))
    if unknown:
        raise ValueError(f"Unknown datasets {unknown}. Available: {sorted(DATASET_LOADERS)}")

    named_datasets: list[Dataset] = []
    for dataset_idx, dataset_key in enumerate(cfg.datasets):
        display_name, root_attr, loader = DATASET_LOADERS[dataset_key]
        root = getattr(cfg, root_attr)
        dataset = loader(
            root=root,
            canonical_action_chunk_size=cfg.action_chunk_size,
            num_frames=cfg.num_frames,
        )
        dataset = NamedDataset(dataset, display_name)
        dataset = split_dataset(dataset, cfg.split, cfg.val_fraction, cfg.seed + dataset_idx)
        dataset = sample_dataset(dataset, cfg.max_samples_per_dataset, cfg.seed + dataset_idx)
        named_datasets.append(dataset)

    dataset: Dataset = ConcatDataset(named_datasets) if len(named_datasets) > 1 else named_datasets[0]
    return sample_dataset(dataset, cfg.max_eval_samples, cfg.seed)


def make_training_config(cfg: EvalConfig) -> TrainingConfig:
    training_cfg = TrainingConfig(
        per_device_batch_size=cfg.per_device_batch_size,
        dataloader_num_workers=cfg.dataloader_num_workers,
        action_chunk_size=cfg.action_chunk_size,
        model_id=cfg.model_id,
        action_vocab_size=cfg.action_vocab_size,
        proprio_vocab_size=cfg.proprio_vocab_size,
        max_tokens_per_image=cfg.max_tokens_per_image,
        max_sequence_length=cfg.max_sequence_length,
        num_frames=cfg.num_frames,
    )
    training_cfg.load_model_weights = cfg.load_model_weights
    training_cfg.enable_image_augmentation = False
    return training_cfg


def load_vlm_backend(cfg: EvalConfig, device: torch.device):
    training_cfg = make_training_config(cfg)
    install_subtask_complementary_key()
    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        trust_remote_code=True,
        max_soft_tokens=cfg.max_tokens_per_image,
        image_seq_length=cfg.max_tokens_per_image,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    action_tokens = [f"<robot_action_{i}>" for i in range(cfg.action_vocab_size)]
    proprio_tokens = [f"<proprio_state_{i}>" for i in range(cfg.proprio_vocab_size)]
    processor.tokenizer.add_tokens(action_tokens + proprio_tokens, special_tokens=True)

    model = AutoModelClass.from_pretrained(
        cfg.model_id,
        torch_dtype=get_dtype(cfg),
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    if hasattr(model, "config"):
        model.config.vocab_size = len(processor.tokenizer)

    if cfg.load_model_weights:
        tensors = {}
        with safe_open(cfg.load_model_weights, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        missing, unexpected = model.load_state_dict(tensors, strict=False)
        print(f"Loaded weights: {len(missing)} missing keys, {len(unexpected)} unexpected keys.")

    model.to(device)
    model.eval()
    policy_preprocessor = make_policy_processor(training_cfg, processor)
    action_token_ids = processor.tokenizer.convert_tokens_to_ids(action_tokens)
    action_token_id_to_index = {
        token_id: token_idx
        for token_idx, token_id in enumerate(action_token_ids)
        if token_id is not None and token_id != processor.tokenizer.unk_token_id
    }
    fast_tokenizer = AutoProcessor.from_pretrained(
        "lerobot/fast-action-tokenizer",
        trust_remote_code=True,
    )
    return model, policy_preprocessor, frozenset(action_token_id_to_index), action_token_id_to_index, fast_tokenizer


def import_object(import_path: str):
    module_name, sep, object_name = import_path.replace(":", ".").rpartition(".")
    if not sep:
        raise ValueError(f"Expected import path like package.module:Class, got {import_path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def load_policy_backend(cfg: EvalConfig, device: torch.device):
    if not cfg.policy_class:
        raise ValueError("--policy-class is required when --backend policy is used.")
    if not cfg.policy_pretrained_path:
        raise ValueError("--policy-pretrained-path is required when --backend policy is used.")
    policy_cls = import_object(cfg.policy_class)
    if not hasattr(policy_cls, "from_pretrained"):
        raise TypeError(f"{cfg.policy_class} does not expose from_pretrained().")
    policy = policy_cls.from_pretrained(cfg.policy_pretrained_path)
    policy.to(device)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()
    return policy


def make_vlm_collate_fn(policy_preprocessor):
    def collate_fn(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        raw_batch = collate_with_observation_image_lists(examples)
        return {
            "raw_batch": raw_batch,
            "model_inputs": policy_preprocessor(raw_batch),
        }

    return collate_fn


def make_policy_collate_fn():
    def collate_fn(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return collate_with_observation_image_lists(examples)

    return collate_fn


def update_vlm_metrics(
    accumulators: dict[str, MetricAccumulator],
    raw_batch: Mapping[str, Any],
    labels: torch.Tensor,
    logits: torch.Tensor,
    action_token_ids: set[int],
    action_token_id_to_index: Mapping[int, int],
    fast_tokenizer: Any,
) -> None:
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100
    if valid_mask.sum().item() == 0:
        return

    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    predictions = shift_logits.argmax(dim=-1)
    correct = (predictions == shift_labels) & valid_mask
    action_mask = valid_mask & torch.isin(
        shift_labels,
        torch.as_tensor(sorted(action_token_ids), device=shift_labels.device),
    )
    action_correct = correct & action_mask

    dataset_names = get_dataset_names(raw_batch)
    for sample_idx, dataset_name in enumerate(dataset_names):
        acc = accumulators[dataset_name]
        acc.add_raw("examples", 1)
        token_count = int(valid_mask[sample_idx].sum().item())
        if token_count:
            acc.add("nll", float(token_losses[sample_idx][valid_mask[sample_idx]].sum().item()), token_count)
            acc.add("token_accuracy", float(correct[sample_idx].sum().item()), token_count)
        action_count = int(action_mask[sample_idx].sum().item())
        if action_count:
            acc.add("action_token_accuracy", float(action_correct[sample_idx].sum().item()), action_count)
            predicted_action_token_ids = predictions[sample_idx][action_mask[sample_idx]].tolist()
            target_action_token_ids = shift_labels[sample_idx][action_mask[sample_idx]].tolist()
            valid_predicted_tokens = sum(
                int(token_id) in action_token_id_to_index
                for token_id in predicted_action_token_ids
            )
            acc.add("action_token_valid_rate", valid_predicted_tokens, action_count)
            decoded_predicted = decode_fast_action_tokens(
                predicted_action_token_ids,
                action_token_id_to_index,
                fast_tokenizer,
            )
            decoded_target = decode_fast_action_tokens(
                target_action_token_ids,
                action_token_id_to_index,
                fast_tokenizer,
            )
            if decoded_predicted is not None and decoded_target is not None:
                update_decoded_action_metrics(
                    acc,
                    raw_batch,
                    sample_idx,
                    decoded_predicted,
                    decoded_target,
                    metric_prefix="decoded_action",
                )
        acc.add_raw("tokens", token_count)
        acc.add_raw("action_tokens", action_count)


def decode_fast_action_tokens(
    token_ids: Sequence[int],
    action_token_id_to_index: Mapping[int, int],
    fast_tokenizer: Any,
) -> torch.Tensor | None:
    try:
        fast_tokens = [action_token_id_to_index[int(token_id)] for token_id in token_ids]
    except KeyError:
        return None
    try:
        decoded = fast_tokenizer.decode([fast_tokens])
    except Exception:
        return None
    decoded_tensor = torch.as_tensor(decoded, dtype=torch.float32)
    if decoded_tensor.ndim >= 3 and decoded_tensor.shape[0] == 1:
        decoded_tensor = decoded_tensor[0]
    return decoded_tensor


def update_decoded_action_metrics(
    accumulator: MetricAccumulator,
    raw_batch: Mapping[str, Any],
    sample_idx: int,
    predicted_action: torch.Tensor,
    target_action: torch.Tensor,
    metric_prefix: str,
) -> None:
    raw_target = raw_batch["action"][sample_idx].float()
    n_action_dims = raw_batch.get("info", {}).get("n_action_dims")
    if n_action_dims is not None:
        if torch.is_tensor(n_action_dims):
            action_dim = int(n_action_dims[sample_idx].item())
        else:
            action_dim = int(n_action_dims[sample_idx])
    else:
        action_dim = raw_target.shape[-1]

    horizon = min(predicted_action.shape[-2], target_action.shape[-2], raw_target.shape[-2])
    action_dim = min(action_dim, predicted_action.shape[-1], target_action.shape[-1], raw_target.shape[-1])
    if horizon <= 0 or action_dim <= 0:
        return

    predicted_action = predicted_action[:horizon, :action_dim]
    target_action = target_action[:horizon, :action_dim]
    raw_target = raw_target[:horizon, :action_dim]

    pred_vs_raw = predicted_action - raw_target
    pred_vs_decoded_target = predicted_action - target_action
    n_values = horizon * action_dim
    accumulator.add(f"{metric_prefix}_mae", float(pred_vs_raw.abs().sum().item()), n_values)
    accumulator.add(f"{metric_prefix}_mse", float(pred_vs_raw.square().sum().item()), n_values)
    accumulator.add(
        f"{metric_prefix}_quantized_target_mae",
        float(pred_vs_decoded_target.abs().sum().item()),
        n_values,
    )
    accumulator.add(
        f"{metric_prefix}_first_step_mae",
        float(pred_vs_raw[0].abs().sum().item()),
        action_dim,
    )


def update_policy_action_metrics(
    accumulators: dict[str, MetricAccumulator],
    raw_batch: Mapping[str, Any],
    predicted_action: torch.Tensor,
) -> None:
    target_action = raw_batch["action"].to(predicted_action.device)
    if predicted_action.ndim == target_action.ndim - 1:
        target_action = target_action[:, 0]
    elif predicted_action.ndim == target_action.ndim:
        horizon = min(predicted_action.shape[-2], target_action.shape[-2])
        predicted_action = predicted_action[:, :horizon]
        target_action = target_action[:, :horizon]
    else:
        raise ValueError(
            f"Cannot compare predicted action shape {tuple(predicted_action.shape)} "
            f"with target shape {tuple(target_action.shape)}."
        )

    action_dim = min(predicted_action.shape[-1], target_action.shape[-1])
    predicted_action = predicted_action[..., :action_dim].float()
    target_action = target_action[..., :action_dim].float()
    dim_mask = get_action_dim_mask(raw_batch, target_action[..., :action_dim])
    if predicted_action.ndim == 3:
        mask = dim_mask.unsqueeze(1).expand_as(predicted_action)
    else:
        mask = dim_mask.expand_as(predicted_action)

    diff = predicted_action - target_action
    abs_error = diff.abs()
    sq_error = diff.square()
    dataset_names = get_dataset_names(raw_batch)
    for sample_idx, dataset_name in enumerate(dataset_names):
        sample_mask = mask[sample_idx]
        count = int(sample_mask.sum().item())
        acc = accumulators[dataset_name]
        acc.add_raw("examples", 1)
        if count:
            acc.add("action_mae", float(abs_error[sample_idx][sample_mask].sum().item()), count)
            acc.add("action_mse", float(sq_error[sample_idx][sample_mask].sum().item()), count)
            if predicted_action.ndim == 3:
                first_mask = dim_mask[sample_idx]
                first_count = int(first_mask.sum().item())
                acc.add(
                    "first_step_action_mae",
                    float(abs_error[sample_idx, 0][first_mask].sum().item()),
                    first_count,
                )


def extract_action_tensor(output: Any) -> torch.Tensor | None:
    if output is None:
        return None
    if torch.is_tensor(output):
        return output
    if isinstance(output, np.ndarray):
        return torch.from_numpy(output)
    if isinstance(output, Mapping):
        for key in ("action", "actions", "pred_action", "predicted_action"):
            if key in output:
                return extract_action_tensor(output[key])
    for key in ("action", "actions", "pred_action", "predicted_action"):
        if hasattr(output, key):
            return extract_action_tensor(getattr(output, key))
    return None


def extract_loss_and_extra(output: Any) -> tuple[Any, Mapping[str, Any] | None]:
    if isinstance(output, Mapping):
        return output.get("loss"), {k: v for k, v in output.items() if k != "loss"}
    if isinstance(output, tuple):
        if len(output) == 2:
            return output
        if len(output) > 0:
            return output[0], None
    if hasattr(output, "loss"):
        return getattr(output, "loss"), None
    return output, None


def merge_metrics(accumulators: dict[str, MetricAccumulator]) -> dict[str, Any]:
    overall = MetricAccumulator()
    for dataset_acc in accumulators.values():
        for key, value in dataset_acc.values.items():
            overall.values[key] += value

    metrics = {
        "overall": overall.to_dict(),
        "by_dataset": {name: acc.to_dict() for name, acc in sorted(accumulators.items())},
    }
    nll = metrics["overall"].get("nll")
    if nll is not None:
        metrics["overall"]["perplexity"] = math.exp(min(nll, 50.0))
    for dataset_metrics in metrics["by_dataset"].values():
        nll = dataset_metrics.get("nll")
        if nll is not None:
            dataset_metrics["perplexity"] = math.exp(min(nll, 50.0))
    return metrics


def add_batch_scalar_metric(
    accumulators: dict[str, MetricAccumulator],
    raw_batch: Mapping[str, Any],
    metric_name: str,
    value: float,
) -> None:
    dataset_names = get_dataset_names(raw_batch)
    unique_names = set(dataset_names)
    if len(unique_names) == 1:
        accumulators[dataset_names[0]].add(metric_name, value, len(dataset_names))
    else:
        accumulators["mixed_batch"].add(metric_name, value, len(dataset_names))
        accumulators["mixed_batch"].add_raw(f"{metric_name}_mixed_batches", 1)


@torch.inference_mode()
def evaluate_vlm(cfg: EvalConfig, dataset: Dataset, device: torch.device) -> dict[str, Any]:
    (
        model,
        policy_preprocessor,
        action_token_ids,
        action_token_id_to_index,
        fast_tokenizer,
    ) = load_vlm_backend(cfg, device)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=make_vlm_collate_fn(policy_preprocessor),
        pin_memory=device.type == "cuda",
    )
    accumulators: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    autocast_enabled = device.type == "cuda" and get_dtype(cfg) != torch.float32
    for batch in tqdm(dataloader, desc="Evaluating VLM"):
        model_inputs = move_to_device(batch["model_inputs"], device)
        with torch.autocast(device_type=device.type, dtype=get_dtype(cfg), enabled=autocast_enabled):
            outputs = model(**model_inputs)
        update_vlm_metrics(
            accumulators,
            batch["raw_batch"],
            model_inputs["labels"],
            outputs.logits.detach(),
            action_token_ids,
            action_token_id_to_index,
            fast_tokenizer,
        )
    return merge_metrics(accumulators)


@torch.inference_mode()
def evaluate_policy(cfg: EvalConfig, dataset: Dataset, device: torch.device) -> dict[str, Any]:
    policy = load_policy_backend(cfg, device)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=make_policy_collate_fn(),
        pin_memory=device.type == "cuda",
    )
    accumulators: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    for raw_batch in tqdm(dataloader, desc="Evaluating policy"):
        device_batch = move_to_device(raw_batch, device)
        output = None
        if hasattr(policy, "forward"):
            output = policy(device_batch)
            loss, extra = extract_loss_and_extra(output)
            if torch.is_tensor(loss):
                add_batch_scalar_metric(
                    accumulators,
                    raw_batch,
                    "policy_loss",
                    float(loss.detach().float().item()),
                )
            if isinstance(extra, Mapping):
                for key, value in extra.items():
                    if isinstance(value, (int, float)):
                        add_batch_scalar_metric(accumulators, raw_batch, str(key), float(value))

        predicted_action = extract_action_tensor(output)
        if hasattr(policy, "predict_action_chunk"):
            predicted_action = policy.predict_action_chunk(device_batch)
        elif predicted_action is None and hasattr(policy, "select_action"):
            predicted_action = policy.select_action(device_batch)
        predicted_action = extract_action_tensor(predicted_action)
        if predicted_action is not None:
            predicted_action = predicted_action.to(device) if torch.is_tensor(predicted_action) else predicted_action
            update_policy_action_metrics(accumulators, raw_batch, predicted_action.detach())
    return merge_metrics(accumulators)


def write_results(cfg: EvalConfig, metrics: dict[str, Any], elapsed_s: float) -> pathlib.Path:
    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.run_name or time.strftime("%Y%m%d-%H%M%S")
    result = {
        "config": asdict(cfg),
        "elapsed_s": elapsed_s,
        "metrics": metrics,
    }
    json_path = output_dir / f"{run_name}.json"
    csv_path = output_dir / f"{run_name}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "metric", "value"])
        for metric, value in metrics["overall"].items():
            writer.writerow(["overall", metric, value])
        for dataset_name, dataset_metrics in metrics["by_dataset"].items():
            for metric, value in dataset_metrics.items():
                writer.writerow([dataset_name, metric, value])
    return json_path


def maybe_log_wandb(cfg: EvalConfig, metrics: dict[str, Any]) -> None:
    if not cfg.use_wandb:
        return
    import wandb

    wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.run_name, config=asdict(cfg))
    wandb.log({f"eval/{key}": value for key, value in metrics["overall"].items()})
    for dataset_name, dataset_metrics in metrics["by_dataset"].items():
        safe_name = dataset_name.replace("/", "_").replace(" ", "_")
        wandb.log({f"eval_by_dataset/{safe_name}/{key}": value for key, value in dataset_metrics.items()})
    wandb.finish()


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device(cfg)
    start = time.time()
    dataset = build_eval_dataset(cfg)
    print(f"Evaluation dataset size: {len(dataset)} samples")
    print(f"Backend: {cfg.backend}; device: {device}")
    if cfg.backend == "vlm":
        metrics = evaluate_vlm(cfg, dataset, device)
    elif cfg.backend == "policy":
        metrics = evaluate_policy(cfg, dataset, device)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")
    elapsed_s = time.time() - start
    metrics["overall"]["elapsed_s"] = elapsed_s
    result_path = write_results(cfg, metrics, elapsed_s)
    maybe_log_wandb(cfg, metrics)
    print(json.dumps(metrics["overall"], indent=2, sort_keys=True))
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
