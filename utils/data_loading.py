from dataclasses import dataclass
import json
import pathlib
import numpy as np
from typing import Callable, Generic, Iterable, Mapping, Any

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, default_collate
from scipy.interpolate import CubicSpline
from tqdm import tqdm

import lerobot.processor
from lerobot.processor.pipeline import PolicyProcessorPipeline, TOutput
from lerobot.configs.types import NormalizationMode, PolicyFeature, FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

class PreprocessedDataset(Dataset, Generic[TOutput]):

    def __init__(self, dataset: Dataset, preprocessor: PolicyProcessorPipeline[dict[str, Any], TOutput]):
        self.dataset = dataset
        self.preprocessor = preprocessor

    def __getitem__(self, index):
        return self.preprocessor(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

@dataclass
@lerobot.processor.ProcessorStepRegistry.register("resample_action_processor")
class ResampleActionProcessorStep(lerobot.processor.ProcessorStep):
    """
    Resample the action tensor from one chunk size to another, by cubic spline interpolation.
    """
    target_chunk_size: int
    state_key: str | None = None

    def __call__(self, transition):
        action = transition['action']
        if self.state_key is not None:
            initial_state = transition['observation'][self.state_key]
        else:
            initial_state = torch.zeros(action.shape[-1])
        orig_chunk_size = action.shape[0]

        if orig_chunk_size == self.target_chunk_size:
            return transition

        new_transition = transition.copy()

        if orig_chunk_size % self.target_chunk_size == 0:
            step_size = orig_chunk_size // self.target_chunk_size
            new_transition['action'] = action[step_size-1::step_size]
            return new_transition
        else:
            new_transition['action'] = self._interpolate(action, initial_state, orig_chunk_size)
            return new_transition

    def _interpolate(self, action, initial_state, orig_chunk_size):
        trajectory = torch.cat([initial_state.unsqueeze(0), action], dim=0)
        old_times = np.linspace(0, 1, orig_chunk_size + 1)
        new_times = np.linspace(1 / self.target_chunk_size, 1, self.target_chunk_size)

        traj_np = trajectory.cpu().numpy()
        cs = CubicSpline(old_times, traj_np)
        resampled = cs(new_times)

        return torch.from_numpy(resampled).to(
            dtype=action.dtype, device=action.device
        )

    def transform_features(self, features):
        original_shape = features['action']['action'].shape
        features['action']['action'] = PolicyFeature(
            FeatureType.ACTION,
            (self.target_chunk_size, original_shape[-1]),
        )
        return features

@dataclass
@lerobot.processor.ProcessorStepRegistry.register("abs2delta_action_processor")
class Abs2DeltaActionProcessorStep(lerobot.processor.ProcessorStep):
    """
    Convert action tensor from absolute space to delta space.
    """
    mask: torch.Tensor
    state_key: str = 'observation.state'

    def __call__(self, transition):
        new_transition = transition.copy()
        action = transition['action']

        assert self.mask.shape[-1] == action.shape[-1]
        deltas = action - transition['observation'][self.state_key].unsqueeze(-2)
        new_transition['action'] = torch.where(self.mask.expand(action.shape), deltas, action)
        return new_transition

    def transform_features(self, features):
        return features


def load_lerobot_dataset_skip_dirty_episodes(
    repo_id: str,
    root: str | pathlib.Path | None = None,
    episodes: list[int] | None = None,
    *args,
    **kwargs,
) -> PreprocessedDataset:
    if episodes is None and root is not None:
        removed_episodes_path = pathlib.Path(root) / 'meta/removed_episodes.json'
        if removed_episodes_path.exists():
            total_episodes = lerobot.datasets.utils.load_info(root)['total_episodes']
            with open(removed_episodes_path, 'r') as f:
                removed_episodes = json.load(f)['dirty_episodes']
            episodes = [i for i in range(total_episodes) if i not in removed_episodes]
    return LeRobotDataset(repo_id, root, episodes, *args, **kwargs)

def load_dataset(
    root: str | pathlib.Path,
    action_keys: Iterable[str],
    load_action_chunk_size: int,
    canonical_action_chunk_size: int,
    raw_fps: int,
    instance_transform: Callable[[dict[str, Any]], dict[str, Any]],
    norm_stats_transform: Callable[[dict[str, dict[str, np.ndarray]]], dict[str, dict[str, np.ndarray]]],
    num_frames: int = 1, 
) -> PreprocessedDataset:
    root = pathlib.Path(root)

    action_delta_timestamps = [i / raw_fps for i in range(load_action_chunk_size)]
    delta_timestamps = {
        action_key: action_delta_timestamps
        for action_key in action_keys
    }
    
    task_roots = [p.parent.parent for p in root.rglob('info.json')]

    if task_roots and num_frames > 1:
        repo_id = str(task_roots[0].relative_to(root))
        meta = LeRobotDatasetMetadata(repo_id, root=task_roots[0])
        
        image_keys = [k for k in meta.features.keys() if 'image' in k.lower()]
        img_timestamps = [float(i - num_frames + 1) for i in range(num_frames)]
        
        for img_k in image_keys:
            delta_timestamps[img_k] = img_timestamps

    dataset = ConcatDataset([
        load_lerobot_dataset_skip_dirty_episodes(
            task_root.relative_to(root),
            root=task_root,
            delta_timestamps=delta_timestamps, 
        )
        for task_root in tqdm(task_roots, desc="Loading datasets")
    ])
    
    raw_norm_stats = lerobot.datasets.io_utils.cast_stats_to_numpy(
        lerobot.datasets.io_utils.load_json(root / 'delta_norm_stats.json')
    )['norm_stats']
    
    norm_stats = norm_stats_transform(raw_norm_stats)

    norm_map = {
        'ACTION': NormalizationMode.QUANTILES,
    }
    
    resample_step_if_necessary = [ResampleActionProcessorStep(
        target_chunk_size = canonical_action_chunk_size,
    )] if load_action_chunk_size != canonical_action_chunk_size else []
    
    preprocessor = PolicyProcessorPipeline(
        steps = [
            Abs2DeltaActionProcessorStep(
                mask = torch.tensor(
                    [
                        True, True, True, True, True, True, True, False,
                        True, True, True, True, True, True, True, False,
                    ],
                    dtype=torch.bool,
                )
            ),
            *resample_step_if_necessary,
            lerobot.processor.NormalizerProcessorStep({}, norm_map , norm_stats),
        ],
        to_transition=lambda batch:
            lerobot.processor.converters.batch_to_transition(instance_transform(batch)),
    )

    dataset = PreprocessedDataset(dataset, preprocessor)
    return dataset


# =================================================================================
# [MODIFIED] Data Loading & Collation architecture based on "in one go" feedback
# =================================================================================

def build_co_training_dataloader(
    robot_dataset: Dataset, 
    math_hf_dataset: Dataset, 
    batch_size: int, 
    policy_processor, # This must be the NoraPolicyProcessorPipeline
    robot_ratio: float = 0.8,
    num_workers: int = 4
):
    """
    Creates a DataLoader that mixes robotics data and math reasoning data.
    Instead of item-level tokenization, this passes the raw batch to the policy_processor
    to utilize Gemma's native batched padding and tile positioning (in one go).
    """
    
    # 1. Directly concatenate the raw datasets (No Wrapper Needed!)
    combined_dataset = ConcatDataset([robot_dataset, math_hf_dataset])
    
    # 2. Calculate weights for WeightedRandomSampler
    total_robot_len = len(robot_dataset)
    total_math_len = len(math_hf_dataset)
    
    weight_robot = robot_ratio / total_robot_len if total_robot_len > 0 else 0
    weight_math = (1.0 - robot_ratio) / total_math_len if total_math_len > 0 else 0
    
    sample_weights = (
        [weight_robot] * total_robot_len + 
        [weight_math] * total_math_len
    )
    
    # 3. Build Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(combined_dataset), 
        replacement=True 
    )
    
    # 4. Build Dataloader
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda raw_batch: policy_processor(raw_batch)
    )
    
    return dataloader