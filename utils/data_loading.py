from dataclasses import dataclass
import json
from typing import Callable, Generic, Iterable, Mapping
from collections.abc import Set
from scipy.interpolate import CubicSpline
import torch
from torch.utils.data import Dataset, default_collate
from typing import Any
import pathlib
import numpy as np
from lerobot.configs.types import NormalizationMode, PolicyFeature, FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import lerobot.processor
import lerobot.datasets.io_utils
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from lerobot.processor.pipeline import PolicyProcessorPipeline, TOutput

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
    """
    Key for the state tensor that corresponds to the action tensor.
    If None, the initial state is assumed to be zero.
    Leave as None if the action tensor is in delta space.
    """

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
            # If the original chunk size is a multiple of the target chunk size, we can simply take every n-th action.
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
class EmbeddedSE3Segmenter:
    """
    Segment an action tensor into SE(3) matrices and real values.
    """
    se3_segment_start_idxs: Set[int] | None = None

    def __post_init__(self):
        split_points = []
        for start_idx in sorted(self.se3_segment_start_idxs or set()):
            split_points.append((start_idx, 'se3_matrix'))
            split_points.append((start_idx + 16, 'real'))
        if len(split_points) == 0 or split_points[0][0] != 0:
            split_points.insert(0, (None, 'real'))
        self.slices = [
            (slice(split_points[i][0], split_points[i+1][0] if i+1 < len(split_points) else None), split_points[i][1])
            for i in range(len(split_points))
        ]

@dataclass
@lerobot.processor.ProcessorStepRegistry.register("abs2delta_action_processor")
class Abs2DeltaActionProcessorStep(lerobot.processor.ProcessorStep):
    """
    Convert action tensor from absolute space to delta space.
    Expects an action shape of [..., chunk_size, degrees_of_freedom].
    Expects transition to have a state tensor in the same vector space as the action tensor.
    """

    mask: torch.Tensor
    """
    Mask of which action tensor dimensions to convert to delta space.
    `True` dimensions output delta space, `False` dimensions keep absolute space.
    """

    se3_segment_start_idxs: Set[int] | None = None

    state_key: str = 'observation.state'
    """
    Key for the state tensor that corresponds to the action tensor.
    """

    def __post_init__(self):
        self.segmenter = EmbeddedSE3Segmenter(self.se3_segment_start_idxs)

    def __call__(self, transition):
        new_transition = transition.copy()

        action = transition['action']
        state = transition['observation'][self.state_key].unsqueeze(-2)
        leading_dims = action.shape[:-1]

        assert self.mask.shape[-1] == action.shape[-1]

        segments = []

        for slice, operand_type in self.segmenter.slices:
            if operand_type == 'real':
                segment = action[..., slice] - state[..., slice]
            elif operand_type == 'se3_matrix':
                action_se3 = action[..., slice].view(*leading_dims, 4, 4)
                state_se3 = state[..., slice].view(*leading_dims[:-1], 1, 4, 4)
                segment = state_se3.inverse().matmul(action_se3)
                segment = segment.view(*leading_dims, 16)
            segments.append(segment)

        deltas = torch.cat(segments, dim = -1)
        new_transition['action'] = torch.where(self.mask.expand(action.shape), deltas, action)
        return new_transition

    def transform_features(self, features):
        return features

@dataclass
@lerobot.processor.ProcessorStepRegistry.register("se3_matrix_to_xyz_angles_processor")
class SE3MatrixToXYZAnglesProcessorStep(lerobot.processor.ProcessorStep):
    """
    Convert SE(3) matrices in the action tensor to XYZ and angles.
    """
    se3_segment_start_idxs: Set[int]
    state_key: str = 'observation.state'

    PER_SE3_MATRIX_DIM_REDUCTION = 10   # 16 dimensions -> 6 dimensions

    def __post_init__(self):
        self.segmenter = EmbeddedSE3Segmenter(self.se3_segment_start_idxs)

    @staticmethod
    def convert(action: torch.Tensor) -> torch.Tensor:
        r = action[..., :3, :3]
        t = action[..., :3, 3]

        sin_pitch = -r[..., 2, 0]
        cos_pitch = torch.sqrt(
            torch.clamp(r[..., 2, 1] ** 2 + r[..., 2, 2] ** 2, min=0.0)
        )
        alpha = torch.atan2(r[..., 2, 1], r[..., 2, 2])
        beta = torch.atan2(sin_pitch, cos_pitch)
        gamma = torch.atan2(r[..., 1, 0], r[..., 0, 0])

        angles = torch.stack((alpha, beta, gamma), dim=-1)
        return torch.cat((t, angles), dim=-1)

    def __call__(self, transition):
        new_transition = transition.copy()
        action = transition['action']
        state = transition['observation'][self.state_key]

        action_segments = []
        state_segments = []

        for slice, operand_type in self.segmenter.slices:
            if operand_type == 'real':
                action_segments.append(action[..., slice])
                state_segments.append(state[..., slice])
            elif operand_type == 'se3_matrix':
                action_se3 = action[..., slice].view(*action.shape[:-1], 4, 4)
                state_se3 = state[..., slice].view(*state.shape[:-1], 4, 4)
                action_segments.append(self.convert(action_se3))
                state_segments.append(self.convert(state_se3))

        new_transition['action'] = torch.cat(action_segments, dim = -1)
        new_transition['observation'][self.state_key] = torch.cat(state_segments, dim = -1)
        return new_transition
    
    def transform_features(self, features):
        old_shape = features['action'].shape
        num_se3_segments = sum(1 for s, t in self.segmenter.slices if t == 'se3_matrix')
        features['action']['action'] = PolicyFeature(
            FeatureType.ACTION,
            old_shape[:-1] + (old_shape[-1] - num_se3_segments * self.PER_SE3_MATRIX_DIM_REDUCTION,),
        )
        old_shape = features['observation'][self.state_key].shape
        features['observation'][self.state_key] = PolicyFeature(
            FeatureType.OBSERVATION,
            old_shape[:-1] + (old_shape[-1] - num_se3_segments * self.PER_SE3_MATRIX_DIM_REDUCTION,),
        )
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

DEFAULT_DELTA_TRANSFORM_MASK = torch.tensor(
    [
        True, True, True, True, True, True, True, False, False, False, False, False, False, False,
        True, True, True, True, True, True, True, False, False, False, False, False, False, False,
    ],
    dtype=torch.bool,
)

def load_dataset(
    root: str | pathlib.Path,
    action_keys: Iterable[str],
    load_action_chunk_size: int,
    canonical_action_chunk_size: int,
    raw_fps: int,
    instance_transform: Callable[[dict[str, Any]], dict[str, Any]],
    norm_stats_transform: Callable[[dict[str, dict[str, np.ndarray]]], dict[str, dict[str, np.ndarray]]],
    se3_segment_start_idxs: Set[int] | None = None,
    delta_transform_mask: torch.Tensor | None = None,
    num_frames: int = 1, 
) -> PreprocessedDataset:
    """
    Loads preprocessed dataset. The following preprocessing steps are applied:
    - Transform the instance from the raw dataset to the desired format (by the `instance_transform` param).
    - Convert action tensor from absolute space to delta space.
    - Convert SE(3) matrices to XYZ and angles if necessary.
    - Resample the action tensor from one chunk size to another if necessary.
    - Normalize the action tensor.
    """
    root = pathlib.Path(root)

    # 1. Action timestamps (Future prediction chunk)
    action_delta_timestamps = [i / raw_fps for i in range(load_action_chunk_size)]
    delta_timestamps = {
        action_key: action_delta_timestamps
        for action_key in action_keys
    }
    
    task_roots = [p.parent.parent for p in root.rglob('info.json')]

    # 2. Image timestamps (Past observation history)
    # We dynamically find the image keys from the first dataset to apply history frames
    if task_roots and num_frames > 1:
        
        repo_id = str(task_roots[0].relative_to(root))
        meta = LeRobotDatasetMetadata(repo_id, root=task_roots[0])
        
        # Find all keys that represent images
        image_keys = [k for k in meta.features.keys() if 'image' in k.lower()]
        
        # Calculate timestamps for 5 past frames + 1 current frame
        img_timestamps = [float(i - num_frames + 1) for i in range(num_frames)]
        
        # Apply the temporal window to all image streams
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
    
    # Load and prepare normalization stats
    raw_norm_stats = lerobot.datasets.io_utils.cast_stats_to_numpy(
        lerobot.datasets.io_utils.load_json(root / 'delta_norm_stats.json')
    )['norm_stats']
    
    # gripper min and max are currently hardcoded to 0 and 1.
    # if changing this to use other statistics, remember that gripper states are all transformed by 1-x,
    # so the stats would have to be transformed as well, and order reversed (e.g. q01 becomes 1-q99)
    norm_stats = norm_stats_transform(raw_norm_stats)

    norm_map = {
        'ACTION': NormalizationMode.QUANTILES,
    }
    
    convert_se3_if_necessary = [SE3MatrixToXYZAnglesProcessorStep(
        se3_segment_start_idxs = se3_segment_start_idxs,
    )] if se3_segment_start_idxs else []

    resample_step_if_necessary = [ResampleActionProcessorStep(
        target_chunk_size = canonical_action_chunk_size,
    )] if load_action_chunk_size != canonical_action_chunk_size else []

    preprocessor = PolicyProcessorPipeline(
        steps = [
            Abs2DeltaActionProcessorStep(
                mask = delta_transform_mask if delta_transform_mask is not None else DEFAULT_DELTA_TRANSFORM_MASK,
                se3_segment_start_idxs = se3_segment_start_idxs,
            ),
            *convert_se3_if_necessary,
            *resample_step_if_necessary,
            lerobot.processor.NormalizerProcessorStep({}, norm_map , norm_stats),
        ],
        to_transition=lambda batch:
            lerobot.processor.converters.batch_to_transition(instance_transform(batch)),
    )

    dataset = PreprocessedDataset(dataset, preprocessor)

    return dataset

def collate_with_observation_image_lists(
    examples: Mapping[str, Any],
) -> Mapping[str, Any]:
    """
    Collate function that collates `observation.images.*` fields as lists rather than tensors.

    This allows for heterogeneous image shapes in the batch.
    """
    images = [
        {k: v for k, v in example.items() if k.startswith('observation.images.')}
        for example in examples
    ]
    collated_images = {
        k: [observation[k] for observation in images]
        for k in images[0].keys()
    }
    no_images = [
        {k: v for k, v in example.items() if not k.startswith('observation.images.')}
        for example in examples
    ]
    collated = {
        **collated_images,
        **default_collate(no_images)
    }
    return collated