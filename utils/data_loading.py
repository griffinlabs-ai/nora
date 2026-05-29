import bisect
from dataclasses import dataclass
import functools
import json
from typing import Callable, Generic, Iterable, Mapping
from collections.abc import Set
from scipy.interpolate import CubicSpline
import torch
from torch.utils.data import Dataset, default_collate
from typing import Any
import pathlib
import numpy as np
from lerobot.configs.types import NormalizationMode, PipelineFeatureType, PolicyFeature, FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import lerobot.processor
import lerobot.datasets.io_utils
from torch.utils.data import ConcatDataset, Dataset
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

class SkipEpisodesLeRobotDataset(Dataset):
    """
    Wrapper for `LeRobotDataset` that skips dirty episodes listed in the `meta/removed_episodes.json` file.
    """
    REMOVED_EPISODES_PATH = "meta/removed_episodes.json"
    REMOVED_EPISODES_KEY = "dirty_episodes"

    def __init__(self, *args, **kwargs):
        self.lerobot_ds = LeRobotDataset(*args, **kwargs)
        
        rm_ep_path = pathlib.Path(self.lerobot_ds.root / self.REMOVED_EPISODES_PATH)
        if rm_ep_path.exists():
            with open(self.lerobot_ds.root / self.REMOVED_EPISODES_PATH) as f:
                self.skip_episodes = json.load(f)[self.REMOVED_EPISODES_KEY]
            self.skip_ranges = [
                range(row["dataset_from_index"], row["dataset_to_index"])
                for row in self.lerobot_ds.meta.episodes if row["episode_index"] in self.skip_episodes
            ]
            # obtain ranges to keep
            self.keep_old_ranges = [
                range(0, self.skip_ranges[0].start),
                *[
                    range(prev_range.stop, curr_range.start)
                    for prev_range, curr_range in zip(self.skip_ranges[:-1], self.skip_ranges[1:])
                    if prev_range.stop < curr_range.start
                ],
                range(self.skip_ranges[-1].stop, len(self.lerobot_ds)),
            ]
        else:
            self.skip_episodes = []
            self.skip_ranges = []
            self.keep_old_ranges = [range(0, len(self.lerobot_ds))]
        # map new indices to old indices using binary-searchable range ends
        self._new_range_ends = []
        self._old_offsets = []
        self._length = 0
        for old_range in self.keep_old_ranges:
            self._new_range_ends.append(self._length + len(old_range))
            self._old_offsets.append(old_range.start - self._length)
            self._length += len(old_range)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range")
        range_idx = bisect.bisect_right(self._new_range_ends, idx)
        if range_idx < len(self._old_offsets):
            return self.lerobot_ds[idx + self._old_offsets[range_idx]]
        raise IndexError(f"Index {idx} out of range")

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
@lerobot.processor.ProcessorStepRegistry.register("se3_matrix_to_xyz_rot6d_processor")
class SE3MatrixToXYZRot6DProcessorStep(lerobot.processor.ProcessorStep):
    """
    Convert SE(3) matrices in the action tensor to XYZ and rot6d.
    """
    se3_segment_start_idxs: Set[int]
    state_key: str = 'observation.state'

    PER_SE3_MATRIX_DIM_REDUCTION = 7   # 16 dimensions -> 9 dimensions

    def __post_init__(self):
        self.segmenter = EmbeddedSE3Segmenter(self.se3_segment_start_idxs)

    @staticmethod
    def convert(action: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [action[..., :3, 3], action[..., :3, :2].flatten(start_dim = -2)],
            dim = -1,
        )

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
        old_shape = features[PipelineFeatureType.ACTION]['action'].shape
        num_se3_segments = sum(1 for _, t in self.segmenter.slices if t == 'se3_matrix')
        features[PipelineFeatureType.ACTION]['action'] = PolicyFeature(
            FeatureType.ACTION,
            old_shape[:-1] + (old_shape[-1] - num_se3_segments * self.PER_SE3_MATRIX_DIM_REDUCTION,),
        )
        old_shape = features[PipelineFeatureType.OBSERVATION][self.state_key].shape
        features[PipelineFeatureType.OBSERVATION][self.state_key] = PolicyFeature(
            FeatureType.STATE,
            old_shape[:-1] + (old_shape[-1] - num_se3_segments * self.PER_SE3_MATRIX_DIM_REDUCTION,),
        )
        return features

@dataclass
@lerobot.processor.ProcessorStepRegistry.register("random_arm_control_mode_processor")
class RandomArmControlModeProcessorStep(lerobot.processor.ProcessorStep):
    """
    Randomly keep either joint-position or EEF-pose arm control dimensions.

    Gripper/finger dimensions should be included in both masks. The selected tensor is
    compacted, then conditionally padded so that both modes have the same width.
    """

    joint_position_mode_mask: torch.Tensor
    eef_pose_mode_mask: torch.Tensor
    state_key: str = 'observation.state'

    def __post_init__(self):
        if self.joint_position_mode_mask.shape != self.eef_pose_mode_mask.shape:
            raise ValueError(
                "joint_position_mode_mask and eef_pose_mode_mask must have the same shape."
            )
        self.target_dim = max(
            self.joint_position_mode_mask.sum().item(),
            self.eef_pose_mode_mask.sum().item(),
        )

    def __call__(self, transition):
        if torch.rand(()) < 0.5:
            arm_control_mode = 'joint_position'
            mask = self.joint_position_mode_mask
        else:
            arm_control_mode = 'eef_pose'
            mask = self.eef_pose_mode_mask

        new_transition = transition.copy()
        new_transition['action'] = self._select_and_pad(transition['action'], mask)

        observation = dict(transition['observation'])
        observation[self.state_key] = self._select_and_pad(
            transition['observation'][self.state_key],
            mask,
        )
        new_transition['observation'] = observation

        info = dict(transition.get('info') or {})
        info['arm_control_mode'] = arm_control_mode
        info['n_action_dims'] = int(mask.sum().item())
        new_transition['info'] = info
        return new_transition

    def _select_and_pad(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = tensor[..., mask]
        return torch.nn.functional.pad(
            selected,
            (0, self.target_dim - selected.shape[-1]),
        )

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_feat = features[PipelineFeatureType.ACTION]['action']
        features[PipelineFeatureType.ACTION]['action'] = PolicyFeature(
            FeatureType.ACTION,
            action_feat.shape[:-1] + (self.target_dim,),
        )
        state_feat = features[PipelineFeatureType.OBSERVATION][self.state_key]
        features[PipelineFeatureType.OBSERVATION][self.state_key] = PolicyFeature(
            FeatureType.STATE,
            state_feat.shape[:-1] + (self.target_dim,),
        )
        return features

@dataclass
@lerobot.processor.ProcessorStepRegistry.register("pad_action_processor")
class PadActionProcessorStep(lerobot.processor.ProcessorStep):
    """
    Pad action and state tensors to a fixed width after normalization.
    Records the pre-pad dimension count in info['n_action_dims'].
    """

    target_dim: int
    state_key: str = 'observation.state'

    def __call__(self, transition):
        new_transition = transition.copy()
        action = transition['action']
        n = action.shape[-1]
        if n > self.target_dim:
            raise ValueError(
                f"Action dim {n} exceeds target_dim {self.target_dim}; cannot pad."
            )

        info = dict(transition.get('info') or {})
        info.setdefault('n_action_dims', n)
        new_transition['info'] = info

        if n < self.target_dim:
            pad_width = self.target_dim - n
            new_transition['action'] = torch.nn.functional.pad(
                action, (0, pad_width),
            )
            observation = dict(transition['observation'])
            observation[self.state_key] = torch.nn.functional.pad(
                transition['observation'][self.state_key],
                (0, pad_width),
            )
            new_transition['observation'] = observation

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_feat = features[PipelineFeatureType.ACTION]['action']
        old_shape = action_feat.shape
        features[PipelineFeatureType.ACTION]['action'] = PolicyFeature(
            FeatureType.ACTION,
            old_shape[:-1] + (self.target_dim,),
        )
        state_feat = features[PipelineFeatureType.OBSERVATION][self.state_key]
        old_state_shape = state_feat.shape
        features[PipelineFeatureType.OBSERVATION][self.state_key] = PolicyFeature(
            FeatureType.STATE,
            old_state_shape[:-1] + (self.target_dim,),
        )
        return features

def load_lerobot_dataset_skip_dirty_episodes(
    repo_id: str,
    root: str | pathlib.Path | None = None,
    episodes: list[int] | None = None,
    *args,
    **kwargs,
) -> LeRobotDataset:
    if episodes is None and root is not None:
        removed_episodes_path = pathlib.Path(root) / 'meta/removed_episodes.json'
        if removed_episodes_path.exists():
            total_episodes = lerobot.datasets.io_utils.load_info(root)['total_episodes']
            with open(removed_episodes_path, 'r') as f:
                removed_episodes = json.load(f)['dirty_episodes']
            episodes = [i for i in range(total_episodes) if i not in removed_episodes]
    return LeRobotDataset(repo_id, root, episodes, *args, **kwargs)


def load_task_config(task_root: pathlib.Path) -> dict[str, Any] | None:
    path = task_root / "meta" / "task_config.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset(
    root: str | pathlib.Path,
    action_keys: Iterable[str],
    load_action_chunk_size: int,
    canonical_action_chunk_size: int,
    raw_fps: int,
    instance_transform: Callable[..., dict[str, Any]],
    norm_stats_transform: Callable[[dict[str, dict[str, np.ndarray]]], dict[str, dict[str, np.ndarray]]],
    delta_transform_mask: torch.Tensor,
    target_action_dim: int,
    se3_segment_start_idxs: Set[int] | None = None,
    joint_position_mode_mask: torch.Tensor | None = None,
    eef_pose_mode_mask: torch.Tensor | None = None,
    num_frames: int = 1,
) -> Dataset:
    """
    Loads an aggregated dataset from a root directory.

    ``instance_transform`` must accept ``(batch, *, meta: LeRobotDatasetMetadata, task_config: dict | None)``.

    All ``PreprocessedDataset`` leaves share the same ``shared_processor_steps`` list (same step
    instances); only ``to_transition`` / per-root ``instance_transform`` differ.

    Preprocessing per sample:
    - Instance transform (dataset-specific merge, subtask, etc.).
    - Absolute to delta actions, optional SE(3) matrix to XYZ and rot6d, optional resample,
      normalization, optional arm-control-mode selection, then pad to target_action_dim.
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

    # Load and prepare normalization stats.
    raw_norm_stats = lerobot.datasets.io_utils.cast_stats_to_numpy(
        lerobot.datasets.io_utils.load_json(root / 'delta_norm_stats.json')
    )['norm_stats']
    
    # gripper min and max are currently hardcoded to 0 and 1.
    # if changing this to use other statistics, remember that gripper states are all transformed by 1-x,
    # so the stats would have to be transformed as well, and order reversed (e.g. q01 becomes 1-q99)
    norm_stats = norm_stats_transform(raw_norm_stats)

    norm_map = {
        'ACTION': NormalizationMode.QUANTILES,
        'STATE': NormalizationMode.QUANTILES,
    }
    normalizer_features = {
        'observation.state': PolicyFeature(
            FeatureType.STATE,
            tuple(norm_stats['observation.state']['q01'].shape),
        ),
    }

    
    convert_se3_if_necessary = [SE3MatrixToXYZRot6DProcessorStep(
        se3_segment_start_idxs = se3_segment_start_idxs,
    )] if se3_segment_start_idxs else []

    resample_step_if_necessary = [
        ResampleActionProcessorStep(
            target_chunk_size=canonical_action_chunk_size,
        )
    ] if load_action_chunk_size != canonical_action_chunk_size else []

    if convert_se3_if_necessary and resample_step_if_necessary:
        raise ValueError(
            "Resampling and SE(3) matrices cannot be used at the same time, "
            "as correct interpolation of SE(3) values is not implemented."
        )

    select_arm_control_mode_if_necessary = [
        RandomArmControlModeProcessorStep(
            joint_position_mode_mask = joint_position_mode_mask,
            eef_pose_mode_mask = eef_pose_mode_mask,
        )
    ] if joint_position_mode_mask is not None and eef_pose_mode_mask is not None else []

    processor_steps = [
        Abs2DeltaActionProcessorStep(
            mask = delta_transform_mask,
            se3_segment_start_idxs = se3_segment_start_idxs,
        ),
        *convert_se3_if_necessary,
        *resample_step_if_necessary,
        lerobot.processor.NormalizerProcessorStep(
            normalizer_features,
            norm_map,
            norm_stats,
            normalize_observation_keys={'observation.state'},
        ),
        *select_arm_control_mode_if_necessary,
        PadActionProcessorStep(target_dim = target_action_dim),
    ]

    preprocessed_subsets: list[PreprocessedDataset] = []
    with tqdm(task_roots, desc=f"Loading ds — {root}") as tqdm_task_roots:
        for task_root in tqdm_task_roots:
            tqdm_task_roots.set_description(f"Loading ds — {root}/{task_root.relative_to(root)}")
            ds = SkipEpisodesLeRobotDataset(
                str(task_root.relative_to(root)),
                root=task_root,
                delta_timestamps=delta_timestamps,
            )
            task_config = load_task_config(task_root)
            subset_inst_transform = functools.partial(
                instance_transform,
                meta=ds.lerobot_ds.meta,
                task_config=task_config,
            )
            preprocessor = PolicyProcessorPipeline(
                steps=processor_steps,
                to_transition=lambda b, inst_transform=subset_inst_transform: 
                    lerobot.processor.converters.batch_to_transition(inst_transform(b))
            )
            preprocessed_subsets.append(PreprocessedDataset(ds, preprocessor))
        tqdm_task_roots.set_description(f"Done loading ds — {root}")

    return ConcatDataset(preprocessed_subsets)

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