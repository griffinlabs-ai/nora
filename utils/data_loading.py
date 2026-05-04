from dataclasses import dataclass
import functools
import json
from typing import Callable, Generic, Iterable, Mapping
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
        # map new indices to old indices
        self.new_ranges_to_old_offsets = {}
        new_range_start = 0
        for old_range in self.keep_old_ranges:
            new_range = range(new_range_start, new_range_start + len(old_range))
            self.new_ranges_to_old_offsets[new_range] = old_range.start - new_range_start
            new_range_start += len(old_range)

    def __len__(self) -> int:
        return list(self.new_ranges_to_old_offsets)[-1].stop

    def __getitem__(self, idx: int) -> dict:
        # find the range that contains the new index
        for new_range, old_offset in self.new_ranges_to_old_offsets.items():
            if idx in new_range:
                return self.lerobot_ds[idx + old_offset]
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

    state_key: str = 'observation.state'
    """
    Key for the state tensor that corresponds to the action tensor.
    """

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
    num_frames: int = 1,
) -> Dataset:
    """
    Loads an aggregated dataset from a root directory.

    ``instance_transform`` must accept ``(batch, *, meta: LeRobotDatasetMetadata, task_config: dict | None)``.

    All ``PreprocessedDataset`` leaves share the same ``shared_processor_steps`` list (same step
    instances); only ``to_transition`` / per-root ``instance_transform`` differ.

    Preprocessing per sample:
    - Instance transform (dataset-specific merge, subtask, etc.).
    - Absolute to delta actions, optional resample, normalization.
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

    resample_step_if_necessary = [
        ResampleActionProcessorStep(
            target_chunk_size=canonical_action_chunk_size,
        )
    ] if load_action_chunk_size != canonical_action_chunk_size else []
    processor_steps = [
        Abs2DeltaActionProcessorStep(
            mask=torch.tensor(
                [
                    True, True, True, True, True, True, True, False,
                    True, True, True, True, True, True, True, False,
                ],
                dtype=torch.bool,
            )
        ),
        *resample_step_if_necessary,
        lerobot.processor.NormalizerProcessorStep({}, norm_map, norm_stats),
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
            # lerobot_ds = load_lerobot_dataset_skip_dirty_episodes(
            #     str(task_root.relative_to(root)),
            #     root=task_root,
            #     delta_timestamps=delta_timestamps,
            # )
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