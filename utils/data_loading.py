from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Mapping
from scipy.interpolate import CubicSpline
import torch
from torch.utils.data import Dataset, default_collate
from typing import Any
import pathlib
import numpy as np
import torchvision.transforms as T_v2
from lerobot.configs.types import NormalizationMode
import lerobot.processor
import lerobot.datasets.utils
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize, Qwen2VLImageProcessor
from .skip_episodes_lerobot_dataset import SkipEpisodesLeRobotDataset

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

    state_key: str = 'observation.state'
    """
    Key for the state tensor that corresponds to the action tensor.
    """

    def __call__(self, transition):
        action = transition['action']
        initial_state = transition['observation'][self.state_key]
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


def load_dataset(
    root: str,
    action_keys: Iterable[str],
    load_action_chunk_size: int,
    canonical_action_chunk_size: int,
    raw_fps: int,
    image_target_pixels: int,
    image_processor: Qwen2VLImageProcessor,
    aspect_ratio: float,
    instance_transform: Callable[[dict[str, Any]], dict[str, Any]],
    norm_stats_transform: Callable[[dict[str, dict[str, np.ndarray]]], dict[str, dict[str, np.ndarray]]],
) -> PreprocessedDataset:
    """
    Loads preprocessed dataset. The following preprocessing steps are applied:
    - Transform the instance from the raw dataset to the desired format (by the `instance_transform` param).
    - Convert action tensor from absolute space to delta space.
    - Resample the action tensor from one chunk size to another if necessary.
    - Normalize the action tensor.
    """
    root = pathlib.Path(root)

    delta_timestamps = [i / raw_fps for i in range(load_action_chunk_size)]
    delta_timestamps = {
        action_key: delta_timestamps
        for action_key in action_keys
    }
    resize_target = smart_resize(
        (image_target_pixels / aspect_ratio)**0.5,
        (image_target_pixels * aspect_ratio)**0.5,
        factor = image_processor.patch_size * image_processor.merge_size,
        min_pixels = image_processor.size["shortest_edge"],
        max_pixels = image_processor.size["longest_edge"],
    )
    image_transforms = T_v2.Compose([
        # Note that the dlimp RandomResizedCrop used by the original RLDS dataloader
        # seems to handle `ratio` differently: ratio is measured in normalized coordinates (between 0.0 and 1.0)
        # hence ratio=(1.0, 1.0) would crop at the same aspect ratio as the original image
        # (https://github.com/kvablack/dlimp/blob/5edaa4691567873d495633f2708982b42edf1972/dlimp/augmentations.py#L6)
        # With torchvision to crop at the original aspect ratio, we need to pass the ratio of actual pixels
        T_v2.RandomResizedCrop(size=resize_target, scale=(0.9, 0.9), ratio=(aspect_ratio, aspect_ratio)),
        T_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])
    task_roots = [p for p in root.iterdir() if p.is_dir()]
    dataset = ConcatDataset([
        SkipEpisodesLeRobotDataset(
            task_root.name,
            root=task_root,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
        )
        for task_root in tqdm(task_roots, desc="Loading datasets")
    ])
    # Load and prepare normalization stats
    raw_norm_stats = lerobot.datasets.utils.cast_stats_to_numpy(
        lerobot.datasets.utils.load_json(root / 'delta_norm_stats.json')
    )['norm_stats']
    # gripper min and max are currently hardcoded to 0 and 1.
    # if changing this to use other statistics, remember that gripper states are all transformed by 1-x,
    # so the stats would have to be transformed as well, and order reversed (e.g. q01 becomes 1-q99)
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
