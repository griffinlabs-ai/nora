from dataclasses import dataclass
from typing import Callable, Generic, Iterable
import torch
from torch.utils.data import Dataset
from typing import Any
import pathlib
import numpy as np
import torchvision.transforms as T_v2
from lerobot.configs.types import NormalizationMode
import lerobot.processor
import lerobot.datasets.utils
from torch.utils.data import ConcatDataset
from tqdm import tqdm
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
    use_image_augmentation: bool,
    raw_fps: int,
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
    if use_image_augmentation:
        image_transforms = T_v2.Compose([
            # Note that the dlimp RandomResizedCrop used by the original RLDS dataloader
            # seems to handle `ratio` differently: ratio is measured in normalized coordinates (between 0.0 and 1.0)
            # hence ratio=(1.0, 1.0) would crop at the same aspect ratio as the original image
            # (https://github.com/kvablack/dlimp/blob/5edaa4691567873d495633f2708982b42edf1972/dlimp/augmentations.py#L6)
            # With torchvision to crop at the original aspect ratio, we need to pass the ratio of actual pixels
            T_v2.RandomResizedCrop(size=(224, 224), scale=(0.9, 0.9), ratio=(aspect_ratio, aspect_ratio)),
            T_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])
    else:
        image_transforms = None
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
        'ACTION': NormalizationMode.MIN_MAX,
    }
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
            lerobot.processor.NormalizerProcessorStep({}, norm_map , norm_stats),
        ],
        to_transition=lambda batch:
            lerobot.processor.converters.batch_to_transition(instance_transform(batch)),
    )

    dataset = PreprocessedDataset(dataset, preprocessor)

    return dataset
