from dataclasses import dataclass
import json
from typing import Callable, Generic, Iterable, Mapping
from scipy.interpolate import CubicSpline
import torch
from torch.utils.data import Dataset, default_collate
from typing import Any
import pathlib
import numpy as np
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, default_collate
from typing import Any, Mapping
import torch
from lerobot.configs.types import NormalizationMode, PolicyFeature, FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import lerobot.processor


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
    """
    Loads preprocessed dataset. The following preprocessing steps are applied:
    - Transform the instance from the raw dataset to the desired format (by the `instance_transform` param).
    - Convert action tensor from absolute space to delta space.
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


class UnifiedVLAWrapper(Dataset):
    """
    A wrapper to unify the output of Robotics Datasets and VL Math Datasets.
    Ensures that both return identical dictionary keys (input_ids, labels, pixel_values, etc.)
    ready for the VLA model's forward pass.
    """
    def __init__(self, dataset, task_type: str, policy_processor=None, text_tokenizer=None):
        self.dataset = dataset
        self.task_type = task_type
        self.policy_processor = policy_processor
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.task_type == "robot":
            # 1. Process robotics trajectory through your existing pipeline
            transition = self.policy_processor(item)
            return transition # This should return the dictionary with input_ids, labels, etc.
            
        elif self.task_type == "vl_math":
            # 2. Process VL Math data
            image = item["image"]
            instruction = item["instruction"]
            text_answer = item["text_answer"]

            # Construct conversational prompt
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
                {"role": "assistant", "content": [{"type": "text", "text": text_answer}]}
            ]
            
            # Use the text_tokenizer/processor to format the strings and mask the loss
            prompt_text = self.text_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            
            inputs = self.text_tokenizer(
                text=prompt_text, 
                images=[image], 
                return_tensors="pt", 
                padding="max_length", 
                max_length=512,
                truncation=True
            )
            
            # Squeeze batch dimension added by the tokenizer
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            
            # Standard Next-Token Prediction Masking: Mask prompt, calculate loss on answer
            labels = inputs["input_ids"].clone()
            
            # Simple heuristic: find the assistant token and mask everything before it with -100
            assistant_token_id = self.text_tokenizer.tokenizer.convert_tokens_to_ids("<|im_start|>assistant")
            if assistant_token_id is not None:
                assistant_idx = (labels == assistant_token_id).nonzero(as_tuple=True)[0]
                if len(assistant_idx) > 0:
                    labels[:assistant_idx[0] + 1] = -100
                    
            labels[labels == self.text_tokenizer.tokenizer.pad_token_id] = -100
            inputs["labels"] = labels
            
            return inputs


def build_co_training_dataloader(
    robot_dataset: Dataset, 
    math_hf_dataset, 
    batch_size: int, 
    robot_ratio: float = 0.8,
    policy_processor = None,
    text_tokenizer = None,
    num_workers: int = 4
):
    """
    Creates a DataLoader that mixes robotics data and math reasoning data
    at a specific ratio (e.g., 80% Robot, 20% Math) to prevent catastrophic forgetting.
    """
    # 1. Wrap datasets
    wrapped_robot = UnifiedVLAWrapper(robot_dataset, task_type="robot", policy_processor=policy_processor)
    wrapped_math = UnifiedVLAWrapper(math_hf_dataset, task_type="vl_math", text_tokenizer=text_tokenizer)
    
    # 2. Concatenate
    combined_dataset = ConcatDataset([wrapped_robot, wrapped_math])
    
    # 3. Calculate weights for WeightedRandomSampler
    total_robot_len = len(wrapped_robot)
    total_math_len = len(wrapped_math)
    
    weight_robot = robot_ratio / total_robot_len
    weight_math = (1.0 - robot_ratio) / total_math_len
    
    sample_weights = (
        [weight_robot] * total_robot_len + 
        [weight_math] * total_math_len
    )
    
    # 4. Build Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(combined_dataset), 
        replacement=True 
    )
    
    # 5. Build Dataloader
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
