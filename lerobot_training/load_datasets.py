import functools
import pathlib
from typing import Any, Iterable
import torch
from torch.utils.data import ConcatDataset
from utils.data_loading import load_dataset
from datasets import load_dataset as hf_load_dataset, concatenate_datasets, Image as HFImage
import random

import numpy as np

ACTION_DIM_IS_PAD = {
    'dual_arm_7dof': torch.zeros(16, dtype=torch.bool),
    'dual_arm_6dof': torch.tensor(
        [False, False, False, False, False, False, True, False] * 2,
        dtype=torch.bool
    ),
    'single_arm_7dof': torch.tensor([False] * 8 + [True] * 8, dtype=torch.bool),
}

MergeSpec = Iterable[tuple[str, slice] | tuple[float, int]]

MERGE_SPECS = {
    'agibot_world': (
        ('joint.position', slice(0, 7)),
        ('effector.position', slice(0, 1)),
        ('joint.position', slice(7, 14)),
        ('effector.position', slice(1, 2)),
    ),
    'galaxea': (
        ('left_arm', slice(None)),
        (0.0, 1),
        ('left_gripper', slice(None)),
        ('right_arm', slice(None)),
        (0.0, 1),
        ('right_gripper', slice(None)),
    ),
    'interndata_a1_franka': (
        ('joint.position', slice(None)),
        ('gripper.position', slice(None)),
        (0.0, 8),
    ),
    'interndata_a1_genie1': (
        ('left_joint.position', slice(None)),
        ('left_gripper.position', slice(None)),
        ('right_joint.position', slice(None)),
        ('right_gripper.position', slice(None)),
    ),
    'interndata_a1_dual_arm_6dof': (
        ('left_joint.position', slice(None)),
        (0.0, 1),
        ('left_gripper.position', slice(None)),
        ('right_joint.position', slice(None)),
        (0.0, 1),
        ('right_gripper.position', slice(None)),
    ),
}

def merge_features(
    inst: dict[str, Any],
    merge_prefix: str,
    merge_spec: MergeSpec,
    merged_feature_key: str | None = None,
):
    """
    Merge features from the instance dictionary into a single feature.
    Feature keys starting with `merge_prefix` are removed from the instance dictionary.
    If `merged_feature_key` is not provided, the merged feature key is the same as the `merge_prefix`.

    Merge spec is a list of tuples of:
    -  feature key (without `merge_prefix`) and a slice object of dimensions to take, OR
    -  a float value (which would be used for padding) and the number of dimensions to pad.

    The merge output is a single tensor with the concatenated features dimensions.
    """
    first_feature = next(feat for feat, _ in merge_spec if feat is not None)
    first_feature_tensor = inst[f"{merge_prefix}.{first_feature}"]
    leading_dims = first_feature_tensor.shape[:-1]
    to_cat = [
        inst[f"{merge_prefix}.{feat}"][..., dims] if isinstance(feat, str) \
            else torch.full((*leading_dims, dims), fill_value = feat, device = first_feature_tensor.device)
        for feat, dims in merge_spec
    ]

    new_inst = {k: v for k, v in inst.items() if not k.startswith(merge_prefix + '.')}
    new_inst[merged_feature_key or merge_prefix] = torch.cat(to_cat, dim = -1)
    return new_inst

def merge_norm_stats(
    norm_stats: dict[str, dict[str, np.ndarray]],
    merge_prefix: str,
    merge_spec: MergeSpec,
) -> dict[str, dict[str, np.ndarray]]:
    norm_stats = {
        'q01': {feat: torch.from_numpy(norm_stats[feat]['q01']).view(-1) for feat in norm_stats},
        'q99': {feat: torch.from_numpy(norm_stats[feat]['q99']).view(-1) for feat in norm_stats},
    }
    merged = {
        'q01': merge_features(norm_stats['q01'], merge_prefix, merge_spec, 'action'),
        'q99': merge_features(norm_stats['q99'], merge_prefix, merge_spec, 'action'),
    }
    merged = {
        feat: {'q01': merged['q01'][feat].numpy(), 'q99': merged['q99'][feat].numpy()}
        for feat in merged['q01']
    }
    return merged

def generic_to_nora_instance(
    batch: dict[str, Any],
    merge_spec: MergeSpec,
    action_prefix: str,
    state_prefix: str,
    action_dim_is_pad: torch.Tensor,
    embodiment_prompt: str,
):
    batch = merge_features(batch, action_prefix, merge_spec, 'action')
    batch = merge_features(batch, state_prefix, merge_spec, 'observation.state')
    batch['action_dim_is_pad'] = action_dim_is_pad
    batch['info'] = {"embodiment_prompt": embodiment_prompt}
    return batch

def agibot_world_to_nora_instance(batch: dict[str, Any]):
    """
    Convert from raw AgiBot World dataset format to format that is ready to be converted to `EnvTransition`:
    - Merge relevant actions into `action`, discarding other actions.
    - Merge relevant states into `observation.state`, discarding other states.
    - Invert the gripper action by 1-x.
    """
    batch['actions.effector.position'] = 1 - batch['actions.effector.position']
    return generic_to_nora_instance(
        batch,
        merge_spec = MERGE_SPECS['agibot_world'],
        action_prefix = 'actions',
        state_prefix = 'observation.states',
        action_dim_is_pad = ACTION_DIM_IS_PAD['dual_arm_7dof'],
        embodiment_prompt = "AgiBot G1 with 2 grippers",
    )

def galaxea_to_nora_instance(batch: dict[str, Any]):
    """
    Convert from raw Galaxea Open World Dataset format to format that is ready to be converted to `EnvTransition`:
    - Merge relevant actions into `action`, discarding other actions.
    - Merge relevant states into `observation.state`, discarding other states.
    """
    batch['observation.state.left_gripper'] = batch['observation.state.left_gripper'].view(-1)
    batch['observation.state.right_gripper'] = batch['observation.state.right_gripper'].view(-1)
    batch['action.left_gripper'] = batch['action.left_gripper'].view(-1, 1)
    batch['action.right_gripper'] = batch['action.right_gripper'].view(-1, 1)
    batch = generic_to_nora_instance(
        batch,
        merge_spec = MERGE_SPECS['galaxea'],
        action_prefix = 'action',
        state_prefix = 'observation.state',
        action_dim_is_pad = ACTION_DIM_IS_PAD['dual_arm_6dof'],
        embodiment_prompt = "Galaxea R1 Lite",
    )
    # rename image keys, drop right head image
    batch['observation.images.head'] = batch['observation.images.head_rgb']
    batch['observation.images.hand_left'] = batch['observation.images.left_wrist_rgb']
    batch['observation.images.hand_right'] = batch['observation.images.right_wrist_rgb']
    batch['observation.images.head_is_pad'] = batch['observation.images.head_rgb_is_pad']
    batch['observation.images.hand_left_is_pad'] = batch['observation.images.left_wrist_rgb_is_pad']
    batch['observation.images.hand_right_is_pad'] = batch['observation.images.right_wrist_rgb_is_pad']
    del batch['observation.images.head_rgb']
    del batch['observation.images.head_right_rgb']
    del batch['observation.images.left_wrist_rgb']
    del batch['observation.images.right_wrist_rgb']
    del batch['observation.images.head_rgb_is_pad']
    del batch['observation.images.head_right_rgb_is_pad']
    del batch['observation.images.left_wrist_rgb_is_pad']
    del batch['observation.images.right_wrist_rgb_is_pad']
    return batch

def interndata_a1_to_nora_instance(
    batch: dict[str, Any],
    merge_spec: MergeSpec,
    action_dim_is_pad: torch.Tensor,
    embodiment_prompt: str,
):
    for key in batch:
        if key.startswith('states.') and key.endswith('gripper.position'):
            batch[key] = batch[key].view(-1)
        elif key.startswith('actions.') and key.endswith('gripper.position'):
            batch[key] = batch[key].view(-1, 1)

    batch = generic_to_nora_instance(
        batch,
        merge_spec = merge_spec,
        action_prefix = 'actions',
        state_prefix = 'states',
        action_dim_is_pad = action_dim_is_pad,
        embodiment_prompt = embodiment_prompt,
    )
    batch = {k.replace('images.rgb.', 'observation.images.'): v for k, v in batch.items()}
    return batch

interndata_a1_genie1_to_nora_instance = functools.partial(
    interndata_a1_to_nora_instance,
    merge_spec = MERGE_SPECS['interndata_a1_genie1'],
    action_dim_is_pad = ACTION_DIM_IS_PAD['dual_arm_7dof'],
    embodiment_prompt = "InternData-A1 simulated Genie1 with 2 grippers",
)
interndata_a1_lift2_to_nora_instance = functools.partial(
    interndata_a1_to_nora_instance,
    merge_spec = MERGE_SPECS['interndata_a1_dual_arm_6dof'],
    action_dim_is_pad = ACTION_DIM_IS_PAD['dual_arm_6dof'],
    embodiment_prompt = "InternData-A1 simulated Lift-2 with R5a arms",
)
interndata_a1_split_aloha_to_nora_instance = functools.partial(
    interndata_a1_to_nora_instance,
    merge_spec = MERGE_SPECS['interndata_a1_dual_arm_6dof'],
    action_dim_is_pad = ACTION_DIM_IS_PAD['dual_arm_6dof'],
    embodiment_prompt = "InternData-A1 simulated Split Aloha with 2 Piper-100 arms",
)

def interndata_a1_franka_to_nora_instance(batch: dict[str, Any]):
    batch = interndata_a1_to_nora_instance(
        batch,
        merge_spec = MERGE_SPECS['interndata_a1_franka'],
        action_dim_is_pad = ACTION_DIM_IS_PAD['single_arm_7dof'],
        embodiment_prompt = "InternData-A1 simulated Franka Emika Panda",
    )
    batch['observation.images.hand_left'] = batch['observation.images.hand']
    batch['observation.images.hand_right'] = None
    batch['observation.images.hand_left_is_pad'] = batch['observation.images.hand_is_pad']
    batch['observation.images.hand_right_is_pad'] = None
    del batch['observation.images.hand']
    del batch['observation.images.hand_is_pad']
    return batch

def load_agibot_world_dataset(
    root: str,
    canonical_action_chunk_size: int,
    num_frames: int = 1,
):
    return load_dataset(
        root,
        ("actions.joint.position", "actions.effector.position"),
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = agibot_world_to_nora_instance,
        norm_stats_transform = functools.partial(
            merge_norm_stats,
            merge_prefix = 'actions',
            merge_spec = MERGE_SPECS['agibot_world'],
        ),
        num_frames = num_frames
    )

def load_galaxea_dataset(
    root: str,
    canonical_action_chunk_size: int,
    num_frames: int = 1,
):
    assert canonical_action_chunk_size % 2 == 0
    return load_dataset(
        root,
        ("action.left_arm", "action.left_gripper", "action.right_arm", "action.right_gripper"),
        canonical_action_chunk_size // 2,
        canonical_action_chunk_size,
        raw_fps = 15,
        instance_transform = galaxea_to_nora_instance,
        norm_stats_transform = functools.partial(
            merge_norm_stats,
            merge_prefix = 'action',
            merge_spec = MERGE_SPECS['galaxea'],
        ),
        num_frames = num_frames
    )

def load_interndata_a1_dataset(
    root: str | pathlib.Path,
    canonical_action_chunk_size: int,
    num_frames: int = 1,
):
    root = pathlib.Path(root)

    franka_datasets = [
        load_dataset(
            root / f'franka-{i}',
            ("actions.joint.position", "actions.gripper.position"),
            canonical_action_chunk_size,
            canonical_action_chunk_size,
            raw_fps = 30,
            instance_transform = interndata_a1_franka_to_nora_instance,
            norm_stats_transform = functools.partial(
                merge_norm_stats,
                merge_prefix = 'actions',
                merge_spec = MERGE_SPECS['interndata_a1_franka'],
            ),
            num_frames = num_frames
        )
        for i in ('1', '2')
    ]
    dual_arm_action_keys = (
        "actions.left_joint.position",
        "actions.left_gripper.position",
        "actions.right_joint.position",
        "actions.right_gripper.position",
    )
    genie1_dataset = load_dataset(
        root / 'genie1',
        dual_arm_action_keys,
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = interndata_a1_genie1_to_nora_instance,
        norm_stats_transform = functools.partial(
            merge_norm_stats,
            merge_prefix = 'actions',
            merge_spec = MERGE_SPECS['interndata_a1_genie1'],
        ),
        num_frames = num_frames
    )
    dual_arm_6dof_norm_stats_transform = functools.partial(
        merge_norm_stats,
        merge_prefix = 'actions',
        merge_spec = MERGE_SPECS['interndata_a1_dual_arm_6dof'],
    )
    lift2_dataset = load_dataset(
        root / 'lift2',
        dual_arm_action_keys,
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = interndata_a1_lift2_to_nora_instance,
        norm_stats_transform = dual_arm_6dof_norm_stats_transform,
        num_frames = num_frames
    )
    split_aloha_dataset = load_dataset(
        root / 'split_aloha',
        dual_arm_action_keys,
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = interndata_a1_split_aloha_to_nora_instance,
        norm_stats_transform = dual_arm_6dof_norm_stats_transform,
        num_frames = num_frames
    )

def load_math_reasoning_datasets(samples_per_dataset: int = 50):
    """
    Loads and standardizes mathematical reasoning datasets from Hugging Face.
    Converts various dataset formats into a unified schema:
    {'image': PIL.Image, 'instruction': str, 'text_answer': str, 'task_type': 'vl_math'}
    """
    standardized_datasets = []

    def standardize_format(example, img_col, inst_col, ans_col):
        img_data = example.get(img_col)
        image = None
        
        try:
            if img_data is not None:
                if isinstance(img_data, list):
                    img_data = img_data[0] if len(img_data) > 0 else None
                
                if hasattr(img_data, "convert"):
                    image = img_data.convert('RGB')
        except Exception:
            image = None

        return {
            "image": image,
            "instruction": str(example.get(inst_col, "")),
            "text_answer": str(example.get(ans_col, "")),
            "task_type": "vl_math"
        }

    # Helper function to cast the image column explicitly
    def cast_image_feature(dataset):
        if len(dataset) > 0:
            return dataset.cast_column("image", HFImage())
        return dataset

    # 1. MathVision
    try:
        mathvision = hf_load_dataset("MathLLMs/MathVision", split="test") 
        mathvision = mathvision.map(
            lambda x: standardize_format(x, "image", "question", "answer"),
            remove_columns=mathvision.column_names
        ).filter(lambda x: x["image"] is not None)
        
        mathvision = mathvision.select(range(min(samples_per_dataset, len(mathvision))))
        mathvision = cast_image_feature(mathvision)
        standardized_datasets.append(mathvision)
    except Exception as e:
        print(f"Skipping MathVision: {e}")

    # 2. MathVista
    try:
        mathvista = hf_load_dataset("AI4Math/MathVista", split="testmini")
        mathvista = mathvista.map(
            lambda x: standardize_format(x, "decoded_image", "query", "answer"),
            remove_columns=mathvista.column_names
        ).filter(lambda x: x["image"] is not None)
        
        mathvista = mathvista.select(range(min(samples_per_dataset, len(mathvista))))
        mathvista = cast_image_feature(mathvista)
        standardized_datasets.append(mathvista)
    except Exception as e:
        print(f"Skipping MathVista: {e}")

    # 3. CLEVR-Math
    try:
        clevr = hf_load_dataset("WaltonFuture/clevr-math", split="train")
        clevr = clevr.map(
            lambda x: standardize_format(x, "images", "problem", "answer"),
            remove_columns=clevr.column_names
        ).filter(lambda x: x["image"] is not None)
        
        clevr = clevr.select(range(min(samples_per_dataset, len(clevr))))
        clevr = cast_image_feature(clevr)
        standardized_datasets.append(clevr)
    except Exception as e:
        print(f"Skipping CLEVR-Math: {e}")

    # 4. GEOQA_8K_R1V
    try:
        geoqa = hf_load_dataset("leonardPKU/GEOQA_8K_R1V", split="train")
        geoqa = geoqa.map(
            lambda x: standardize_format(x, "images", "problem", "answer"),
            remove_columns=geoqa.column_names
        ).filter(lambda x: x["image"] is not None)
        
        geoqa = geoqa.select(range(min(samples_per_dataset, len(geoqa))))
        geoqa = cast_image_feature(geoqa)
        standardized_datasets.append(geoqa)
    except Exception as e:
        print(f"Skipping GEOQA: {e}")

    if not standardized_datasets:
        raise RuntimeError("Failed to load any math reasoning datasets.")
        
    mixed_math_dataset = concatenate_datasets(standardized_datasets)
    mixed_math_dataset = mixed_math_dataset.shuffle(seed=42)
    return mixed_math_dataset
