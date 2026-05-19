import functools
import pathlib
from typing import Any, Literal, NamedTuple, Sequence
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from torch.utils.data import ConcatDataset
from utils.data_loading import load_dataset
import numpy as np

CANONICAL_ACTION_DIMS = 32

ActionFeatureType = Literal['arm_joints', 'eef_pose', 'gripper_joints']
ActionFeatureSlice = slice[int | None, int, None] | Literal['se3_matrix']

class ActionTensorSegmentSpec(NamedTuple):
    feature_name: str
    feature_slice: ActionFeatureSlice
    feature_type: ActionFeatureType

ActionTensorSpec = Sequence[ActionTensorSegmentSpec]

ACTION_TENSOR_SPECS = {
    'agibot_world': (
        ActionTensorSegmentSpec('joint.position', slice(0, 7), 'arm_joints'),
        ActionTensorSegmentSpec('effector.position', slice(0, 1), 'gripper_joints'),
        ActionTensorSegmentSpec('joint.position', slice(7, 14), 'arm_joints'),
        ActionTensorSegmentSpec('effector.position', slice(1, 2), 'gripper_joints'),
    ),
    'galaxea': (
        ActionTensorSegmentSpec('left_arm', slice(0, 6), 'arm_joints'),
        ActionTensorSegmentSpec('left_gripper', slice(0, 1), 'gripper_joints'),
        ActionTensorSegmentSpec('right_arm', slice(0, 6), 'arm_joints'),
        ActionTensorSegmentSpec('right_gripper', slice(0, 1), 'gripper_joints'),
    ),
    'interndata_a1_franka': (
        ActionTensorSegmentSpec('joint.position', slice(0, 7), 'arm_joints'),
        ActionTensorSegmentSpec('gripper.position', slice(0, 1), 'gripper_joints'),
    ),
    'interndata_a1_genie1': (
        ActionTensorSegmentSpec('left_joint.position', slice(0, 7), 'arm_joints'),
        ActionTensorSegmentSpec('left_gripper.position', slice(0, 1), 'gripper_joints'),
        ActionTensorSegmentSpec('right_joint.position', slice(0, 7), 'arm_joints'),
        ActionTensorSegmentSpec('right_gripper.position', slice(0, 1), 'gripper_joints'),
    ),
    'interndata_a1_dual_arm_6dof': (
        ActionTensorSegmentSpec('left_joint.position', slice(0, 6), 'arm_joints'),
        ActionTensorSegmentSpec('left_gripper.position', slice(0, 1), 'gripper_joints'),
        ActionTensorSegmentSpec('right_joint.position', slice(0, 6), 'arm_joints'),
        ActionTensorSegmentSpec('right_gripper.position', slice(0, 1), 'gripper_joints'),
    ),
    'egodex': (
        ActionTensorSegmentSpec('leftHand', 'se3_matrix', 'eef_pose'),
        ActionTensorSegmentSpec('leftFingers', slice(0, 7), 'gripper_joints'),
        ActionTensorSegmentSpec('rightHand', 'se3_matrix', 'eef_pose'),
        ActionTensorSegmentSpec('rightFingers', slice(0, 7), 'gripper_joints'),
    ),
}


def _get_delta_transform_mask_segment_dim(action_tensor_segment_spec: ActionTensorSegmentSpec) -> int:
    feature_slice = action_tensor_segment_spec.feature_slice
    if feature_slice == 'se3_matrix':
        return 16
    if isinstance(feature_slice, slice):
        return feature_slice.stop - (feature_slice.start or 0)
    raise ValueError(f"Invalid action tensor spec item: {action_tensor_segment_spec}")

@functools.cache
def build_delta_transform_mask(action_tensor_spec: ActionTensorSpec) -> torch.Tensor:
    parts = [
        torch.full(
            (_get_delta_transform_mask_segment_dim(segment_spec),),
            segment_spec.feature_type in ('arm_joints', 'eef_pose'),
            dtype=torch.bool,
        )
        for segment_spec in action_tensor_spec
    ]
    return torch.cat(parts)

def _agibot_subtask_from_meta(
    meta: LeRobotDatasetMetadata,
    episode_index: int,
    frame_index: int,
) -> str:
    ep_row = meta.episodes[episode_index]
    segments = ep_row.get("action_config")
    if not segments:
        return ""
    for seg in segments:
        sf, ef = seg.get("start_frame"), seg.get("end_frame")
        if sf is None or ef is None:
            continue
        if int(sf) <= frame_index < int(ef):
            return seg.get("action_text") or ""
    return ""

def _make_merge_segment(
    inst: dict[str, Any],
    merge_prefix: str,
    segment_spec: ActionTensorSegmentSpec,
    leading_dims: tuple[int, ...],
) -> torch.Tensor:
    feature_slice = segment_spec.feature_slice
    feature = inst[f"{merge_prefix}.{segment_spec.feature_name}"]
    if isinstance(feature_slice, slice):
        return feature[..., feature_slice]
    if feature_slice == 'se3_matrix':
        # flatten based on leading dimensions, this can handle
        # both the SE(3) matrix form (raw action / state) and the XYZ and rot6d form (norm stats)
        return feature.view(*leading_dims, -1)
    raise ValueError(f"Invalid action tensor spec item: {segment_spec}")

def merge_features(
    inst: dict[str, Any],
    merge_prefix: str,
    action_tensor_spec: ActionTensorSpec,
    merged_feature_key: str | None = None,
):
    """
    Merge features from the instance dictionary into a single feature.
    Feature keys starting with `merge_prefix` are removed from the instance dictionary.
    If `merged_feature_key` is not provided, the merged feature key is the same as the `merge_prefix`.

    The merge output is a single tensor with the concatenated features dimensions.
    """
    first_segment_spec = action_tensor_spec[0]
    first_feature_tensor = inst[f"{merge_prefix}.{first_segment_spec.feature_name}"]
    leading_dims = first_feature_tensor.shape[:-1 if first_segment_spec.feature_slice != 'se3_matrix' else -2]
    to_cat = [
        _make_merge_segment(inst, merge_prefix, segment_spec, leading_dims)
        for segment_spec in action_tensor_spec
    ]

    new_inst = {k: v for k, v in inst.items() if not k.startswith(merge_prefix + '.')}
    new_inst[merged_feature_key or merge_prefix] = torch.cat(to_cat, dim = -1)
    return new_inst

def merge_norm_stats(
    norm_stats: dict[str, dict[str, np.ndarray]],
    merge_prefix: str,
    action_tensor_spec: ActionTensorSpec,
) -> dict[str, dict[str, np.ndarray]]:
    norm_stats = {
        'q01': {feat: torch.from_numpy(norm_stats[feat]['q01']).view(-1) for feat in norm_stats},
        'q99': {feat: torch.from_numpy(norm_stats[feat]['q99']).view(-1) for feat in norm_stats},
    }
    merged = {
        'q01': merge_features(norm_stats['q01'], merge_prefix, action_tensor_spec, 'action'),
        'q99': merge_features(norm_stats['q99'], merge_prefix, action_tensor_spec, 'action'),
    }
    merged = {
        feat: {'q01': merged['q01'][feat].numpy(), 'q99': merged['q99'][feat].numpy()}
        for feat in merged['q01']
    }
    return merged

def generic_to_nora_instance(
    batch: dict[str, Any],
    action_tensor_spec: ActionTensorSpec,
    action_prefix: str,
    state_prefix: str,
    embodiment_prompt: str,
):
    batch = merge_features(batch, action_prefix, action_tensor_spec, 'action')
    batch = merge_features(batch, state_prefix, action_tensor_spec, 'observation.state')
    batch['info'] = {"embodiment_prompt": embodiment_prompt}
    if 'subtask' not in batch:
        batch['subtask'] = ""
    return batch

def agibot_world_to_nora_instance(
    batch: dict[str, Any],
    *,
    meta: LeRobotDatasetMetadata,
    task_config: object = None,
):
    """
    Convert from raw AgiBot World dataset format to format that is ready to be converted to `EnvTransition`:
    - Merge relevant actions into `action`, discarding other actions.
    - Merge relevant states into `observation.state`, discarding other states.
    - Invert the gripper action by 1-x.
    - Subtask from episode `action_config` when the current frame falls in a segment.
    """
    batch['actions.effector.position'] = 1 - batch['actions.effector.position']
    ep_idx = batch['episode_index'].item()
    frame_idx = batch['frame_index'].item()
    batch['subtask'] = _agibot_subtask_from_meta(meta, ep_idx, frame_idx)
    return generic_to_nora_instance(
        batch,
        action_tensor_spec = ACTION_TENSOR_SPECS['agibot_world'],
        action_prefix = 'actions',
        state_prefix = 'observation.states',
        embodiment_prompt = "AgiBot G1 with 2 grippers",
    )

def galaxea_to_nora_instance(
    batch: dict[str, Any],
    *,
    meta: LeRobotDatasetMetadata,
    task_config: dict[str, Any] | None = None,
):
    """
    Convert from raw Galaxea Open World Dataset format to format that is ready to be converted to `EnvTransition`:
    - Merge relevant actions into `action`, discarding other actions.
    - Merge relevant states into `observation.state`, discarding other states.
    - Conditional subtask setup based on task config.
    """
    batch['observation.state.left_gripper'] = batch['observation.state.left_gripper'].view(-1)
    batch['observation.state.right_gripper'] = batch['observation.state.right_gripper'].view(-1)
    batch['action.left_gripper'] = batch['action.left_gripper'].view(-1, 1)
    batch['action.right_gripper'] = batch['action.right_gripper'].view(-1, 1)

    task_config = task_config or {}
    fine_task = batch['task'].split('@')[-1]    # keep the English part ("[chinese task]@[english task]")
    if fine_task == 'null':
        fine_task = ""
    if task_config.get('coarse_task_as_main_task'):
        coarse_task = task_config.get('rename_coarse_task')
        if not coarse_task:
            coarse_task_idx = batch['coarse_task_index'].item()
            coarse_task = meta.tasks.iloc[coarse_task_idx].name
        batch['task'] = coarse_task
        batch['subtask'] = fine_task
    else:
        batch['task'] = fine_task
        batch['subtask'] = ""

    batch = generic_to_nora_instance(
        batch,
        action_tensor_spec = ACTION_TENSOR_SPECS['galaxea'],
        action_prefix = 'action',
        state_prefix = 'observation.state',
        embodiment_prompt = "Galaxea R1 Lite",
    )
    # rename image keys, drop right head image
    batch['observation.images.head'] = batch['observation.images.head_rgb']
    batch['observation.images.hand_left'] = batch['observation.images.left_wrist_rgb']
    batch['observation.images.hand_right'] = batch['observation.images.right_wrist_rgb']
    del batch['observation.images.head_rgb']
    del batch['observation.images.head_right_rgb']
    del batch['observation.images.left_wrist_rgb']
    del batch['observation.images.right_wrist_rgb']
    if 'observation.images.head_rgb_is_pad' in batch:
        batch['observation.images.head_is_pad'] = batch['observation.images.head_rgb_is_pad']
        batch['observation.images.hand_left_is_pad'] = batch['observation.images.left_wrist_rgb_is_pad']
        batch['observation.images.hand_right_is_pad'] = batch['observation.images.right_wrist_rgb_is_pad']
        del batch['observation.images.head_rgb_is_pad']
        del batch['observation.images.head_right_rgb_is_pad']
        del batch['observation.images.left_wrist_rgb_is_pad']
        del batch['observation.images.right_wrist_rgb_is_pad']
    return batch

def interndata_a1_to_nora_instance(
    batch: dict[str, Any],
    action_tensor_spec: ActionTensorSpec,
    embodiment_prompt: str,
    *,
    meta: object = None,
    task_config: object = None,
):
    for key in batch:
        if key.startswith('states.') and key.endswith('gripper.position'):
            batch[key] = batch[key].view(-1)
        elif key.startswith('actions.') and key.endswith('gripper.position'):
            batch[key] = batch[key].view(-1, 1)

    batch = generic_to_nora_instance(
        batch,
        action_tensor_spec = action_tensor_spec,
        action_prefix = 'actions',
        state_prefix = 'states',
        embodiment_prompt = embodiment_prompt,
    )
    batch = {k.replace('images.rgb.', 'observation.images.'): v for k, v in batch.items()}
    return batch

interndata_a1_genie1_to_nora_instance = functools.partial(
    interndata_a1_to_nora_instance,
    action_tensor_spec = ACTION_TENSOR_SPECS['interndata_a1_genie1'],
    embodiment_prompt = "InternData-A1 simulated Genie1 with 2 grippers",
)
interndata_a1_lift2_to_nora_instance = functools.partial(
    interndata_a1_to_nora_instance,
    action_tensor_spec = ACTION_TENSOR_SPECS['interndata_a1_dual_arm_6dof'],
    embodiment_prompt = "InternData-A1 simulated Lift-2 with R5a arms",
)
interndata_a1_split_aloha_to_nora_instance = functools.partial(
    interndata_a1_to_nora_instance,
    action_tensor_spec = ACTION_TENSOR_SPECS['interndata_a1_dual_arm_6dof'],
    embodiment_prompt = "InternData-A1 simulated Split Aloha with 2 Piper-100 arms",
)

def interndata_a1_franka_to_nora_instance(
    batch: dict[str, Any],
    *,
    meta: object = None,
    task_config: object = None,
):
    batch = interndata_a1_to_nora_instance(
        batch,
        action_tensor_spec = ACTION_TENSOR_SPECS['interndata_a1_franka'],
        embodiment_prompt = "InternData-A1 simulated Franka Emika Panda",
    )
    batch['observation.images.hand_left'] = batch['observation.images.hand']
    batch['observation.images.hand_right'] = None
    del batch['observation.images.hand']
    if 'observation.images.hand_is_pad' in batch:
        batch['observation.images.hand_left_is_pad'] = batch['observation.images.hand_is_pad']
        batch['observation.images.hand_right_is_pad'] = None
        del batch['observation.images.hand_is_pad']
    return batch

def egodex_to_nora_instance(batch: dict[str, Any]):
    batch = generic_to_nora_instance(
        batch,
        action_tensor_spec = ACTION_TENSOR_SPECS['egodex'],
        action_prefix = 'action',
        state_prefix = 'observation.state',
        embodiment_prompt = "simplified real human hands (from Apple Vision Pro tracking)",
    )
    batch['observation.images.head'] = batch['observation.images.camera']
    batch['observation.images.hand_left'] = None
    batch['observation.images.hand_right'] = None
    batch['observation.images.head_is_pad'] = batch['observation.images.camera_is_pad']
    batch['observation.images.hand_left_is_pad'] = None
    batch['observation.images.hand_right_is_pad'] = None
    del batch['observation.images.camera']
    del batch['observation.images.camera_is_pad']
    return batch

def load_agibot_world_dataset(
    root: str,
    canonical_action_chunk_size: int,
    num_frames: int = 1,
):
    action_tensor_spec = ACTION_TENSOR_SPECS['agibot_world']
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
            action_tensor_spec = action_tensor_spec,
        ),
        delta_transform_mask = build_delta_transform_mask(action_tensor_spec),
        target_action_dim = CANONICAL_ACTION_DIMS,
        num_frames = num_frames,
    )

def load_galaxea_dataset(
    root: str,
    canonical_action_chunk_size: int,
    num_frames: int = 1,
):
    assert canonical_action_chunk_size % 2 == 0
    action_tensor_spec = ACTION_TENSOR_SPECS['galaxea']
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
            action_tensor_spec = action_tensor_spec,
        ),
        delta_transform_mask = build_delta_transform_mask(action_tensor_spec),
        target_action_dim = CANONICAL_ACTION_DIMS,
        num_frames = num_frames,
    )

def load_interndata_a1_dataset(
    root: str | pathlib.Path,
    canonical_action_chunk_size: int,
    num_frames: int = 1,
):
    root = pathlib.Path(root)
    franka_action_tensor_spec = ACTION_TENSOR_SPECS['interndata_a1_franka']
    franka_delta_mask = build_delta_transform_mask(franka_action_tensor_spec)

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
                action_tensor_spec = franka_action_tensor_spec,
            ),
            delta_transform_mask = franka_delta_mask,
            target_action_dim = CANONICAL_ACTION_DIMS,
            num_frames = num_frames,
        )
        for i in ('1', '2')
    ]
    dual_arm_action_keys = (
        "actions.left_joint.position",
        "actions.left_gripper.position",
        "actions.right_joint.position",
        "actions.right_gripper.position",
    )
    genie1_action_tensor_spec = ACTION_TENSOR_SPECS['interndata_a1_genie1']
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
            action_tensor_spec = genie1_action_tensor_spec,
        ),
        delta_transform_mask = build_delta_transform_mask(genie1_action_tensor_spec),
        target_action_dim = CANONICAL_ACTION_DIMS,
        num_frames = num_frames,
    )
    dual_arm_6dof_action_tensor_spec = ACTION_TENSOR_SPECS['interndata_a1_dual_arm_6dof']
    dual_arm_6dof_delta_mask = build_delta_transform_mask(dual_arm_6dof_action_tensor_spec)
    dual_arm_6dof_norm_stats_transform = functools.partial(
        merge_norm_stats,
        merge_prefix = 'actions',
        action_tensor_spec = dual_arm_6dof_action_tensor_spec,
    )
    lift2_dataset = load_dataset(
        root / 'lift2',
        dual_arm_action_keys,
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = interndata_a1_lift2_to_nora_instance,
        norm_stats_transform = dual_arm_6dof_norm_stats_transform,
        delta_transform_mask = dual_arm_6dof_delta_mask,
        target_action_dim = CANONICAL_ACTION_DIMS,
        num_frames = num_frames,
    )
    split_aloha_dataset = load_dataset(
        root / 'split_aloha',
        dual_arm_action_keys,
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = interndata_a1_split_aloha_to_nora_instance,
        norm_stats_transform = dual_arm_6dof_norm_stats_transform,
        delta_transform_mask = dual_arm_6dof_delta_mask,
        target_action_dim = CANONICAL_ACTION_DIMS,
        num_frames = num_frames,
    )
    return ConcatDataset([*franka_datasets, genie1_dataset, lift2_dataset, split_aloha_dataset])

def load_egodex_dataset(
    root: str | pathlib.Path,
    canonical_action_chunk_size: int,
    num_frames: int = 1,
):
    action_tensor_spec = ACTION_TENSOR_SPECS['egodex']
    return load_dataset(
        root,
        ("action.leftHand", "action.rightHand", "action.leftFingers", "action.rightFingers"),
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = egodex_to_nora_instance,
        norm_stats_transform = functools.partial(
            merge_norm_stats,
            merge_prefix = 'action',
            action_tensor_spec = action_tensor_spec,
        ),
        se3_segment_start_idxs = frozenset((0, 24)),
        delta_transform_mask = build_delta_transform_mask(action_tensor_spec),
        target_action_dim = CANONICAL_ACTION_DIMS,
        num_frames = num_frames,
    )
