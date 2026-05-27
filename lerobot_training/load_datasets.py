import functools
import pathlib
from typing import Any, Literal, NamedTuple, Sequence
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from torch.utils.data import ConcatDataset
from utils.data_loading import load_dataset
from utils.se3 import position_quaternion_xyzw_to_se3, pose_xyz_wxyz_to_se3
import numpy as np

CANONICAL_ACTION_DIMS = 32

ActionFeatureType = Literal['arm_joints', 'eef_pose', 'gripper_joints', 'base_velocity']
ActionFeatureSlice = slice | Literal['se3_matrix']

class ActionTensorSegmentSpec(NamedTuple):
    feature_name: str
    feature_slice: ActionFeatureSlice
    feature_type: ActionFeatureType

ActionTensorSpec = Sequence[ActionTensorSegmentSpec]

ACTION_TENSOR_SPECS = {
    'agibot_world': (
        ActionTensorSegmentSpec('robot.velocity', slice(0, 2), 'base_velocity'),
        ActionTensorSegmentSpec('joint.position', slice(0, 7), 'arm_joints'),
        ActionTensorSegmentSpec('left_eef_pose', 'se3_matrix', 'eef_pose'),
        ActionTensorSegmentSpec('effector.position', slice(0, 1), 'gripper_joints'),
        ActionTensorSegmentSpec('joint.position', slice(7, 14), 'arm_joints'),
        ActionTensorSegmentSpec('right_eef_pose', 'se3_matrix', 'eef_pose'),
        ActionTensorSegmentSpec('effector.position', slice(1, 2), 'gripper_joints'),
    ),
    'galaxea': (
        ActionTensorSegmentSpec('chassis.velocities', slice(0, 3), 'base_velocity'),
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
        ActionTensorSegmentSpec('left_eef_pose', 'se3_matrix', 'eef_pose'),
        ActionTensorSegmentSpec('left_gripper.position', slice(0, 1), 'gripper_joints'),
        ActionTensorSegmentSpec('right_joint.position', slice(0, 7), 'arm_joints'),
        ActionTensorSegmentSpec('right_eef_pose', 'se3_matrix', 'eef_pose'),
        ActionTensorSegmentSpec('right_gripper.position', slice(0, 1), 'gripper_joints'),
    ),
    'interndata_a1_dual_arm_6dof': (
        ActionTensorSegmentSpec('left_joint.position', slice(0, 6), 'arm_joints'),
        ActionTensorSegmentSpec('left_eef_pose', 'se3_matrix', 'eef_pose'),
        ActionTensorSegmentSpec('left_gripper.position', slice(0, 1), 'gripper_joints'),
        ActionTensorSegmentSpec('right_joint.position', slice(0, 6), 'arm_joints'),
        ActionTensorSegmentSpec('right_eef_pose', 'se3_matrix', 'eef_pose'),
        ActionTensorSegmentSpec('right_gripper.position', slice(0, 1), 'gripper_joints'),
    ),
    'egodex': (
        ActionTensorSegmentSpec('leftHand', 'se3_matrix', 'eef_pose'),
        ActionTensorSegmentSpec('leftFingers', slice(0, 7), 'gripper_joints'),
        ActionTensorSegmentSpec('rightHand', 'se3_matrix', 'eef_pose'),
        ActionTensorSegmentSpec('rightFingers', slice(0, 7), 'gripper_joints'),
    ),
    'droid': (
        ActionTensorSegmentSpec('action_all', slice(0, 7), 'arm_joints'),
        ActionTensorSegmentSpec('action_all', slice(7, 8), 'gripper_joints'),
    ),
}

def _get_action_tensor_segment_dim(action_tensor_segment_spec: ActionTensorSegmentSpec, se3_dim: int) -> int:
    feature_slice = action_tensor_segment_spec.feature_slice
    if feature_slice == 'se3_matrix':
        return se3_dim
    if isinstance(feature_slice, slice):
        return feature_slice.stop - (feature_slice.start or 0)
    raise ValueError(f"Invalid action tensor spec item: {action_tensor_segment_spec}")

def build_delta_transform_mask(action_tensor_spec: ActionTensorSpec) -> torch.Tensor:
    parts = [
        torch.full(
            (_get_action_tensor_segment_dim(segment_spec, 16),),
            segment_spec.feature_type in ('arm_joints', 'eef_pose'),
            dtype=torch.bool,
        )
        for segment_spec in action_tensor_spec
    ]
    return torch.cat(parts)

def build_se3_segment_start_idxs(action_tensor_spec: ActionTensorSpec) -> frozenset[int]:
    start_idxs = set()
    start_idx = 0
    for segment_spec in action_tensor_spec:
        if segment_spec.feature_slice == 'se3_matrix':
            start_idxs.add(start_idx)
        start_idx += _get_action_tensor_segment_dim(segment_spec, 16)
    return frozenset(start_idxs)

def build_arm_control_mode_masks(action_tensor_spec: ActionTensorSpec) -> tuple[torch.Tensor, torch.Tensor]:
    joint_position_mask_parts = []
    eef_pose_mask_parts = []
    for segment_spec in action_tensor_spec:
        dim = _get_action_tensor_segment_dim(segment_spec, 9)
        always_include = segment_spec.feature_type in ('gripper_joints', 'base_velocity')
        joint_position_mask_parts.append(
            torch.full((dim,), segment_spec.feature_type == 'arm_joints' or always_include, dtype=torch.bool)
        )
        eef_pose_mask_parts.append(
            torch.full((dim,), segment_spec.feature_type == 'eef_pose' or always_include, dtype=torch.bool)
        )
    return torch.cat(joint_position_mask_parts), torch.cat(eef_pose_mask_parts)

def _add_agibot_eef_pose_features(batch: dict[str, Any], prefix: str) -> None:
    position = batch[f'{prefix}.end.position']
    orientation = batch[f'{prefix}.end.orientation']
    batch[f'{prefix}.left_eef_pose'] = position_quaternion_xyzw_to_se3(
        position[..., 0, :],
        orientation[..., 0, :],
    )
    batch[f'{prefix}.right_eef_pose'] = position_quaternion_xyzw_to_se3(
        position[..., 1, :],
        orientation[..., 1, :],
    )

def _add_interndata_a1_eef_pose_features(batch: dict[str, Any], prefix: str) -> None:
    batch[f'{prefix}.left_eef_pose'] = pose_xyz_wxyz_to_se3(
        batch[f'{prefix}.left_ee_to_robot_pose']
    )
    batch[f'{prefix}.right_eef_pose'] = pose_xyz_wxyz_to_se3(
        batch[f'{prefix}.right_ee_to_robot_pose']
    )

def _make_se3_features_relative_to(
    batch: dict[str, Any],
    reference: torch.Tensor,
    feature_keys: Sequence[str],
) -> None:
    reference_inv = reference.inverse()
    for feature_key in feature_keys:
        batch[feature_key] = reference_inv.matmul(batch[feature_key])

def _add_arm_control_mode(batch: dict[str, Any], arm_control_mode: str | None) -> None:
    if arm_control_mode is not None:
        info = batch.get('info') or {}
        info['arm_control_mode'] = arm_control_mode
        batch['info'] = info

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
    zero_fill_feature_types: frozenset[ActionFeatureType] = frozenset(),
    reference_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    feature_slice = segment_spec.feature_slice
    if segment_spec.feature_type in zero_fill_feature_types:
        return torch.zeros(
            *leading_dims,
            _get_action_tensor_segment_dim(segment_spec, 16),
            dtype=reference_tensor.dtype,
            device=reference_tensor.device,
        )
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
    zero_fill_feature_types: frozenset[ActionFeatureType] = frozenset(),
):
    """
    Merge features from the instance dictionary into a single feature.
    Feature keys starting with `merge_prefix` are removed from the instance dictionary.
    If `merged_feature_key` is not provided, the merged feature key is the same as the `merge_prefix`.

    The merge output is a single tensor with the concatenated features dimensions.
    """
    first_segment_spec = next(
        segment_spec
        for segment_spec in action_tensor_spec
        if segment_spec.feature_type not in zero_fill_feature_types
    )
    first_feature_tensor = inst[f"{merge_prefix}.{first_segment_spec.feature_name}"]
    leading_dims = first_feature_tensor.shape[:-1 if first_segment_spec.feature_slice != 'se3_matrix' else -2]
    to_cat = [
        _make_merge_segment(
            inst,
            merge_prefix,
            segment_spec,
            leading_dims,
            zero_fill_feature_types,
            first_feature_tensor,
        )
        for segment_spec in action_tensor_spec
    ]

    new_inst = {k: v for k, v in inst.items() if not k.startswith(merge_prefix + '.')}
    new_inst[merged_feature_key or merge_prefix] = torch.cat(to_cat, dim = -1)
    return new_inst

def merge_norm_stats(
    norm_stats: dict[str, dict[str, np.ndarray]],
    action_prefix: str,
    state_prefix: str,
    action_tensor_spec: ActionTensorSpec,
) -> dict[str, dict[str, np.ndarray]]:
    norm_stats = {
        'q01': {feat: torch.from_numpy(norm_stats[feat]['q01']).view(-1) for feat in norm_stats},
        'q99': {feat: torch.from_numpy(norm_stats[feat]['q99']).view(-1) for feat in norm_stats},
    }
    # merge actions
    merged = {
        stat_name: merge_features(stat, action_prefix, action_tensor_spec, 'action')
        for stat_name, stat in norm_stats.items()
    }
    # merge states
    merged = {
        stat_name: merge_features(
            stat,
            state_prefix,
            action_tensor_spec,
            'observation.state',
            zero_fill_feature_types=frozenset(('base_velocity',)),
        )
        for stat_name, stat in merged.items()
    }
    # patch state norm stats for base velocity which should normalize to 0
    zero_fill_state_mask = torch.cat([
        torch.full(
            (_get_action_tensor_segment_dim(segment_spec, 9),),
            segment_spec.feature_type == 'base_velocity',
            dtype=torch.bool,
        )
        for segment_spec in action_tensor_spec
    ])
    if zero_fill_state_mask.any():
        merged['q01']['observation.state'][zero_fill_state_mask] = -1
        merged['q99']['observation.state'][zero_fill_state_mask] = 1
    # transpose back to original format
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
    arm_control_mode: str | None = None,
):
    batch = merge_features(batch, action_prefix, action_tensor_spec, 'action')
    batch = merge_features(
        batch,
        state_prefix,
        action_tensor_spec,
        'observation.state',
        zero_fill_feature_types=frozenset(('base_velocity',)),
    )
    batch['info'] = {"embodiment_prompt": embodiment_prompt}
    _add_arm_control_mode(batch, arm_control_mode)
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
    _add_agibot_eef_pose_features(batch, 'actions')
    _add_agibot_eef_pose_features(batch, 'observation.states')
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
        arm_control_mode = 'joint_position',
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
    arm_control_mode: str | None = None,
    *,
    meta: object = None,
    task_config: object = None,
):
    for key in batch:
        if key.startswith('states.') and key.endswith('gripper.position'):
            batch[key] = batch[key].view(-1)
        elif key.startswith('actions.') and key.endswith('gripper.position'):
            batch[key] = batch[key].view(-1, 1)

    if any(segment.feature_type == 'eef_pose' for segment in action_tensor_spec):
        _add_interndata_a1_eef_pose_features(batch, 'actions')
        _add_interndata_a1_eef_pose_features(batch, 'states')
        head_pose = pose_xyz_wxyz_to_se3(batch['head_camera_to_robot_extrinsics'])
        _make_se3_features_relative_to(
            batch,
            head_pose,
            (
                'actions.left_eef_pose',
                'actions.right_eef_pose',
                'states.left_eef_pose',
                'states.right_eef_pose',
            ),
        )


    batch = generic_to_nora_instance(
        batch,
        action_tensor_spec = action_tensor_spec,
        action_prefix = 'actions',
        state_prefix = 'states',
        embodiment_prompt = embodiment_prompt,
        arm_control_mode = arm_control_mode,
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
        arm_control_mode = 'joint_position',
    )
    batch['observation.images.hand_left'] = batch['observation.images.hand']
    batch['observation.images.hand_right'] = None
    del batch['observation.images.hand']
    if 'observation.images.hand_is_pad' in batch:
        batch['observation.images.hand_left_is_pad'] = batch['observation.images.hand_is_pad']
        batch['observation.images.hand_right_is_pad'] = None
        del batch['observation.images.hand_is_pad']
    return batch

def egodex_to_nora_instance(batch: dict[str, Any], *, meta: object = None, task_config: object = None):
    _make_se3_features_relative_to(
        batch,
        batch['observation.state.camera'],
        (
            'action.leftHand',
            'action.rightHand',
            'observation.state.leftHand',
            'observation.state.rightHand',
        ),
    )
    batch = generic_to_nora_instance(
        batch,
        action_tensor_spec = ACTION_TENSOR_SPECS['egodex'],
        action_prefix = 'action',
        state_prefix = 'observation.state',
        embodiment_prompt = "simplified real human hands (from Apple Vision Pro tracking)",
        arm_control_mode = 'eef_pose',
    )
    batch['observation.images.head'] = batch['observation.images.camera']
    batch['observation.images.hand_left'] = None
    batch['observation.images.hand_right'] = None
    del batch['observation.images.camera']
    if 'observation.images.camera_is_pad' in batch:
        batch['observation.images.head_is_pad'] = batch['observation.images.camera_is_pad']
        batch['observation.images.hand_left_is_pad'] = None
        batch['observation.images.hand_right_is_pad'] = None
        del batch['observation.images.camera_is_pad']
    return batch

def droid_to_nora_instance(
    batch: dict[str, Any],
    *,
    meta: object = None,
    task_config: object = None,
):
    # 1. Map flat keys to dummy nested keys
    if 'action' in batch:
        batch['droid_actions.action_all'] = batch.pop('action')
    if 'observation.state' in batch:
        batch['droid_states.action_all'] = batch.pop('observation.state')

    batch = generic_to_nora_instance(
        batch,
        action_tensor_spec = ACTION_TENSOR_SPECS['droid'],
        action_prefix = 'droid_actions',
        state_prefix = 'droid_states',
        embodiment_prompt = "DROID platform with Franka Emika Panda, 1 gripper",
        arm_control_mode = 'joint_position',
    )

    # 2. Language and task mapping
    if meta is not None and hasattr(meta, 'tasks') and 'task_index' in batch:
        task_idx = batch.pop('task_index')
        if isinstance(task_idx, torch.Tensor):
            task_idx = task_idx.item()
        
        try:
            batch['task'] = meta.tasks.iloc[task_idx].name
        except Exception:
            batch['task'] = ""
    else:
        batch['task'] = ""
        batch.pop('task_index', None)
            
    batch['subtask'] = ""

    for lang_key in [
        'language_instruction',
        'language_instruction_2',
        'language_instruction_3'
    ]:
        batch.pop(lang_key, None)

    # 3. Handle Official DROID Image Tensors
    batch['observation.images.head'] = batch.pop('observation.images.exterior_1_left', None)
    batch['observation.images.hand_left'] = batch.pop('observation.images.wrist_left', None)
    batch.pop('observation.images.exterior_2_left', None)
    
    batch['observation.images.hand_right'] = None

    # 4. Conditional Padding Mask logic
    has_pad = False
    if 'observation.images.exterior_1_left_is_pad' in batch:
        batch['observation.images.head_is_pad'] = batch.pop('observation.images.exterior_1_left_is_pad')
        has_pad = True
        
    if 'observation.images.wrist_left_is_pad' in batch:
        batch['observation.images.hand_left_is_pad'] = batch.pop('observation.images.wrist_left_is_pad')
        has_pad = True
        
    batch.pop('observation.images.exterior_2_left_is_pad', None)

    if has_pad:
        batch['observation.images.hand_right_is_pad'] = None

    return batch

def droid_norm_stats_transform(
    norm_stats: dict[str, dict[str, np.ndarray]],
    action_tensor_spec: ActionTensorSpec
) -> dict[str, dict[str, np.ndarray]]:
    if 'action' in norm_stats:
        norm_stats['droid_actions.action_all'] = norm_stats.pop('action')
    return merge_norm_stats(norm_stats, 'droid_actions', action_tensor_spec)

def load_agibot_world_dataset(
    root: str,
    canonical_action_chunk_size: int,
    num_frames: int = 1,
):
    action_tensor_spec = ACTION_TENSOR_SPECS['agibot_world']
    joint_position_mode_mask, eef_pose_mode_mask = build_arm_control_mode_masks(action_tensor_spec)
    return load_dataset(
        root,
        (
            "actions.robot.velocity",
            "actions.joint.position",
            "actions.end.position",
            "actions.end.orientation",
            "actions.effector.position",
        ),
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = agibot_world_to_nora_instance,
        norm_stats_transform = functools.partial(
            merge_norm_stats,
            action_prefix = 'actions',
            state_prefix = 'observation.states',
            action_tensor_spec = action_tensor_spec,
        ),
        se3_segment_start_idxs = build_se3_segment_start_idxs(action_tensor_spec),
        delta_transform_mask = build_delta_transform_mask(action_tensor_spec),
        joint_position_mode_mask = joint_position_mode_mask,
        eef_pose_mode_mask = eef_pose_mode_mask,
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
        (
            "action.chassis.velocities",
            "action.left_arm",
            "action.left_gripper",
            "action.right_arm",
            "action.right_gripper",
        ),
        canonical_action_chunk_size // 2,
        canonical_action_chunk_size,
        raw_fps = 15,
        instance_transform = galaxea_to_nora_instance,
        norm_stats_transform = functools.partial(
            merge_norm_stats,
            action_prefix = 'action',
            state_prefix = 'observation.state',
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
                action_prefix = 'actions',
                state_prefix = 'states',
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
        "actions.left_ee_to_robot_pose",
        "actions.left_gripper.position",
        "actions.right_joint.position",
        "actions.right_ee_to_robot_pose",
        "actions.right_gripper.position",
    )
    genie1_action_tensor_spec = ACTION_TENSOR_SPECS['interndata_a1_genie1']
    genie1_joint_position_mode_mask, genie1_eef_pose_mode_mask = build_arm_control_mode_masks(
        genie1_action_tensor_spec
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
            action_prefix = 'actions',
            state_prefix = 'states',
            action_tensor_spec = genie1_action_tensor_spec,
        ),
        se3_segment_start_idxs = build_se3_segment_start_idxs(genie1_action_tensor_spec),
        delta_transform_mask = build_delta_transform_mask(genie1_action_tensor_spec),
        joint_position_mode_mask = genie1_joint_position_mode_mask,
        eef_pose_mode_mask = genie1_eef_pose_mode_mask,
        target_action_dim = CANONICAL_ACTION_DIMS,
        num_frames = num_frames,
    )
    dual_arm_6dof_action_tensor_spec = ACTION_TENSOR_SPECS['interndata_a1_dual_arm_6dof']
    dual_arm_6dof_delta_mask = build_delta_transform_mask(dual_arm_6dof_action_tensor_spec)
    dual_arm_6dof_joint_position_mode_mask, dual_arm_6dof_eef_pose_mode_mask = build_arm_control_mode_masks(
        dual_arm_6dof_action_tensor_spec
    )
    dual_arm_6dof_norm_stats_transform = functools.partial(
        merge_norm_stats,
        action_prefix = 'actions',
        state_prefix = 'states',
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
        se3_segment_start_idxs = build_se3_segment_start_idxs(dual_arm_6dof_action_tensor_spec),
        delta_transform_mask = dual_arm_6dof_delta_mask,
        joint_position_mode_mask = dual_arm_6dof_joint_position_mode_mask,
        eef_pose_mode_mask = dual_arm_6dof_eef_pose_mode_mask,
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
        se3_segment_start_idxs = build_se3_segment_start_idxs(dual_arm_6dof_action_tensor_spec),
        delta_transform_mask = dual_arm_6dof_delta_mask,
        joint_position_mode_mask = dual_arm_6dof_joint_position_mode_mask,
        eef_pose_mode_mask = dual_arm_6dof_eef_pose_mode_mask,
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
            action_prefix = 'action',
            state_prefix = 'observation.state',
            action_tensor_spec = action_tensor_spec,
        ),
        se3_segment_start_idxs = frozenset((0, 24)),
        delta_transform_mask = build_delta_transform_mask(action_tensor_spec),
        target_action_dim = CANONICAL_ACTION_DIMS,
        num_frames = num_frames,
    )

def load_droid_dataset(
    root: str | pathlib.Path,
    canonical_action_chunk_size: int,
    num_frames: int = 1,
):
    root = pathlib.Path(root)
    droid_action_tensor_spec = ACTION_TENSOR_SPECS['droid']
    droid_delta_mask = build_delta_transform_mask(droid_action_tensor_spec)
    
    action_keys = ("action",)
    
    return load_dataset(
        root,
        action_keys,
        canonical_action_chunk_size // 2,
        canonical_action_chunk_size,
        raw_fps = 15,
        instance_transform = droid_to_nora_instance,
        norm_stats_transform = functools.partial(
            droid_norm_stats_transform,
            action_tensor_spec = droid_action_tensor_spec,
        ),
        delta_transform_mask = droid_delta_mask,
        target_action_dim = CANONICAL_ACTION_DIMS,
        num_frames = num_frames,
    )