from dataclasses import dataclass, field
from functools import cache
from typing import TypeVar
from collections.abc import Set

import numpy as np
import torch
import torchvision.transforms as T_v2
from scipy.interpolate import CubicSpline
from transformers import AutoProcessor

import lerobot.processor
from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_griffin_alpha import GriffinAlphaConfig


T = TypeVar('T')

def _index_optional_list(l: list[T] | None, index: int) -> T | None:
    return l[index] if l is not None else None


def map_fast_token_to_vlm_action(tokens: list[str]) -> str:
    return "".join(f"<robot_action_{token}>" for token in tokens)


def make_proprio_state_tokens(vocab_size: int) -> list[str]:
    return [f"<proprio_state_{i}>" for i in range(vocab_size)]


def map_normalized_state_to_vlm_proprio(state: torch.Tensor, vocab_size: int) -> str:
    if vocab_size <= 0:
        raise ValueError(f"proprio_vocab_size must be positive, got {vocab_size}")

    clipped = state.clamp(-1.0, 1.0)
    bucket_ids = torch.floor((clipped + 1.0) * (vocab_size / 2.0)).to(torch.long)
    bucket_ids = bucket_ids.clamp(0, vocab_size - 1)
    return "".join(f"<proprio_state_{bucket_id}>" for bucket_id in bucket_ids.reshape(-1).tolist())


class GriffinAlphaImageTransform:
    BRIGHTNESS_FACTOR = 0.2
    CONTRAST_FACTOR = 0.2
    SATURATION_FACTOR = 0.2
    HUE_FACTOR = 0.05
    CROP_SCALE = 0.9

    def __init__(self) -> None:
        self.color_jitter = T_v2.ColorJitter(
            brightness=self.BRIGHTNESS_FACTOR,
            contrast=self.CONTRAST_FACTOR,
            saturation=self.SATURATION_FACTOR,
            hue=self.HUE_FACTOR,
        )

    @staticmethod
    @cache
    def get_random_crop_transform(source_size: tuple[int, int], scale: float) -> T_v2.RandomCrop:
        linear_scale = scale**0.5
        target_size = round(source_size[0] * linear_scale), round(source_size[1] * linear_scale)
        return T_v2.RandomCrop(target_size)

    @staticmethod
    @cache
    def get_center_crop_transform(source_size: tuple[int, int], scale: float) -> T_v2.CenterCrop:
        linear_scale = scale**0.5
        target_size = round(source_size[0] * linear_scale), round(source_size[1] * linear_scale)
        return T_v2.CenterCrop(target_size)

    def center_crop(self, image: torch.Tensor) -> torch.Tensor:
        return self.get_center_crop_transform(image.shape[-2:], self.CROP_SCALE)(image)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = self.get_random_crop_transform(image.shape[-2:], self.CROP_SCALE)(image)
        return self.color_jitter(image)


@dataclass
@ProcessorStepRegistry.register("griffinlabs/griffin_alpha_add_batch_dimension")
class GriffinAlphaAddBatchDimensionProcessorStep(ProcessorStep):
    """Add a batch dimension to unbatched transitions, keyed on observation.state."""

    @staticmethod
    def _add_batch_dim_to_dict(d: dict) -> dict:
        batched = d.copy()
        for key, value in d.items():
            if isinstance(value, torch.Tensor) and not key.startswith(f"{OBS_IMAGES}."):
                batched[key] = value.unsqueeze(0)
            else:
                batched[key] = [value]
        return batched

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition[TransitionKey.OBSERVATION]

        if observation[OBS_STATE].ndim == 2:
            return transition.copy()

        new_transition = transition.copy()

        new_transition[TransitionKey.OBSERVATION] = self._add_batch_dim_to_dict(observation)

        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            new_transition[TransitionKey.ACTION] = action.unsqueeze(0)

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is not None:
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = \
                self._add_batch_dim_to_dict(complementary_data)

        new_transition[TransitionKey.INFO] = \
            self._add_batch_dim_to_dict(transition[TransitionKey.INFO])

        return new_transition

    def transform_features(self, features):
        return features


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("griffinlabs/resample_action_processor")
class ResampleActionProcessorStep(lerobot.processor.ProcessorStep):
    """
    Resample the action tensor from any chunk size to a target chunk size,
    by cubic spline interpolation.

    Args:
     - target_chunk_size: The target chunk size to resample the action tensor to.
     - state_key: Key for the state tensor that corresponds to the action tensor.
       If None, the initial state is assumed to be zero.
       Leave as None if the action tensor is in delta space.
    """

    target_chunk_size: int
    state_key: str | None = None

    def __call__(self, transition):
        action = transition.get("action")
        if action is None:
            return transition

        if self.state_key is not None:
            initial_state = transition["observation"][self.state_key].unsqueeze(-2)
        else:
            initial_state = torch.zeros_like(action[..., :1, :])
        orig_chunk_size = action.shape[-2]

        if orig_chunk_size == self.target_chunk_size:
            return transition

        new_transition = transition.copy()

        if orig_chunk_size % self.target_chunk_size == 0:
            # If the original chunk size is a multiple of the target chunk size,
            # we can simply take every n-th action.
            step_size = orig_chunk_size // self.target_chunk_size
            new_transition["action"] = action[..., step_size - 1 :: step_size, :]
            return new_transition

        new_transition["action"] = self._interpolate(action, initial_state, orig_chunk_size)
        return new_transition

    def _interpolate(self, action, initial_state, orig_chunk_size):
        trajectory = torch.cat([initial_state, action], dim=-2)
        old_times = np.linspace(0, 1, orig_chunk_size + 1)
        new_times = np.linspace(1 / self.target_chunk_size, 1, self.target_chunk_size)

        traj_np = trajectory.cpu().numpy()
        cs = CubicSpline(old_times, traj_np, axis=-2)
        resampled = cs(new_times)

        return torch.from_numpy(resampled).to(dtype=action.dtype, device=action.device)

    def transform_features(self, features):
        original_shape = features["action"]["action"].shape
        features["action"]["action"] = PolicyFeature(
            FeatureType.ACTION,
            (self.target_chunk_size, original_shape[-1]),
        )
        return features

    def get_config(self):
        return {
            "target_chunk_size": self.target_chunk_size,
            "state_key": self.state_key,
        }


@dataclass
class EmbeddedSE3Segmenter:
    se3_segment_start_idxs: Set[int] | list[int] | None = None

    def __post_init__(self):
        split_points = []
        for start_idx in sorted(self.se3_segment_start_idxs or set()):
            split_points.append((start_idx, "se3_matrix"))
            split_points.append((start_idx + 16, "real"))
        if len(split_points) == 0 or split_points[0][0] != 0:
            split_points.insert(0, (None, "real"))
        self.slices = [
            (
                slice(
                    split_points[i][0],
                    split_points[i + 1][0] if i + 1 < len(split_points) else None,
                ),
                split_points[i][1],
            )
            for i in range(len(split_points))
        ]


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("griffinlabs/relative_action_with_se3_processor")
class RelativeActionWithSE3ProcessorStep(lerobot.processor.ProcessorStep):
    """
    Convert action tensor from absolute space to relative space.
    Expects an action shape of [..., chunk_size, degrees_of_freedom].
    Expects transition to have a state tensor in the same vector space as the action tensor.

    Args:
     - mask: Mask of which action tensor dimensions to convert to relative space.
       `True` dimensions output relative space, `False` dimensions keep absolute space.
     - se3_segment_start_idxs: Indices of the action tensor dimensions that correspond to SE(3) matrices.
       If None, all action tensor dimensions are assumed to be real.
     - state_key: Key for the state tensor that corresponds to the action tensor.
    """

    mask: list[bool]
    se3_segment_start_idxs: list[int] | None = None
    state_key: str = OBS_STATE
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._mask = torch.tensor(self.mask, dtype=torch.bool)
        self._segmenter = EmbeddedSE3Segmenter(self.se3_segment_start_idxs)

    def __call__(self, transition):
        action = transition.get("action")
        if action is None:
            self._last_state = transition["observation"].get(self.state_key)
            return transition

        new_transition = transition.copy()

        state = transition["observation"][self.state_key].unsqueeze(-2)
        leading_dims = action.shape[:-1]

        assert self._mask.shape[-1] == action.shape[-1]

        segments = []

        for segment_slice, operand_type in self._segmenter.slices:
            if operand_type == "real":
                segment = action[..., segment_slice] - state[..., segment_slice]
            elif operand_type == "se3_matrix":
                action_se3 = action[..., segment_slice].view(*leading_dims, 4, 4)
                state_se3 = state[..., segment_slice].view(*leading_dims[:-1], 1, 4, 4)
                segment = state_se3.inverse().matmul(action_se3)
                segment = segment.view(*leading_dims, 16)
            segments.append(segment)

        deltas = torch.cat(segments, dim=-1)
        new_transition["action"] = torch.where(self._mask.expand(action.shape), deltas, action)
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self):
        return {
            "mask": self.mask,
            "se3_segment_start_idxs": self.se3_segment_start_idxs,
            "state_key": self.state_key,
        }


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("griffinlabs/se3_mat_to_xyz_rot6d_processor")
class SE3MatrixToXYZRot6DProcessorStep(lerobot.processor.ProcessorStep):
    """
    Convert SE(3) matrices in the action tensor to XYZ and rot6d.
    """

    se3_segment_start_idxs: list[int]
    state_key: str = OBS_STATE

    PER_SE3_MATRIX_DIM_REDUCTION = 7    # 16 dimensions -> 9 dimensions

    def __post_init__(self):
        self._segmenter = EmbeddedSE3Segmenter(self.se3_segment_start_idxs)

    @staticmethod
    def convert(action: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [action[..., :3, 3], action[..., :3, :2].flatten(start_dim=-2)],
            dim=-1,
        )

    def __call__(self, transition):
        new_transition = transition.copy()
        action = transition.get("action")
        state = transition["observation"][self.state_key]

        action_segments = []
        state_segments = []

        for segment_slice, operand_type in self._segmenter.slices:
            if operand_type == "real":
                if action is not None:
                    action_segments.append(action[..., segment_slice])
                state_segments.append(state[..., segment_slice])
            elif operand_type == "se3_matrix":
                if action is not None:
                    action_se3 = action[..., segment_slice].view(*action.shape[:-1], 4, 4)
                    action_segments.append(self.convert(action_se3))
                state_se3 = state[..., segment_slice].view(*state.shape[:-1], 4, 4)
                state_segments.append(self.convert(state_se3))

        if action is not None:
            new_transition["action"] = torch.cat(action_segments, dim=-1)
        new_transition["observation"] = transition["observation"].copy()
        new_transition["observation"][self.state_key] = torch.cat(state_segments, dim=-1)
        return new_transition

    def transform_features(self, features):
        old_shape = features[PipelineFeatureType.ACTION]["action"].shape
        num_se3_segments = sum(1 for _, t in self._segmenter.slices if t == "se3_matrix")
        features[PipelineFeatureType.ACTION]["action"] = PolicyFeature(
            FeatureType.ACTION,
            old_shape[:-1] + (old_shape[-1] - num_se3_segments * self.PER_SE3_MATRIX_DIM_REDUCTION,),
        )
        old_shape = features[PipelineFeatureType.OBSERVATION][self.state_key].shape
        features[PipelineFeatureType.OBSERVATION][self.state_key] = PolicyFeature(
            FeatureType.STATE,
            old_shape[:-1] + (old_shape[-1] - num_se3_segments * self.PER_SE3_MATRIX_DIM_REDUCTION,),
        )
        return features

    def get_config(self):
        return {
            "se3_segment_start_idxs": self.se3_segment_start_idxs,
            "state_key": self.state_key,
        }


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("griffinlabs/xyz_rot6d_to_se3_mat_processor")
class XYZRot6DToSE3MatrixProcessorStep(lerobot.processor.ProcessorStep):
    se3_segment_start_idxs: list[int]
    state_key: str = OBS_STATE

    PER_SE3_MATRIX_DIM_REDUCTION = 7    # 16 dimensions -> 9 dimensions

    def __post_init__(self):
        self._segmenter = EmbeddedSE3Segmenter(self.se3_segment_start_idxs)
        self._num_se3_segments = 0
        self._reduced_slices: list[tuple[slice, str]] = []
        reduced_start = 0
        for segment_slice, operand_type in self._segmenter.slices:
            if operand_type == "se3_matrix":
                self._num_se3_segments += 1
                reduced_stop = reduced_start + 9
                self._reduced_slices.append((slice(reduced_start, reduced_stop), operand_type))
                reduced_start = reduced_stop
                continue

            if segment_slice.stop is None:
                self._reduced_slices.append((slice(reduced_start, None), operand_type))
                continue

            segment_start = segment_slice.start or 0
            width = segment_slice.stop - segment_start
            reduced_stop = reduced_start + width
            self._reduced_slices.append((slice(reduced_start, reduced_stop), operand_type))
            reduced_start = reduced_stop

    def __call__(self, transition):
        action = transition.get("action")
        if action is None:
            return transition

        new_transition = transition.copy()
        segments = []
        for segment_slice, operand_type in self._reduced_slices:
            segment = action[..., segment_slice]
            if operand_type == "real":
                segments.append(segment)
                continue

            xyz = segment[..., :3]
            rot6d = segment[..., 3:9]
            rot_cols = rot6d.view(*segment.shape[:-1], 3, 2)
            a1 = rot_cols[..., :, 0]
            a2 = rot_cols[..., :, 1]
            b1 = torch.nn.functional.normalize(a1, dim=-1)
            b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
            b3 = torch.cross(b1, b2, dim=-1)
            rotation = torch.stack((b1, b2, b3), dim=-1)

            se3 = torch.zeros(*segment.shape[:-1], 4, 4, dtype=segment.dtype, device=segment.device)
            se3[..., :3, :3] = rotation
            se3[..., :3, 3] = xyz
            se3[..., 3, 3] = 1
            segments.append(se3.flatten(start_dim=-2))

        new_transition["action"] = torch.cat(segments, dim=-1)
        return new_transition

    def transform_features(self, features):
        old_shape = features[PipelineFeatureType.ACTION]["action"].shape
        features[PipelineFeatureType.ACTION]["action"] = PolicyFeature(
            FeatureType.ACTION,
            old_shape[:-1] + (old_shape[-1] + self._num_se3_segments * self.PER_SE3_MATRIX_DIM_REDUCTION,),
        )
        return features

    def get_config(self):
        return {
            "se3_segment_start_idxs": self.se3_segment_start_idxs,
            "state_key": self.state_key,
        }


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("griffinlabs/absolute_action_with_se3_processor")
class AbsoluteActionWithSE3ProcessorStep(lerobot.processor.ProcessorStep):
    mask: list[bool]
    se3_segment_start_idxs: list[int] | None = None
    state_key: str = OBS_STATE
    relative_step: "RelativeActionWithSE3ProcessorStep | None" = field(default=None, repr=False)

    def __post_init__(self):
        self._mask = torch.tensor(self.mask, dtype=torch.bool)
        self._segmenter = EmbeddedSE3Segmenter(self.se3_segment_start_idxs)

    def __call__(self, transition):
        action = transition.get("action")
        if action is None:
            return transition
        if self.relative_step is None:
            raise RuntimeError(
                "AbsoluteActionWithSE3ProcessorStep requires a paired RelativeActionWithSE3ProcessorStep."
            )

        state = self.relative_step._last_state
        if state is None:
            raise RuntimeError(
                "No cached state found in paired RelativeActionWithSE3ProcessorStep. "
                "Run the preprocessor before this postprocessor."
            )

        state = state.to(device=action.device, dtype=action.dtype)
        state = state.unsqueeze(-2)
        leading_dims = action.shape[:-1]

        assert self._mask.shape[-1] == action.shape[-1]

        segments = []
        for segment_slice, operand_type in self._segmenter.slices:
            if operand_type == "real":
                segment = action[..., segment_slice] + state[..., segment_slice]
            elif operand_type == "se3_matrix":
                action_se3 = action[..., segment_slice].view(*leading_dims, 4, 4)
                state_se3 = state[..., segment_slice].view(*leading_dims[:-1], 1, 4, 4)
                segment = state_se3.matmul(action_se3).view(*leading_dims, 16)
            segments.append(segment)

        new_transition = transition.copy()
        absolutes = torch.cat(segments, dim=-1)
        new_transition["action"] = torch.where(self._mask.expand(action.shape), absolutes, action)
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self):
        return {
            "mask": self.mask,
            "se3_segment_start_idxs": self.se3_segment_start_idxs,
            "state_key": self.state_key,
        }


@dataclass
@ProcessorStepRegistry.register("griffinlabs/griffin_alpha_vlm_input")
class GriffinAlphaVLMInputProcessorStep(ProcessorStep):
    fast_action_tokenizer_name: str = "lerobot/fast-action-tokenizer"
    base_vlm_processor_name: str = "google/gemma-4-E4B-it"
    max_tokens_per_image: int = 70
    max_sequence_length: int = 500
    proprio_vocab_size: int = 256
    action_vocab_size: int = 2048
    apply_image_augmentation: bool = True
    apply_inference_center_crop: bool = False
    image_keys: tuple[str, ...] = (
        "observation.images.head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    )
    embodiment_prompt: str | None = None
    arm_control_mode: str | None = None
    predict_subtask: bool | None = None

    def __post_init__(self) -> None:
        # JSON round-trips turn the tuple into a list; normalize so equality and
        # downstream tuple usage stay stable after save/load.
        self.image_keys = tuple(self.image_keys)

        self._fast_tokenizer = AutoProcessor.from_pretrained(
            self.fast_action_tokenizer_name,
            trust_remote_code=True,
        )

        self._vla_processor = self._make_vla_processor()

        proprio_tokens = make_proprio_state_tokens(self.proprio_vocab_size)
        proprio_ids = self._vla_processor.tokenizer.convert_tokens_to_ids(proprio_tokens)
        proprio_ids = [
            token_id
            for token_id in proprio_ids
            if token_id is not None and token_id != self._vla_processor.tokenizer.unk_token_id
        ]
        if len(proprio_ids) != self.proprio_vocab_size:
            raise ValueError("Proprio state token count mismatch")

        action_tokens = [f"<robot_action_{i}>" for i in range(self.action_vocab_size)]
        action_ids = self._vla_processor.tokenizer.convert_tokens_to_ids(action_tokens)
        action_ids = [
            token_id
            for token_id in action_ids
            if token_id is not None and token_id != self._vla_processor.tokenizer.unk_token_id
        ]
        if len(action_ids) != self.action_vocab_size:
            raise ValueError("Action token count mismatch")

        self._image_transform = GriffinAlphaImageTransform()
    
    def _make_vla_processor(self) -> AutoProcessor:
        processor = AutoProcessor.from_pretrained(
            self.base_vlm_processor_name,
            trust_remote_code=True,
            max_soft_tokens=self.max_tokens_per_image,
            image_seq_length=self.max_tokens_per_image,
        )

        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"

        # add action and proprio state tokens to the tokenizer
        action_tokens = [f"<robot_action_{i}>" for i in range(self.action_vocab_size)]
        proprio_tokens = make_proprio_state_tokens(self.proprio_vocab_size)
        processor.tokenizer.add_tokens(action_tokens + proprio_tokens, special_tokens=True)

        return processor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition[TransitionKey.OBSERVATION]
        proprio_state = observation[OBS_STATE]
        batch_size = proprio_state.shape[0]
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        info = transition[TransitionKey.INFO]
        action = transition.get(TransitionKey.ACTION)
        has_action_labels = isinstance(action, torch.Tensor)
        apply_augmentation = self.apply_image_augmentation and has_action_labels and torch.is_grad_enabled()
        apply_center_crop = self.apply_inference_center_crop and not apply_augmentation

        text_prompts = []
        batch_images = []

        for i in range(batch_size):
            imgs_for_this_sample = []

            for image_key in self.image_keys:
                img_tensor = observation[image_key][i]
                if img_tensor is not None:
                    if apply_augmentation:
                        img_tensor = self._image_transform(img_tensor)
                    elif apply_center_crop:
                        img_tensor = self._image_transform.center_crop(img_tensor)
                    imgs_for_this_sample.append(img_tensor)

            batch_images.append(imgs_for_this_sample)

            n_action_dims = _index_optional_list(info.get("n_action_dims"), i)
            vlm_proprio = map_normalized_state_to_vlm_proprio(
                proprio_state[i][:n_action_dims],
                self.proprio_vocab_size,
            )

            task = _index_optional_list(complementary_data.get("task"), i) or ""
            subtask = _index_optional_list(complementary_data.get("subtask"), i) or ""
            predict_subtask = (
                _index_optional_list(info.get("predict_subtask"), i)
                or self.predict_subtask
                or bool(subtask)
            )
            embodiment = _index_optional_list(info.get("embodiment_prompt"), i) or self.embodiment_prompt or ""
            arm_control_mode = _index_optional_list(info.get("arm_control_mode"), i) or self.arm_control_mode
            if arm_control_mode is None:
                raise ValueError(
                    "arm_control_mode must be provided via transition info or "
                    "GriffinAlphaConfig.arm_control_mode, but both are missing."
                )

            content = [
                {
                    "type": "text",
                    "text": f"[embodiment: {embodiment}; arm control mode: {arm_control_mode}] {vlm_proprio}",
                }
            ]
            for _ in imgs_for_this_sample:
                content.append({"type": "image"})
            content.append(
                {
                    "type": "text",
                    "text": f"{task}\npredict subtask: {'true' if predict_subtask else 'false'}",
                }
            )

            if has_action_labels:
                sample_action = action[i][:, :n_action_dims]
                fast_tokens = self._fast_tokenizer(sample_action.cpu())[0]
                vlm_action = map_fast_token_to_vlm_action(fast_tokens)
                subtask_segment = f"subtask: {subtask}\n" if subtask else ""
                messages = [
                    {"role": "user", "content": content},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"{subtask_segment}action: {vlm_action}"}],
                    },
                ]
                prompt = self._vla_processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                messages = [{"role": "user", "content": content}]
                prompt = self._vla_processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            text_prompts.append(prompt)

        batch_input = self._vla_processor(
            text=text_prompts,
            images=batch_images,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

        batch_input["n_action_dims"] = info.get("n_action_dims")

        if has_action_labels:
            labels = batch_input["input_ids"].clone()
            sot_token_id = self._vla_processor.tokenizer.sot_token_id
            for i in range(labels.size(0)):
                seq = labels[i]
                sot_indices = (seq == sot_token_id).nonzero(as_tuple=False)
                if sot_indices.numel() > 0:
                    last_sot_index = sot_indices[-1].item()
                    seq[:last_sot_index] = -100
                else:
                    seq[:] = -100

            pad_token_id = self._vla_processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
            batch_input["labels"] = labels

        return lerobot.processor.create_transition(complementary_data = batch_input)

    def transform_features(self, features):
        return {
            PipelineFeatureType.ACTION: {},
            PipelineFeatureType.OBSERVATION: {},
        }

    def get_config(self):
        return {
            "fast_action_tokenizer_name": self.fast_action_tokenizer_name,
            "base_vlm_processor_name": self.base_vlm_processor_name,
            "max_tokens_per_image": self.max_tokens_per_image,
            "max_sequence_length": self.max_sequence_length,
            "proprio_vocab_size": self.proprio_vocab_size,
            "action_vocab_size": self.action_vocab_size,
            "apply_image_augmentation": self.apply_image_augmentation,
            "apply_inference_center_crop": self.apply_inference_center_crop,
            "image_keys": self.image_keys,
            "embodiment_prompt": self.embodiment_prompt,
            "arm_control_mode": self.arm_control_mode,
            "predict_subtask": self.predict_subtask,
        }
    
    @classmethod
    def from_griffin_alpha_config(cls, config: GriffinAlphaConfig) -> "GriffinAlphaVLMInputProcessorStep":
        return cls(
            fast_action_tokenizer_name=config.fast_action_tokenizer_name,
            base_vlm_processor_name=config.base_vlm_processor_name,
            max_tokens_per_image=config.max_tokens_per_image,
            max_sequence_length=config.max_sequence_length,
            proprio_vocab_size=config.proprio_vocab_size,
            action_vocab_size=config.action_vocab_size,
            apply_image_augmentation=config.apply_image_augmentation,
            apply_inference_center_crop=config.apply_inference_center_crop,
            image_keys=config.image_keys,
            embodiment_prompt=config.embodiment_prompt,
            arm_control_mode=config.arm_control_mode,
            predict_subtask=config.predict_subtask,
        )

def make_griffin_alpha_pre_post_processors(
    config: GriffinAlphaConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[EnvTransition, EnvTransition],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    if config.se3_segment_start_idxs and config.resample_action_chunk_size is not None:
        raise ValueError(
            "Resampling and SE(3) matrices cannot be used at the same time, "
            "as correct interpolation of SE(3) values is not implemented."
        )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        GriffinAlphaAddBatchDimensionProcessorStep(),
    ]

    relative_step = None
    if config.relative_action_mask is not None:
        relative_step = RelativeActionWithSE3ProcessorStep(
            mask=config.relative_action_mask,
            se3_segment_start_idxs=config.se3_segment_start_idxs,
            state_key=config.state_key,
        )
        input_steps.append(relative_step)

    if config.se3_segment_start_idxs:
        input_steps.append(
            SE3MatrixToXYZRot6DProcessorStep(
                se3_segment_start_idxs=config.se3_segment_start_idxs,
                state_key=config.state_key,
            )
        )

    if config.resample_action_chunk_size is not None:
        input_steps.append(
            ResampleActionProcessorStep(target_chunk_size=config.horizon),
        )

    if dataset_stats is not None:
        norm_stats = dataset_stats
        norm_map: dict[str, NormalizationMode] = config.normalization_mapping
        normalizer_features = {
            OBS_STATE: PolicyFeature(FeatureType.STATE, tuple(norm_stats[OBS_STATE]["q01"].shape))
        }
        input_steps.append(
            NormalizerProcessorStep(
                normalizer_features,
                norm_map,
                norm_stats,
                normalize_observation_keys={OBS_STATE},
            )
        )

    input_steps.extend([
        GriffinAlphaVLMInputProcessorStep.from_griffin_alpha_config(config),
        DeviceProcessorStep(device=config.device),
    ])

    output_steps = []

    if dataset_stats is not None:
        output_steps.append(
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            )
        )

    if config.se3_segment_start_idxs:
        output_steps.append(
            XYZRot6DToSE3MatrixProcessorStep(
                se3_segment_start_idxs=config.se3_segment_start_idxs,
                state_key=config.state_key,
            )
        )

    if config.relative_action_mask is not None:
        output_steps.append(
            AbsoluteActionWithSE3ProcessorStep(
                mask=config.relative_action_mask,
                se3_segment_start_idxs=config.se3_segment_start_idxs,
                state_key=config.state_key,
                relative_step=relative_step,
            )
        )

    output_steps.append(DeviceProcessorStep(device="cpu"))

    return (
        PolicyProcessorPipeline[EnvTransition, EnvTransition](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_output=lambda tr: tr[TransitionKey.COMPLEMENTARY_DATA],
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
