from dataclasses import dataclass
from functools import cache
from typing import TypeVar

import torch
import torchvision.transforms as T_v2
from transformers import AutoProcessor

import lerobot.processor
from lerobot.configs.types import PipelineFeatureType
from lerobot.processor import (
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep
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
            predict_subtask = _index_optional_list(info.get("predict_subtask"), i) or bool(subtask)
            embodiment = _index_optional_list(info.get("embodiment_prompt"), i) or ""
            arm_control_mode = info["arm_control_mode"][i]

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
        )

def make_griffin_alpha_pre_post_processors(
    config: GriffinAlphaConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[EnvTransition, EnvTransition],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        GriffinAlphaAddBatchDimensionProcessorStep(),
        GriffinAlphaVLMInputProcessorStep.from_griffin_alpha_config(config),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps = [
        DeviceProcessorStep(device="cpu"),
    ]

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
