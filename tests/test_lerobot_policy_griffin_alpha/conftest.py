import sys
from unittest.mock import MagicMock


def _stub_lerobot_groot_imports() -> None:
    """Work around lerobot 0.5.1 groot dataclass import failure on Python 3.12."""
    if "lerobot.policies.groot.configuration_groot" in sys.modules:
        return

    config_mod = MagicMock()
    config_mod.GrootConfig = MagicMock()
    modeling_mod = MagicMock()
    modeling_mod.GrootPolicy = MagicMock()
    groot_mod = MagicMock()
    groot_mod.GrootConfig = config_mod.GrootConfig
    groot_mod.GrootPolicy = modeling_mod.GrootPolicy

    sys.modules["lerobot.policies.groot.groot_n1"] = MagicMock()
    sys.modules["lerobot.policies.groot.configuration_groot"] = config_mod
    sys.modules["lerobot.policies.groot.modeling_groot"] = modeling_mod
    sys.modules["lerobot.policies.groot"] = groot_mod


_stub_lerobot_groot_imports()

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.types import EnvTransition
from lerobot.utils.constants import ACTION
from transformers import AutoProcessor, Gemma4Config, Gemma4ForConditionalGeneration

from lerobot_policy_griffin_alpha.configuration_griffin_alpha import GriffinAlphaConfig
from lerobot_policy_griffin_alpha.processor_griffin_alpha import make_proprio_state_tokens

from .helpers import make_sample_env_transition


@pytest.fixture
def tiny_gemma4_config() -> Gemma4Config:
    return Gemma4Config(
        architectures=["Gemma4ForConditionalGeneration"],
        text_config={
            "hidden_size": 64,
            "hidden_size_per_layer_input": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            "num_kv_shared_layers": 2,
            "dtype": "bfloat16",
        },
        vision_config={
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "num_channels": 3,
            "image_size": 64,
            "patch_size": 16,
            "dtype": "bfloat16",
        },
        audio_config={
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "dtype": "bfloat16",
        },
        dtype="bfloat16",
    )


@pytest.fixture
def tiny_gemma4_model(tiny_gemma4_config: Gemma4Config) -> Gemma4ForConditionalGeneration:
    # Mirror a real checkpoint loaded via HF from_pretrained, which is fully bfloat16
    # (the plain constructor would otherwise yield vision and audio embed layers in float32).
    return Gemma4ForConditionalGeneration(tiny_gemma4_config).to(torch.bfloat16)


@pytest.fixture
def griffin_alpha_config(tiny_gemma4_config: Gemma4Config) -> GriffinAlphaConfig:
    return GriffinAlphaConfig(
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        image_keys=("observation.images.head",),
        gemma4_config=tiny_gemma4_config,
        gradient_checkpointing=False,
        device="cpu",
        horizon=4,
        n_action_steps=4,
        apply_image_augmentation=False,
    )


@pytest.fixture(scope="session")
def fast_action_tokenizer():
    tokenizer = AutoProcessor.from_pretrained(
        "lerobot/fast-action-tokenizer",
        trust_remote_code=True,
    )
    return tokenizer


@pytest.fixture(scope="session")
def gemma4_vlm_processor():
    processor = AutoProcessor.from_pretrained(
        "google/gemma-4-E4B-it",
        trust_remote_code=True,
        max_soft_tokens=70,
        image_seq_length=70,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    action_tokens = [f"<robot_action_{i}>" for i in range(2048)]
    proprio_tokens = make_proprio_state_tokens(256)
    processor.tokenizer.add_tokens(action_tokens + proprio_tokens, special_tokens=True)
    return processor


@pytest.fixture
def sample_env_transition(griffin_alpha_config: GriffinAlphaConfig) -> EnvTransition:
    return make_sample_env_transition(griffin_alpha_config, batched=True, with_action=False)


@pytest.fixture
def sample_training_transition(griffin_alpha_config: GriffinAlphaConfig) -> EnvTransition:
    return make_sample_env_transition(griffin_alpha_config, batched=True, with_action=True)
