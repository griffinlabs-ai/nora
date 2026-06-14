import dataclasses

import draccus
import pytest
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from transformers.models.gemma4 import Gemma4Config

from lerobot_policy_griffin_alpha import (
    GriffinAlphaConfig,
    GriffinAlphaPolicy,
    make_griffin_alpha_pre_post_processors,
)


def test_package_exports():
    assert GriffinAlphaConfig is not None
    assert GriffinAlphaPolicy is not None
    assert make_griffin_alpha_pre_post_processors is not None


def test_n_action_steps_cannot_exceed_horizon():
    with pytest.raises(ValueError, match="n_action_steps"):
        GriffinAlphaConfig(
            horizon=40,
            n_action_steps=60,
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )


def test_validate_features_adds_missing_action(griffin_alpha_config: GriffinAlphaConfig):
    griffin_alpha_config.output_features.pop(ACTION, None)
    griffin_alpha_config.validate_features()
    assert ACTION in griffin_alpha_config.output_features


def test_get_optimizer_preset(griffin_alpha_config: GriffinAlphaConfig):
    preset = griffin_alpha_config.get_optimizer_preset()
    assert isinstance(preset, AdamWConfig)
    assert preset.lr == griffin_alpha_config.optimizer_lr
    assert preset.betas == griffin_alpha_config.optimizer_betas
    assert preset.eps == griffin_alpha_config.optimizer_eps
    assert preset.weight_decay == griffin_alpha_config.optimizer_weight_decay
    assert preset.grad_clip_norm == griffin_alpha_config.optimizer_grad_clip_norm


def test_get_scheduler_preset(griffin_alpha_config: GriffinAlphaConfig):
    preset = griffin_alpha_config.get_scheduler_preset()
    assert isinstance(preset, CosineDecayWithWarmupSchedulerConfig)
    assert preset.peak_lr == griffin_alpha_config.optimizer_lr
    assert preset.num_warmup_steps == griffin_alpha_config.scheduler_warmup_steps
    assert preset.num_decay_steps == griffin_alpha_config.scheduler_decay_steps
    assert preset.decay_lr == griffin_alpha_config.scheduler_decay_lr


def test_action_delta_indices(griffin_alpha_config: GriffinAlphaConfig):
    assert griffin_alpha_config.action_delta_indices == list(range(griffin_alpha_config.horizon))


def test_reward_delta_indices_is_none(griffin_alpha_config: GriffinAlphaConfig):
    assert griffin_alpha_config.reward_delta_indices is None


def test_observation_delta_indices_is_none(griffin_alpha_config: GriffinAlphaConfig):
    assert griffin_alpha_config.observation_delta_indices is None


def test_max_action_dim(griffin_alpha_config: GriffinAlphaConfig):
    assert griffin_alpha_config.max_action_dim == griffin_alpha_config.output_features[ACTION].shape[-1]


def test_gemma4_config_encode_decode_hook():
    cfg = Gemma4Config()
    encoded = draccus.encode(cfg)
    assert isinstance(encoded, dict)

    assert encoded == cfg.to_dict()

    decoded = draccus.decode(Gemma4Config, encoded)
    assert isinstance(decoded, Gemma4Config)
    assert decoded.to_dict() == cfg.to_dict()


def test_gemma4_config_draccus_round_trip(griffin_alpha_config: GriffinAlphaConfig, tmp_path):
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f, draccus.config_type("json"):
        draccus.dump(griffin_alpha_config, f, indent=2)

    with draccus.config_type("json"):
        restored = draccus.parse(PreTrainedConfig, str(config_file), args=[])

    assert isinstance(restored, GriffinAlphaConfig)
    assert isinstance(restored.gemma4_config, Gemma4Config)
    assert restored.gemma4_config.to_dict() == griffin_alpha_config.gemma4_config.to_dict()


def test_config_save_load_round_trip(griffin_alpha_config: GriffinAlphaConfig, tmp_path):
    griffin_alpha_config._save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(restored, GriffinAlphaConfig)
    assert isinstance(restored.gemma4_config, Gemma4Config)
    # A plain `restored == griffin_alpha_config` (dataclass __eq__) fails only on the gemma4_config
    # field. transformers' PretrainedConfig.__eq__ compares __dict__, which carries volatile
    # bookkeeping that differs between an in-memory config (transformers_version is None and there is
    # no model_type entry) and one that has been serialized then reloaded (transformers_version gets
    # filled in by to_dict, and model_type is materialized). The serialization itself is lossless:
    # to_dict() matches exactly, so compare gemma4_config that way and every other field directly.
    assert restored.gemma4_config.to_dict() == griffin_alpha_config.gemma4_config.to_dict()
    for f in dataclasses.fields(GriffinAlphaConfig):
        if f.name == "gemma4_config":
            continue
        assert getattr(restored, f.name) == getattr(griffin_alpha_config, f.name), f.name
