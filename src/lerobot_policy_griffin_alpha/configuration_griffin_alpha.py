from dataclasses import dataclass, field

import draccus
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from transformers.models.gemma4 import Gemma4Config


# `Gemma4Config` is a transformers dataclass whose annotations contain forward
# references that draccus cannot resolve when it recurses into the type. Register
# leaf encode/decode hooks so draccus serializes it as a plain dict instead of
# introspecting its fields.
@draccus.encode.register
def _encode_gemma4_config(cfg: Gemma4Config) -> dict:
    return cfg.to_dict()


draccus.decode.register(Gemma4Config, lambda value: Gemma4Config.from_dict(value))


DEFAULT_ACTION_TOKEN_MIN = 262144   # 262144 is Gemma 4's default vocab size
DEFAULT_ACTION_VOCAB_SIZE = 2048
MAX_ACTION_DIM = 32
ACTION_HORIZON = 50

@PreTrainedConfig.register_subclass("griffin_alpha")
@dataclass
class GriffinAlphaConfig(PreTrainedConfig):
    """Configuration class for GriffinAlphaPolicy."""

    output_features: dict[str, PolicyFeature] = field(default_factory=lambda: {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(MAX_ACTION_DIM,)),
    })

    horizon: int = ACTION_HORIZON
    n_action_steps: int = ACTION_HORIZON

    gemma4_config: Gemma4Config = field(default_factory=Gemma4Config)

    action_vocab_size: int = DEFAULT_ACTION_VOCAB_SIZE
    action_token_min: int = DEFAULT_ACTION_TOKEN_MIN
    action_token_max: int = DEFAULT_ACTION_TOKEN_MIN + DEFAULT_ACTION_VOCAB_SIZE - 1

    proprio_vocab_size: int = 256

    max_tokens_per_image: int = 70
    # Max total token length per sample; longer sequences are right-truncated
    max_sequence_length: int = 500

    fast_action_tokenizer_name: str = "lerobot/fast-action-tokenizer"
    base_vlm_processor_name: str = "google/gemma-4-E4B-it"
    apply_image_augmentation: bool = True
    apply_inference_center_crop: bool = False
    gradient_checkpointing: bool = True

    image_keys: tuple[str, ...] = (
        "observation.images.head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    )

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )
    relative_action_mask: list[bool] | None = None
    se3_segment_start_idxs: list[int] | None = None
    resample_action_to_horizon: bool = False
    state_key: str = OBS_STATE

    optimizer_lr: float = 3e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-8
    optimizer_grad_clip_norm: float = 50.0

    scheduler_warmup_steps: int = 50_000
    scheduler_decay_steps: int = 675_000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.horizon:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than horizon ({self.horizon})"
            )

    def validate_features(self) -> None:
        """Validate input/output feature compatibility."""
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(MAX_ACTION_DIM,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def max_action_dim(self) -> int:
        return self.output_features[ACTION].shape[-1]
