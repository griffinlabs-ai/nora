import builtins
import logging
from collections import deque
from pathlib import Path
from typing import Sequence, TypeVar

import torch
from torch import Tensor
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Gemma4ForConditionalGeneration, Gemma4Config

from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_griffin_alpha import GriffinAlphaConfig
from .processor_griffin_alpha import ResampleActionProcessorStep

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="GriffinAlphaPolicy")

DEFAULT_MAX_NEW_TOKENS = 512

MODEL_INPUT_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "image_position_ids",
        "mm_token_type_ids",
    }
)


class GriffinAlphaPolicy(PreTrainedPolicy):
    config_class = GriffinAlphaConfig
    name = "griffin_alpha"

    config: GriffinAlphaConfig

    def __init__(self, config: GriffinAlphaConfig, gemma4_model: Gemma4ForConditionalGeneration = None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        config.validate_features()

        self.model = gemma4_model or Gemma4ForConditionalGeneration(config.gemma4_config)

        self.config.gemma4_config = self.model.config

        # Reload (PreTrainedPolicy.from_pretrained) builds the model with the plain
        # Gemma4ForConditionalGeneration(config) constructor and then copies weights in via
        # load_state_dict. Unlike HF from_pretrained, the plain constructor does not build under
        # the config dtype (e.g. the embed_vision/embed_audio projection layers stay float32), and
        # load_state_dict keeps the destination dtype, so a bfloat16 checkpoint would silently
        # reload as (mixed) float32. Explicitly cast to the config dtype to keep precision uniform.
        model_dtype = self.config.gemma4_config.dtype
        if model_dtype is not None:
            self.model = self.model.to(model_dtype)

        self.model.model.audio_tower.requires_grad_(False)
        self.model.model.embed_audio.requires_grad_(False)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

        self.fast_tokenizer = AutoProcessor.from_pretrained(
            config.fast_action_tokenizer_name,
            trust_remote_code=True,
        )
        self.fast_tokenizer.action_dim = config.max_action_dim
        self.fast_tokenizer.time_horizon = config.horizon
        self._action_resampler = (
            ResampleActionProcessorStep(target_chunk_size=config.resample_action_chunk_size)
            if config.resample_action_chunk_size
            else None
        )
        self.reset()

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque()

    def get_optim_params(self) -> dict:
        return {"params": [p for p in self.parameters() if p.requires_grad]}

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        self.model.generation_config.save_pretrained(save_directory)

    @staticmethod
    def _filter_model_inputs(batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {key: value for key, value in batch.items() if key in MODEL_INPUT_KEYS and value is not None}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        model_inputs = self._filter_model_inputs(batch)
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        return loss, {"loss": loss.item()}

    def _decode_action_tokens(self, generated_ids: Tensor, n_action_dims: Sequence[int] | None = None) -> Tensor:
        batch_size = generated_ids.shape[0]
        max_action_dim = self.config.max_action_dim
        horizon = self.config.horizon
        decoded_actions = []

        for i in range(batch_size):
            seq = generated_ids[i]
            action_mask = (seq >= self.config.action_token_min) & (seq <= self.config.action_token_max)
            action_token_ids = seq[action_mask]
            if action_token_ids.numel() == 0:
                decoded_actions.append(
                    torch.zeros(
                        horizon,
                        max_action_dim,
                        dtype=torch.float32,
                        device=generated_ids.device,
                    )
                )
                continue

            fast_ids = (action_token_ids - self.config.action_token_min).tolist()
            self.fast_tokenizer.action_dim = n_action_dims[i] if n_action_dims is not None else max_action_dim
            action = self.fast_tokenizer.decode([fast_ids])[0]
            action_tensor = torch.as_tensor(action, dtype=torch.float32, device=generated_ids.device)
            if action_tensor.shape[-1] < max_action_dim:
                pad = torch.zeros(
                    horizon,
                    max_action_dim - action_tensor.shape[-1],
                    dtype=action_tensor.dtype,
                    device=action_tensor.device,
                )
                action_tensor = torch.cat([action_tensor, pad], dim=-1)
            decoded_actions.append(action_tensor)

        return torch.stack(decoded_actions, dim=0)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        model_inputs = self._filter_model_inputs(batch)
        model_inputs.pop("labels", None)
        generated_ids = self.model.generate(**model_inputs)
        chunk = self._decode_action_tokens(generated_ids, batch.get("n_action_dims"))
        if self._action_resampler is not None:
            chunk = self._action_resampler({"action": chunk})["action"]
        return chunk

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @classmethod
    def from_gemma4_model(
        cls: builtins.type[T],
        gemma4_model: Gemma4ForConditionalGeneration,
        config: GriffinAlphaConfig | None = None,
        *,
        resize_embeddings: bool = True,
        **kwargs,
    ) -> T:
        """Convert a Gemma 4 transformers model into a GriffinAlphaPolicy.

        When ``resize_embeddings`` is True (the default), the token embeddings are
        resized to make room for the action and proprio state tokens. This is the
        path for a fresh base Gemma 4 model.

        When ``resize_embeddings`` is False, the embeddings are left untouched and
        the action token range is derived from the model's existing vocabulary.
        Use this for a model whose embeddings were already resized.
        """
        base_config = gemma4_model.config
        config = config or cls.config_class(gemma4_config=base_config)
        if resize_embeddings:
            # A fresh base model: action and proprio tokens are appended right after the
            # base vocab, so the base vocab size is the action token range start.
            base_vocab = base_config.text_config.vocab_size
            new_vocab_size = base_vocab + config.action_vocab_size + config.proprio_vocab_size
            gemma4_model.resize_token_embeddings(new_vocab_size)
        else:
            # An already-resized model: recover the base vocab from the enlarged vocab.
            final_vocab = gemma4_model.config.text_config.vocab_size
            base_vocab = final_vocab - config.action_vocab_size - config.proprio_vocab_size
            if base_vocab <= 0:
                raise ValueError(
                    f"Model vocab size ({final_vocab}) is too small to contain "
                    f"{config.action_vocab_size} action and {config.proprio_vocab_size} "
                    "proprio tokens; the model does not appear to be an adapted Griffin "
                    "Alpha checkpoint. Pass resize_embeddings=True to adapt a base model."
                )
        # Derive the action token range from the base vocab in both cases so it never
        # relies on the action_token_min default matching the base model's vocab size.
        config.action_token_min = base_vocab
        config.action_token_max = base_vocab + config.action_vocab_size - 1
        config.gemma4_config = gemma4_model.config
        gemma4_model.generation_config.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        gemma4_model.generation_config.do_sample = False
        policy = cls(config, gemma4_model, **kwargs)
        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def from_pretrained(cls: builtins.type[T], pretrained_name_or_path: str | Path, **kwargs) -> T:
        policy = super().from_pretrained(pretrained_name_or_path, **kwargs)
        download_keys = (
            "cache_dir",
            "force_download",
            "resume_download",
            "proxies",
            "token",
            "revision",
            "local_files_only",
        )
        gen_kwargs = {key: kwargs[key] for key in download_keys if key in kwargs}
        try:
            policy.model.generation_config = GenerationConfig.from_pretrained(
                pretrained_name_or_path, **gen_kwargs
            )
        except OSError:
            logger.info(
                "No generation_config found for %s; keeping defaults.", pretrained_name_or_path
            )
        return policy

    @classmethod
    def from_gemma4_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: Gemma4Config | None = None,
        resize_embeddings: bool = True,
        **load_kwargs,
    ) -> T:
        """Load a pretrained Gemma 4 checkpoint and convert it into a GriffinAlphaPolicy.

        Pass ``resize_embeddings=False`` when loading a model whose embeddings were
        already resized.
        """
        gemma4_config = config or AutoConfig.from_pretrained(
            pretrained_name_or_path,
            **load_kwargs,
        )
        gemma4_model = Gemma4ForConditionalGeneration.from_pretrained(
            pretrained_name_or_path,
            config=gemma4_config,
            trust_remote_code=True,
            **load_kwargs,
        )
        return cls.from_gemma4_model(gemma4_model, resize_embeddings=resize_embeddings)
