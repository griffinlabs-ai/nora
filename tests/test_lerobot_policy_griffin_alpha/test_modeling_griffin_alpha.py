from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from lerobot.utils.constants import ACTION
from transformers import Gemma4Config, Gemma4ForConditionalGeneration

from lerobot_policy_griffin_alpha.configuration_griffin_alpha import GriffinAlphaConfig
from lerobot_policy_griffin_alpha.modeling_griffin_alpha import (
    DEFAULT_MAX_NEW_TOKENS,
    MODEL_INPUT_KEYS,
    GriffinAlphaPolicy,
)


@pytest.fixture
def griffin_alpha_policy(griffin_alpha_config, tiny_gemma4_model, fast_action_tokenizer):
    with patch(
        "lerobot_policy_griffin_alpha.modeling_griffin_alpha.AutoProcessor.from_pretrained",
        return_value=fast_action_tokenizer,
    ):
        policy = GriffinAlphaPolicy(griffin_alpha_config, gemma4_model=tiny_gemma4_model)
    return policy


class TestFilterModelInputs:
    def test_filters_unknown_keys_and_none(self):
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "labels": None,
            "unknown_key": torch.tensor([0]),
        }
        filtered = GriffinAlphaPolicy._filter_model_inputs(batch)
        assert set(filtered.keys()) == {"input_ids", "attention_mask"}
        assert "labels" not in filtered
        assert "unknown_key" not in filtered

    def test_model_input_keys_complete(self):
        assert "input_ids" in MODEL_INPUT_KEYS
        assert "pixel_values" in MODEL_INPUT_KEYS


class TestGriffinAlphaPolicy:
    def test_reset_clears_action_queue(self, griffin_alpha_policy):
        griffin_alpha_policy._action_queue.append(torch.zeros(7))
        griffin_alpha_policy.reset()
        assert len(griffin_alpha_policy._action_queue) == 0

    def test_get_optim_params_excludes_frozen_audio(self, griffin_alpha_policy):
        optim_params = griffin_alpha_policy.get_optim_params()["params"]
        frozen_param_ids = {
            id(p)
            for p in griffin_alpha_policy.model.model.audio_tower.parameters()
        } | {
            id(p)
            for p in griffin_alpha_policy.model.model.embed_audio.parameters()
        }
        optim_param_ids = {id(p) for p in optim_params}
        assert optim_param_ids.isdisjoint(frozen_param_ids)
        assert len(optim_params) > 0
        assert not any(p.requires_grad for p in griffin_alpha_policy.model.model.audio_tower.parameters())
        assert not any(p.requires_grad for p in griffin_alpha_policy.model.model.embed_audio.parameters())

    def test_decode_action_tokens_empty_sequence(self, griffin_alpha_policy):
        generated_ids = torch.tensor([[1, 2, 3, 4]])
        actions = griffin_alpha_policy._decode_action_tokens(generated_ids)
        horizon = griffin_alpha_policy.config.horizon
        action_dim = griffin_alpha_policy.config.output_features[ACTION].shape[0]
        assert actions.shape == (1, horizon, action_dim)
        assert torch.all(actions == 0)

    def test_decode_action_tokens_with_action_tokens(self, griffin_alpha_policy, fast_action_tokenizer):
        horizon = griffin_alpha_policy.config.horizon
        action_dim = griffin_alpha_policy.config.output_features[ACTION].shape[0]
        sample_action = torch.randn(1, horizon, action_dim)
        fast_action_tokenizer.action_dim = action_dim
        fast_action_tokenizer.time_horizon = horizon
        fast_ids = fast_action_tokenizer(sample_action)[0]
        reference_decoded = fast_action_tokenizer.decode([fast_ids])[0]
        reference_decoded = torch.as_tensor(reference_decoded, dtype=torch.float32)

        token_min = griffin_alpha_policy.config.action_token_min
        action_token_ids = [token_min + int(t) for t in fast_ids]
        padding = [0, 1, 2]
        generated_ids = torch.tensor([padding + action_token_ids])

        decoded = griffin_alpha_policy._decode_action_tokens(generated_ids)
        assert decoded.shape == (1, horizon, action_dim)
        assert not torch.all(decoded == 0)
        assert torch.allclose(decoded, reference_decoded)

    def test_forward_returns_loss(self, griffin_alpha_policy):
        fake_loss = torch.tensor(1.5, requires_grad=True)
        mock_outputs = SimpleNamespace(loss=fake_loss)

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        with patch.object(griffin_alpha_policy.model, "forward", return_value=mock_outputs):
            loss, metrics = griffin_alpha_policy.forward(batch)
        assert loss == fake_loss
        assert loss.requires_grad
        assert metrics["loss"] == pytest.approx(1.5)

    def test_select_action_drains_queue(self, griffin_alpha_policy):
        n_steps = griffin_alpha_policy.config.n_action_steps
        horizon = griffin_alpha_policy.config.horizon
        action_dim = griffin_alpha_policy.config.max_action_dim
        fake_chunk = torch.randn(1, horizon, action_dim)
        griffin_alpha_policy.predict_action_chunk = MagicMock(return_value=fake_chunk)

        batch = {"input_ids": torch.tensor([[1]])}
        actions = [griffin_alpha_policy.select_action(batch) for _ in range(n_steps)]

        assert griffin_alpha_policy.predict_action_chunk.call_count == 1
        for action in actions:
            assert action.shape == (1, action_dim)

    def test_predict_action_chunk_resamples_to_native_chunk_size(
        self,
        griffin_alpha_config,
        tiny_gemma4_model,
        fast_action_tokenizer,
    ):
        native_chunk_size = griffin_alpha_config.horizon + 2
        config = replace(griffin_alpha_config, resample_action_chunk_size=native_chunk_size)
        with patch(
            "lerobot_policy_griffin_alpha.modeling_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            policy = GriffinAlphaPolicy(config, gemma4_model=tiny_gemma4_model)

        action_dim = policy.config.max_action_dim
        decoded_chunk = torch.randn(1, policy.config.horizon, action_dim)
        with patch.object(policy.model, "generate", return_value=torch.tensor([[1, 2, 3]])):
            with patch.object(policy, "_decode_action_tokens", return_value=decoded_chunk):
                chunk = policy.predict_action_chunk({"input_ids": torch.tensor([[1]])})

        assert chunk.shape == (1, native_chunk_size, action_dim)

    def test_save_and_load_round_trip(
        self, griffin_alpha_config, tiny_gemma4_model, fast_action_tokenizer, tmp_path
    ):
        with patch(
            "lerobot_policy_griffin_alpha.modeling_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            # Build a resized policy (production shape) and round-trip it through disk.
            policy = GriffinAlphaPolicy.from_gemma4_model(tiny_gemma4_model, config=griffin_alpha_config)
            expected_vocab = policy.config.gemma4_config.text_config.vocab_size
            policy.save_pretrained(tmp_path)
            reloaded = GriffinAlphaPolicy.from_pretrained(tmp_path)

        assert isinstance(reloaded, GriffinAlphaPolicy)
        assert reloaded.config.gemma4_config.text_config.vocab_size == expected_vocab

        # The generation config is persisted alongside the weights and restored on reload.
        assert (tmp_path / "generation_config.json").is_file()
        assert reloaded.model.generation_config.do_sample is False
        assert reloaded.model.generation_config.max_new_tokens == DEFAULT_MAX_NEW_TOKENS

        # The source policy is bfloat16 (base model is bf16), and reload must preserve that:
        # reload rebuilds the model with the plain float32 constructor, so this guards the cast.
        assert all(p.dtype == torch.bfloat16 for p in policy.parameters())
        assert all(p.dtype == torch.bfloat16 for p in reloaded.parameters())

        # Every weight is identical.
        orig_sd = policy.state_dict()
        reloaded_sd = reloaded.state_dict()
        assert orig_sd.keys() == reloaded_sd.keys()
        for key in orig_sd:
            assert torch.equal(orig_sd[key].cpu(), reloaded_sd[key].cpu()), key

    def test_from_gemma4_model_converts(
        self,
        tiny_gemma4_model,
        tiny_gemma4_config,
        fast_action_tokenizer,
    ):
        target_vocab_size = tiny_gemma4_config.text_config.vocab_size + 2048 + 256  # action + proprio vocab

        with patch(
            "lerobot_policy_griffin_alpha.modeling_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            policy = GriffinAlphaPolicy.from_gemma4_model(tiny_gemma4_model)

        # resize_token_embeddings mutates the shared config in place, so derive the
        # original base vocab from the (pre-computed) target size instead of re-reading it.
        base_vocab = target_vocab_size - 2048 - 256

        assert isinstance(policy, GriffinAlphaPolicy)
        assert policy.config.gemma4_config is tiny_gemma4_config
        assert policy.config.gemma4_config.text_config.vocab_size == target_vocab_size
        assert policy.model is tiny_gemma4_model
        assert policy.fast_tokenizer is fast_action_tokenizer
        # The action token range is derived from the base vocab, not the 262144 default.
        assert policy.config.action_token_min == base_vocab
        assert policy.config.action_token_max == base_vocab + 2048 - 1
        # Conversion preserves the base model's bfloat16 precision (no upcast).
        assert all(p.dtype == torch.bfloat16 for p in policy.parameters())
        # Conversion seeds deterministic decoding params on the generation config.
        assert policy.model.generation_config.do_sample is False
        assert policy.model.generation_config.max_new_tokens == DEFAULT_MAX_NEW_TOKENS

    def test_from_gemma4_pretrained_converts(
        self,
        tiny_gemma4_model,
        tiny_gemma4_config,
        fast_action_tokenizer,
    ):
        target_vocab_size = tiny_gemma4_config.text_config.vocab_size + 2048 + 256  # action + proprio vocab

        with patch(
            "lerobot_policy_griffin_alpha.modeling_griffin_alpha.AutoConfig.from_pretrained",
            return_value=tiny_gemma4_config,
        ):
            with patch(
                "lerobot_policy_griffin_alpha.modeling_griffin_alpha.Gemma4ForConditionalGeneration.from_pretrained",
                return_value=tiny_gemma4_model,
            ):
                with patch(
                    "lerobot_policy_griffin_alpha.modeling_griffin_alpha.AutoProcessor.from_pretrained",
                    return_value=fast_action_tokenizer,
                ):
                    policy = GriffinAlphaPolicy.from_gemma4_pretrained("fake/path")

        assert isinstance(policy, GriffinAlphaPolicy)
        assert policy.config.gemma4_config is tiny_gemma4_config
        assert policy.config.gemma4_config.text_config.vocab_size == target_vocab_size
        assert policy.model is tiny_gemma4_model
        assert policy.fast_tokenizer is fast_action_tokenizer

    def test_from_gemma4_model_no_resize_keeps_vocab_and_derives_token_range(
        self,
        tiny_gemma4_model,
        fast_action_tokenizer,
    ):
        action_vocab_size = 2048
        proprio_vocab_size = 256
        base_vocab = tiny_gemma4_model.config.text_config.vocab_size
        # Simulate a model trained via lerobot_training.py: its embeddings were already
        # resized before training, so the vocab already contains the action/proprio tokens.
        adapted_vocab = base_vocab + action_vocab_size + proprio_vocab_size
        tiny_gemma4_model.resize_token_embeddings(adapted_vocab)

        with patch(
            "lerobot_policy_griffin_alpha.modeling_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            policy = GriffinAlphaPolicy.from_gemma4_model(
                tiny_gemma4_model, resize_embeddings=False
            )

        assert isinstance(policy, GriffinAlphaPolicy)
        assert policy.model is tiny_gemma4_model
        # No second resize: the vocab is unchanged from the adapted checkpoint.
        assert policy.config.gemma4_config.text_config.vocab_size == adapted_vocab
        # The action token range is derived from the existing vocab, not the 262144 default.
        assert policy.config.action_token_min == base_vocab
        assert policy.config.action_token_max == base_vocab + action_vocab_size - 1
        assert policy.model.generation_config.do_sample is False
        assert policy.model.generation_config.max_new_tokens == DEFAULT_MAX_NEW_TOKENS

    def test_from_gemma4_model_no_resize_rejects_unadapted_model(
        self,
        tiny_gemma4_config,
        fast_action_tokenizer,
    ):
        # A base model whose vocab is smaller than action + proprio tokens cannot be a
        # trained Griffin Alpha checkpoint.
        small_config = Gemma4Config.from_dict(tiny_gemma4_config.to_dict())
        small_config.text_config.vocab_size = 100
        small_model = Gemma4ForConditionalGeneration(small_config).to(torch.bfloat16)

        with patch(
            "lerobot_policy_griffin_alpha.modeling_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            with pytest.raises(ValueError, match="does not appear to be an adapted"):
                GriffinAlphaPolicy.from_gemma4_model(small_model, resize_embeddings=False)

    def test_from_gemma4_pretrained_no_resize_converts(
        self,
        tiny_gemma4_model,
        tiny_gemma4_config,
        fast_action_tokenizer,
    ):
        action_vocab_size = 2048
        proprio_vocab_size = 256
        base_vocab = tiny_gemma4_model.config.text_config.vocab_size
        adapted_vocab = base_vocab + action_vocab_size + proprio_vocab_size
        # Mirror a consolidated trained checkpoint: weights and config already carry the
        # enlarged vocab, so from_pretrained returns an already-resized model.
        tiny_gemma4_model.resize_token_embeddings(adapted_vocab)

        with patch(
            "lerobot_policy_griffin_alpha.modeling_griffin_alpha.AutoConfig.from_pretrained",
            return_value=tiny_gemma4_config,
        ):
            with patch(
                "lerobot_policy_griffin_alpha.modeling_griffin_alpha.Gemma4ForConditionalGeneration.from_pretrained",
                return_value=tiny_gemma4_model,
            ):
                with patch(
                    "lerobot_policy_griffin_alpha.modeling_griffin_alpha.AutoProcessor.from_pretrained",
                    return_value=fast_action_tokenizer,
                ):
                    policy = GriffinAlphaPolicy.from_gemma4_pretrained(
                        "fake/path", resize_embeddings=False
                    )

        assert isinstance(policy, GriffinAlphaPolicy)
        assert policy.model is tiny_gemma4_model
        assert policy.config.gemma4_config.text_config.vocab_size == adapted_vocab
        assert policy.config.action_token_min == base_vocab
        assert policy.config.action_token_max == base_vocab + action_vocab_size - 1
