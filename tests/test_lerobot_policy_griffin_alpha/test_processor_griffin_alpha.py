from unittest.mock import MagicMock, patch

import pytest
import torch
from lerobot.processor import DeviceProcessorStep, RenameObservationsProcessorStep
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from lerobot_policy_griffin_alpha.configuration_griffin_alpha import GriffinAlphaConfig
from lerobot_policy_griffin_alpha.processor_griffin_alpha import (
    GriffinAlphaAddBatchDimensionProcessorStep,
    GriffinAlphaImageTransform,
    GriffinAlphaVLMInputProcessorStep,
    _index_optional_list,
    make_griffin_alpha_pre_post_processors,
    make_proprio_state_tokens,
    map_fast_token_to_vlm_action,
    map_normalized_state_to_vlm_proprio,
)
from .helpers import make_sample_env_transition


class TestHelperFunctions:
    def test_index_optional_list_none(self):
        assert _index_optional_list(None, 0) is None

    def test_index_optional_list_indexing(self):
        assert _index_optional_list(["a", "b"], 1) == "b"

    def test_map_fast_token_to_vlm_action(self):
        assert map_fast_token_to_vlm_action(["0", "1"]) == "<robot_action_0><robot_action_1>"

    def test_make_proprio_state_tokens(self):
        tokens = make_proprio_state_tokens(3)
        assert tokens == ["<proprio_state_0>", "<proprio_state_1>", "<proprio_state_2>"]

    def test_map_normalized_state_to_vlm_proprio_clamps_and_buckets(self):
        state = torch.tensor([-2.0, 0.0, 2.0])
        result = map_normalized_state_to_vlm_proprio(state, vocab_size=4)
        assert result.startswith("<proprio_state_")
        assert result.endswith(">")

    def test_map_normalized_state_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="proprio_vocab_size must be positive"):
            map_normalized_state_to_vlm_proprio(torch.tensor([0.0]), vocab_size=0)


class TestGriffinAlphaImageTransform:
    def test_get_random_crop_transform_size(self):
        crop = GriffinAlphaImageTransform.get_random_crop_transform((224, 224), 0.9)
        linear_scale = 0.9**0.5
        expected_h = round(224 * linear_scale)
        expected_w = round(224 * linear_scale)
        assert crop.size == (expected_h, expected_w)

    def test_call_preserves_channels(self):
        transform = GriffinAlphaImageTransform()
        image = torch.rand(3, 224, 224)
        output = transform(image)
        assert output.shape[0] == 3
        assert output.shape[1] <= 224
        assert output.shape[2] <= 224


class TestGriffinAlphaAddBatchDimensionProcessorStep:
    def test_unbatched_adds_batch_dimension(self, griffin_alpha_config: GriffinAlphaConfig):
        step = GriffinAlphaAddBatchDimensionProcessorStep()
        transition = make_sample_env_transition(griffin_alpha_config, batched=False)
        result = step(transition)

        # check state
        assert result[TransitionKey.OBSERVATION][OBS_STATE].ndim == 2
        assert result[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 7)
        # check images
        for image_key in griffin_alpha_config.image_keys:
            assert isinstance(result[TransitionKey.OBSERVATION][image_key], list)
            assert len(result[TransitionKey.OBSERVATION][image_key]) == 1
            assert result[TransitionKey.OBSERVATION][image_key][0].shape == (3, 64, 64)
        # check info
        assert result[TransitionKey.INFO]["arm_control_mode"] == ["joint"]
        assert result[TransitionKey.INFO]["embodiment_prompt"] == ["test_robot"]
        assert result[TransitionKey.INFO]["n_action_dims"] == [7]
        # check complementary data
        assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == ["pick up the cup"]
        assert result[TransitionKey.COMPLEMENTARY_DATA]["subtask"] == [""]
        # check action
        assert result[TransitionKey.ACTION] is None


    def test_already_batched_unchanged(self, griffin_alpha_config: GriffinAlphaConfig, sample_env_transition):
        step = GriffinAlphaAddBatchDimensionProcessorStep()
        result = step(sample_env_transition)
        assert result[TransitionKey.OBSERVATION][OBS_STATE].shape == sample_env_transition[TransitionKey.OBSERVATION][OBS_STATE].shape

    def test_serialization_methods(self):
        step = GriffinAlphaAddBatchDimensionProcessorStep()
        assert step.get_config() == {}
        assert step.state_dict() == {}
        step.load_state_dict({})


class TestGriffinAlphaVLMInputProcessorStep:
    @pytest.fixture
    def vlm_step(self, griffin_alpha_config: GriffinAlphaConfig, fast_action_tokenizer, gemma4_vlm_processor):
        with patch(
            "lerobot_policy_griffin_alpha.processor_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            with patch.object(
                GriffinAlphaVLMInputProcessorStep,
                "_make_vla_processor",
                return_value=gemma4_vlm_processor,
            ):
                return GriffinAlphaVLMInputProcessorStep.from_griffin_alpha_config(griffin_alpha_config)

    @pytest.fixture
    def vlm_step_mocks(self, fast_action_tokenizer, gemma4_vlm_processor):
        return patch(
            "lerobot_policy_griffin_alpha.processor_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ), patch.object(
            GriffinAlphaVLMInputProcessorStep,
            "_make_vla_processor",
            return_value=gemma4_vlm_processor,
        )

    def test_unlabeled_path_output_keys(self, vlm_step, sample_env_transition):
        result = vlm_step(sample_env_transition)
        batch_input = result[TransitionKey.COMPLEMENTARY_DATA]
        assert "input_ids" in batch_input
        assert "attention_mask" in batch_input
        assert "labels" not in batch_input

    def test_labeled_path_has_labels(self, vlm_step, sample_training_transition):
        result = vlm_step(sample_training_transition)
        batch_input = result[TransitionKey.COMPLEMENTARY_DATA]
        assert "labels" in batch_input
        labels = batch_input["labels"]
        sot_token_id = vlm_step._vla_processor.tokenizer.sot_token_id
        sot_indices = (labels[0] == sot_token_id).nonzero(as_tuple=False)
        if sot_indices.numel() > 0:
            last_sot = sot_indices[-1].item()
            assert (labels[0, :last_sot] == -100).all()

    def test_proprio_tokens_in_prompt(self, vlm_step, sample_env_transition):
        result = vlm_step(sample_env_transition)
        batch_input = result[TransitionKey.COMPLEMENTARY_DATA]
        decoded = vlm_step._vla_processor.tokenizer.decode(batch_input["input_ids"][0])
        assert "<proprio_state_" in decoded

    def test_action_tokens_in_prompt(self, vlm_step, sample_training_transition):
        result = vlm_step(sample_training_transition)
        batch_input = result[TransitionKey.COMPLEMENTARY_DATA]
        decoded = vlm_step._vla_processor.tokenizer.decode(batch_input["input_ids"][0])
        assert "<robot_action_" in decoded

    def test_round_trip_from_get_config(
        self, vlm_step, vlm_step_mocks, sample_env_transition
    ):
        auto_patch, vla_patch = vlm_step_mocks
        with auto_patch, vla_patch:
            restored = GriffinAlphaVLMInputProcessorStep(**vlm_step.get_config())
            result = restored(sample_env_transition)
        assert "input_ids" in result[TransitionKey.COMPLEMENTARY_DATA]
        assert vlm_step.get_config() == restored.get_config()
        assert vlm_step == restored

    def test_get_config_includes_max_sequence_length(
        self, vlm_step, griffin_alpha_config: GriffinAlphaConfig
    ):
        config = vlm_step.get_config()
        assert "max_sequence_length" in config
        assert config["max_sequence_length"] == griffin_alpha_config.max_sequence_length

    def test_forwards_truncation_to_max_sequence_length(self, vlm_step, sample_env_transition):
        spy = MagicMock(wraps=vlm_step._vla_processor)
        vlm_step._vla_processor = spy
        vlm_step(sample_env_transition)
        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == vlm_step.max_sequence_length

    def test_serialization_methods(self, vlm_step):
        assert vlm_step.state_dict() == {}
        vlm_step.load_state_dict({})


class TestMakeGriffinAlphaPrePostProcessors:
    def test_returns_two_pipelines(self, griffin_alpha_config: GriffinAlphaConfig, fast_action_tokenizer, gemma4_vlm_processor):
        with patch(
            "lerobot_policy_griffin_alpha.processor_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            with patch.object(
                GriffinAlphaVLMInputProcessorStep,
                "_make_vla_processor",
                return_value=gemma4_vlm_processor,
            ):
                pre, post = make_griffin_alpha_pre_post_processors(griffin_alpha_config)

        assert isinstance(pre, DataProcessorPipeline)
        assert isinstance(post, DataProcessorPipeline)
        assert pre.name == POLICY_PREPROCESSOR_DEFAULT_NAME
        assert post.name == POLICY_POSTPROCESSOR_DEFAULT_NAME
        assert isinstance(pre.steps[0], RenameObservationsProcessorStep)
        assert isinstance(pre.steps[1], GriffinAlphaAddBatchDimensionProcessorStep)
        assert isinstance(pre.steps[2], GriffinAlphaVLMInputProcessorStep)
        assert isinstance(pre.steps[3], DeviceProcessorStep)
        assert isinstance(post.steps[0], DeviceProcessorStep)

    def test_preprocessor_save_and_load_round_trip(
        self,
        griffin_alpha_config: GriffinAlphaConfig,
        fast_action_tokenizer,
        gemma4_vlm_processor,
        tmp_path,
    ):
        with patch(
            "lerobot_policy_griffin_alpha.processor_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            with patch.object(
                GriffinAlphaVLMInputProcessorStep,
                "_make_vla_processor",
                return_value=gemma4_vlm_processor,
            ):
                pre, _ = make_griffin_alpha_pre_post_processors(griffin_alpha_config)
                pre.save_pretrained(tmp_path)

                loaded = DataProcessorPipeline.from_pretrained(
                    tmp_path,
                    config_filename="policy_preprocessor.json",
                    to_transition=lambda data: data,
                    to_output=lambda tr: tr[TransitionKey.COMPLEMENTARY_DATA],
                )

                unbatched = make_sample_env_transition(griffin_alpha_config, batched=False)
                result = loaded(unbatched)

        assert isinstance(loaded.steps[1], GriffinAlphaAddBatchDimensionProcessorStep)
        assert isinstance(loaded.steps[2], GriffinAlphaVLMInputProcessorStep)
        assert "input_ids" in result
        assert isinstance(loaded.steps[2].image_keys, tuple)
        assert loaded.steps[2].image_keys == griffin_alpha_config.image_keys

    def test_postprocessor_save_and_load_round_trip(
        self,
        griffin_alpha_config: GriffinAlphaConfig,
        fast_action_tokenizer,
        gemma4_vlm_processor,
        tmp_path,
    ):
        with patch(
            "lerobot_policy_griffin_alpha.processor_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            with patch.object(
                GriffinAlphaVLMInputProcessorStep,
                "_make_vla_processor",
                return_value=gemma4_vlm_processor,
            ):
                _, post = make_griffin_alpha_pre_post_processors(griffin_alpha_config)
                post.save_pretrained(tmp_path)

        loaded = DataProcessorPipeline.from_pretrained(
            tmp_path,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        action = torch.randn(1, griffin_alpha_config.max_action_dim)
        result = loaded(action)

        assert isinstance(loaded.steps[0], DeviceProcessorStep)
        assert isinstance(result, torch.Tensor)
        assert result.shape == action.shape
