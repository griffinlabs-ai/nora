from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from lerobot_policy_griffin_alpha.configuration_griffin_alpha import GriffinAlphaConfig
from lerobot_policy_griffin_alpha.processor_griffin_alpha import (
    GriffinAlphaAddBatchDimensionProcessorStep,
    GriffinAlphaImageTransform,
    GriffinAlphaVLMInputProcessorStep,
    RelativeActionWithSE3ProcessorStep,
    ResampleActionProcessorStep,
    SE3MatrixToXYZRot6DProcessorStep,
    _index_optional_list,
    make_griffin_alpha_pre_post_processors,
    make_proprio_state_tokens,
    map_fast_token_to_vlm_action,
    map_normalized_state_to_vlm_proprio,
)
from .helpers import make_sample_env_transition


@pytest.fixture
def vlm_step(griffin_alpha_config: GriffinAlphaConfig, fast_action_tokenizer, gemma4_vlm_processor):
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
def vlm_step_mocks(fast_action_tokenizer, gemma4_vlm_processor):
    return patch(
        "lerobot_policy_griffin_alpha.processor_griffin_alpha.AutoProcessor.from_pretrained",
        return_value=fast_action_tokenizer,
    ), patch.object(
        GriffinAlphaVLMInputProcessorStep,
        "_make_vla_processor",
        return_value=gemma4_vlm_processor,
    )


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

    def test_get_center_crop_transform_size(self):
        crop = GriffinAlphaImageTransform.get_center_crop_transform((224, 224), 0.9)
        linear_scale = 0.9**0.5
        expected_h = round(224 * linear_scale)
        expected_w = round(224 * linear_scale)
        assert crop.size == (expected_h, expected_w)

    def test_center_crop_reduces_size_and_preserves_channels(self):
        transform = GriffinAlphaImageTransform()
        image = torch.rand(3, 224, 224)
        output = transform.center_crop(image)
        assert output.shape[0] == 3
        assert output.shape[1] < 224
        assert output.shape[2] < 224


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


class TestRelativeActionWithSE3ProcessorStep:
    def test_real_only_subtracts_state(self):
        step = RelativeActionWithSE3ProcessorStep(
            mask=[True, True, True],
            se3_segment_start_idxs=None,
            state_key="observation.state",
        )
        action = torch.tensor([[[1.0, 2.0, 3.0]]])
        state = torch.tensor([[1.0, 3.0, 1.0]])
        transition = {
            "action": action,
            "observation": {"observation.state": state},
        }

        result = step(transition)
        expected = action - state.unsqueeze(-2)
        assert torch.equal(result["action"], expected)

    def test_action_none_returns_identity(self):
        step = RelativeActionWithSE3ProcessorStep(
            mask=[True, True, True],
            state_key="observation.state",
        )
        transition = {
            "action": None,
            "observation": {"observation.state": torch.tensor([[1.0, 2.0, 3.0]])},
        }

        result = step(transition)
        assert result is transition

    def test_get_config_round_trip_via_registry(self):
        step = RelativeActionWithSE3ProcessorStep(
            mask=[True, False, True],
            se3_segment_start_idxs=[0],
            state_key="observation.state",
        )

        cfg = step.get_config()
        assert isinstance(cfg["mask"], list)
        assert isinstance(cfg["se3_segment_start_idxs"], list)
        restored = ProcessorStepRegistry.get("griffinlabs/relative_action_with_se3_processor")(**cfg)

        assert restored.mask == step.mask
        assert torch.equal(restored._mask, torch.tensor(step.mask, dtype=torch.bool))
        assert restored.se3_segment_start_idxs == step.se3_segment_start_idxs
        assert restored.state_key == step.state_key


class TestSE3MatrixToXYZRot6DProcessorStep:
    def test_converts_action_and_state_segments(self):
        step = SE3MatrixToXYZRot6DProcessorStep(
            se3_segment_start_idxs=[0],
            state_key="observation.state",
        )
        se3_flat = torch.eye(4).reshape(1, 1, 16)
        state = torch.eye(4).reshape(1, 16)
        transition = {
            "action": se3_flat.repeat(1, 2, 1),
            "observation": {"observation.state": state},
        }

        result = step(transition)
        assert result["action"].shape[-1] == 9
        assert result["observation"]["observation.state"].shape[-1] == 9

    def test_action_none_still_converts_state(self):
        step = SE3MatrixToXYZRot6DProcessorStep(
            se3_segment_start_idxs=[0],
            state_key="observation.state",
        )
        transition = {
            "action": None,
            "observation": {"observation.state": torch.eye(4).reshape(1, 16)},
        }

        result = step(transition)
        assert result["action"] is None
        assert result["observation"]["observation.state"].shape[-1] == 9

    def test_get_config_round_trip_via_registry(self):
        step = SE3MatrixToXYZRot6DProcessorStep(
            se3_segment_start_idxs=[0],
            state_key="observation.state",
        )
        cfg = step.get_config()
        assert isinstance(cfg["se3_segment_start_idxs"], list)

        restored = ProcessorStepRegistry.get("griffinlabs/se3_mat_to_xyz_rot6d_processor")(**cfg)
        assert restored.se3_segment_start_idxs == step.se3_segment_start_idxs
        assert restored.state_key == step.state_key


class TestResampleActionProcessorStep:
    def test_divisible_chunk_subsamples(self):
        step = ResampleActionProcessorStep(target_chunk_size=2)
        action = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
        transition = {"action": action, "observation": {"observation.state": torch.zeros(1, 3)}}

        result = step(transition)
        expected = action[:, 1::2, :]
        assert result["action"].shape == (1, 2, 3)
        assert torch.equal(result["action"], expected)

    def test_non_divisible_chunk_interpolates(self):
        step = ResampleActionProcessorStep(target_chunk_size=3)
        action = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
        transition = {"action": action, "observation": {"observation.state": torch.zeros(1, 3)}}

        result = step(transition)
        assert result["action"].shape == (1, 3, 3)

    def test_equal_chunk_size_returns_unchanged(self):
        step = ResampleActionProcessorStep(target_chunk_size=4)
        action = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
        transition = {"action": action, "observation": {"observation.state": torch.zeros(1, 3)}}

        result = step(transition)
        assert result is transition

    def test_action_none_returns_unchanged(self):
        step = ResampleActionProcessorStep(target_chunk_size=2)
        transition = {"action": None, "observation": {"observation.state": torch.zeros(1, 3)}}

        result = step(transition)
        assert result is transition

    def test_get_config_round_trip_via_registry(self):
        step = ResampleActionProcessorStep(target_chunk_size=2, state_key="observation.state")
        cfg = step.get_config()
        restored = ProcessorStepRegistry.get("griffinlabs/resample_action_processor")(**cfg)

        assert restored.target_chunk_size == step.target_chunk_size
        assert restored.state_key == step.state_key


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

    def test_get_config_includes_conditioning_fields(self, vlm_step):
        config = vlm_step.get_config()
        assert "embodiment_prompt" in config
        assert "arm_control_mode" in config
        assert "predict_subtask" in config
        assert config["embodiment_prompt"] == vlm_step.embodiment_prompt
        assert config["arm_control_mode"] == vlm_step.arm_control_mode
        assert config["predict_subtask"] == vlm_step.predict_subtask

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
        assert len(pre.steps) == 4
        assert isinstance(pre.steps[0], RenameObservationsProcessorStep)
        assert isinstance(pre.steps[1], GriffinAlphaAddBatchDimensionProcessorStep)
        assert isinstance(pre.steps[2], GriffinAlphaVLMInputProcessorStep)
        assert isinstance(pre.steps[3], DeviceProcessorStep)
        assert isinstance(post.steps[0], DeviceProcessorStep)

    def test_relative_action_mask_inserts_step(
        self,
        griffin_alpha_config: GriffinAlphaConfig,
        fast_action_tokenizer,
        gemma4_vlm_processor,
    ):
        config_with_relative = replace(griffin_alpha_config, relative_action_mask=[True] * 7)
        with patch(
            "lerobot_policy_griffin_alpha.processor_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            with patch.object(
                GriffinAlphaVLMInputProcessorStep,
                "_make_vla_processor",
                return_value=gemma4_vlm_processor,
            ):
                pre, _ = make_griffin_alpha_pre_post_processors(config_with_relative)

        assert len(pre.steps) == 5
        assert isinstance(pre.steps[2], RelativeActionWithSE3ProcessorStep)
        assert isinstance(pre.steps[3], GriffinAlphaVLMInputProcessorStep)
        assert isinstance(pre.steps[4], DeviceProcessorStep)

    def test_resample_action_step_inserted_before_vlm(
        self,
        griffin_alpha_config: GriffinAlphaConfig,
        fast_action_tokenizer,
        gemma4_vlm_processor,
    ):
        config_with_resample = replace(griffin_alpha_config, resample_action_to_horizon=True)
        with patch(
            "lerobot_policy_griffin_alpha.processor_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            with patch.object(
                GriffinAlphaVLMInputProcessorStep,
                "_make_vla_processor",
                return_value=gemma4_vlm_processor,
            ):
                pre, _ = make_griffin_alpha_pre_post_processors(config_with_resample)

        assert isinstance(pre.steps[2], ResampleActionProcessorStep)
        assert isinstance(pre.steps[3], GriffinAlphaVLMInputProcessorStep)

    def test_resample_and_se3_enabled_raises_value_error(
        self,
        griffin_alpha_config: GriffinAlphaConfig,
    ):
        invalid_config = replace(
            griffin_alpha_config,
            se3_segment_start_idxs=[0],
            resample_action_to_horizon=True,
        )
        with pytest.raises(ValueError, match="Resampling and SE\\(3\\) matrices cannot be used at the same time"):
            make_griffin_alpha_pre_post_processors(invalid_config)

    def test_dataset_stats_inserts_normalizer_before_vlm(
        self,
        griffin_alpha_config: GriffinAlphaConfig,
        fast_action_tokenizer,
        gemma4_vlm_processor,
    ):
        dataset_stats = {
            OBS_STATE: {"q01": torch.zeros(7), "q99": torch.ones(7)},
            "action": {"q01": torch.zeros(7), "q99": torch.ones(7)},
        }
        with patch(
            "lerobot_policy_griffin_alpha.processor_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            with patch.object(
                GriffinAlphaVLMInputProcessorStep,
                "_make_vla_processor",
                return_value=gemma4_vlm_processor,
            ):
                pre, _ = make_griffin_alpha_pre_post_processors(
                    griffin_alpha_config,
                    dataset_stats=dataset_stats,
                )

        assert isinstance(pre.steps[2], NormalizerProcessorStep)
        assert isinstance(pre.steps[3], GriffinAlphaVLMInputProcessorStep)

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

    def test_preprocessor_save_and_load_round_trip_with_relative_action_mask(
        self,
        griffin_alpha_config: GriffinAlphaConfig,
        fast_action_tokenizer,
        gemma4_vlm_processor,
        tmp_path,
    ):
        config_with_relative = replace(griffin_alpha_config, relative_action_mask=[True] * 7)
        with patch(
            "lerobot_policy_griffin_alpha.processor_griffin_alpha.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            with patch.object(
                GriffinAlphaVLMInputProcessorStep,
                "_make_vla_processor",
                return_value=gemma4_vlm_processor,
            ):
                pre, _ = make_griffin_alpha_pre_post_processors(config_with_relative)
                pre.save_pretrained(tmp_path)

                loaded = DataProcessorPipeline.from_pretrained(
                    tmp_path,
                    config_filename="policy_preprocessor.json",
                    to_transition=lambda data: data,
                    to_output=lambda tr: tr[TransitionKey.COMPLEMENTARY_DATA],
                )

        assert isinstance(loaded.steps[2], RelativeActionWithSE3ProcessorStep)
        assert loaded.steps[2].mask == [True] * 7
        assert torch.equal(loaded.steps[2]._mask, torch.tensor([True] * 7, dtype=torch.bool))

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


class TestGriffinAlphaVLMInputProcessorConditioning:
    def _build_step(self, config, vlm_step_mocks):
        auto_patch, vla_patch = vlm_step_mocks
        with auto_patch, vla_patch:
            return GriffinAlphaVLMInputProcessorStep.from_griffin_alpha_config(config)

    def _run_and_get_prompt(self, step, transition):
        spy = MagicMock(wraps=step._vla_processor)
        step._vla_processor = spy
        step(transition)
        return spy.call_args.kwargs["text"][0]

    def test_arm_control_mode_from_config(
        self, griffin_alpha_config, vlm_step_mocks, sample_env_transition
    ):
        step = self._build_step(
            replace(griffin_alpha_config, arm_control_mode="eef_pose"), vlm_step_mocks
        )
        del sample_env_transition[TransitionKey.INFO]["arm_control_mode"]
        prompt = self._run_and_get_prompt(step, sample_env_transition)
        assert "arm control mode: eef_pose" in prompt

    def test_info_arm_control_mode_overrides_config(
        self, griffin_alpha_config, vlm_step_mocks, sample_env_transition
    ):
        step = self._build_step(
            replace(griffin_alpha_config, arm_control_mode="eef_pose"), vlm_step_mocks
        )
        # helper default info already sets arm_control_mode == ["joint"]
        prompt = self._run_and_get_prompt(step, sample_env_transition)
        assert "arm control mode: joint" in prompt

    def test_arm_control_mode_missing_raises(self, vlm_step, sample_env_transition):
        del sample_env_transition[TransitionKey.INFO]["arm_control_mode"]
        with pytest.raises(ValueError, match="arm_control_mode must be provided"):
            vlm_step(sample_env_transition)

    def test_embodiment_prompt_from_config(
        self, griffin_alpha_config, vlm_step_mocks, sample_env_transition
    ):
        step = self._build_step(
            replace(griffin_alpha_config, embodiment_prompt="ConfigRobot"), vlm_step_mocks
        )
        del sample_env_transition[TransitionKey.INFO]["embodiment_prompt"]
        prompt = self._run_and_get_prompt(step, sample_env_transition)
        assert "embodiment: ConfigRobot" in prompt

    def test_predict_subtask_from_config_true(
        self, griffin_alpha_config, vlm_step_mocks, sample_env_transition
    ):
        step = self._build_step(
            replace(griffin_alpha_config, predict_subtask=True), vlm_step_mocks
        )
        prompt = self._run_and_get_prompt(step, sample_env_transition)
        assert "predict subtask: true" in prompt

    def test_predict_subtask_defaults_false(self, vlm_step, sample_env_transition):
        prompt = self._run_and_get_prompt(vlm_step, sample_env_transition)
        assert "predict subtask: false" in prompt
