
import os
import logging
from typing import List, Any, Optional
from dataclasses import dataclass
import pathlib

import torch
from torch.utils.data import DataLoader, default_collate

from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import get_scheduler

import lerobot.processor
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset
from lerobot.configs.types import  NormalizationMode, PipelineFeatureType
from qwen_vl_utils import process_vision_info
import numpy as np
from tqdm import tqdm


import torchvision



logger = get_logger(__name__)

# --- 1. Configuration ---
class TrainingConfig:
    def __init__(
        self,
        per_device_batch_size: int = 1,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 1,
        num_warmup_steps: int = 1000,
        max_train_steps: int = 60000,
        output_dir: str = './nora_finetune_object',
        resume_from_checkpoint: str = '',
        load_model_weights: Optional[str] = None,
        lerobot_dataset_repo_id: str | None = None,
        lerobot_dataset_root: str = "/home/ubuntu/agibot-world-lerobot-us-east-1/data/sample_dataset_lerobot_depth/agibotworld/",
        wandb_project_name: str = "Nora VLA with LeRobotDataset",
        checkpoint_save_frequency: int = 20000,
        logging_frequency: int = 100,
        gradient_clipping: Optional[float] = None,
        invert_grippler_action: bool = True,
    ):
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.max_train_steps = max_train_steps
        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.load_model_weights = load_model_weights
        self.lerobot_dataset_repo_id = lerobot_dataset_repo_id
        self.lerobot_dataset_root = lerobot_dataset_root
        self.wandb_project_name = wandb_project_name
        self.checkpoint_save_frequency = checkpoint_save_frequency
        self.logging_frequency = logging_frequency
        self.gradient_clipping = gradient_clipping
        ## In Nora's pretraining, the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open. While some environments have -1 = open, +1 = close. Setting this to True will invert the gripper action(map -1 to 1, +1 to 0)
        self.invert_grippler_action = invert_grippler_action
        self.image_key = 'observation.images.head'
        self.action_key = 'action'
        self.task_key = 'task'
        self.fps = 30
        self.action_chunk_size = 50

# --- 2. Data Loading and Preprocessing ---
def load_and_prepare_dataset(config: TrainingConfig) -> MultiLeRobotDataset:
    """Loads and prepares the LeRobot dataset."""
    delta_timestamps = [i / config.fps for i in range(config.action_chunk_size)]
    return MultiLeRobotDataset(
        [p.name for p in pathlib.Path(config.lerobot_dataset_root).glob("task_*")],
        root = config.lerobot_dataset_root,
        delta_timestamps = {
            "actions.joint.position": delta_timestamps,
            "actions.effector.position": delta_timestamps,
        }
    )

def agibot_world_to_nora_instance(batch: dict[str, Any], img_key):
    """
    Convert from raw dataset batch format to format that is ready to be converted to `EnvTransition`.
    Merges `actions.joint.position` and `actions.effector.position` into `action`, discarding other actions.

    Also discards all `observation.*` except the one matching `img_key`.
    """
    image = batch[img_key]
    prev_dim_sizes = batch['actions.joint.position'].shape[:2]
    action = torch.cat(
        [
            batch['actions.joint.position'].view(*prev_dim_sizes, 2, 7),
            1 - batch['actions.effector.position'].view(*prev_dim_sizes, 2, 1)
        ],
        dim = -1
    ).view(*prev_dim_sizes, 16)
    batch = {k:v for k, v in batch.items() if not k.startswith('actions.') and not k.startswith('observation.')}
    batch[img_key] = image
    batch['action'] = action
    return batch

def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format."""
    return ''.join([f"<robot_action_{token}>" for token in tokens])

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

def inverse_transform_gripper_action(action, binarized_input=True):
    """
    Maps the gripper action  to the range [0, 1].
    Args:
        action (torch.Tensor): The action vector with the gripper action as the last dimension,
                             which has been transformed by invert_gripper_action and then
                             normalize_gripper_action.
        binarized_input (bool): Whether the input to normalize_gripper_action was binarized.
                                This affects the inverse transformation.

    """

    action[..., -1] = action[..., -1] * -1.0

    if binarized_input:
        # If the input was binarized, the values are -1 or +1.
        # Just map -1 to 0 and +1 to 1. Note that the previous line we have already flipped the sign.
        action[..., -1] = torch.where(action[..., -1] == -1, 0.0, 1.0)
    else:
        action[..., -1] = (action[..., -1] + 1) / 2

    return action

@dataclass
@lerobot.processor.ProcessorStepRegistry.register("nora_processor")
class NoraPolicyProcessorStep(lerobot.processor.ProcessorStep):

    config: TrainingConfig

    def __post_init__(self):
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        self.transformer_processor = AutoProcessor.from_pretrained('declare-lab/nora')
        self.transformer_processor.tokenizer.padding_side = 'left'

    def __call__(self, transition: lerobot.processor.EnvTransition) -> lerobot.processor.EnvTransition:
        pixel_values = transition['observation'][self.config.image_key]
        pixel_values = [
            torchvision.transforms.functional.to_pil_image(pv)
            for pv in pixel_values
        ]

        action = transition['action']
        lang = transition['complementary_data']['task']
        fast_tokens = self.fast_tokenizer(action.cpu())
        vlm_action = [map_fast_token_to_vlm_action(ft) for ft in fast_tokens]

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pv,
                        "resized_height": 224,
                        "resized_width": 224,},
                        {"type": "text", "text": l},

                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": act},
                    ],
                }
            ]
            for pv, l, act in zip(pixel_values, lang, vlm_action)
        ]

        text = self.transformer_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.transformer_processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        action_token_min = 151665
        action_token_max = 153712
        labels = batch_input['input_ids'].clone()

        for i in range(labels.size(0)):
            seq = labels[i]
            mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
            nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
            if nonzero_indices.numel() > 0:
                first_action_index = nonzero_indices[0].item()
                seq[:first_action_index] = -100
            else:
                seq[:] = -100

        labels[labels == self.transformer_processor.tokenizer.pad_token_id] = -100
        batch_input['labels'] = labels

        return lerobot.processor.create_transition(complementary_data = batch_input)

    def transform_features(self, features):
        return {
            PipelineFeatureType.ACTION: {},
            PipelineFeatureType.OBSERVATION: {},
        }

def make_policy_processor(
        config: TrainingConfig,
        norm_stats: dict[str, dict[str, np.ndarray]],
) -> lerobot.processor.PolicyProcessorPipeline:
    # if changing this to use other statistics, remember that gripper states are all transformed by 1-x,
    # so the stats would have to be transformed as well, and order reversed (e.g. p10 becomes p90)
    norm_stats = {
        "action": {
            "min": np.append(np.insert(norm_stats['actions.joint.position']['min'], 7, 0), 0),
            "max": np.append(np.insert(norm_stats['actions.joint.position']['max'], 7, 1), 1),
        }
    }

    norm_map = {
        'ACTION': NormalizationMode.MIN_MAX,
    }

    return lerobot.processor.PolicyProcessorPipeline(
        steps = [
            lerobot.processor.NormalizerProcessorStep({}, norm_map , norm_stats),
            NoraPolicyProcessorStep(config),
        ],
        to_transition=lambda batch:
            lerobot.processor.converters.batch_to_transition(agibot_world_to_nora_instance(batch, config.image_key)),
        to_output=lerobot.processor.converters.transition_to_batch,
    )


# --- 3. Model Initialization ---
def load_model_and_processor(config: TrainingConfig, accelerator: Accelerator) -> Qwen2_5_VLForConditionalGeneration:
    """Loads the model and processor."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'declare-lab/nora',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    if config.load_model_weights:
        tensors = {}
        from safetensors import safe_open
        with safe_open(config.load_model_weights, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
        accelerator.print("Pretrained weights loaded.")

    return model

# --- 4. Training Loop ---
def train(config: TrainingConfig):
    """Main training loop."""
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps,log_with="wandb")
    accelerator.dataloader_config.dispatch_batches = False
    logger.info(accelerator.state, main_process_only=False)

    accelerator.init_trackers(config.wandb_project_name, config=config)
        #wandb.init(project=config.wandb_project_name)

    model = load_model_and_processor(config, accelerator)

    with accelerator.main_process_first():
        dataset = load_and_prepare_dataset(config)
        policy_preprocessor = make_policy_processor(config, dataset.stats)

    train_dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=lambda examples: policy_preprocessor(default_collate(examples)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
        eps=1e-8,
    )

    max_train_steps = config.max_train_steps
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps*accelerator.num_processes,
        num_training_steps=config.max_train_steps*accelerator.num_processes
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader,lr_scheduler
    )

    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        accelerator.print(f"Resumed from local checkpoint: {config.resume_from_checkpoint}")

    total_batch_size = config.per_device_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num steps = {config.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    completed_steps = 0
    progress_bar = tqdm(range(completed_steps,max_train_steps), disable=not accelerator.is_local_main_process)

    while completed_steps < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if config.gradient_clipping is not None:
                        accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)
                    completed_steps += 1

                    optimizer.step()
                    lr_scheduler.step()

                    if completed_steps % config.logging_frequency == 0:
                        if accelerator.is_main_process:
                            total_norm = 0.0
                            for p in model.parameters():
                                if p.grad is not None:
                                    total_norm += p.grad.data.norm(2).item() ** 2

                            total_norm = total_norm**0.5
                            lr = lr_scheduler.get_last_lr()[0]

                            logger.info(f"Step {completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm}", main_process_only=True)
                            accelerator.log({"train_loss": loss.item(), "learning_rate": lr,"grad_norm":total_norm}, step=completed_steps)
                    #logger.info(f"Step {completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm}", main_process_only=True)
                    #accelerator.log({"train_loss": loss.item(), "learning_rate": lr,"grad_norm":total_norm}, step=completed_steps)

            if completed_steps % config.checkpoint_save_frequency == 0 and completed_steps > 0:
                accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))

            if completed_steps >= max_train_steps:
                break

    accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))
    if accelerator.is_main_process:
        checkpoint_path = os.path.join(config.output_dir, f"steps_{completed_steps}")
        logger.info(f"Training finished. Final checkpoint saved at {checkpoint_path}")


def main():
    config = TrainingConfig()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    train(config)

if __name__ == "__main__":
    main()
