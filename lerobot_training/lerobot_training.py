from functools import lru_cache
import sys
import pathlib

from torch.utils.data.dataset import ConcatDataset

_root = pathlib.Path(__file__).resolve().parent.parent  # repo root
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import math
import os
import logging
from typing import List, Any, Optional, Tuple, Dict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T_v2

from accelerate import Accelerator
from accelerate.logging import get_logger

# Use dynamic import to support different transformers versions for VLM base classes
from transformers import AutoProcessor, get_scheduler
from transformers import AutoModelForImageTextToText as AutoModelClass

import lerobot.processor
from lerobot.configs.types import PipelineFeatureType
from tqdm import tqdm

import load_datasets
from utils.data_loading import build_co_training_dataloader, collate_with_observation_image_lists

logger = get_logger(__name__)

# --- 1. Configuration ---
@dataclass
class TrainingConfig:
    per_device_batch_size: int = 256
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    num_warmup_steps: int = 1000
    max_epochs: int = 5
    output_dir: str = './nora_finetune_object'
    resume_from_checkpoint: str = ''
    load_model_weights: Optional[str] = None
    agibot_world_root: str = "data/agibot-world/tasks"
    galaxea_open_world_ds_root: str = "data/galaxea-open-world-dataset/subsets"
    interndata_a1_root: str = "data/interndata-a1/"
    wandb_project_name: str = "Nora VLA with LeRobotDataset"
    checkpoint_save_frequency: int = 20000
    logging_frequency: int = 100
    gradient_clipping: Optional[float] = None
    dataloader_num_workers: int = 4
    action_chunk_size: int = 50
    
    # Updated to use Gemma-4 E4B
    model_id: str = "google/gemma-4-E4B-it" 
    action_vocab_size: int = 2048

    # Gemma 4 image token budget
    max_tokens_per_image: int = 70
    
    # Number of frames to input (5 past + 1 current = 6)
    num_frames: int = 6
    
    # Flag for image augmentation
    image_augmentation: bool = False

    # --- New Config for Math VL Data ---
    math_vl_mixing_ratio: float = 0.2  # Mix 20% Math VL data
    
    math_vl_samples_per_dataset: Optional[int] = None 


# --- 2. Data Preprocessing & Transforms ---
def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format."""
    return ''.join([f"<robot_action_{token}>" for token in tokens])

class NoraImageTransform:
    """
    Image augmentation transforms for Nora policy training.
    Applies relative-size random crop and color jitter.
    """

    BRIGHTNESS_FACTOR = 0.2
    CONTRAST_FACTOR = 0.2
    SATURATION_FACTOR = 0.2
    HUE_FACTOR = 0.05
    CROP_SCALE = 0.9

    def __init__(self):
        self.color_jitter = T_v2.ColorJitter(
            brightness=self.BRIGHTNESS_FACTOR,
            contrast=self.CONTRAST_FACTOR,
            saturation=self.SATURATION_FACTOR,
            hue=self.HUE_FACTOR,
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def get_random_crop_transform(source_size: Tuple[int, int], scale: float) -> T_v2.RandomCrop:
        linear_scale = scale ** 0.5
        target_size = round(source_size[0] * linear_scale), round(source_size[1] * linear_scale)
        return T_v2.RandomCrop(target_size)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = self.get_random_crop_transform(image.shape[-2:], self.CROP_SCALE)(image)
        image = self.color_jitter(image)
        return image


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("nora_processor")
class NoraPolicyProcessorStep(lerobot.processor.ProcessorStep):
    config: TrainingConfig 
    transformer_processor: Any 
    action_token_min: int = -1
    action_token_max: int = -1

    IMAGE_KEYS = (
        'observation.images.head',
        'observation.images.hand_left',
        'observation.images.hand_right',
    )

    def __post_init__(self):
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            "lerobot/fast-action-tokenizer", trust_remote_code=True,
        )
        
        # Dynamically calculate Action Token range
        action_tokens = [f"<robot_action_{i}>" for i in range(self.config.action_vocab_size)]
        action_ids = self.transformer_processor.tokenizer.convert_tokens_to_ids(action_tokens)
        action_ids = [id for id in action_ids if id is not None and id != self.transformer_processor.tokenizer.unk_token_id]
        
        if action_ids:
            self.action_token_min = min(action_ids)
            self.action_token_max = max(action_ids)
        else:
            raise ValueError("Action Tokens not found in the Gemma vocabulary!")

        self.nora_image_transform = NoraImageTransform()
    def _process_robot_data(self, batch: lerobot.processor.EnvTransition, i: int, imgs_for_this_sample: List):
        first_cam = batch['observation.images.head'][i]
        T = first_cam.shape[0] if first_cam.dim() == 4 else 1
        
        # Extract historical frames and apply Transform
        for t in range(T):
            for k in self.IMAGE_KEYS:
                img_list = batch.get(k)
                if img_list is not None and img_list[i] is not None:
                    img_tensor = img_list[i]
                    frame = img_tensor[t] if img_tensor.dim() == 4 else img_tensor
                    transformed_frame = self.nora_image_transform(frame)
                    imgs_for_this_sample.append(transformed_frame)

        # Extract Action Tokens
        action = batch['action'][i]
        action_dim_is_pad = batch['action_dim_is_pad'][i]
        action = action[:, action_dim_is_pad.logical_not()] if action.dim() == 2 else action[action_dim_is_pad.logical_not()]
        fast_tokens = self.fast_tokenizer(action.cpu())[0]
        vlm_action = map_fast_token_to_vlm_action(fast_tokens)
        
        task = batch['task'][i]
        embodiment = batch['info'][i].get('embodiment_prompt', 'Generic Robot')

        content = [{"type": "text", "text": f"[embodiment: {embodiment}]\n"}]
        for _ in imgs_for_this_sample:
            content.append({"type": "image"})
        content.append({"type": "text", "text": task})

        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": vlm_action}]}
        ]
        return messages, imgs_for_this_sample
    def _process_math_vl_data(self, batch: lerobot.processor.EnvTransition, i: int, imgs_for_this_sample: List):
        head_img = batch['observation.images.head'][i]
        if head_img is not None:
            # Revert unsqueeze(0) if present depending on how pure collate works
            frame = head_img[0] if head_img.dim() == 4 else head_img
            imgs_for_this_sample.append(frame)

        task_instruction = batch['task'][i]
        comp_data = batch['complementary_data'][i] if batch.get('complementary_data') else {}
        text_answer = comp_data.get('text_answer', '')

        content = []
        for _ in imgs_for_this_sample:
            content.append({"type": "image"})
        content.append({"type": "text", "text": task_instruction})

        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": text_answer}]}
        ]
        return messages, imgs_for_this_sample
    def __call__(self, batch: lerobot.processor.EnvTransition) -> lerobot.processor.EnvTransition:
        """Processes a single batched EnvTransition."""
        batch_size = len(batch['task'])
        text_prompts = []
        batch_images = []
        is_robot_flags = []
        messages_list = []
        
        # 1. Route batched items to Robot Pipeline or Math VL Pipeline
        for i in range(batch_size):
            imgs_for_this_sample = []
            task_type = batch['info'][i].get('task_type', 'robot')
            is_robot = (task_type != 'vl_math')
            is_robot_flags.append(is_robot)

            if is_robot:
                messages, imgs_for_this_sample = self._process_robot_data(batch, i, imgs_for_this_sample)
            else:
                messages, imgs_for_this_sample = self._process_math_vl_data(batch, i, imgs_for_this_sample)

            batch_images.append(imgs_for_this_sample)
            messages_list.append(messages)

            # Get purely textual prompt with <image> placeholders
            prompt = self.transformer_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            text_prompts.append(prompt)

        # 2. Pass separated text and images into the processor
        batch_input = self.transformer_processor(
            text=text_prompts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        )
        
        labels = batch_input['input_ids'].clone()
        pad_token_id = self.transformer_processor.tokenizer.pad_token_id

        # 3. Dynamic Loss Masking
        for i in range(batch_size):
            seq = labels[i]
            if is_robot_flags[i]:
                # Robot: Mask out everything before the action tokens
                mask_seq = (seq >= self.action_token_min) & (seq <= self.action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    seq[:first_action_index] = -100
                else:
                    seq[:] = -100
            else:
                # Math VL: Dynamically find the prompt length and mask it to isolate answer loss
                prompt_messages = [messages_list[i][0]] # User message only
                prompt_str = self.transformer_processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                
                # Tokenize prefix to get its length
                prefix_inputs = self.transformer_processor(text=[prompt_str], images=[batch_images[i]] if batch_images[i] else None, return_tensors="pt")
                prefix_len = prefix_inputs['input_ids'].shape[1]
                
                pad_count = (seq == pad_token_id).sum().item() if pad_token_id is not None else 0
                if self.transformer_processor.tokenizer.padding_side == 'left':
                    seq[:pad_count + prefix_len] = -100
                else:
                    seq[:prefix_len] = -100

        # Mask out padding tokens
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
            
        batch_input['labels'] = labels

        return lerobot.processor.create_transition(complementary_data = batch_input)

    def transform_features(self, features):
        return {
            PipelineFeatureType.ACTION: {},
            PipelineFeatureType.OBSERVATION: {},
        }

def make_policy_processor(config: TrainingConfig, transformer_processor: Any) -> lerobot.processor.PolicyProcessorPipeline:
    return lerobot.processor.PolicyProcessorPipeline(
        steps = [NoraPolicyProcessorStep(config=config, transformer_processor=transformer_processor)],
        to_output=lambda tr: tr['complementary_data'],
    )


# --- 3. Model Initialization ---
def load_model_and_processor(config: TrainingConfig, accelerator: Accelerator):
    """Loads Gemma-4, its processor, injects action tokens, and applies embedding hotfix."""
    transformer_processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        max_soft_tokens=config.max_tokens_per_image,
        image_seq_length=config.max_tokens_per_image,
    )
    
    # Ensure pad token exists
    if transformer_processor.tokenizer.pad_token is None:
        transformer_processor.tokenizer.pad_token = transformer_processor.tokenizer.eos_token
    transformer_processor.tokenizer.padding_side = 'left'

    old_vocab_size = len(transformer_processor.tokenizer)
    
    # Inject action tokens
    action_tokens = [f"<robot_action_{i}>" for i in range(config.action_vocab_size)]
    transformer_processor.tokenizer.add_tokens(action_tokens, special_tokens=True)
    new_vocab_size = len(transformer_processor.tokenizer)
    
    accelerator.print(f"Added {len(action_tokens)} action tokens to Gemma-4 vocabulary.")
    accelerator.print(f"Vocab size resized: {old_vocab_size} -> {new_vocab_size}")

    model = AutoModelClass.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Standard API resizing
    model.resize_token_embeddings(new_vocab_size)
    if hasattr(model, 'config'):
        model.config.vocab_size = new_vocab_size

    # =====================================================================
    # [HOTFIX] Scan and force resize nested embedding layers missed by API
    # =====================================================================
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) and module.num_embeddings == old_vocab_size:
            accelerator.print(f"-> Hotfix: Resizing nested layer {name} from {old_vocab_size} to {new_vocab_size}")
            
            old_weight = module.weight.data
            new_weight = torch.empty((new_vocab_size, module.embedding_dim), 
                                     dtype=old_weight.dtype, 
                                     device=old_weight.device)
            new_weight[:old_vocab_size, :] = old_weight
            
            mean = old_weight.mean(dim=0)
            std = old_weight.std(dim=0)
            new_weight[old_vocab_size:, :] = torch.normal(
                mean.expand(new_vocab_size - old_vocab_size, -1), 
                std.expand(new_vocab_size - old_vocab_size, -1)
            )
            
            module.num_embeddings = new_vocab_size
            module.weight = torch.nn.Parameter(new_weight)
    # =====================================================================

    if config.load_model_weights:
        tensors = {}
        from safetensors import safe_open
        with safe_open(config.load_model_weights, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
        accelerator.print("Pretrained weights loaded.")

    return model, transformer_processor

# --- 4. Training Loop ---
def train(config: TrainingConfig):
    """Main training loop."""
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps, log_with="wandb")
    accelerator.dataloader_config.dispatch_batches = False
    logger.info(accelerator.state, main_process_only=False)

    accelerator.init_trackers(config.wandb_project_name, config=config)

    model, transformer_processor = load_model_and_processor(config, accelerator)

    with accelerator.main_process_first():
        agibot_world = load_datasets.load_agibot_world_dataset(
            root = config.agibot_world_root,
            canonical_action_chunk_size = config.action_chunk_size,
            num_frames = config.num_frames, 
        )
        galaxea_open_world_ds = load_datasets.load_galaxea_dataset(
            root = config.galaxea_open_world_ds_root,
            canonical_action_chunk_size = config.action_chunk_size,
            num_frames = config.num_frames, 
        )
        interndata_a1 = load_datasets.load_interndata_a1_dataset(
            root = config.interndata_a1_root,
            canonical_action_chunk_size = config.action_chunk_size,
            num_frames = config.num_frames, 
        )
        
        # Merge robot datasets
        robot_dataset = ConcatDataset([agibot_world, galaxea_open_world_ds, interndata_a1])
        
        # Load new Math VL dataset (now correctly defaults to full dataset)
        math_vl_dataset = load_datasets.load_math_reasoning_datasets(
            samples_per_dataset=config.math_vl_samples_per_dataset
        )
        
        policy_preprocessor = make_policy_processor(config, transformer_processor)
    def composed_collate_fn(instances: List[Dict]):
        # The imported original pure collator only batches
        batched_transition = collate_with_observation_image_lists(instances)
        # Process handles EnvTransition
        return policy_preprocessor(batched_transition)

    train_dataloader = build_co_training_dataloader(
        robot_dataset=robot_dataset,
        math_hf_dataset=math_vl_dataset,
        batch_size=config.per_device_batch_size,
        policy_processor=composed_collate_fn, 
        robot_ratio=(1.0 - config.math_vl_mixing_ratio), 
        num_workers=config.dataloader_num_workers
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
        eps=1e-8,
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    max_train_steps = len(train_dataloader) * config.max_epochs
    max_optim_steps = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps) * config.max_epochs
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=math.ceil(config.num_warmup_steps / config.gradient_accumulation_steps),
        num_training_steps=max_optim_steps
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)

    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        accelerator.print(f"Resumed from local checkpoint: {config.resume_from_checkpoint}")

    total_batch_size = config.per_device_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_optim_steps}")

    completed_steps = 0
    progress_bar = tqdm(range(completed_steps, max_train_steps), disable=not accelerator.is_local_main_process)

    while completed_steps < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)

                progress_bar.update(1)
                completed_steps += 1

                if accelerator.sync_gradients:
                    if config.gradient_clipping is not None:
                        accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)

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