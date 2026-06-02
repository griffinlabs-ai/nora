from datetime import timedelta
from functools import lru_cache
import sys
import resource
import pathlib
from collections import defaultdict

from torch.utils.data.dataset import ConcatDataset
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

_root = pathlib.Path(__file__).resolve().parent.parent  # repo root
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import math
import os
import logging
from typing import Iterator, List, Any, Optional, Sequence, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Sampler
import torchvision.transforms as T_v2

from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from accelerate.utils import TorchDynamoPlugin, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger

# Use dynamic import to support different transformers versions for VLM base classes
from transformers import AutoProcessor, get_scheduler
from transformers import AutoModelForImageTextToText as AutoModelClass

import lerobot.processor
from lerobot.configs.types import PipelineFeatureType
from tqdm import tqdm

import load_datasets
from utils.data_loading import collate_with_observation_image_lists

logger = get_logger(__name__)

# --- 1. Configuration ---
@dataclass
class TrainingConfig:
    per_device_batch_size: int = 32
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    num_warmup_steps: int = 25000
    max_epochs: int = 0.25
    output_dir: str = './griffin_alpha_finetune_object'
    resume_from_checkpoint: str = ''
    load_model_weights: Optional[str] = None
    agibot_world_root: str = "data/agibot-world/tasks"
    galaxea_open_world_ds_root: str = "data/galaxea-open-world-dataset"
    interndata_a1_root: str = "data/interndata-a1/"
    egodex_root: str = "data/egodex/train"
    droid_root: str = "data/droid_1.0.1"
    wandb_project_name: str = "Griffin Alpha"
    checkpoint_save_frequency: int = 20000
    logging_frequency: int = 100
    gradient_clipping: Optional[float] = None
    dataloader_num_workers: int = 8
    action_chunk_size: int = 50
    model_id: str = "google/gemma-4-E4B-it"
    action_vocab_size: int = 2048
    proprio_vocab_size: int = 256    
    # Standard vision target size
    image_target_size: Tuple[int, int] = (224, 224) 
    # Number of image frames to input (5 past + 1 current = 6)
    num_frames: int = 1
    dataset_sample_ratios: Tuple[float, float, float, float] = (0.70, 0.05, 0.05, 0.15)
    dataloader_sampler_seed: int = 42


class WeightedConcatRandomSampler(Sampler[int]):
    """
    Sample ConcatDataset child datasets by ratio without storing per-sample weights.

    Dataset ids are sampled with replacement according to ``sample_ratios``. Once a dataset
    id is chosen, the local sample index is sampled uniformly from that child dataset.
    """

    def __init__(
        self,
        datasets: Sequence[torch.utils.data.Dataset],
        sample_ratios: Sequence[float],
        num_samples: int | None = None,
        seed: int = 42,
    ):
        if len(datasets) != len(sample_ratios):
            raise ValueError(
                f"Expected one sample ratio per dataset, got {len(sample_ratios)} ratios "
                f"for {len(datasets)} datasets."
            )

        self.lengths = [len(dataset) for dataset in datasets]
        if any(length <= 0 for length in self.lengths):
            raise ValueError(f"All datasets must be non-empty, got lengths {self.lengths}.")

        ratio_tensor = torch.as_tensor(sample_ratios, dtype=torch.float64)
        if not torch.isfinite(ratio_tensor).all():
            raise ValueError(f"Dataset sample ratios must be finite, got {sample_ratios}.")
        if (ratio_tensor < 0).any():
            raise ValueError(f"Dataset sample ratios must be non-negative, got {sample_ratios}.")
        if ratio_tensor.sum().item() <= 0:
            raise ValueError(f"At least one dataset sample ratio must be positive, got {sample_ratios}.")

        self.probabilities = ratio_tensor / ratio_tensor.sum()
        self.cumulative_offsets = [0]
        for length in self.lengths[:-1]:
            self.cumulative_offsets.append(self.cumulative_offsets[-1] + length)

        self.num_samples = sum(self.lengths) if num_samples is None else num_samples
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}.")

        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        epoch = self.epoch
        self.epoch += 1

        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)

        for _ in range(self.num_samples):
            segment_index = int(torch.multinomial(self.probabilities, 1, generator=generator).item())
            local_index = int(torch.randint(self.lengths[segment_index], (), generator=generator).item())
            yield self.cumulative_offsets[segment_index] + local_index


# --- 2. Data Preprocessing & Transforms ---
def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format."""
    return ''.join([f"<robot_action_{token}>" for token in tokens])

def make_proprio_state_tokens(vocab_size: int) -> List[str]:
    return [f"<proprio_state_{i}>" for i in range(vocab_size)]

def map_normalized_state_to_vlm_proprio(state: torch.Tensor, vocab_size: int) -> str:
    """Quantizes normalized proprio state values and maps them to VLM tokens."""
    if vocab_size <= 0:
        raise ValueError(f"proprio_vocab_size must be positive, got {vocab_size}")

    clipped = state.clamp(-1.0, 1.0)
    bucket_ids = torch.floor((clipped + 1.0) * (vocab_size / 2.0)).to(torch.long)
    bucket_ids = bucket_ids.clamp(0, vocab_size - 1)
    return ''.join(f"<proprio_state_{bucket_id}>" for bucket_id in bucket_ids.reshape(-1).tolist())

@dataclass
class NoraImageTransform:
    target_size: Tuple[int, int]

    def __post_init__(self):
        self.color_jitter = T_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        self.crop_transform = T_v2.RandomResizedCrop(
            size=self.target_size,
            scale=(0.9, 1.0),
            ratio=(0.9, 1.1),
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = self.crop_transform(image)
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

        proprio_tokens = make_proprio_state_tokens(self.config.proprio_vocab_size)
        proprio_ids = self.transformer_processor.tokenizer.convert_tokens_to_ids(proprio_tokens)
        proprio_ids = [
            id for id in proprio_ids
            if id is not None and id != self.transformer_processor.tokenizer.unk_token_id
        ]
        if len(proprio_ids) != self.config.proprio_vocab_size:
            raise ValueError("Proprio state tokens not found in the Gemma vocabulary!")

        self.nora_image_transform = NoraImageTransform(
            target_size = self.config.image_target_size
        )

    def __call__(self, transition) -> lerobot.processor.EnvTransition:
        batch_size = transition['action'].shape[0]
        obs_dict = transition['observation'] if 'observation' in transition else transition
        
        text_prompts = []
        batch_images = []
        
        # 1. Process images and construct text prompts
        for i in range(batch_size):
            imgs_for_this_sample = []
            
            first_cam = obs_dict[self.IMAGE_KEYS[0]][i]
            T = first_cam.shape[0] if first_cam.dim() == 4 else 1
            
            # Extract historical frames and apply Transform
            for t in range(T):
                for k in self.IMAGE_KEYS:
                    img_tensor = obs_dict[k][i]
                    if img_tensor is not None:
                        frame = img_tensor[t] if img_tensor.dim() == 4 else img_tensor
                        transformed_frame = self.nora_image_transform(frame)
                        imgs_for_this_sample.append(transformed_frame)
            
            batch_images.append(imgs_for_this_sample)

            # 2. Extract Action Tokens
            action = transition['action'][i]
            n_action_dims = int(transition['info']['n_action_dims'][i])
            action = action[:, :n_action_dims]
            fast_tokens = self.fast_tokenizer(action.cpu())[0]
            vlm_action = map_fast_token_to_vlm_action(fast_tokens)

            proprio_state = obs_dict['observation.state'][i][:n_action_dims]
            vlm_proprio = map_normalized_state_to_vlm_proprio(
                proprio_state,
                self.config.proprio_vocab_size,
            )
            
            task = transition['complementary_data']['task'][i]
            subtask = transition['complementary_data']['subtask'][i]
            embodiment = transition['info']['embodiment_prompt'][i]
            arm_control_mode = transition['info']['arm_control_mode'][i]

            # 3. Construct the message with image placeholders ONLY
            content = [{
                "type": "text",
                "text": f"[embodiment: {embodiment}; arm control mode: {arm_control_mode}] {vlm_proprio}",
            }]
            for _ in imgs_for_this_sample:
                content.append({"type": "image"})
            content.append({"type": "text", "text": f"{task}\npredict subtask: {'true' if subtask else 'false'}"})

            subtask_segment = f"subtask: {subtask}\n" if subtask else ""
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [
                    {"type": "text", "text": f"{subtask_segment}action: {vlm_action}"},
                ]}
            ]
            
            # Get purely textual prompt with <image> placeholders
            prompt = self.transformer_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            text_prompts.append(prompt)

        # 4. Pass separated text and images into the processor
        batch_input = self.transformer_processor(
            text=text_prompts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        )
        
        labels = batch_input['input_ids'].clone()

        # Mask out everything before the final start-of-turn token to calculate loss only on model output
        sot_token_id = self.transformer_processor.tokenizer.sot_token_id
        for i in range(labels.size(0)):
            seq = labels[i]
            sot_indices = (seq == sot_token_id).nonzero(as_tuple=False)
            if sot_indices.numel() > 0:
                last_sot_index = sot_indices[-1].item()
                seq[:last_sot_index] = -100
            else:
                seq[:] = -100

        # Mask out padding tokens
        pad_token_id = self.transformer_processor.tokenizer.pad_token_id
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
    with accelerator.main_process_first():
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
    
    # Inject action and proprio state tokens
    action_tokens = [f"<robot_action_{i}>" for i in range(config.action_vocab_size)]
    proprio_tokens = make_proprio_state_tokens(config.proprio_vocab_size)
    transformer_processor.tokenizer.add_tokens(action_tokens + proprio_tokens, special_tokens=True)
    new_vocab_size = len(transformer_processor.tokenizer)
    
    accelerator.print(
        f"Added {len(action_tokens)} action and {len(proprio_tokens)} proprio state tokens to Gemma-4 vocabulary."
    )
    accelerator.print(f"Vocab size resized: {old_vocab_size} -> {new_vocab_size}")

    with accelerator.main_process_first():
        model = AutoModelClass.from_pretrained(
            config.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=defaultdict(lambda: accelerator.device),
        )

    # Resize token embedding layers
    accelerator.print("Resizing token embedding layer.")
    # Make sure other processes will be busy too,
    # so they won't be waiting for the main process and end up timing out
    accelerator.wait_for_everyone()
    model.resize_token_embeddings(new_vocab_size)
    if hasattr(model, 'config'):
        model.config.vocab_size = new_vocab_size
    accelerator.print("Done resizing token embedding layer.")

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

    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model, transformer_processor


# --- 4. Training Loop ---
def train(config: TrainingConfig):
    """Main training loop."""
    # increase the number of open files limit to accommodate large datasets
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, hard))

    # try to use faster float32 matmul
    torch.set_float32_matmul_precision('high')

    dynamo_plugin = TorchDynamoPlugin(
        backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
        mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
        fullgraph=False,
        dynamic=False
    )
    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version = 2,
        reshard_after_forward=False,
        auto_wrap_policy="transformer_based_wrap",
        state_dict_type="SHARDED_STATE_DICT",
        transformer_cls_names_to_wrap=[
            "Gemma4TextModel",
            "Gemma4TextScaledWordEmbedding",
            "Gemma4VisionEncoderLayer",
            "Gemma4AudioLayer",
        ]
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
        dynamo_plugin=dynamo_plugin,
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(seconds=1800)),
            ## If switching to DDP:
            # DistributedDataParallelKwargs(find_unused_parameters=True),
        ],
    )
    accelerator.dataloader_config.dispatch_batches = False
    logger.info(accelerator.state, main_process_only=False)

    accelerator.init_trackers(config.wandb_project_name, config=config)

    model, transformer_processor = load_model_and_processor(config, accelerator)

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
    droid = load_datasets.load_droid_dataset(
        root = config.droid_root,
        canonical_action_chunk_size = config.action_chunk_size,
        num_frames = config.num_frames, 
    )
    dataset_names = ("AgiBotWorld-Beta", "Galaxea A1", "InternVLA simulation", "DROID")
    source_datasets = [agibot_world, galaxea_open_world_ds, interndata_a1, droid]
    dataset = ConcatDataset(source_datasets)
    train_sampler = WeightedConcatRandomSampler(
        source_datasets,
        config.dataset_sample_ratios,
        seed=config.dataloader_sampler_seed,
    )
    accelerator.print(f"Total number of frames in dataset: {len(dataset)}")
    for name, source_dataset, sample_ratio in zip(
        dataset_names,
        source_datasets,
        train_sampler.probabilities.tolist(),
        strict=True,
    ):
        accelerator.print(f"  {name}: {len(source_dataset)} frames, target sample ratio {sample_ratio:.1%}")
    accelerator.print(f"Weighted sampler epoch length: {len(train_sampler)} samples")

    with accelerator.main_process_first():
        policy_preprocessor = make_policy_processor(config, transformer_processor)

    train_dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=lambda examples: policy_preprocessor(collate_with_observation_image_lists(examples)),
        sampler=train_sampler,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
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

    n_units = sum(1 for m in model.modules() if isinstance(m, FSDPModule))
    accelerator.print(f"FSDP units: {n_units}")

    # Per-rank local parameter count (DTensor .to_local() gives the shard)
    local = sum(p.to_local().numel() if isinstance(p, DTensor) else p.numel()
                for p in model.parameters())
    accelerator.print(f"Local params on rank {accelerator.process_index}: {local/1e9:.2f}B")

    max_train_steps = math.floor(len(train_dataloader) * config.max_epochs)
    max_optim_steps = math.floor(math.ceil(len(train_dataloader) / config.gradient_accumulation_steps) * config.max_epochs)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=math.ceil(config.num_warmup_steps / config.gradient_accumulation_steps * accelerator.num_processes),
        num_training_steps=max_optim_steps
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)

    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        accelerator.print(f"Resumed from local checkpoint: {config.resume_from_checkpoint}")

    total_batch_size = config.per_device_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
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