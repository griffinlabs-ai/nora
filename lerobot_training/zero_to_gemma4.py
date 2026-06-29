import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers import AutoModelForImageTextToText, AutoProcessor
import pathlib
import sys

from sys import argv

script_dir = pathlib.Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from lerobot_training import TrainingConfig, make_proprio_state_tokens

checkpoint_dir = argv[1]
config = TrainingConfig()

processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
action_tokens = [f"<robot_action_{i}>" for i in range(config.action_vocab_size)]
proprio_tokens = make_proprio_state_tokens(config.proprio_vocab_size)
processor.tokenizer.add_tokens(action_tokens + proprio_tokens, special_tokens=True)
new_vocab_size = len(processor.tokenizer)

model = AutoModelForImageTextToText.from_pretrained(
    config.model_id, dtype=torch.bfloat16
)
model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
model.config.vocab_size = new_vocab_size

state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)  # fp32
model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state_dict.items()}, strict=False)
model.save_pretrained("gemma-4-griffin-alpha")
