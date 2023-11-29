# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import transformers

# %%
from dataset import JSONLDataset

print("Loading Dataset")

tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 128
dataset = JSONLDataset('data/pile_data_truncated.jsonl', tokenizer)
# dataset = JSONLDataset('data/proj2_data.jsonl', tokenizer)

# print(len(dataset))
# print(dataset[60000])

print("Dataset Loaded")

# %%
from mingpt.model import GPT
from mingpt.trainer import Trainer

print("Instantiating Model")

model_config = GPT.get_default_config()
model_config.model_type = "gpt-mini"
model_config.vocab_size = tokenizer.vocab_size
model_config.block_size = tokenizer.model_max_length
model = GPT(model_config)

print("Instantiating Trainer")

trainer_config = Trainer.get_default_config()
trainer_config.max_iters = 40000
trainer_config.device = "cuda"
trainer_config.load_checkpoint = True
trainer_config.checkpoint_dir = "checkpoints"
trainer_config.checkpoint_iters = 25
trainer = Trainer(trainer_config, model, dataset)

def print_loss(trainer):
    if trainer.iter_num % 5 == 0:
        print(f"Batch: {trainer.iter_num}, Loss: {trainer.loss.item()}")

trainer.add_callback("on_batch_end", print_loss)

print("Starting Training")
# %%
trainer.run()

# %%



