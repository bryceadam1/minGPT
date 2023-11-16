# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import transformers

# %%
from dataset import JSONLDataset

tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 64
dataset = JSONLDataset('data/proj2_data.jsonl', tokenizer)

print(len(dataset))
print(dataset[60000])

# %%
from mingpt.model import GPT
from mingpt.trainer import Trainer

model_config = GPT.get_default_config()
model_config.model_type = "gpt-nano"
model_config.vocab_size = tokenizer.vocab_size
model_config.block_size = tokenizer.model_max_length
model = GPT(model_config)

trainer_config = Trainer.get_default_config()
trainer_config.max_iters = 10
trainer_config.device = "mps"
trainer_config.load_checkpoint = True
trainer_config.checkpoint_dir = "checkpoints"
trainer_config.checkpoint_iters = 1
trainer = Trainer(trainer_config, model, dataset)

def print_loss(trainer):
    if trainer.iter_num % 2 == 0:
        print(f"Batch: {trainer.iter_num}, Loss: {trainer.loss.item()}")

trainer.add_callback("on_batch_end", print_loss)

# %%
trainer.run()

# %%



