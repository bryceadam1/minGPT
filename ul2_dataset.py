import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import json

class UL2Dataset(Dataset):
    def __init__(self, path, tokenizer):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        
        self.len = 0
        self.line_starts = [0]
        # self.line_num_items = []
        # self.line_idx = [0]
        with open(self.path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = json.loads(line)
                # text = line['text']
                # num_tokens = len(self.tokenizer(text, max_length=int(1e30), truncation=True)['input_ids'])
                # # There are num_tokens - 1 ways to predict the next token
                # self.len += num_tokens - 1
                # self.line_num_items.append(num_tokens - 1)
                # self.line_idx.append(self.len)
                self.line_starts.append(f.tell())
                self.len += 1

        self.line_starts.pop()
        # self.line_idx.pop()

    def get_line_loc(self, idx):
        # Find the largest element in line_idx that is smaller than or equal to idx
        # This is the line that contains idx
        line_idx = self.line_idx
        line_starts = self.line_starts
        lo = 0
        hi = len(line_idx) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_idx[mid] > idx:
                hi = mid - 1
            else:
                lo = mid
        return line_starts[lo], idx - line_idx[lo]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # line_start, idx_in_line = self.get_line_loc(idx)
        line_start = self.line_starts[idx]
        with open(self.path, 'r') as f:
            f.seek(line_start)
            line = f.readline()
        line = json.loads(line)
        text = line['text']
        # tokens = self.tokenizer(text, return_tensors='pt', max_length=int(1e30), truncation=True)['input_ids'][0]
        tokens = self.tokenizer(text, max_length=3*(self.tokenizer.model_max_length)//4, truncation=True)['input_ids']
        
        denoising_task = np.random.choice(["R", "X", "S"], p=[0.25, 0.25, 0.5])

        if denoising_task == "S":
            input = self.tokenizer("<S>")["input_ids"] + self.tokenizer("<extra_id_0>")["input_ids"]
            random_index = np.random.randint(len(tokens)//4, 3*len(tokens)//4)
            input = input + tokens[random_index:] + self.tokenizer("<B>")["input_ids"]
            input = input + self.tokenizer("<extra_id_0>")["input_ids"] + tokens[:random_index] + self.tokenizer("<E>")["input_ids"]
            input = torch.Tensor(input).long()
            x = input[:-1]
            y = input[1:]
        else:
            if denoising_task == "R":
                mu = 3
                span = 2
                corruption = .15
            elif denoising_task == "X":
                if np.random.choice([True, False]) and len(tokens) > 55:
                    mu = 32
                    span = 16
                    corruption = .15
                else:
                    mu = 3
                    span = 2
                    corruption = .5

            input = self.tokenizer("<R>")["input_ids"]
            output = self.tokenizer("<B>")["input_ids"]
            remove = 0
            extra_id = 0
            for i in range(len(tokens)):
                if remove == 0 and np.random.rand() < corruption:
                    remove = np.random.randint(mu-span, mu+span+1)
                    input = input + self.tokenizer(f"<extra_id_{extra_id}>")["input_ids"]
                    output = output + self.tokenizer(f"<extra_id_{extra_id}>")["input_ids"]
                    extra_id += 1
                
                if remove > 0:
                    remove -= 1
                    output.append(tokens[i])
                else:
                    input.append(tokens[i])
            output = output + self.tokenizer("<E>")["input_ids"]
                    
            sequence = torch.tensor(input + output).long()
            x = sequence[:-1]
            y = sequence[1:]
        
        # Truncate x and y if necessary
        if len(x) > self.tokenizer.model_max_length:
            x = x[:self.tokenizer.model_max_length]
            y = y[:self.tokenizer.model_max_length]
        return x, y
    
    def pad_collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return x, y
