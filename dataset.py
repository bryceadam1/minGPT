import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json

class JSONLDataset(Dataset):
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
        tokens = self.tokenizer(text, return_tensors='pt', max_length=self.tokenizer.model_max_length + 1, truncation=True)['input_ids'][0]
        x = tokens[:-1]
        y = tokens[1:]
        # Pad or truncate each on the left to the length self.tokenizer.model_max_length
        # if len(x) < self.tokenizer.model_max_length:
        #     x = F.pad(x, (self.tokenizer.model_max_length - len(x), 0), value=self.tokenizer.pad_token_id)
        # elif len(x) > self.tokenizer.model_max_length:
        #     x = x[-self.tokenizer.model_max_length:]
        # if len(y) < self.tokenizer.model_max_length:
        #     y = F.pad(y, (self.tokenizer.model_max_length - len(y), 0), value=self.tokenizer.pad_token_id)
        # elif len(y) > self.tokenizer.model_max_length:
        #     y = y[-self.tokenizer.model_max_length:]
        
        # Truncate each on the left to the length self.tokenizer.model_max_length
        # if len(x) > self.tokenizer.model_max_length:
        #     x = x[-self.tokenizer.model_max_length:]
        # if len(y) > self.tokenizer.model_max_length:
        #     y = y[-self.tokenizer.model_max_length:]
        return x, y
    
    def pad_collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return x, y
