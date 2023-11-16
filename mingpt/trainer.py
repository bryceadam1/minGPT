"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict
import os

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.checkpoint_iters = 100 # save a checkpoint every N iterations
        C.checkpoint_dir = None # directory to save checkpoints in
        C.load_checkpoint = True # load the latest checkpoint from the checkpoint path
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        assert config.checkpoint_iters is None or config.checkpoint_dir is not None, \
            "must specify a checkpoint path to save model checkpoints"
        
        assert config.load_checkpoint or config.checkpoint_dir is None, \
            "must specify a checkpoint path to load model checkpoints from"
        
        # load the latest checkpoint from the checkpoint path
        if config.load_checkpoint:
            if os.path.exists(os.path.join(config.checkpoint_dir, 'model.pt')):
                model.load_checkpoint(os.path.join(config.checkpoint_dir, 'model.pt'))
            
            if os.path.exists(os.path.join(config.checkpoint_dir, 'optim.pt')):
                self.optimizer.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, 'optim.pt')))
            
            # Load the random state
            if os.path.exists(os.path.join(config.checkpoint_dir, 'rng.pt')):
                torch.random.set_rng_state(torch.load(os.path.join(config.checkpoint_dir, 'rng.pt')))

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            if config.checkpoint_iters is not None and self.iter_num % config.checkpoint_iters == 0 and self.iter_num > 0:
                if not os.path.exists(config.checkpoint_dir):
                    os.makedirs(config.checkpoint_dir)
                model.save_checkpoint(os.path.join(config.checkpoint_dir, 'model.pt'))
                torch.save(self.optimizer.state_dict(), os.path.join(config.checkpoint_dir, 'optim.pt'))
                torch.save(torch.random.get_rng_state(), os.path.join(config.checkpoint_dir, 'rng.pt'))

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
        
        # Save the model and optimizer state
        if config.checkpoint_dir is not None:
            if not os.path.exists(config.checkpoint_dir):
                os.makedirs(config.checkpoint_dir)
            model.save_checkpoint(os.path.join(config.checkpoint_dir, 'model.pt'))
            torch.save(self.optimizer.state_dict(), os.path.join(config.checkpoint_dir, 'optim.pt'))
