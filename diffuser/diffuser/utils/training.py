import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        # dataset,
        # renderer,
        ema_decay=0.995,
        # train_batch_size=32,
        train_lr=2e-4,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=1000,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./diffuser_results',
        n_reference=8,
        bucket=None,
        warmup_steps=10000,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        # self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        # self.dataset = dataset
        # self.dataloader = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        # ))
        # self.dataloader_vis = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        # ))
        # self.renderer = renderer
        self.optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=train_lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps+1)/warmup_steps, 1)
        )

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps, batch):

        timer = Timer()
        for step in range(n_train_steps):
            for _ in range(self.gradient_accumulate_every):
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # if self.step % self.save_freq == 0:
            #     label = self.step // self.label_freq * self.label_freq
            #     self.save(label)

            # if self.step % self.log_freq == 0:
            #     infos_str = ' | '.join([f'{key}: {val:8.10f}' for key, val in infos.items()])
            #     print(f'{self.step}: loss: {loss:8.10f} | {infos_str} | t: {timer():8.10f}', flush=True)

            # if self.step == 0 and self.sample_freq:
            #     self.render_reference(self.n_reference) # NOT IMPLEMENTED

            # if self.sample_freq and self.step % self.sample_freq == 0:
            #     self.render_samples() # NOT IMPLEMENTED

            self.step += 1
            
        return loss.item(), infos.items()

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#
