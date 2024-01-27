import numpy as np
import torch
from modt.training.trainer import Trainer
from diffuser import utils
from collections import namedtuple
from copy import deepcopy

Batch = namedtuple('Batch', 'trajs, conds')

class DiffuserTrainer(Trainer):
    def __init__(
        self,
        model, # MODiffuser model
        optimizer,
        get_batch,
        loss_fn,
        dataset_min_prefs,
        dataset_max_prefs,
        dataset_min_raw_r,
        dataset_max_raw_r,
        dataset_min_final_r,
        dataset_max_final_r,
        scheduler=None,
        eval_fns=...,
        max_iter=0,
        n_steps_per_iter=0,
        eval_only=False,
        concat_rtg_pref=0,
        concat_act_pref=0,
        logsdir="./",
        use_p_bar=False,
    ):
        super().__init__(
            model,
            optimizer,
            get_batch,
            loss_fn,
            dataset_min_prefs,
            dataset_max_prefs,
            dataset_min_raw_r,
            dataset_max_raw_r,
            dataset_min_final_r,
            dataset_max_final_r,
            scheduler,
            eval_fns,
            max_iter,
            n_steps_per_iter,
            eval_only,
            concat_rtg_pref,
            concat_act_pref,
            logsdir,
            use_p_bar,
        )
        
        self.args = args = self.model.args
        trainer_config = utils.Config(
            utils.Trainer,
            savepath=(args.savepath, 'trainer_config.pkl'),
            # train_batch_size=args.batch_size,
            train_lr=args.learning_rate,
            gradient_accumulate_every=args.gradient_accumulate_every,
            ema_decay=args.ema_decay,
            sample_freq=args.sample_freq,
            save_freq=args.save_freq,
            label_freq=int(args.n_train_steps // args.n_saves),
            save_parallel=args.save_parallel,
            results_folder=args.savepath,
            bucket=args.bucket,
            n_reference=args.n_reference,
        )
        
        self.trainer = trainer_config(model.diffusion)

    def train_step(self):
        (
            states,
            actions,
            raw_return,
            rtg,
            timesteps,
            attention_mask,
            pref,
        ) = self.get_batch()
        # Prepare training batch
        as_trajs = torch.cat([actions, states], axis=-1) # TODO may also use rtg, timessteps, pref, ... {the only training setting}
        s_conds = {0: deepcopy(states[:, 0, :])}
        batch = Batch(trajs=as_trajs, conds=s_conds)
        
        # Invoke diffusion trainer
        # n_train_steps = int(self.args.n_train_steps)
        loss, infos = self.trainer.train(1, batch)

        # update logs
        # ...
        # print(f"infos: {infos}")
        
        return loss, infos
