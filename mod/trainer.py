import numpy as np
import torch
from modt.training.trainer import Trainer
from diffuser import utils
from collections import namedtuple
from copy import deepcopy

Batch = namedtuple('Batch', 'trajs conds returns') # return can be reward or rtg
aBatch = namedtuple('ActionBatch', 'trajs actions conds returns') # for invdyn

class DiffuserTrainer(Trainer):
    def __init__(
        self,
        model,  # MODiffuser model
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
            savepath=(args.savepath, 'trainer_config'),
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
        self.diffuser = model # MODiffuser

        self.mod_type = self.model.mod_type
        if self.mod_type == 'bc':
            self.batch_fn = self._bc_get_batch
        elif self.mod_type == 'dd':
            self.batch_fn = self._dd_get_batch
        elif self.mod_type == 'dt':
            self.batch_fn = self._dt_get_batch
        elif self.mod_type == 'td':
            self.batch_fn = self._td_get_batch
            
        self.infer_N = self.model.infer_N
        self.cond_M = self.model.cond_M

    def train_step(self):
        s, a, r, g, t, mask, p = self.get_batch()
        g = g[:, :-1]
        traj_returns = r.sum(1) / r.shape[1]

        # Prepare training batch
        batch = self.batch_fn(s, a, r, g, t, mask, p, traj_returns)

        # Invoke diffusion trainer
        loss, infos = self.trainer.train(1, batch)

        # update logs
        # ...
        # print(f"infos: {infos}")

        return loss, infos

    def _bc_get_batch(self, s, a, r, g, t, mask, p, traj_r):
        as_trajs = torch.cat([a, s], dim=-1)
        conds = self.diffuser._make_cond(a, s, None)
        return Batch(trajs=as_trajs, conds=conds, returns=traj_r)

    def _dd_get_batch(self, s, a, r, g, t, mask, p, traj_r):
        sg_trajs = torch.cat([s, g], dim=-1)
        conds = self.diffuser._make_cond(None, s, g)
        return aBatch(trajs=sg_trajs, actions=a, conds=conds, returns=traj_r)

    def _dt_get_batch(self, s, a, r, g, t, mask, p, traj_r):
        asg_trajs = torch.cat([a, s, g], dim=-1)
        conds = self.diffuser._make_cond(a, s, g)
        return Batch(trajs=asg_trajs, conds=conds, returns=traj_r)

