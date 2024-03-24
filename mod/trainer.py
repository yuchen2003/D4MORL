import numpy as np
import torch
from modt.training.trainer import Trainer
from diffuser import utils
from collections import namedtuple
from copy import deepcopy
from mod.model import MODiffuser

Batch = namedtuple('Batch', 'trajs conds prefs returns') # return can be reward or rtg
aBatch = namedtuple('ActionBatch', 'trajs actions conds prefs returns') # for invdyn
from mod.model import Inpaint

class DiffuserTrainer(Trainer):
    def __init__(
        self,
        model : MODiffuser,
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
        datapath=None,
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
            datapath=datapath,
        )

        self.args = args = self.model.args
        trainer_config = utils.Config(
            utils.Trainer,
            savepath=(args.savepath, 'trainer_config'),
            # train_batch_size=args.batch_size,
            train_lr=float(args.learning_rate),
            gradient_accumulate_every=args.gradient_accumulate_every,
            ema_decay=args.ema_decay,
            sample_freq=args.sample_freq,
            save_freq=args.save_freq,
            label_freq=int(args.n_train_steps // args.n_saves),
            save_parallel=args.save_parallel,
            results_folder=args.savepath,
            bucket=args.bucket,
            n_reference=args.n_reference,
            warmup_steps=self.model.warmup_steps,
        )

        self.trainer = trainer_config(model.diffusion)
        self.diffuser = model # MODiffuser
        self.state_dim = self.diffuser.state_dim
        self.act_dim = self.diffuser.act_dim
        self.rtg_dim = self.diffuser.rtg_dim
        self.pref_dim = self.diffuser.pref_dim

        self.mod_type = self.model.mod_type
        if self.mod_type == 'bc':
            self.batch_fn = self._bc_get_batch
        elif self.mod_type == 'dd':
            self.batch_fn = self._dd_get_batch
        elif self.mod_type == 'dt':
            self.batch_fn = self._dt_get_batch
            
        self.infer_N = self.model.infer_N
        self.cond_M = self.model.cond_M
        self.concat_on = self.model.concat_on
        self.mixup_step = self.diffuser.mixup_step
        self.mixup_num = self.diffuser.mixup_num

    def train_step(self, cur_step):
        use_mixup = (cur_step < self.mixup_step)
        s, a, r, g, t, mask, p = self.get_batch(use_mixup, self.mixup_num) # r, g is divided by scale
        g = g[:, :-1]

        # 1. all average
        # traj_returns = r.sum(1) / r.shape[1] # unweighted
        # traj_weighted_returns = torch.multiply(traj_returns, p[:, 0, :])
        
        # or 2. weighted average
        cur_r_weight = 10
        traj_returns = (r.sum(1) + (cur_r_weight - 1) * r[:, self.cond_M - 1, :]) / (r.shape[1] + cur_r_weight - 1)
        # max_r = torch.tensor(self.dataset_max_raw_r / 1000 / 500, dtype=torch.float32, device=s.device) # FIXME
        # max_returns = max_r.repeat(len(p) - len(traj_returns), 1)
        # traj_returns = torch.cat([traj_returns, max_returns])
        
        traj_weighted_returns = torch.multiply(traj_returns, p[:, 0, :])
        
        if self.concat_rtg_pref != 0:
            g = torch.cat((g, torch.cat([p] * self.concat_rtg_pref, dim=2)), dim=2)
            r = torch.cat((r, torch.cat([p] * self.concat_rtg_pref, dim=2)), dim=2)
        if self.concat_act_pref != 0:
            a = torch.cat((a, torch.cat([p] * self.concat_act_pref, dim=2)), dim=2)

        # Prepare training batch
        guidance_term = torch.cat([traj_weighted_returns], dim=-1) # weighted returns, rtg, pref
        # guidance_term = g[:, self.cond_M - 1]
        batch = self.batch_fn(s, a, r, g, t, mask, p[:, 0, :], guidance_term)

        # Invoke diffusion trainer
        loss, infos = self.trainer.train(1, batch)

        # update logs
        # ...
        # print(f"infos: {infos}")

        return loss, infos

    def _make_cond(self, a=None, s=None, g=None):
        conds = {}
        dim_start, dim_end = 0, 0
        
        if a is not None:
            dim_start = dim_end
            dim_end += self.act_dim
            conds.update({'a' : Inpaint(0, self.cond_M - 1, dim_start, dim_end, a)})
        else:
            conds.update({'a': None})
            
        if s is not None:
            dim_start = dim_end
            dim_end += self.state_dim
            conds.update({'s' : Inpaint(0, self.cond_M, dim_start, dim_end, s)})
        else:
            conds.update({'s': None})
            
        if g is not None:
            dim_start = dim_end
            dim_end += self.rtg_dim
            conds.update({'g' : Inpaint(0, self.cond_M - 1, dim_start, dim_end, g)})
        else:
            conds.update({'g': None})

        return conds
    
    def _bc_get_batch(self, s, a, r, g, t, mask, p, traj_r):
        as_trajs = torch.cat([a, s], dim=-1)
        conds = self._make_cond(a, s, None)
        return Batch(trajs=as_trajs, conds=conds, prefs=p, returns=traj_r)

    def _dd_get_batch(self, s, a, r, g, t, mask, p, traj_r):
        if self.concat_on == 'r':
            sg_trajs = torch.cat([s, r], dim=-1)
            conds = self._make_cond(None, s, r)
        elif self.concat_on == 'g':
            sg_trajs = torch.cat([s, g], dim=-1)
            conds = self._make_cond(None, s, g)
        elif self.concat_on == 's':
            sg_trajs = s
            conds = self._make_cond(None, s, None) # as that in DD
        else:
            raise ValueError
        return aBatch(trajs=sg_trajs, actions=a, conds=conds, prefs=p, returns=traj_r)

    def _dt_get_batch(self, s, a, r, g, t, mask, p, traj_r):
        if self.concat_on == 'r':
            asg_trajs = torch.cat([a, s, r], dim=-1)
            conds = self._make_cond(a, s, r)
        elif self.concat_on == 'g':
            asg_trajs = torch.cat([a, s, g], dim=-1)
            conds = self._make_cond(a, s, g)
        else:
            raise ValueError
        return Batch(trajs=asg_trajs, conds=conds, prefs=p, returns=traj_r)

