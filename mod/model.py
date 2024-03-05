from logging import warning
import numpy as np
import torch
import torch.nn as nn

import transformers
from modt.models.model import TrajectoryModel
from diffuser import utils

from collections import namedtuple

DESIGNED_MOD_TYPE = ['bc', 'dd', 'dt']

Sample = namedtuple("Sample", "trajectories values chains")
Inpaint = namedtuple("InpaintConfig", "traj_start traj_end dim_start dim_end target")

class MODiffuser(TrajectoryModel):

    """
    This model uses Decision Diffuser to model ([state pref] action) * maxlen
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        pref_dim,
        hidden_size,
        max_length,
        eval_context_length,
        max_ep_len,
        act_scale,
        scale,
        use_pref,
        concat_state_pref,
        concat_act_pref,
        concat_rtg_pref,
        diffuser_args=None,
        mod_type='bc',
        infer_N=1,
        cond_M=0,
        batch_size=64,
        returns_condition=False,
        condition_guidance_w=0.1,
        concat_on='r', 
        verbose=False,
        warmup_steps=10000,
    ):
        super().__init__(state_dim, act_dim, pref_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.use_pref = use_pref
        self.act_scale = act_scale
        self.scale = scale
        self.concat_state_pref = concat_state_pref
        self.concat_act_pref = concat_act_pref
        self.concat_rtg_pref = concat_rtg_pref
        self.eval_context_length = eval_context_length
        self.verbose = verbose
        self.warmup_steps = warmup_steps

        self.state_dim = state_dim
        self.act_dim = act_dim + concat_act_pref * pref_dim   
        self.pref_dim = pref_dim
        self.rtg_dim = pref_dim + concat_rtg_pref * pref_dim 
        
        self.concat_on = concat_on 

        assert mod_type in DESIGNED_MOD_TYPE, f'Should set MO Diffuser type as one of {DESIGNED_MOD_TYPE}'
        self.ar_inv = False
        self.mod_type = mod_type
        if self.mod_type == 'bc':
            self.act_fn = self._bc_get_action
            trans_dim = self.act_dim + self.state_dim # a, s
        elif self.mod_type == 'dd':
            self.act_fn = self._dd_get_action
            trans_dim = self.state_dim + self.rtg_dim  # s, g (a from InvDyn)
            # self.ar_inv = True
        elif self.mod_type == 'dt':
            self.act_fn = self._dt_get_action
            trans_dim = self.act_dim + self.state_dim + self.rtg_dim  # a, s, g

        assert infer_N + cond_M == max_length
        self.infer_N = infer_N
        self.cond_M = cond_M
        self.gen_H = max_length
        
        self.batch_size = batch_size

        self.args = args = diffuser_args
        self.device = args.device
        
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, "model_config"),
            horizon=max_length,
            transition_dim=trans_dim,
            dim=args.dim,
            cond_dim=self.pref_dim, # weighted returns, rtg, pref(and dont use concat_rtg_pref)
            pref_dim=self.pref_dim,
            dim_mults=args.dim_mults,
            attention=args.attention,
            device=args.device,
            returns_condition=returns_condition,
        )  # Unet

        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, "diffusion_config"),
            horizon=args.horizon,
            cond_M = self.cond_M,
            observation_dim=self.state_dim,
            action_dim=self.act_dim,
            pref_dim=self.pref_dim,
            rtg_dim=self.rtg_dim,
            trans_dim=trans_dim,
            hidden_dim=hidden_size,
            mod_type=mod_type,
            n_timesteps=args.n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            # loss weighting
            action_weight=args.action_weight,
            loss_weights=args.loss_weights,
            loss_discount=args.loss_discount,
            returns_condition=returns_condition,
            condition_guidance_w=condition_guidance_w,
            ar_inv=self.ar_inv,
            device=args.device,
        )

        self._model = model_config()

        self.diffusion = diffusion_config(self._model)

    def forward(self, cond, prefs, target_return, *args, **kwargs) -> Sample:
        # return -> Sample(<denoised traj>, <some? value>, <trajs chain>)
        # Just for sampling
        batch_size = target_return.shape[0]
        return self.diffusion.forward(cond, batch_size, prefs, target_return, *args, **kwargs)

    def get_action(self, states, actions, rtg, prefs, timesteps, max_r):
        ''' Predict a_t using s_{t-N:t} and a_{t-N:t-1}, as in DMBP '''
        rtg = rtg[-self.max_length : ]
        # print(prefs.shape)
        if self.concat_on == 'r':
            target_r = torch.ones(1, self.max_length, self.rtg_dim, device=states.device, dtype=torch.float32)
            target_r[:, :, :self.pref_dim] = torch.multiply(max_r / self.scale, target_r[:, :, :self.pref_dim])
        elif self.concat_on == 'g':
            target_r = torch.zeros(1, self.max_length, self.rtg_dim, device=states.device, dtype=torch.float32)
            target_r[0, -rtg.shape[0]:, :rtg.shape[1]] = rtg
        else:
            raise ValueError
        
        target_weighted_returns = torch.multiply(max_r / self.scale, prefs[-1]) # == ones x pref, size==(bs, rtg_dim); as DD does
        if self.concat_rtg_pref != 0:
            target_r[:, :, -self.pref_dim:] = prefs[0]
            # target_weighted_returns = torch.cat((target_weighted_returns, torch.cat([prefs] * self.concat_rtg_pref, dim=1)), dim=1)
        if self.concat_act_pref != 0:
            actions = torch.cat((actions, torch.cat([prefs] * self.concat_rtg_pref, dim=1)), dim=1)
        
        guidance_terms = torch.cat([target_weighted_returns], dim=-1).view(1, -1) # weighted returns, rtg, pref
        action = self.act_fn(states, actions, target_r, prefs[[-1]], timesteps, guidance_terms)
        if self.concat_act_pref:
            return action[ : -self.pref_dim]
        else:
            return action

    def _pad_or_clip(self, traj):
        traj_dim = traj.shape[-1]
        traj = traj.reshape(1, -1, traj_dim)
        if traj.shape[1] < self.max_length:
            traj = torch.cat(
                [
                    torch.zeros(
                        (1, self.max_length - traj.shape[1], traj_dim),
                        dtype=torch.float32,
                        device=traj.device,
                    ),
                    traj,  # the last one(s)
                ],
                dim=1,
            )
        elif traj.shape[1] > self.max_length:
            traj = traj[:, -self.max_length:, :]

        return traj

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
            conds.update({'s' : Inpaint(0, self.cond_M - int(a is None), dim_start, dim_end, s)})
        else:
            conds.update({'s': None})
            
        # if g is not None:
        #     dim_start = dim_end
        #     dim_end += self.rtg_dim
        #     conds.update({'g' : Inpaint(0, self.cond_M, dim_start, dim_end, g)})
        # else:
        #     conds.update({'g': None})
        conds.update({'g': None})

        return conds

    def _bc_get_action(self, states, actions, target_r, pref, t, target_return):
        # Preprocess
        actions = self._pad_or_clip(actions)
        states = self._pad_or_clip(states)
        
        conds = self._make_cond(actions, states, None)
        traj_gen = self.forward(conds, pref, target_return, self.gen_H, verbose=self.verbose).trajectories
        action = traj_gen[0, self.cond_M - 1, : self.act_dim]

        return action

    def _dd_get_action(self, states, actions, target_r, pref, t, target_return): # Refer DD:Alg.1
        # Preprocess
        states = self._pad_or_clip(states)
        target_r = self._pad_or_clip(target_r)
        
        conds = self._make_cond(None, states, target_r)
        traj_gen = self.forward(conds, pref, target_return, self.gen_H, verbose=self.verbose).trajectories
        state_traj = traj_gen[:, :, :self.state_dim]
        s_t, s_t_1 = state_traj[[-1], -2, :], state_traj[[-1], -1, :]
        s_comb_t = torch.cat([s_t, s_t_1], dim=-1)
        action = self.diffusion.inv_model(s_comb_t)[-1]
        return action

    def _dt_get_action(self, states, actions, target_r, pref, t, target_return):
        # Preprocess
        actions = self._pad_or_clip(actions)
        states = self._pad_or_clip(states)
        target_r = self._pad_or_clip(target_r)
        
        conds = self._make_cond(actions, states, target_r)
        traj_gen = self.forward(conds, pref, target_return, self.gen_H, verbose=self.verbose).trajectories
        action = traj_gen[0, self.cond_M - 1, : self.act_dim]
        
        return action
    
    # Save model parameters
    def save_model(self, file_name):
        # logger.info('Saving models to {}'.format(file_name))
        print("Saving models to {}".format(file_name))
        torch.save(
            {
                "diffusion": self.diffusion,
            },
            file_name,
        )

    # Load model parameters
    def load_model(self, filename, device_idx=0, evaluate=False):
        # logger.info(f'Loading models from {filename}')
        print(f"Loading models from {filename}")
        if filename is not None:
            if device_idx == -1:
                checkpoint = torch.load(filename, map_location=f"cpu")
            else:
                checkpoint = torch.load(filename, map_location=f"cuda:{device_idx}")

            self.diffusion.load_state_dict(checkpoint["diffusion"])
            if evaluate:
                self.diffusion.eval()
            else:
                self.diffusion.train()
