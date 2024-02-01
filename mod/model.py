from logging import warning
import numpy as np
import torch
import torch.nn as nn

import transformers
from modt.models.model import TrajectoryModel
from diffuser import utils
from diffuser.models import Inpaint

from collections import namedtuple

DESIGNED_MOD_TYPE = ['bc', 'dd', 'dt']

Sample = namedtuple("Sample", "trajectories values chains")


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
        use_pref,
        concat_state_pref,
        concat_act_pref,
        concat_rtg_pref,
        n_layer, # TODO simplify these
        n_head,
        n_inner,
        n_positions,
        resid_pdrop,
        attn_pdrop,
        diffuser_args=None,
        mod_type='bc',
        infer_N=1,
        cond_M=0,
        batch_size=64,
        returns_condition=False,
        verbose=False,
    ):
        super().__init__(state_dim, act_dim, pref_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.use_pref = use_pref
        self.act_scale = act_scale
        self.concat_state_pref = concat_state_pref
        self.concat_act_pref = concat_act_pref
        self.eval_context_length = eval_context_length
        self.verbose = verbose

        # //suppose nothing is concated in advance (this means 'concat_<state|act>_pref' has a slightly different semantic)
        # self.state_dim = (
        #     state_dim + concat_state_pref * pref_dim
        # )  # already done in experiment.py
        self.state_dim = state_dim
        self.act_dim = act_dim + concat_act_pref * pref_dim   # TODO need concat a, p
        self.rtg_dim = pref_dim + concat_rtg_pref * pref_dim  # TODO need concat g, p

        assert mod_type in DESIGNED_MOD_TYPE, f'Should set MO Diffuser type as one of {DESIGNED_MOD_TYPE}'
        self.ar_inv = False
        self.mod_type = mod_type
        if self.mod_type == 'bc':
            self.act_fn = self._bc_get_action
            trans_dim = self.act_dim + self.state_dim # a, s (merely s may lead to low perf)
            # self.ar_inv = True
        elif self.mod_type == 'dd':
            self.act_fn = self._dd_get_action
            trans_dim = self.state_dim + self.rtg_dim  # s, g (a from InvDyn)
            self.ar_inv = True
        elif self.mod_type == 'dt':
            self.act_fn = self._dt_get_action
            trans_dim = self.act_dim + self.state_dim + self.rtg_dim  # a, s, g

        assert infer_N + cond_M == max_length
        self.infer_N = infer_N
        self.cond_M = cond_M
        
        self.batch_size = batch_size

        self.args = args = diffuser_args
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, "model_config"),
            horizon=max_length,
            transition_dim=trans_dim,
            dim=max_length,
            cond_dim=0,  # not used in TemporalUnet and ValueUnet
            rtg_dim=self.rtg_dim,
            dim_mults=args.dim_mults,
            attention=args.attention,
            device=args.device,
            returns_condition=returns_condition,
        )  # Unet

        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, "diffusion_config"),
            horizon=args.horizon,
            observation_dim=state_dim,
            action_dim=act_dim,
            pref_dim=pref_dim,
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
            condition_guidance_w=0.1,
            ar_inv=self.ar_inv,
            device=args.device,
        )

        self._model = model_config()

        self.diffusion = diffusion_config(self._model)

    def forward(self, cond, target_return, *args, **kwargs) -> Sample:
        # return -> Sample(<denoised traj>, <some? value>, <trajs chain>)
        # Just for sampling
        batch_size = target_return.shape[0]
        return self.diffusion.forward(cond, batch_size, target_return, *args, **kwargs)

    def get_action(self, states, actions, rtg, pref, timesteps):
        ''' Predict a_t using s_{t-N:t} and a_{t-N:t-1}, as in DMBP (# TODO: may need train also like in DMBP, i.e. a non-Markovian loss) '''
        rtg = rtg[-1]
        target_return = torch.ones((states.shape[0], rtg.shape[-1]), dtype=torch.float32, device=rtg.device) # Maximizing traj returns
        action = self.act_fn(states, actions, rtg, pref, timesteps, target_return)
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
        if a is not None:
            conds.update({'a' : Inpaint(0, self.cond_M - 1, a)})
        else:
            conds.update({'a': None})
            
        if s is not None:
            conds.update({'s' : Inpaint(0, self.cond_M - int(a is None), s)})
        else:
            conds.update({'s': None})
            
        if g is not None:
            conds.update({'g' : Inpaint(0, self.cond_M, g)})
        else:
            conds.update({'g': None})

        return conds

    def _bc_get_action(self, states, actions, rtg, pref, t, target_return):
        # Preprocess
        actions = self._pad_or_clip(actions)
        states = self._pad_or_clip(states)
        rtg = self._pad_or_clip(rtg)
        
        conds = self._make_cond(actions, states, None)
        traj_gen = self.forward(conds, target_return, verbose=self.verbose).trajectories
        action = traj_gen[0, self.cond_M - 1, : self.act_dim]

        return action

    def _dd_get_action(self, states, actions, rtg, pref, t, target_return): # Refer DD:Alg.1
        # Preprocess
        states = self._pad_or_clip(states)
        rtg = self._pad_or_clip(rtg)
        
        conds = self._make_cond(None, states, rtg)
        traj_gen = self.forward(conds, target_return, verbose=self.verbose).trajectories
        state_traj = traj_gen[:, :, :self.state_dim]
        s_t, s_t_1 = state_traj[:, -2, :], state_traj[:, -1, :]
        s_comb_t = torch.cat([s_t, s_t_1], dim=-1)
        action = self.diffusion.inv_model(s_comb_t)
        return action

    def _dt_get_action(self, states, actions, rtg, pref, t, target_return):
        # Preprocess
        actions = self._pad_or_clip(actions)
        states = self._pad_or_clip(states)
        rtg = self._pad_or_clip(rtg)
        
        conds = self._make_cond(actions, states, rtg)
        traj_gen = self.forward(conds, target_return, verbose=self.verbose).trajectories
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
