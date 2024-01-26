from logging import warning
import numpy as np
import torch
import torch.nn as nn

import transformers
from modt.models.model import TrajectoryModel
from diffuser import utils
from diffuser.models import GaussianDiffusion

from collections import namedtuple

Sample = namedtuple("Sample", "trajectories values chains")


class DecisionDiffuser(TrajectoryModel):

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
        n_layer,
        n_head,
        n_inner,
        n_positions,
        resid_pdrop,
        attn_pdrop,
        diffuser_args=None,
    ):
        super().__init__(state_dim, act_dim, pref_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.use_pref = use_pref
        self.act_scale = act_scale
        self.concat_state_pref = concat_state_pref
        self.concat_act_pref = concat_act_pref
        self.eval_context_length = eval_context_length

        # //suppose nothing is concated in advance (this means 'concat_<state|act>_pref' has a slightly different semantic)
        # self.state_dim = (
        #     state_dim + concat_state_pref * pref_dim
        # )  # already done in experiment.py
        self.act_dim = act_dim + concat_act_pref * pref_dim

        self.args = args = diffuser_args
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, "model_config.pkl"),
            horizon=max_length,
            transition_dim=state_dim + act_dim,
            dim=max_length,
            cond_dim=0,
            dim_mults=args.dim_mults,
            attention=args.attention,
            device=args.device,
        )  # Unet

        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, "diffusion_config.pkl"),
            horizon=args.horizon,
            observation_dim=state_dim,
            action_dim=act_dim,
            n_timesteps=args.n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            # loss weighting
            action_weight=args.action_weight,
            loss_weights=args.loss_weights,
            loss_discount=args.loss_discount,
            device=args.device,
        )

        self._model = model_config()

        self.diffusion = diffusion_config(self._model)

    def forward(self, cond) -> Sample:
        # return -> Sample(<denoised traj>, <some? value>, <trajs chain>)
        # Just for sampling
        return self.diffusion.forward(cond)

    def get_action(self, states, actions, rtg, pref, t): # TODO: add a diffuion step for evaluation (much less than that in training)
        ''' Predict a_t using s_{t-N:t} and a_{t-N:t-1}, as in DMBP (TODO: may need train also like in DMBP, i.e. a non-Markovian loss) '''
        # obtain action from generated traj
        cond = {0: states} # TODO: to be further modified
        traj = self.forward(cond).trajectories
        # [[a, s] * H] * bs <=> (bs, H, a+s)
        action = traj[0, -1, : self.act_dim]

        return action
